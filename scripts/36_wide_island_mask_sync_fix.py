#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import statistics
import sys
import time
import types
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import mlx.core as mx
import torch
import transformers
from PIL import Image
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor


def load_module(path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load helper script: {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    try:
        spec.loader.exec_module(mod)
    except Exception:
        sys.modules.pop(module_name, None)
        raise
    return mod


def percentile(xs: list[float], p: float) -> float:
    ys = sorted(xs)
    if len(ys) == 1:
        return ys[0]
    k = (len(ys) - 1) * p
    f, c = math.floor(k), math.ceil(k)
    if f == c:
        return ys[f]
    return ys[f] * (c - k) + ys[c] * (k - f)


def stats(xs: list[float]) -> dict[str, float]:
    return {
        "n": len(xs),
        "mean_ms": statistics.fmean(xs),
        "median_ms": statistics.median(xs),
        "p90_ms": percentile(xs, 0.90),
        "p95_ms": percentile(xs, 0.95),
        "min_ms": min(xs),
        "max_ms": max(xs),
    }


def tensor_error(a: torch.Tensor, b: torch.Tensor) -> dict[str, Any]:
    af = a.detach().float()
    bf = b.detach().float()
    d = (af - bf).abs()
    return {
        "shape": list(a.shape),
        "max_abs": float(d.max().item()),
        "mean_abs": float(d.mean().item()),
        "rmse": float(torch.sqrt(torch.mean((af - bf) ** 2)).item()),
        "allclose_1e-5": bool(torch.allclose(a, b, rtol=1e-5, atol=1e-5)),
        "allclose_1e-4": bool(torch.allclose(a, b, rtol=1e-4, atol=1e-4)),
        "finite": bool(torch.isfinite(a).all().item() and torch.isfinite(b).all().item()),
    }


def mask_aware_error(a: torch.Tensor, b: torch.Tensor) -> dict[str, Any]:
    a = a.detach()
    b = b.detach()
    fa = torch.isfinite(a)
    fb = torch.isfinite(b)
    common = fa & fb

    if bool(common.any().item()):
        af = a[common].float()
        bf = b[common].float()
        d = (af - bf).abs()
        max_abs = float(d.max().item())
        mean_abs = float(d.mean().item())
        rmse = float(torch.sqrt(torch.mean((af - bf) ** 2)).item())
    else:
        max_abs = mean_abs = rmse = 0.0

    return {
        "shape": list(a.shape),
        "finite_mask_equal": bool(torch.equal(fa, fb)),
        "nan_mask_equal": bool(torch.equal(torch.isnan(a), torch.isnan(b))),
        "posinf_mask_equal": bool(torch.equal(torch.isposinf(a), torch.isposinf(b))),
        "neginf_mask_equal": bool(torch.equal(torch.isneginf(a), torch.isneginf(b))),
        "finite_max_abs": max_abs,
        "finite_mean_abs": mean_abs,
        "finite_rmse": rmse,
        "allclose_1e-5": bool(torch.allclose(a, b, rtol=1e-5, atol=1e-5)),
        "allclose_1e-4": bool(torch.allclose(a, b, rtol=1e-4, atol=1e-4)),
    }


class MlxFusionDeformableWideIslandFixed:
    """
    Corrected wide island.

    Critical change from Experiment 0034:
      - import the already-produced key_padding_mask after a producer sync
      - derive deformable's valid-token mask INSIDE MLX

    This avoids:
      torch.mps.synchronize()
      -> create (~key_padding_mask) on MPS
      -> immediately import that still-pending tensor into MLX

    which violates cross-framework producer/consumer ordering.
    """

    def __init__(self, fusion_layer, fusion_core, deform_core):
        if fusion_layer.training:
            raise ValueError("Wide island requires model.eval().")

        self.fusion_core = fusion_core
        self.deform_core = deform_core

        self.v_ln_w = mx.asarray(fusion_layer.layer_norm_vision.weight, copy=False)
        self.v_ln_b = mx.asarray(fusion_layer.layer_norm_vision.bias, copy=False)
        self.t_ln_w = mx.asarray(fusion_layer.layer_norm_text.weight, copy=False)
        self.t_ln_b = mx.asarray(fusion_layer.layer_norm_text.bias, copy=False)
        self.v_ln_eps = float(fusion_layer.layer_norm_vision.eps)
        self.t_ln_eps = float(fusion_layer.layer_norm_text.eps)

        self.vision_param = mx.asarray(fusion_layer.vision_param, copy=False)
        self.text_param = mx.asarray(fusion_layer.text_param, copy=False)

        mx.eval(
            self.v_ln_w,
            self.v_ln_b,
            self.t_ln_w,
            self.t_ln_b,
            self.vision_param,
            self.text_param,
        )
        mx.synchronize()

        self.compiled_min = mx.compile(self._forward_min)

    def _fusion(
        self,
        vision,
        text,
        key_padding_mask,
        text_mask,
    ):
        v = mx.fast.layer_norm(
            vision, self.v_ln_w, self.v_ln_b, self.v_ln_eps
        )
        t = mx.fast.layer_norm(
            text, self.t_ln_w, self.t_ln_b, self.t_ln_eps
        )

        delta_v, _vision_attn, delta_t, _text_attn = (
            self.fusion_core._forward_impl(
                v,
                t,
                key_padding_mask,
                text_mask,
            )
        )

        v = v + self.vision_param * delta_v
        t = t + self.text_param * delta_t
        return v, t

    def _forward_min(
        self,
        vision,
        text,
        key_padding_mask,
        text_mask,
        position,
        reference_points,
    ):
        v, t = self._fusion(
            vision,
            text,
            key_padding_mask,
            text_mask,
        )

        # HF GroundingDinoEncoderLayer passes:
        #   fusion:      key_padding_mask      (True = padding)
        #   deformable: ~key_padding_mask      (True = valid)
        #
        # Do the inversion inside MLX so there is no new pending PyTorch/MPS
        # producer op after the producer synchronization.
        deform_valid_mask = mx.logical_not(key_padding_mask)

        v_out, _deform_attn = self.deform_core._forward_impl(
            v,
            deform_valid_mask,
            position,
            reference_points,
        )
        return v_out, t


@dataclass
class LayerCallState:
    baseline_calls: int = 0
    fixed_calls: int = 0


class FixedWideLayerDispatcher:
    def __init__(self, layer, fixed_island, state: LayerCallState):
        self.layer = layer
        self.fixed_island = fixed_island
        self.state = state
        self.original_forward = layer.forward
        self.mode = "baseline"

        def dispatched(
            _layer_self,
            vision_features,
            vision_position_embedding,
            spatial_shapes,
            spatial_shapes_list,
            level_start_index,
            key_padding_mask,
            reference_points,
            text_features=None,
            text_attention_mask=None,
            text_position_embedding=None,
            text_self_attention_masks=None,
            text_position_ids=None,
        ):
            if self.mode == "baseline":
                self.state.baseline_calls += 1
                return self.original_forward(
                    vision_features=vision_features,
                    vision_position_embedding=vision_position_embedding,
                    spatial_shapes=spatial_shapes,
                    spatial_shapes_list=spatial_shapes_list,
                    level_start_index=level_start_index,
                    key_padding_mask=key_padding_mask,
                    reference_points=reference_points,
                    text_features=text_features,
                    text_attention_mask=text_attention_mask,
                    text_position_embedding=text_position_embedding,
                    text_self_attention_masks=text_self_attention_masks,
                    text_position_ids=text_position_ids,
                )

            self.state.fixed_calls += 1

            text_position_embedding_resolved = (
                _layer_self.get_text_position_embeddings(
                    text_features,
                    text_position_embedding,
                    text_position_ids,
                )
            )

            # IMPORTANT: do not create any new PyTorch/MPS producer tensors
            # after this synchronization before importing into MLX.
            torch.mps.synchronize()

            v_mx = mx.asarray(vision_features, copy=False)
            t_mx = mx.asarray(text_features, copy=False)
            key_padding_mx = mx.asarray(key_padding_mask, copy=False)
            text_mask_mx = (
                None
                if text_attention_mask is None
                else mx.asarray(text_attention_mask, copy=False)
            )
            pos_mx = mx.asarray(vision_position_embedding, copy=False)
            ref_mx = mx.asarray(reference_points, copy=False)

            v_out_mx, t_fused_mx = self.fixed_island.compiled_min(
                v_mx,
                t_mx,
                key_padding_mx,
                text_mask_mx,
                pos_mx,
                ref_mx,
            )
            mx.eval(v_out_mx, t_fused_mx)
            mx.synchronize()

            vision_out = torch.as_tensor(v_out_mx)
            text_fused = torch.as_tensor(t_fused_mx)
            torch.mps.synchronize()

            text_out, text_enhanced_attn = _layer_self.text_enhancer_layer(
                hidden_states=text_fused,
                attention_masks=~text_self_attention_masks,
                position_embeddings=(
                    text_position_embedding_resolved
                    if text_position_embedding_resolved is not None
                    else None
                ),
            )

            return (
                (vision_out, text_out),
                (None, None, text_enhanced_attn, None),
            )

        layer.forward = types.MethodType(dispatched, layer)

    def set_mode(self, mode: str):
        if mode not in ("baseline", "fixed"):
            raise ValueError(mode)
        self.mode = mode

    def restore(self):
        self.layer.forward = self.original_forward


def set_mode(dispatchers, mode: str):
    for d in dispatchers:
        d.set_mode(mode)


def run_timed(model, inputs, sync_fn):
    sync_fn()
    t0 = time.perf_counter_ns()
    with torch.inference_mode():
        out = model(**inputs)
    sync_fn()
    return out, (time.perf_counter_ns() - t0) / 1e6


def pair_audit(a, b, model, h14):
    return {
        "logits": mask_aware_error(a.logits, b.logits),
        "pred_boxes": mask_aware_error(a.pred_boxes, b.pred_boxes),
        "topk": h14.topk_audit(a, b, model.config.num_queries),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="IDEA-Research/grounding-dino-tiny")
    ap.add_argument("--image", type=Path, default=Path("assets/input.jpg"))
    ap.add_argument("--prompt", nargs="+", default=["a cat", "a dog"])
    ap.add_argument("--threadgroup", type=int, default=256)
    ap.add_argument("--warmup-per-mode", type=int, default=3)
    ap.add_argument("--pairs", type=int, default=20)
    ap.add_argument(
        "--exp13-helper",
        type=Path,
        default=Path("scripts/13_bench_fusion_algebraic_specialization.py"),
    )
    ap.add_argument(
        "--exp14-helper",
        type=Path,
        default=Path("scripts/14_full_model_fusion_algebraic.py"),
    )
    ap.add_argument(
        "--exp17-helper",
        type=Path,
        default=Path("scripts/17_bench_deformable_mlx_island.py"),
    )
    ap.add_argument(
        "--exp18-helper",
        type=Path,
        default=Path("scripts/18_full_model_deformable_islands.py"),
    )
    ap.add_argument(
        "--exp30-helper",
        type=Path,
        default=Path("scripts/30_bench_fully_folded_fusion_mlx.py"),
    )
    ap.add_argument(
        "--exp31-helper",
        type=Path,
        default=Path("scripts/31_full_model_mlx_fusion_paired.py"),
    )
    args = ap.parse_args()

    for p in (
        args.exp13_helper,
        args.exp14_helper,
        args.exp17_helper,
        args.exp18_helper,
        args.exp30_helper,
        args.exp31_helper,
    ):
        if not p.exists():
            raise SystemExit(f"Missing helper: {p}")

    if not torch.backends.mps.is_available():
        raise SystemExit("MPS unavailable.")

    h13 = load_module(args.exp13_helper, "mg36_h13")
    h14 = load_module(args.exp14_helper, "mg36_h14")
    h17 = load_module(args.exp17_helper, "mg36_h17")
    h18 = load_module(args.exp18_helper, "mg36_h18")
    h30 = load_module(args.exp30_helper, "mg36_h30")
    h31 = load_module(args.exp31_helper, "mg36_h31")

    mx.set_default_device(mx.gpu)

    print("Loading model...", flush=True)
    processor = AutoProcessor.from_pretrained(args.model)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(args.model)
    model.eval().to("mps")
    h14.sync()

    if bool(model.config.output_attentions):
        raise RuntimeError("Experiment 0036 requires output_attentions=False.")

    image = Image.open(args.image).convert("RGB")
    text_labels = [args.prompt]
    cpu_inputs = processor(images=image, text=text_labels, return_tensors="pt")
    inputs = {k: v.to("mps") for k, v in cpu_inputs.items()}
    h14.sync()

    print("Original PyTorch/MPS oracle...", flush=True)
    with torch.inference_mode():
        original_ref = model(**inputs)
        h14.sync()

    msda_state = h14.patch_msda(model, args.threadgroup)

    print("Building six robust compiled-MLX fusion paths...", flush=True)
    fusion_dispatchers = []
    for i, layer in enumerate(model.model.encoder.layers):
        torch_spec = h13.AlgebraicBiMHA(layer.fusion_layer.attn)
        mlx_spec = h30.MlxFullyFoldedFusion(torch_spec)
        fd = h31.FusionDispatcher(
            layer.fusion_layer.attn,
            torch_spec,
            mlx_spec,
        )
        fd.set_mode("mlx")
        fusion_dispatchers.append(fd)
        print(f"  fusion {i}: ready", flush=True)

    # Capture geometry before deformable patching.
    layer0_deform = model.model.encoder.layers[0].deformable_layer
    original_deform_forward = layer0_deform.forward
    captured = {}

    def capture_deform(
        self,
        hidden_states,
        attention_mask,
        position_embeddings=None,
        reference_points=None,
        spatial_shapes=None,
        spatial_shapes_list=None,
        level_start_index=None,
        output_attentions=False,
    ):
        if not captured:
            captured["spatial_shapes_list"] = [
                (int(h), int(w)) for h, w in spatial_shapes_list
            ]
        return original_deform_forward(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_embeddings=position_embeddings,
            reference_points=reference_points,
            spatial_shapes=spatial_shapes,
            spatial_shapes_list=spatial_shapes_list,
            level_start_index=level_start_index,
            output_attentions=output_attentions,
        )

    layer0_deform.forward = types.MethodType(
        capture_deform, layer0_deform
    )
    with torch.inference_mode():
        _ = model(**inputs)
        h14.sync()
    layer0_deform.forward = original_deform_forward

    if not captured:
        raise RuntimeError("Failed to capture spatial shapes.")

    print("Building six current deformable MLX islands...", flush=True)
    island_state, deform_islands = h18.patch_encoder_deformable_islands(
        model,
        h17.MlxDeformableIsland,
        activation_name=model.config.activation_function,
        spatial_shapes_list=captured["spatial_shapes_list"],
        threadgroup=args.threadgroup,
    )

    print("Building six corrected wide islands...", flush=True)
    fixed_islands = []
    for i, layer in enumerate(model.model.encoder.layers):
        fixed_islands.append(
            MlxFusionDeformableWideIslandFixed(
                layer.fusion_layer,
                fusion_dispatchers[i].mlx_specializer,
                deform_islands[i],
            )
        )
        print(f"  fixed wide {i}: ready", flush=True)

    state = LayerCallState()
    layer_dispatchers = [
        FixedWideLayerDispatcher(layer, fixed, state)
        for layer, fixed in zip(model.model.encoder.layers, fixed_islands)
    ]

    try:
        # Stable separate baseline.
        set_mode(layer_dispatchers, "baseline")
        with torch.inference_mode():
            separate_ref = model(**inputs)
            h14.sync()

        # --------------------------------------------------------------
        # Same-input local audit for all six corrected layers.
        # Return baseline downstream to prevent trajectory contamination.
        # --------------------------------------------------------------
        local_audit = []
        print("\n=== Corrected same-input local audit ===", flush=True)

        for i, (layer, dispatcher) in enumerate(
            zip(model.model.encoder.layers, layer_dispatchers)
        ):
            set_mode(layer_dispatchers, "baseline")
            dispatched_forward = layer.forward
            record = {}

            def make_wrapper(bound_forward, bound_dispatcher, rec):
                def wrapper(_self, *f_args, **f_kwargs):
                    bound_dispatcher.set_mode("baseline")
                    base = bound_forward(*f_args, **f_kwargs)
                    h14.sync()

                    bound_dispatcher.set_mode("fixed")
                    cand = bound_forward(*f_args, **f_kwargs)
                    h14.sync()

                    bound_dispatcher.set_mode("baseline")

                    base_v, base_t = base[0]
                    cand_v, cand_t = cand[0]
                    rec["vision"] = tensor_error(cand_v, base_v)
                    rec["text"] = tensor_error(cand_t, base_t)
                    return base

                return wrapper

            layer.forward = types.MethodType(
                make_wrapper(dispatched_forward, dispatcher, record),
                layer,
            )

            with torch.inference_mode():
                out = model(**inputs)
                h14.sync()

            layer.forward = dispatched_forward

            if not record:
                raise RuntimeError(f"Local audit layer {i} did not fire.")

            downstream_topk = h14.topk_audit(
                out, separate_ref, model.config.num_queries
            )
            row = {
                "layer": i,
                "same_input_fixed_vs_separate": record,
                "returned_baseline_downstream_topk": downstream_topk,
            }
            local_audit.append(row)

            print(
                f"layer {i}: "
                f"vision max={record['vision']['max_abs']:.3e}, "
                f"vision 1e-4={record['vision']['allclose_1e-4']}, "
                f"text max={record['text']['max_abs']:.3e}, "
                f"text 1e-4={record['text']['allclose_1e-4']}",
                flush=True,
            )

        local_pass = all(
            row["same_input_fixed_vs_separate"]["vision"]["allclose_1e-4"]
            and row["same_input_fixed_vs_separate"]["text"]["allclose_1e-4"]
            for row in local_audit
        )

        # --------------------------------------------------------------
        # Full-model corrected wide candidate.
        # --------------------------------------------------------------
        print("\n=== Corrected six-wide full-model preflight ===", flush=True)
        set_mode(layer_dispatchers, "fixed")
        with torch.inference_mode():
            fixed_ref = model(**inputs)
            h14.sync()

        correctness = {
            "fixed_vs_separate": pair_audit(
                fixed_ref, separate_ref, model, h14
            ),
            "separate_vs_original": pair_audit(
                separate_ref, original_ref, model, h14
            ),
            "fixed_vs_original": pair_audit(
                fixed_ref, original_ref, model, h14
            ),
        }

        print(
            json.dumps(
                {
                    k: v["topk"]
                    for k, v in correctness.items()
                },
                indent=2,
            ),
            flush=True,
        )

        topk_pass = (
            correctness["fixed_vs_separate"]["topk"]["set_overlap"] == 900
        )

        # Only time if the correctness bug is actually fixed.
        latency = None
        derived = None
        call_validation = None

        if local_pass and topk_pass:
            print(
                "\nCorrectness gate passed; running paired latency benchmark...",
                flush=True,
            )

            for i in range(args.warmup_per_mode):
                set_mode(layer_dispatchers, "baseline")
                _ = run_timed(model, inputs, h14.sync)
                set_mode(layer_dispatchers, "fixed")
                _ = run_timed(model, inputs, h14.sync)
                print(
                    f"  warmup pair {i+1}/{args.warmup_per_mode}",
                    flush=True,
                )

            state.baseline_calls = 0
            state.fixed_calls = 0
            for fd in fusion_dispatchers:
                fd.reset_counts()
            island_state.calls = 0
            msda_state.call_count = 0

            baseline_samples = []
            fixed_samples = []
            deltas = []

            for i in range(args.pairs):
                order = (
                    ("baseline", "fixed")
                    if i % 2 == 0
                    else ("fixed", "baseline")
                )
                local = {}

                for mode in order:
                    set_mode(layer_dispatchers, mode)
                    _out, dt = run_timed(model, inputs, h14.sync)
                    local[mode] = dt
                    if mode == "baseline":
                        baseline_samples.append(dt)
                    else:
                        fixed_samples.append(dt)

                delta = local["baseline"] - local["fixed"]
                deltas.append(delta)
                print(
                    f"  pair {i+1:02d}/{args.pairs}: "
                    f"separate={local['baseline']:.3f} ms, "
                    f"fixed={local['fixed']:.3f} ms, "
                    f"delta={delta:+.3f} ms",
                    flush=True,
                )

            b = stats(baseline_samples)
            f = stats(fixed_samples)
            dd = stats(deltas)
            latency = {
                "separate_baseline": b,
                "corrected_wide": f,
                "paired_delta_ms_separate_minus_fixed": dd,
            }
            derived = {
                "speedup_from_medians": b["median_ms"] / f["median_ms"],
                "median_difference_of_marginals_ms":
                    b["median_ms"] - f["median_ms"],
                "paired_delta_median_ms": dd["median_ms"],
                "paired_delta_mean_ms": dd["mean_ms"],
            }

            fusion_counts = h31.sum_counts(fusion_dispatchers)
            expected_each = args.pairs * 6
            expected_decoder = args.pairs * 2 * 6

            call_validation = {
                "baseline_layer_calls": state.baseline_calls,
                "fixed_layer_calls": state.fixed_calls,
                "baseline_fusion_mlx_calls": fusion_counts["mlx_calls"],
                "baseline_deformable_island_calls": island_state.calls,
                "decoder_metal_msda_calls": msda_state.call_count,
                "expected_each_encoder_mode": expected_each,
                "expected_decoder_metal_msda_calls": expected_decoder,
            }

            if state.baseline_calls != expected_each:
                raise RuntimeError(
                    f"baseline layer calls {state.baseline_calls} != {expected_each}"
                )
            if state.fixed_calls != expected_each:
                raise RuntimeError(
                    f"fixed layer calls {state.fixed_calls} != {expected_each}"
                )
            if fusion_counts["mlx_calls"] != expected_each:
                raise RuntimeError(
                    f"baseline fusion calls {fusion_counts['mlx_calls']} != {expected_each}"
                )
            if island_state.calls != expected_each:
                raise RuntimeError(
                    f"baseline deform calls {island_state.calls} != {expected_each}"
                )
            if msda_state.call_count != expected_decoder:
                raise RuntimeError(
                    f"decoder MSDA calls {msda_state.call_count} != {expected_decoder}"
                )
        else:
            print(
                "\nCorrectness gate FAILED; latency timing intentionally skipped.",
                flush=True,
            )

        result = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "experiment": "0036",
            "purpose": (
                "Causal fix test for Experiment 0034/0035: eliminate an "
                "unsynchronized PyTorch-MPS mask inversion created after the "
                "producer sync by deriving the deformable valid-token mask "
                "inside the compiled MLX wide island."
            ),
            "model": args.model,
            "device": "mps",
            "dtype": "float32",
            "software": {
                "torch": torch.__version__,
                "transformers": transformers.__version__,
            },
            "bug_hypothesis": {
                "old_sequence": [
                    "torch.mps.synchronize()",
                    "create ~key_padding_mask on MPS",
                    "mx.asarray(copy=False) immediately imports new mask",
                    "MLX consumes buffer without producer synchronization",
                ],
                "fixed_sequence": [
                    "torch.mps.synchronize()",
                    "import already-produced key_padding_mask into MLX",
                    "derive logical_not(key_padding_mask) inside MLX graph",
                ],
            },
            "configuration": {
                "prompt_cache": False,
                "output_attentions": False,
                "warmup_per_mode": args.warmup_per_mode,
                "paired_iterations_if_correct": args.pairs,
                "threadgroup": args.threadgroup,
                "approximation": False,
                "retraining": False,
                "reduced_precision": False,
            },
            "corrected_local_same_input_audit": local_audit,
            "local_all_layers_allclose_1e-4": local_pass,
            "full_model_correctness": correctness,
            "full_model_topk_set_preserved": topk_pass,
            "latency": latency,
            "derived": derived,
            "call_validation": call_validation,
            "decision_rule": (
                "The synchronization hypothesis is supported if corrected "
                "same-input vision/text outputs return to the established "
                "1e-4 envelope in all six layers and full-model top-900 "
                "membership is restored. If so, retain the corrected wide "
                "island and evaluate its paired latency; otherwise continue "
                "semantic debugging and do not adopt."
            ),
            "notes": [
                "HF GroundingDinoEncoderLayer uses key_padding_mask for fusion and its logical inverse for deformable attention.",
                "MLX DLPack import does not synchronize pending Metal producer work; explicit producer synchronization is required before sharing.",
                "The fixed path performs the mask inversion inside MLX after importing the synchronized padding mask.",
                "No approximation, quantization, pruning, retraining, or reduced precision is used.",
                "Dataset-level accuracy equivalence remains unmeasured."
            ],
        }

        out = Path(
            "results/metalground_wide_island_mask_sync_fix.json"
        )
        out.write_text(
            json.dumps(result, indent=2, ensure_ascii=False) + "\n"
        )

        print("\n=== Experiment 0036 summary ===", flush=True)
        print(f"local all-layer 1e-4 pass: {local_pass}", flush=True)
        print(f"full-model top-k set preserved: {topk_pass}", flush=True)
        if derived is not None:
            print(
                f"paired delta median: "
                f"{derived['paired_delta_median_ms']:+.3f} ms",
                flush=True,
            )
        print(f"Saved: {out}", flush=True)

    finally:
        for d in layer_dispatchers:
            d.restore()
        for fd in fusion_dispatchers:
            fd.restore()


if __name__ == "__main__":
    main()
