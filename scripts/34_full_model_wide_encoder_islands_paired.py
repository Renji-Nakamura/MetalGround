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


@dataclass
class WideState:
    baseline_layer_calls: int = 0
    wide_layer_calls: int = 0


class EncoderLayerWideDispatcher:
    """
    Baseline:
        original HF encoder-layer forward, with:
          - compiled MLX fully-folded fusion attention
          - compiled MLX deformable island

    Wide:
        text-position construction (PyTorch)
        -> one MLX fusion+deformable island
        -> PyTorch text enhancer
        -> return

    In the original layer, after fusion:
        text enhancer reads/writes only the text branch;
        deformable reads/writes only the vision branch.
    Therefore those two post-fusion branches commute in inference.
    """

    def __init__(self, layer, wide_island, state: WideState):
        self.layer = layer
        self.wide_island = wide_island
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
                self.state.baseline_layer_calls += 1
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

            self.state.wide_layer_calls += 1

            # Preserve the encoder layer's text-position semantics.
            text_position_embedding_resolved = (
                _layer_self.get_text_position_embeddings(
                    text_features,
                    text_position_embedding,
                    text_position_ids,
                )
            )

            # One conservative Torch -> MLX boundary for both branches.
            torch.mps.synchronize()

            v_mx = mx.asarray(vision_features, copy=False)
            t_mx = mx.asarray(text_features, copy=False)
            vm_mx = mx.asarray(key_padding_mask, copy=False)
            tm_mx = (
                None
                if text_attention_mask is None
                else mx.asarray(text_attention_mask, copy=False)
            )
            deform_mask_mx = mx.asarray(~key_padding_mask, copy=False)
            pos_mx = mx.asarray(vision_position_embedding, copy=False)
            ref_mx = mx.asarray(reference_points, copy=False)

            v_out_mx, t_fused_mx = self.wide_island.compiled_min(
                v_mx,
                t_mx,
                vm_mx,
                tm_mx,
                deform_mask_mx,
                pos_mx,
                ref_mx,
            )
            mx.eval(v_out_mx, t_fused_mx)
            mx.synchronize()

            vision_out = torch.as_tensor(v_out_mx)
            text_fused = torch.as_tensor(t_fused_mx)
            torch.mps.synchronize()

            # Original order is fusion -> text enhancer -> deformable.
            # After fusion, text enhancer and deformable operate on independent
            # text/vision branches, so executing the vision branch inside the
            # wide island before this text enhancer is semantically equivalent.
            text_out, text_enhanced_attn = _layer_self.text_enhancer_layer(
                hidden_states=text_fused,
                attention_masks=~text_self_attention_masks,
                position_embeddings=(
                    text_position_embedding_resolved
                    if text_position_embedding_resolved is not None
                    else None
                ),
            )

            # Default inference has output_attentions=False at encoder level,
            # so the outer encoder ignores these placeholders. The text
            # enhancer attention is kept because it is already produced by the
            # untouched PyTorch text enhancer.
            attentions = (
                None,
                None,
                text_enhanced_attn,
                None,
            )
            return (vision_out, text_out), attentions

        layer.forward = types.MethodType(dispatched, layer)

    def set_mode(self, mode: str):
        if mode not in ("baseline", "wide"):
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


def correctness_bundle(candidate, baseline, original, model, h14, processor, inputs, text_labels, image_size, box_threshold, text_threshold):
    def pair(a, b):
        return {
            "logits": mask_aware_error(a.logits, b.logits),
            "pred_boxes": mask_aware_error(a.pred_boxes, b.pred_boxes),
            "topk": h14.topk_audit(a, b, model.config.num_queries),
        }

    return {
        "wide_vs_separate_baseline": pair(candidate, baseline),
        "separate_baseline_vs_original": pair(baseline, original),
        "wide_vs_original": pair(candidate, original),
        "detections": {
            "original": h14.detection_summary(
                processor,
                original,
                inputs["input_ids"],
                text_labels,
                image_size,
                box_threshold,
                text_threshold,
            ),
            "separate_baseline": h14.detection_summary(
                processor,
                baseline,
                inputs["input_ids"],
                text_labels,
                image_size,
                box_threshold,
                text_threshold,
            ),
            "wide": h14.detection_summary(
                processor,
                candidate,
                inputs["input_ids"],
                text_labels,
                image_size,
                box_threshold,
                text_threshold,
            ),
        },
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="IDEA-Research/grounding-dino-tiny")
    ap.add_argument("--image", type=Path, default=Path("assets/input.jpg"))
    ap.add_argument("--prompt", nargs="+", default=["a cat", "a dog"])
    ap.add_argument("--warmup-per-mode", type=int, default=3)
    ap.add_argument("--pairs", type=int, default=20)
    ap.add_argument("--threadgroup", type=int, default=256)
    ap.add_argument("--box-threshold", type=float, default=0.3)
    ap.add_argument("--text-threshold", type=float, default=0.25)
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
    ap.add_argument(
        "--exp33-helper",
        type=Path,
        default=Path("scripts/33_bench_fusion_deformable_wide_island.py"),
    )
    args = ap.parse_args()

    for p in (
        args.exp13_helper,
        args.exp14_helper,
        args.exp17_helper,
        args.exp18_helper,
        args.exp30_helper,
        args.exp31_helper,
        args.exp33_helper,
    ):
        if not p.exists():
            raise SystemExit(f"Missing helper: {p}")

    if not torch.backends.mps.is_available():
        raise SystemExit("MPS unavailable.")

    h13 = load_module(args.exp13_helper, "mg34_h13")
    h14 = load_module(args.exp14_helper, "mg34_h14")
    h17 = load_module(args.exp17_helper, "mg34_h17")
    h18 = load_module(args.exp18_helper, "mg34_h18")
    h30 = load_module(args.exp30_helper, "mg34_h30")
    h31 = load_module(args.exp31_helper, "mg34_h31")
    h33 = load_module(args.exp33_helper, "mg34_h33")

    mx.set_default_device(mx.gpu)

    print("Loading model...", flush=True)
    processor = AutoProcessor.from_pretrained(args.model)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(args.model)
    model.eval().to("mps")
    h14.sync()

    # This patch is designed specifically for the default inference path.
    if bool(model.config.output_attentions):
        raise RuntimeError(
            "Experiment 0034 requires config.output_attentions=False."
        )

    image = Image.open(args.image).convert("RGB")
    text_labels = [args.prompt]
    cpu_inputs = processor(images=image, text=text_labels, return_tensors="pt")
    inputs = {k: v.to("mps") for k, v in cpu_inputs.items()}
    h14.sync()

    print("Original PyTorch/MPS correctness reference...", flush=True)
    with torch.inference_mode():
        original_ref = model(**inputs)
        h14.sync()

    # Decoder keeps the existing custom Metal MSDA. Encoder MSDA patches will
    # be bypassed by the deformable islands.
    msda_state = h14.patch_msda(model, args.threadgroup)

    # Build six robust compiled-MLX fusion paths from Experiment 0032.
    print("Building six compiled-MLX fully-folded fusion paths...", flush=True)
    fusion_dispatchers = []
    for i, layer in enumerate(model.model.encoder.layers):
        torch_spec = h13.AlgebraicBiMHA(layer.fusion_layer.attn)
        mlx_spec = h30.MlxFullyFoldedFusion(torch_spec)
        dispatcher = h31.FusionDispatcher(
            layer.fusion_layer.attn,
            torch_spec,
            mlx_spec,
        )
        dispatcher.set_mode("mlx")
        fusion_dispatchers.append(dispatcher)
        print(f"  fusion layer {i}: ready", flush=True)

    # Capture the shared multiscale geometry before replacing deformable layers.
    layer0_deform = model.model.encoder.layers[0].deformable_layer
    original_layer0_deform_forward = layer0_deform.forward
    captured = {}

    def capture_forward(
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
        return original_layer0_deform_forward(
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
        capture_forward, layer0_deform
    )
    print("Capturing encoder spatial geometry...", flush=True)
    with torch.inference_mode():
        _ = model(**inputs)
        h14.sync()
    layer0_deform.forward = original_layer0_deform_forward

    if not captured:
        raise RuntimeError("Failed to capture spatial_shapes_list.")

    print(f"spatial_shapes={captured['spatial_shapes_list']}", flush=True)

    # Build and patch the current separate six deformable islands.
    h14.sync()
    print("Building six current deformable MLX islands...", flush=True)
    island_state, deform_islands = h18.patch_encoder_deformable_islands(
        model,
        h17.MlxDeformableIsland,
        activation_name=model.config.activation_function,
        spatial_shapes_list=captured["spatial_shapes_list"],
        threadgroup=args.threadgroup,
    )

    # Build a wide island for each encoder layer, reusing the exact same MLX
    # fusion and deformable objects as the baseline.
    print("Building six wide fusion+deformable islands...", flush=True)
    wide_islands = []
    for i, layer in enumerate(model.model.encoder.layers):
        wide = h33.MlxFusionDeformableWideIsland(
            layer.fusion_layer,
            fusion_dispatchers[i].mlx_specializer,
            deform_islands[i],
        )
        wide_islands.append(wide)
        print(f"  wide layer {i}: ready", flush=True)

    # Patch whole encoder layers with a baseline/wide dispatcher.
    wide_state = WideState()
    layer_dispatchers = [
        EncoderLayerWideDispatcher(layer, wide, wide_state)
        for layer, wide in zip(model.model.encoder.layers, wide_islands)
    ]

    try:
        # First pass in each mode excludes compilation/JIT effects and provides
        # a three-way correctness audit.
        print("Correctness baseline: separate MLX fusion + deformable...", flush=True)
        set_mode(layer_dispatchers, "baseline")
        with torch.inference_mode():
            separate_ref = model(**inputs)
            h14.sync()

        print("Correctness candidate: six wide islands...", flush=True)
        set_mode(layer_dispatchers, "wide")
        with torch.inference_mode():
            wide_ref = model(**inputs)
            h14.sync()

        correctness = correctness_bundle(
            wide_ref,
            separate_ref,
            original_ref,
            model,
            h14,
            processor,
            inputs,
            text_labels,
            image.size,
            args.box_threshold,
            args.text_threshold,
        )

        print(
            "Top-k audit:",
            json.dumps(
                {
                    "wide_vs_separate": correctness[
                        "wide_vs_separate_baseline"
                    ]["topk"],
                    "separate_vs_original": correctness[
                        "separate_baseline_vs_original"
                    ]["topk"],
                    "wide_vs_original": correctness[
                        "wide_vs_original"
                    ]["topk"],
                },
                indent=2,
            ),
            flush=True,
        )

        # Warm both paths.
        print("Warming both full-model modes...", flush=True)
        for i in range(args.warmup_per_mode):
            set_mode(layer_dispatchers, "baseline")
            _ = run_timed(model, inputs, h14.sync)
            set_mode(layer_dispatchers, "wide")
            _ = run_timed(model, inputs, h14.sync)
            print(
                f"  warmup pair {i+1}/{args.warmup_per_mode}",
                flush=True,
            )

        # Benchmark-only counters.
        wide_state.baseline_layer_calls = 0
        wide_state.wide_layer_calls = 0
        for d in fusion_dispatchers:
            d.reset_counts()
        island_state.calls = 0
        msda_state.call_count = 0

        baseline_samples = []
        wide_samples = []
        deltas = []

        print(f"Paired full-model benchmark x{args.pairs}...", flush=True)
        for i in range(args.pairs):
            order = (
                ("baseline", "wide")
                if i % 2 == 0
                else ("wide", "baseline")
            )
            local = {}

            for mode in order:
                set_mode(layer_dispatchers, mode)
                _out, dt = run_timed(model, inputs, h14.sync)
                local[mode] = dt
                if mode == "baseline":
                    baseline_samples.append(dt)
                else:
                    wide_samples.append(dt)

            delta = local["baseline"] - local["wide"]
            deltas.append(delta)
            print(
                f"  pair {i+1:02d}/{args.pairs}: "
                f"separate={local['baseline']:.3f} ms, "
                f"wide={local['wide']:.3f} ms, "
                f"delta={delta:+.3f} ms",
                flush=True,
            )

        baseline_stats = stats(baseline_samples)
        wide_stats = stats(wide_samples)
        delta_stats = stats(deltas)

        expected_layer_calls_each = args.pairs * 6
        expected_fusion_baseline_calls = args.pairs * 6
        expected_deform_baseline_calls = args.pairs * 6
        expected_decoder_msda_calls = args.pairs * 2 * 6

        fusion_counts = h31.sum_counts(fusion_dispatchers)

        if wide_state.baseline_layer_calls != expected_layer_calls_each:
            raise RuntimeError(
                f"Baseline layer calls {wide_state.baseline_layer_calls} "
                f"!= {expected_layer_calls_each}"
            )
        if wide_state.wide_layer_calls != expected_layer_calls_each:
            raise RuntimeError(
                f"Wide layer calls {wide_state.wide_layer_calls} "
                f"!= {expected_layer_calls_each}"
            )
        if fusion_counts["mlx_calls"] != expected_fusion_baseline_calls:
            raise RuntimeError(
                f"Baseline fusion MLX calls {fusion_counts['mlx_calls']} "
                f"!= {expected_fusion_baseline_calls}"
            )
        if island_state.calls != expected_deform_baseline_calls:
            raise RuntimeError(
                f"Baseline deformable island calls {island_state.calls} "
                f"!= {expected_deform_baseline_calls}"
            )
        if msda_state.call_count != expected_decoder_msda_calls:
            raise RuntimeError(
                f"Decoder Metal MSDA calls {msda_state.call_count} "
                f"!= {expected_decoder_msda_calls}"
            )

        result = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "experiment": "0034",
            "purpose": (
                "Same-process paired full-model A/B integration of six wide "
                "fusion+deformable MLX encoder islands, compared with the "
                "robust Experiment-0032 separate MLX fusion + deformable path."
            ),
            "model": args.model,
            "device": "mps",
            "dtype": "float32",
            "software": {
                "torch": torch.__version__,
                "transformers": transformers.__version__,
            },
            "configuration": {
                "baseline": (
                    "six compiled MLX fully-folded fusion attentions + "
                    "PyTorch fusion wrapper + six compiled MLX deformable islands"
                ),
                "candidate": (
                    "six single-boundary compiled MLX islands containing "
                    "fusion LayerNorm/residual + fully-folded fusion + "
                    "deformable attention/FFN"
                ),
                "text_enhancer": "PyTorch in both modes",
                "prompt_cache": False,
                "output_attentions": False,
                "warmup_per_mode": args.warmup_per_mode,
                "paired_iterations": args.pairs,
                "pair_order": "alternating separate->wide / wide->separate",
                "threadgroup": args.threadgroup,
                "approximation": False,
                "retraining": False,
                "reduced_precision": False,
            },
            "latency": {
                "separate_mlx_boundaries": baseline_stats,
                "wide_mlx_islands": wide_stats,
                "paired_delta_ms_separate_minus_wide": delta_stats,
            },
            "derived": {
                "speedup_from_medians": (
                    baseline_stats["median_ms"]
                    / wide_stats["median_ms"]
                ),
                "median_difference_of_marginals_ms": (
                    baseline_stats["median_ms"]
                    - wide_stats["median_ms"]
                ),
                "paired_delta_median_ms": delta_stats["median_ms"],
                "paired_delta_mean_ms": delta_stats["mean_ms"],
                "paired_delta_percent_of_baseline_median": (
                    100.0
                    * delta_stats["median_ms"]
                    / baseline_stats["median_ms"]
                ),
            },
            "correctness": correctness,
            "call_validation": {
                "baseline_encoder_layer_calls":
                    wide_state.baseline_layer_calls,
                "wide_encoder_layer_calls":
                    wide_state.wide_layer_calls,
                "baseline_fusion_mlx_calls":
                    fusion_counts["mlx_calls"],
                "baseline_deformable_island_calls":
                    island_state.calls,
                "decoder_metal_msda_calls":
                    msda_state.call_count,
                "expected_encoder_calls_each_mode":
                    expected_layer_calls_each,
                "expected_baseline_fusion_calls":
                    expected_fusion_baseline_calls,
                "expected_baseline_deformable_calls":
                    expected_deform_baseline_calls,
                "expected_decoder_metal_msda_calls":
                    expected_decoder_msda_calls,
            },
            "decision_rule": (
                "Proceed to multi-process replication and adopt the wide "
                "encoder island if the same-process paired median saving is "
                "materially positive (target >=10 ms full-model) and the "
                "top-900 proposal set is preserved with final outputs within "
                "the established FP32 backend envelope. Otherwise retain the "
                "robust separate MLX fusion + deformable implementation."
            ),
            "notes": [
                "The candidate reorders only two post-fusion independent branches: text enhancement and vision deformable processing. They share no post-fusion data dependency.",
                "Both modes use the same trained weights, same algebraic fusion specialization, same Metal MSDA kernel, and same PyTorch text enhancer.",
                "The candidate returns no fusion/deformable attention tensors across the MLX boundary because default inference has output_attentions=False.",
                "Prompt caching is disabled to isolate wide-island effects.",
                "No approximation, quantization, pruning, retraining, or reduced precision is used.",
                "Dataset-level accuracy equivalence remains unmeasured."
            ],
        }

        out = Path("results/metalground_six_wide_encoder_islands_paired.json")
        out.write_text(
            json.dumps(result, indent=2, ensure_ascii=False) + "\n"
        )

        print("\n=== Experiment 0034 summary ===", flush=True)
        print(
            f"separate median: {baseline_stats['median_ms']:.3f} ms",
            flush=True,
        )
        print(
            f"wide median:     {wide_stats['median_ms']:.3f} ms",
            flush=True,
        )
        print(
            f"paired delta:    {delta_stats['median_ms']:+.3f} ms",
            flush=True,
        )
        print(
            f"speedup:         "
            f"{result['derived']['speedup_from_medians']:.3f}x",
            flush=True,
        )
        print(f"Saved: {out}", flush=True)

    finally:
        for d in layer_dispatchers:
            d.restore()
        for d in fusion_dispatchers:
            d.restore()


if __name__ == "__main__":
    main()
