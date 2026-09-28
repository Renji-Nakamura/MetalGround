#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import types
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


def output_audit(out, baseline, original, model, h14):
    def pair(a, b):
        return {
            "logits": mask_aware_error(a.logits, b.logits),
            "pred_boxes": mask_aware_error(a.pred_boxes, b.pred_boxes),
            "topk": h14.topk_audit(a, b, model.config.num_queries),
        }

    return {
        "vs_separate_baseline": pair(out, baseline),
        "vs_original": pair(out, original),
    }


def set_modes(dispatchers, wide_indices: set[int]):
    for i, d in enumerate(dispatchers):
        d.set_mode("wide" if i in wide_indices else "baseline")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="IDEA-Research/grounding-dino-tiny")
    ap.add_argument("--image", type=Path, default=Path("assets/input.jpg"))
    ap.add_argument("--prompt", nargs="+", default=["a cat", "a dog"])
    ap.add_argument("--threadgroup", type=int, default=256)
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
    ap.add_argument(
        "--exp34-helper",
        type=Path,
        default=Path("scripts/34_full_model_wide_encoder_islands_paired.py"),
    )
    args = ap.parse_args()

    helpers = (
        args.exp13_helper,
        args.exp14_helper,
        args.exp17_helper,
        args.exp18_helper,
        args.exp30_helper,
        args.exp31_helper,
        args.exp33_helper,
        args.exp34_helper,
    )
    for p in helpers:
        if not p.exists():
            raise SystemExit(f"Missing helper: {p}")

    if not torch.backends.mps.is_available():
        raise SystemExit("MPS unavailable.")

    h13 = load_module(args.exp13_helper, "mg35_h13")
    h14 = load_module(args.exp14_helper, "mg35_h14")
    h17 = load_module(args.exp17_helper, "mg35_h17")
    h18 = load_module(args.exp18_helper, "mg35_h18")
    h30 = load_module(args.exp30_helper, "mg35_h30")
    h31 = load_module(args.exp31_helper, "mg35_h31")
    h33 = load_module(args.exp33_helper, "mg35_h33")
    h34 = load_module(args.exp34_helper, "mg35_h34")

    mx.set_default_device(mx.gpu)

    print("Loading model...", flush=True)
    processor = AutoProcessor.from_pretrained(args.model)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(args.model)
    model.eval().to("mps")
    h14.sync()

    if bool(model.config.output_attentions):
        raise RuntimeError("Experiment 0035 requires output_attentions=False.")

    image = Image.open(args.image).convert("RGB")
    text_labels = [args.prompt]
    cpu_inputs = processor(images=image, text=text_labels, return_tensors="pt")
    inputs = {k: v.to("mps") for k, v in cpu_inputs.items()}
    h14.sync()

    print("Original PyTorch/MPS oracle...", flush=True)
    with torch.inference_mode():
        original_ref = model(**inputs)
        h14.sync()

    # Decoder custom Metal MSDA, same as current runtime.
    _msda_state = h14.patch_msda(model, args.threadgroup)

    # Robust compiled-MLX fusion implementation from 0032.
    fusion_dispatchers = []
    for i, layer in enumerate(model.model.encoder.layers):
        torch_spec = h13.AlgebraicBiMHA(layer.fusion_layer.attn)
        mlx_spec = h30.MlxFullyFoldedFusion(torch_spec)
        fd = h31.FusionDispatcher(layer.fusion_layer.attn, torch_spec, mlx_spec)
        fd.set_mode("mlx")
        fusion_dispatchers.append(fd)
        print(f"fusion {i}: ready", flush=True)

    # Capture multiscale geometry.
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

    layer0_deform.forward = types.MethodType(capture_deform, layer0_deform)
    with torch.inference_mode():
        _ = model(**inputs)
        h14.sync()
    layer0_deform.forward = original_deform_forward

    if not captured:
        raise RuntimeError("Failed to capture spatial shapes.")

    # Current separate deformable islands.
    _island_state, deform_islands = h18.patch_encoder_deformable_islands(
        model,
        h17.MlxDeformableIsland,
        activation_name=model.config.activation_function,
        spatial_shapes_list=captured["spatial_shapes_list"],
        threadgroup=args.threadgroup,
    )

    # Build six wide islands and whole-layer dispatchers.
    wide_islands = []
    for i, layer in enumerate(model.model.encoder.layers):
        wide_islands.append(
            h33.MlxFusionDeformableWideIsland(
                layer.fusion_layer,
                fusion_dispatchers[i].mlx_specializer,
                deform_islands[i],
            )
        )

    wide_state = h34.WideState()
    layer_dispatchers = [
        h34.EncoderLayerWideDispatcher(layer, wide, wide_state)
        for layer, wide in zip(model.model.encoder.layers, wide_islands)
    ]

    try:
        # Stable separate baseline.
        set_modes(layer_dispatchers, set())
        with torch.inference_mode():
            separate_ref = model(**inputs)
            h14.sync()

        print(
            "Separate baseline top-k vs original:",
            json.dumps(
                h14.topk_audit(
                    separate_ref, original_ref, model.config.num_queries
                ),
                indent=2,
            ),
            flush=True,
        )

        # ------------------------------------------------------------------
        # 1) Local dual-execution audit.
        #
        # Each layer receives the exact separate-baseline trajectory input.
        # We execute baseline and wide on that same input, record both outputs,
        # but return the baseline output downstream. Hence later layers never
        # see candidate drift.
        # ------------------------------------------------------------------
        local_audit = []

        print("\n=== Local same-input layer audit ===", flush=True)
        for target_i, (layer, dispatcher) in enumerate(
            zip(model.model.encoder.layers, layer_dispatchers)
        ):
            set_modes(layer_dispatchers, set())
            dispatched_forward = layer.forward
            record = {}

            def make_wrapper(bound_forward, bound_dispatcher, bound_record):
                def wrapper(_self, *f_args, **f_kwargs):
                    bound_dispatcher.set_mode("baseline")
                    base = bound_forward(*f_args, **f_kwargs)
                    h14.sync()

                    bound_dispatcher.set_mode("wide")
                    cand = bound_forward(*f_args, **f_kwargs)
                    h14.sync()

                    bound_dispatcher.set_mode("baseline")

                    base_v, base_t = base[0]
                    cand_v, cand_t = cand[0]
                    bound_record["vision"] = tensor_error(cand_v, base_v)
                    bound_record["text"] = tensor_error(cand_t, base_t)
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
                raise RuntimeError(f"Layer {target_i} local audit did not fire.")

            downstream = output_audit(
                out, separate_ref, original_ref, model, h14
            )

            row = {
                "layer": target_i,
                "same_input_wide_vs_separate": record,
                "downstream_returned_baseline_check": downstream,
            }
            local_audit.append(row)

            print(
                f"layer {target_i}: "
                f"vision max={record['vision']['max_abs']:.3e} "
                f"(1e-4={record['vision']['allclose_1e-4']}), "
                f"text max={record['text']['max_abs']:.3e} "
                f"(1e-4={record['text']['allclose_1e-4']})",
                flush=True,
            )

        # ------------------------------------------------------------------
        # 2) Single-layer substitutions.
        # ------------------------------------------------------------------
        single_layer = []
        print("\n=== One wide layer at a time ===", flush=True)
        for i in range(6):
            set_modes(layer_dispatchers, {i})
            with torch.inference_mode():
                out = model(**inputs)
                h14.sync()
            audit = output_audit(
                out, separate_ref, original_ref, model, h14
            )
            row = {"wide_layers": [i], **audit}
            single_layer.append(row)
            tk = audit["vs_separate_baseline"]["topk"]
            print(
                f"layer {i}: overlap={tk['set_overlap']}/900, "
                f"rankwise={tk['rankwise_identical']}/900, "
                f"changed={tk['changed_membership_each_side']}",
                flush=True,
            )

        # ------------------------------------------------------------------
        # 3) Prefix sweep: first N wide layers.
        # ------------------------------------------------------------------
        prefix = []
        print("\n=== Prefix-wide sweep ===", flush=True)
        for n in range(0, 7):
            indices = set(range(n))
            set_modes(layer_dispatchers, indices)
            with torch.inference_mode():
                out = model(**inputs)
                h14.sync()
            audit = output_audit(
                out, separate_ref, original_ref, model, h14
            )
            row = {"prefix_wide_layers": n, **audit}
            prefix.append(row)
            tk = audit["vs_separate_baseline"]["topk"]
            print(
                f"prefix {n}: overlap={tk['set_overlap']}/900, "
                f"rankwise={tk['rankwise_identical']}/900, "
                f"changed={tk['changed_membership_each_side']}",
                flush=True,
            )

        set_modes(layer_dispatchers, set(range(6)))
        with torch.inference_mode():
            all_wide = model(**inputs)
            h14.sync()

        result = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "experiment": "0035",
            "purpose": (
                "Numerical-causality audit for Experiment 0034: distinguish "
                "local semantic mismatch from accumulation/amplification of "
                "small per-layer FP32 differences in the six wide encoder islands."
            ),
            "model": args.model,
            "device": "mps",
            "dtype": "float32",
            "software": {
                "torch": torch.__version__,
                "transformers": transformers.__version__,
            },
            "configuration": {
                "separate_baseline": (
                    "robust compiled-MLX fusion + PyTorch fusion wrapper + "
                    "compiled-MLX deformable island"
                ),
                "wide_candidate": (
                    "compiled-MLX fusion LayerNorm/residual + fusion + "
                    "deformable island"
                ),
                "prompt_cache": False,
                "timing": False,
                "approximation": False,
                "retraining": False,
                "reduced_precision": False,
            },
            "separate_baseline_vs_original": output_audit(
                separate_ref, separate_ref, original_ref, model, h14
            )["vs_original"],
            "local_same_input_audit": local_audit,
            "single_layer_substitution": single_layer,
            "prefix_wide_sweep": prefix,
            "all_six_wide_final": output_audit(
                all_wide, separate_ref, original_ref, model, h14
            ),
            "decision_rule": (
                "If any layer's same-input vision/text output fails allclose "
                "1e-4, treat Experiment 0034 as a local implementation/semantic "
                "bug and fix that layer/path. If all six local outputs remain "
                "within 1e-4 but full-model divergence grows under single/prefix "
                "substitution, classify the failure as trajectory amplification "
                "from repeated backend-association differences; do not adopt "
                "the wide island without a stronger numerical-stability strategy."
            ),
            "notes": [
                "The local audit returns the separate-baseline output downstream, so each audited layer sees the exact baseline trajectory input.",
                "The single-layer sweep tests sensitivity to one wide layer while all other layers remain on the robust separate path.",
                "The prefix sweep identifies how divergence accumulates as more wide layers are composed.",
                "No performance timing is collected; this experiment is correctness-only.",
                "No approximation, quantization, pruning, retraining, or reduced precision is used."
            ],
        }

        out = Path("results/metalground_wide_island_numerical_audit.json")
        out.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
        print(f"\nSaved: {out}", flush=True)

    finally:
        for d in layer_dispatchers:
            d.restore()
        for d in fusion_dispatchers:
            d.restore()


if __name__ == "__main__":
    main()
