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
from datetime import datetime, timezone
from pathlib import Path

import mlx.core as mx
import torch
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


def percentile(xs, p):
    ys = sorted(xs)
    if len(ys) == 1:
        return ys[0]
    k = (len(ys) - 1) * p
    f, c = math.floor(k), math.ceil(k)
    if f == c:
        return ys[f]
    return ys[f] * (c - k) + ys[c] * (k - f)


def stats(xs):
    return {
        "n": len(xs),
        "mean_ms": statistics.fmean(xs),
        "median_ms": statistics.median(xs),
        "p90_ms": percentile(xs, 0.90),
        "p95_ms": percentile(xs, 0.95),
        "min_ms": min(xs),
        "max_ms": max(xs),
    }


def mask_aware_error(a: torch.Tensor, b: torch.Tensor):
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


class FusionDispatcher:
    def __init__(self, attn, torch_specializer, mlx_specializer):
        self.attn = attn
        self.torch_specializer = torch_specializer
        self.mlx_specializer = mlx_specializer
        self.original_forward = attn.forward
        self.mode = "torch"
        self.calls = 0
        self.torch_calls = 0
        self.mlx_calls = 0

        def dispatched(
            _self_module,
            vision_features,
            text_features,
            vision_attention_mask=None,
            text_attention_mask=None,
        ):
            self.calls += 1

            if self.mode == "torch":
                self.torch_calls += 1
                return self.torch_specializer.fully_folded(
                    vision_features,
                    text_features,
                    vision_attention_mask=vision_attention_mask,
                    text_attention_mask=text_attention_mask,
                )

            self.mlx_calls += 1

            # Conservative producer synchronization before sharing MPS tensors
            # with MLX. This matches the safe bridge policy used in prior work.
            torch.mps.synchronize()

            v = mx.asarray(vision_features, copy=False)
            t = mx.asarray(text_features, copy=False)
            vm = (
                None
                if vision_attention_mask is None
                else mx.asarray(vision_attention_mask, copy=False)
            )
            tm = (
                None
                if text_attention_mask is None
                else mx.asarray(text_attention_mask, copy=False)
            )

            out = self.mlx_specializer.compiled_call(v, t, vm, tm)
            mx.eval(*out)
            mx.synchronize()

            dv = torch.as_tensor(out[0])
            va = torch.as_tensor(out[1])
            dt = torch.as_tensor(out[2])
            ta = torch.as_tensor(out[3])
            torch.mps.synchronize()

            return (dv, va), (dt, ta)

        attn.forward = types.MethodType(dispatched, attn)

    def set_mode(self, mode: str):
        if mode not in ("torch", "mlx"):
            raise ValueError(mode)
        self.mode = mode

    def reset_counts(self):
        self.calls = self.torch_calls = self.mlx_calls = 0

    def restore(self):
        self.attn.forward = self.original_forward


def set_all(dispatchers, mode):
    for d in dispatchers:
        d.set_mode(mode)


def reset_all_counts(dispatchers):
    for d in dispatchers:
        d.reset_counts()


def sum_counts(dispatchers):
    return {
        "calls": sum(d.calls for d in dispatchers),
        "torch_calls": sum(d.torch_calls for d in dispatchers),
        "mlx_calls": sum(d.mlx_calls for d in dispatchers),
    }


def run_timed(model, inputs, sync_fn):
    sync_fn()
    t0 = time.perf_counter_ns()
    with torch.inference_mode():
        out = model(**inputs)
    sync_fn()
    return out, (time.perf_counter_ns() - t0) / 1e6


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="IDEA-Research/grounding-dino-tiny")
    ap.add_argument("--image", type=Path, default=Path("assets/input.jpg"))
    ap.add_argument("--prompt", nargs="+", default=["a cat", "a dog"])
    ap.add_argument("--warmup-per-mode", type=int, default=3)
    ap.add_argument("--pairs", type=int, default=20)
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
    args = ap.parse_args()

    for p in (
        args.exp13_helper,
        args.exp14_helper,
        args.exp17_helper,
        args.exp18_helper,
        args.exp30_helper,
    ):
        if not p.exists():
            raise SystemExit(f"Missing helper: {p}")

    if not torch.backends.mps.is_available():
        raise SystemExit("MPS unavailable.")

    h13 = load_module(args.exp13_helper, "mg_h13_exp31")
    h14 = load_module(args.exp14_helper, "mg_h14_exp31")
    h17 = load_module(args.exp17_helper, "mg_h17_exp31")
    h18 = load_module(args.exp18_helper, "mg_h18_exp31")
    h30 = load_module(args.exp30_helper, "mg_h30_exp31")

    mx.set_default_device(mx.gpu)

    print("Loading model...", flush=True)
    processor = AutoProcessor.from_pretrained(args.model)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(args.model)
    model.eval().to("mps")
    h14.sync()

    image = Image.open(args.image).convert("RGB")
    text_labels = [args.prompt]
    cpu_inputs = processor(images=image, text=text_labels, return_tensors="pt")
    inputs = {k: v.to("mps") for k, v in cpu_inputs.items()}
    h14.sync()

    # Keep decoder MSDA on MetalGround v0 as in v2.
    msda_state = h14.patch_msda(model, args.threadgroup)

    # Build both exact algebraic fusion implementations for all six layers.
    print("Building six PyTorch/MLX fully-folded fusion dispatchers...", flush=True)
    dispatchers = []
    h14.sync()
    for i, layer in enumerate(model.model.encoder.layers):
        attn = layer.fusion_layer.attn
        torch_spec = h13.AlgebraicBiMHA(attn)
        mlx_spec = h30.MlxFullyFoldedFusion(torch_spec)
        dispatchers.append(FusionDispatcher(attn, torch_spec, mlx_spec))
        print(f"  layer {i}: ready", flush=True)

    # Capture geometry while using the current PyTorch fully-folded fusion path.
    set_all(dispatchers, "torch")
    layer0 = model.model.encoder.layers[0].deformable_layer
    original_layer0_forward = layer0.forward
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
        return original_layer0_forward(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_embeddings=position_embeddings,
            reference_points=reference_points,
            spatial_shapes=spatial_shapes,
            spatial_shapes_list=spatial_shapes_list,
            level_start_index=level_start_index,
            output_attentions=output_attentions,
        )

    layer0.forward = types.MethodType(capture_forward, layer0)
    print("Capturing encoder multiscale geometry...", flush=True)
    with torch.inference_mode():
        _ = model(**inputs)
        h14.sync()
    layer0.forward = original_layer0_forward

    if not captured:
        raise RuntimeError("Failed to capture spatial_shapes_list.")

    # Patch six encoder deformable layers with the compiled v2 islands.
    h14.sync()
    island_state, _islands = h18.patch_encoder_deformable_islands(
        model,
        h17.MlxDeformableIsland,
        activation_name=model.config.activation_function,
        spatial_shapes_list=captured["spatial_shapes_list"],
        threadgroup=args.threadgroup,
    )

    try:
        # First forward in each mode excluded from timing/JIT effects.
        print("Correctness oracle: PyTorch fully-folded fusion mode...", flush=True)
        set_all(dispatchers, "torch")
        with torch.inference_mode():
            torch_ref = model(**inputs)
            h14.sync()

        print("Correctness candidate: MLX fully-folded fusion mode...", flush=True)
        set_all(dispatchers, "mlx")
        with torch.inference_mode():
            mlx_ref = model(**inputs)
            h14.sync()

        correctness = {
            "logits": mask_aware_error(mlx_ref.logits, torch_ref.logits),
            "pred_boxes": mask_aware_error(
                mlx_ref.pred_boxes, torch_ref.pred_boxes
            ),
            "topk": h14.topk_audit(
                mlx_ref, torch_ref, model.config.num_queries
            ),
        }
        print(json.dumps(correctness, indent=2), flush=True)

        # Warm both modes.
        print("Warming both full-model modes...", flush=True)
        for i in range(args.warmup_per_mode):
            set_all(dispatchers, "torch")
            _ = run_timed(model, inputs, h14.sync)
            set_all(dispatchers, "mlx")
            _ = run_timed(model, inputs, h14.sync)
            print(
                f"  warmup pair {i+1}/{args.warmup_per_mode}",
                flush=True,
            )

        reset_all_counts(dispatchers)
        baseline_samples = []
        candidate_samples = []
        deltas = []

        print(f"Paired full-model benchmark x{args.pairs}...", flush=True)
        for i in range(args.pairs):
            order = (
                ("torch", "mlx")
                if i % 2 == 0
                else ("mlx", "torch")
            )
            local = {}

            for mode in order:
                set_all(dispatchers, mode)
                _out, dt = run_timed(model, inputs, h14.sync)
                local[mode] = dt
                if mode == "torch":
                    baseline_samples.append(dt)
                else:
                    candidate_samples.append(dt)

            delta = local["torch"] - local["mlx"]
            deltas.append(delta)
            print(
                f"  pair {i+1:02d}/{args.pairs}: "
                f"torch={local['torch']:.3f} ms, "
                f"mlx={local['mlx']:.3f} ms, "
                f"delta={delta:+.3f} ms",
                flush=True,
            )

        baseline_stats = stats(baseline_samples)
        candidate_stats = stats(candidate_samples)
        delta_stats = stats(deltas)
        counts = sum_counts(dispatchers)

        expected_each_mode_calls = args.pairs * 6
        expected_total_calls = args.pairs * 2 * 6
        if counts["calls"] != expected_total_calls:
            raise RuntimeError(
                f"Fusion call mismatch: got {counts['calls']}, "
                f"expected {expected_total_calls}"
            )
        if counts["torch_calls"] != expected_each_mode_calls:
            raise RuntimeError(
                f"Torch fusion call mismatch: got {counts['torch_calls']}, "
                f"expected {expected_each_mode_calls}"
            )
        if counts["mlx_calls"] != expected_each_mode_calls:
            raise RuntimeError(
                f"MLX fusion call mismatch: got {counts['mlx_calls']}, "
                f"expected {expected_each_mode_calls}"
            )

        result = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "experiment": "0031",
            "purpose": (
                "Same-process paired full-model A/B test replacing all six "
                "PyTorch fully-folded fusion attention calls in MetalGround v2 "
                "with the exact compiled-MLX algebra validated in Experiment 0030."
            ),
            "model": args.model,
            "device": "mps",
            "dtype": "float32",
            "configuration": {
                "baseline": (
                    "MetalGround v2 with six PyTorch/MPS fully-folded fusion attentions"
                ),
                "candidate": (
                    "same runtime with six compiled-MLX fully-folded fusion attentions"
                ),
                "encoder_deformable_islands": 6,
                "decoder_metal_msda": True,
                "fusion_calls_per_forward": 6,
                "warmup_per_mode": args.warmup_per_mode,
                "paired_iterations": args.pairs,
                "order": "alternating torch->mlx / mlx->torch",
                "conservative_sync_at_each_mlx_fusion_boundary": True,
                "prompt_cache": False,
                "approximation": False,
                "retraining": False,
                "reduced_precision": False,
            },
            "latency": {
                "pytorch_fusion_full_model": baseline_stats,
                "mlx_fusion_full_model": candidate_stats,
                "paired_delta_ms_torch_minus_mlx": delta_stats,
            },
            "derived": {
                "speedup_from_medians": (
                    baseline_stats["median_ms"]
                    / candidate_stats["median_ms"]
                ),
                "median_difference_of_marginals_ms": (
                    baseline_stats["median_ms"]
                    - candidate_stats["median_ms"]
                ),
                "paired_delta_median_ms": delta_stats["median_ms"],
                "paired_delta_mean_ms": delta_stats["mean_ms"],
            },
            "correctness": correctness,
            "call_validation": {
                **counts,
                "expected_total_calls": expected_total_calls,
                "expected_calls_each_mode": expected_each_mode_calls,
            },
            "decision_rule": (
                "If full-model paired delta is materially positive (target >=20 ms) "
                "and correctness remains within the established FP32 backend envelope, "
                "adopt MLX fusion and next test a wider fusion+deformable execution "
                "island to remove the extra per-layer bridge. Otherwise keep the "
                "PyTorch fully-folded implementation."
            ),
            "notes": [
                "Only the six fusion attention implementations differ between paired modes.",
                "Fusion LayerNorm, layer-scale residuals, text enhancer, deformable islands, decoder, and backbone are identical.",
                "Each MLX fusion call uses conservative explicit cross-runtime synchronization, so a wider island can only reduce boundary count if this candidate is successful.",
                "Prompt caching is intentionally disabled to isolate the fusion replacement.",
                "No approximation, quantization, pruning, retraining, or reduced precision is used."
            ],
        }

        out = Path("results/metalground_v2_mlx_fusion_paired.json")
        out.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")

        print("\n=== Experiment 0031 summary ===", flush=True)
        print(
            f"PyTorch-fusion median: {baseline_stats['median_ms']:.3f} ms",
            flush=True,
        )
        print(
            f"MLX-fusion median:     {candidate_stats['median_ms']:.3f} ms",
            flush=True,
        )
        print(
            f"paired delta median:   {delta_stats['median_ms']:+.3f} ms",
            flush=True,
        )
        print(
            f"speedup from medians:  "
            f"{result['derived']['speedup_from_medians']:.3f}x",
            flush=True,
        )
        print(f"Saved: {out}", flush=True)

    finally:
        for d in dispatchers:
            d.restore()


if __name__ == "__main__":
    main()
