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


def compare(a: torch.Tensor, b: torch.Tensor) -> dict[str, Any]:
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
        "finite_mask_equal": bool(
            torch.equal(torch.isfinite(a), torch.isfinite(b))
        ),
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
    ap.add_argument("--pairs", type=int, default=30)
    ap.add_argument("--threadgroup", type=int, default=256)
    ap.add_argument("--v2-authoritative-ms", type=float, default=542.305167)
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
    args = ap.parse_args()

    for p in (args.exp14_helper, args.exp17_helper, args.exp18_helper):
        if not p.exists():
            raise SystemExit(f"Missing helper script: {p}")
    if not torch.backends.mps.is_available():
        raise SystemExit("MPS unavailable.")

    h14 = load_module(args.exp14_helper, "metalground_exp14_helpers_exp26")
    h17 = load_module(args.exp17_helper, "metalground_exp17_helpers_exp26")
    h18 = load_module(args.exp18_helper, "metalground_exp18_helpers_exp26")

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

    print("Building MetalGround v2...", flush=True)
    fusion_items = h14.collect_fusion_specializers(model)
    msda_state = h14.patch_msda(model, args.threadgroup)
    h14.set_fusion_mode(fusion_items, "fully_folded")

    # Capture encoder spatial shapes for deformable islands.
    layer0 = model.model.encoder.layers[0].deformable_layer
    original_layer0_forward = layer0.forward
    enc_capture = {}

    def capture_encoder(
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
        if not enc_capture:
            enc_capture["spatial_shapes_list"] = [
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

    layer0.forward = types.MethodType(capture_encoder, layer0)
    with torch.inference_mode():
        _ = model(**inputs)
        h14.sync()
    layer0.forward = original_layer0_forward

    if not enc_capture:
        raise RuntimeError("Failed to capture encoder spatial shapes.")

    h14.sync()
    island_state, _islands = h18.patch_encoder_deformable_islands(
        model,
        h17.MlxDeformableIsland,
        activation_name=model.config.activation_function,
        spatial_shapes_list=enc_capture["spatial_shapes_list"],
        threadgroup=args.threadgroup,
    )

    # Capture the exact runtime dimensions for stage0 block1.
    block = model.model.backbone.conv_encoder.model.swin.encoder.layers[0].blocks[1]
    original_block_forward = block.forward
    block_capture = {}

    def capture_block(
        self,
        hidden_states,
        input_dimensions,
        always_partition=False,
        **kwargs,
    ):
        if not block_capture:
            block_capture["input_dimensions"] = (
                int(input_dimensions[0]),
                int(input_dimensions[1]),
            )
            block_capture["hidden_dtype"] = hidden_states.dtype
            block_capture["hidden_device"] = hidden_states.device
        return original_block_forward(
            hidden_states,
            input_dimensions,
            always_partition=always_partition,
            **kwargs,
        )

    block.forward = types.MethodType(capture_block, block)
    with torch.inference_mode():
        _ = model(**inputs)
        h14.sync()
    block.forward = original_block_forward

    if not block_capture:
        raise RuntimeError("Failed to capture stage0 block1 dimensions.")

    dims = block_capture["input_dimensions"]
    block.set_shift_and_window_size(dims)
    h, w = dims
    ws = int(block.window_size)
    pad_right = (ws - w % ws) % ws
    pad_bottom = (ws - h % ws) % ws
    hp = h + pad_bottom
    wp = w + pad_right

    original_get_mask = block.get_attn_mask
    with torch.inference_mode():
        cached_mask = original_get_mask(
            hp,
            wp,
            dtype=block_capture["hidden_dtype"],
            device=block_capture["hidden_device"],
        )
        if cached_mask is None:
            raise RuntimeError("stage0 block1 unexpectedly has no shift mask.")
        cached_mask = cached_mask.detach()
        h14.sync()

    def cached_get_mask(self, height, width, dtype, device):
        if (
            int(height) != hp
            or int(width) != wp
            or dtype != cached_mask.dtype
            or device != cached_mask.device
            or int(self.window_size) != ws
            or int(self.shift_size) <= 0
        ):
            return original_get_mask(height, width, dtype, device)
        return cached_mask

    cached_method = types.MethodType(cached_get_mask, block)

    def set_baseline():
        block.get_attn_mask = original_get_mask

    def set_cached():
        block.get_attn_mask = cached_method

    print(
        f"stage0 block1: dims={dims}, padded={(hp, wp)}, "
        f"mask_shape={tuple(cached_mask.shape)}, "
        f"mask_bytes={cached_mask.numel() * cached_mask.element_size()}",
        flush=True,
    )

    # Correctness on the exact same MetalGround-v2 process/state.
    set_baseline()
    with torch.inference_mode():
        baseline_correct = model(**inputs)
        h14.sync()

    set_cached()
    with torch.inference_mode():
        cached_correct = model(**inputs)
        h14.sync()

    correctness = {
        "logits": compare(cached_correct.logits, baseline_correct.logits),
        "pred_boxes": compare(
            cached_correct.pred_boxes, baseline_correct.pred_boxes
        ),
        "topk_baseline_vs_cached": {
            "baseline": h14.topk_audit(
                baseline_correct, baseline_correct, model.config.num_queries
            ),
            "cached_vs_baseline": h14.topk_audit(
                cached_correct, baseline_correct, model.config.num_queries
            ),
        },
    }

    # Warm both modes.
    print("Warming both full-model modes...", flush=True)
    for i in range(args.warmup_per_mode):
        set_baseline()
        _ = run_timed(model, inputs, h14.sync)
        set_cached()
        _ = run_timed(model, inputs, h14.sync)
        print(f"  warmup pair {i+1}/{args.warmup_per_mode}", flush=True)

    baseline_samples = []
    cached_samples = []
    deltas = []

    print(f"Paired full-model benchmark x{args.pairs}...", flush=True)
    for i in range(args.pairs):
        order = ("baseline", "cached") if i % 2 == 0 else ("cached", "baseline")
        local = {}

        for mode in order:
            if mode == "baseline":
                set_baseline()
            else:
                set_cached()

            _out, dt = run_timed(model, inputs, h14.sync)
            local[mode] = dt

            if mode == "baseline":
                baseline_samples.append(dt)
            else:
                cached_samples.append(dt)

        delta = local["baseline"] - local["cached"]
        deltas.append(delta)

        print(
            f"  pair {i+1:02d}/{args.pairs}: "
            f"baseline={local['baseline']:.3f} ms, "
            f"cached={local['cached']:.3f} ms, "
            f"delta={delta:+.3f} ms",
            flush=True,
        )

    set_baseline()

    baseline_stats = stats(baseline_samples)
    cached_stats = stats(cached_samples)
    delta_stats = stats(deltas)

    result = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "experiment": "0026",
        "purpose": (
            "Same-process paired full-model A/B test of the only robust Swin "
            "mask-cache candidate: stage0 block1 on MetalGround v2."
        ),
        "model": args.model,
        "device": "mps",
        "dtype": "float32",
        "software": {
            "torch": torch.__version__,
            "transformers": transformers.__version__,
        },
        "configuration": {
            "base_runtime": "MetalGround v2",
            "cached_block": (
                "model.backbone.conv_encoder.model.swin.encoder.layers.0.blocks.1"
            ),
            "input_dimensions": list(dims),
            "padded_dimensions": [hp, wp],
            "mask_shape": list(cached_mask.shape),
            "mask_bytes": int(
                cached_mask.numel() * cached_mask.element_size()
            ),
            "warmup_per_mode": args.warmup_per_mode,
            "paired_iterations": args.pairs,
            "order": "alternating baseline->cached / cached->baseline",
        },
        "latency": {
            "baseline_v2_same_process": baseline_stats,
            "stage0_mask_cache_same_process": cached_stats,
            "paired_delta_ms_baseline_minus_cached": delta_stats,
        },
        "derived": {
            "speedup_from_medians": (
                baseline_stats["median_ms"] / cached_stats["median_ms"]
            ),
            "median_reduction_ms": (
                baseline_stats["median_ms"] - cached_stats["median_ms"]
            ),
            "median_reduction_percent": (
                100.0
                * (baseline_stats["median_ms"] - cached_stats["median_ms"])
                / baseline_stats["median_ms"]
            ),
            "authoritative_v2_median_ms": args.v2_authoritative_ms,
            "authoritative_projection_if_delta_transfers_ms": (
                args.v2_authoritative_ms - delta_stats["median_ms"]
            ),
        },
        "correctness": correctness,
        "notes": [
            "Baseline and cached full-model executions share one process and alternate order each pair.",
            "Only stage0 block1 get_attn_mask is replaced; all other MetalGround-v2 code paths are identical.",
            "The cached mask is exact for the captured fixed resolution and falls back to original construction for mismatched geometry/dtype/device.",
            "The paired result is the causal performance estimate; the 542.305 ms Experiment 0018 value remains the cross-process authoritative v2 headline unless this optimization is adopted and re-benchmarked fresh.",
            "No approximation, retraining, pruning, quantization, or reduced precision is used."
        ],
    }

    out = Path("results/metalground_stage0_mask_cache_full_model_paired.json")
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")

    print("\n=== Experiment 0026 summary ===", flush=True)
    print(
        f"baseline median: {baseline_stats['median_ms']:.3f} ms",
        flush=True,
    )
    print(
        f"cached median:   {cached_stats['median_ms']:.3f} ms",
        flush=True,
    )
    print(
        f"median reduction: "
        f"{result['derived']['median_reduction_ms']:+.3f} ms "
        f"({result['derived']['speedup_from_medians']:.4f}x)",
        flush=True,
    )
    print(
        f"paired delta median: {delta_stats['median_ms']:+.3f} ms",
        flush=True,
    )
    print(f"Saved: {out}", flush=True)


if __name__ == "__main__":
    main()
