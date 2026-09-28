#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import statistics
import time
import types
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import transformers
from PIL import Image
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor


def sync() -> None:
    torch.mps.synchronize()


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
        "finite": bool(torch.isfinite(a).all().item()),
    }


def capture_shifted_blocks(model, inputs):
    swin = model.model.backbone.conv_encoder.model.swin
    targets = []
    captured: dict[str, dict[str, Any]] = {}
    originals = {}

    for stage_i, stage in enumerate(swin.encoder.layers):
        for block_i, block in enumerate(stage.blocks):
            if int(block.shift_size) <= 0:
                continue
            name = (
                f"model.backbone.conv_encoder.model.swin.encoder."
                f"layers.{stage_i}.blocks.{block_i}"
            )
            targets.append((name, stage_i, block_i, block))
            originals[name] = block.forward
            orig = block.forward

            def make_capture(tag, original_forward):
                def capture(
                    self,
                    hidden_states,
                    input_dimensions,
                    always_partition=False,
                    **kwargs,
                ):
                    if tag not in captured:
                        captured[tag] = {
                            "hidden_states": hidden_states.detach(),
                            "input_dimensions": (
                                int(input_dimensions[0]),
                                int(input_dimensions[1]),
                            ),
                        }
                    return original_forward(
                        hidden_states,
                        input_dimensions,
                        always_partition=always_partition,
                        **kwargs,
                    )
                return capture

            block.forward = types.MethodType(make_capture(name, orig), block)

    with torch.inference_mode():
        _ = model(**inputs)
        sync()

    for name, _, _, block in targets:
        block.forward = originals[name]

    expected = {x[0] for x in targets}
    if set(captured) != expected:
        raise RuntimeError(
            f"Capture mismatch: expected {sorted(expected)}, "
            f"got {sorted(captured)}"
        )

    return targets, captured


def build_cached_mask(block, dims):
    block.set_shift_and_window_size(dims)
    h, w = int(dims[0]), int(dims[1])
    ws = int(block.window_size)
    pad_right = (ws - w % ws) % ws
    pad_bottom = (ws - h % ws) % ws
    hp = h + pad_bottom
    wp = w + pad_right

    # Match the dtype/device requested by the actual block path. Hidden-state
    # dtype/device are passed separately by caller; get_attn_mask itself only
    # needs those values.
    return hp, wp


def run_block(block, hidden, dims):
    with torch.inference_mode():
        return block(hidden, dims, always_partition=False)


def timed_call(block, hidden, dims):
    sync()
    t0 = time.perf_counter_ns()
    out = run_block(block, hidden, dims)
    sync()
    return out, (time.perf_counter_ns() - t0) / 1e6


def paired_benchmark(
    block,
    hidden,
    dims,
    cached_mask,
    *,
    warmup: int,
    pairs: int,
):
    original_get_mask = block.get_attn_mask

    def cached_get_mask(self, height, width, dtype, device):
        return cached_mask

    cached_method = types.MethodType(cached_get_mask, block)

    def set_baseline():
        block.get_attn_mask = original_get_mask

    def set_cached():
        block.get_attn_mask = cached_method

    # Warm both code paths before paired timing.
    for _ in range(warmup):
        set_baseline()
        _ = timed_call(block, hidden, dims)
        set_cached()
        _ = timed_call(block, hidden, dims)

    baseline_samples = []
    cached_samples = []
    paired_deltas = []

    baseline_last = None
    cached_last = None

    try:
        for i in range(pairs):
            # Alternate order every pair to reduce drift/order bias.
            order = ("baseline", "cached") if i % 2 == 0 else ("cached", "baseline")
            local = {}

            for mode in order:
                if mode == "baseline":
                    set_baseline()
                    out, dt = timed_call(block, hidden, dims)
                    baseline_last = out
                    baseline_samples.append(dt)
                else:
                    set_cached()
                    out, dt = timed_call(block, hidden, dims)
                    cached_last = out
                    cached_samples.append(dt)
                local[mode] = dt

            paired_deltas.append(local["baseline"] - local["cached"])
            print(
                f"    pair {i+1:02d}/{pairs}: "
                f"baseline={local['baseline']:.3f} ms, "
                f"cached={local['cached']:.3f} ms, "
                f"delta={local['baseline']-local['cached']:+.3f} ms",
                flush=True,
            )
    finally:
        set_baseline()

    return {
        "baseline": stats(baseline_samples),
        "cached": stats(cached_samples),
        "paired_delta_ms": stats(paired_deltas),
        "speedup_from_medians": (
            statistics.median(baseline_samples)
            / statistics.median(cached_samples)
        ),
        "median_reduction_ms": (
            statistics.median(baseline_samples)
            - statistics.median(cached_samples)
        ),
        "correctness": compare(cached_last[0], baseline_last[0]),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="IDEA-Research/grounding-dino-tiny")
    ap.add_argument("--image", type=Path, default=Path("assets/input.jpg"))
    ap.add_argument("--prompt", nargs="+", default=["a cat", "a dog"])
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--pairs", type=int, default=20)
    args = ap.parse_args()

    if not torch.backends.mps.is_available():
        raise SystemExit("MPS unavailable.")

    print("Loading model...", flush=True)
    processor = AutoProcessor.from_pretrained(args.model)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(args.model)
    model.eval().to("mps")
    sync()

    image = Image.open(args.image).convert("RGB")
    text_labels = [args.prompt]
    cpu_inputs = processor(images=image, text=text_labels, return_tensors="pt")
    inputs = {k: v.to("mps") for k, v in cpu_inputs.items()}
    sync()

    print("Capturing all real shifted-Swin block inputs...", flush=True)
    targets, captured = capture_shifted_blocks(model, inputs)

    blocks = {}
    total_positive_median_reduction = 0.0
    total_all_median_reduction = 0.0

    for name, stage_i, block_i, block in targets:
        cap = captured[name]
        hidden = cap["hidden_states"]
        dims = cap["input_dimensions"]

        block.set_shift_and_window_size(dims)
        hp, wp = build_cached_mask(block, dims)

        with torch.inference_mode():
            cached_mask = block.get_attn_mask(
                hp,
                wp,
                dtype=hidden.dtype,
                device=hidden.device,
            )
            if cached_mask is not None:
                cached_mask = cached_mask.detach()
            sync()

        if cached_mask is None:
            print(
                f"\n=== {name}: runtime shift clamped to zero; skipping ===",
                flush=True,
            )
            blocks[name] = {
                "stage": stage_i,
                "block": block_i,
                "input_shape": list(hidden.shape),
                "input_dimensions": list(dims),
                "runtime_window_size": int(block.window_size),
                "runtime_shift_size": int(block.shift_size),
                "skipped": True,
                "reason": "get_attn_mask returned None",
            }
            continue

        print(f"\n=== {name} ===", flush=True)
        print(
            f"  input={tuple(hidden.shape)} dims={dims} "
            f"window={block.window_size} shift={block.shift_size} "
            f"mask_shape={tuple(cached_mask.shape)}",
            flush=True,
        )

        row = paired_benchmark(
            block,
            hidden,
            dims,
            cached_mask,
            warmup=args.warmup,
            pairs=args.pairs,
        )

        reduction = row["median_reduction_ms"]
        total_all_median_reduction += reduction
        if reduction > 0:
            total_positive_median_reduction += reduction

        blocks[name] = {
            "stage": stage_i,
            "block": block_i,
            "input_shape": list(hidden.shape),
            "input_dimensions": list(dims),
            "runtime_window_size": int(block.window_size),
            "runtime_shift_size": int(block.shift_size),
            "mask_shape": list(cached_mask.shape),
            "mask_numel": int(cached_mask.numel()),
            "mask_bytes": int(cached_mask.numel() * cached_mask.element_size()),
            "skipped": False,
            **row,
        }

        print(
            f"  median: {row['baseline']['median_ms']:.3f} -> "
            f"{row['cached']['median_ms']:.3f} ms "
            f"({row['speedup_from_medians']:.3f}x), "
            f"reduction={row['median_reduction_ms']:+.3f} ms",
            flush=True,
        )

    ranked = sorted(
        [
            {
                "module": name,
                "stage": row["stage"],
                "block": row["block"],
                "median_reduction_ms": row.get("median_reduction_ms"),
                "speedup": row.get("speedup_from_medians"),
            }
            for name, row in blocks.items()
            if not row.get("skipped", False)
        ],
        key=lambda x: x["median_reduction_ms"],
        reverse=True,
    )

    result = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "experiment": "0025",
        "purpose": (
            "Paired same-process A/B benchmark of exact shifted-window mask "
            "caching across every shifted Swin block, to explain why the "
            "all-block full-model cache did not transfer Experiment 0023's gain."
        ),
        "model": args.model,
        "device": "mps",
        "dtype": "float32",
        "software": {
            "torch": torch.__version__,
            "transformers": transformers.__version__,
        },
        "protocol": {
            "warmup_per_mode": args.warmup,
            "paired_iterations": args.pairs,
            "order": "alternating baseline->cached / cached->baseline",
            "same_process": True,
            "same_captured_real_inputs": True,
        },
        "blocks": blocks,
        "derived": {
            "ranked_by_median_reduction": ranked,
            "diagnostic_sum_all_block_median_reductions_ms":
                total_all_median_reduction,
            "diagnostic_sum_positive_block_median_reductions_ms":
                total_positive_median_reduction,
        },
        "notes": [
            "Each block uses its exact captured real input and runtime spatial dimensions.",
            "Baseline and cached executions are paired and order-alternated to reduce thermal/scheduling drift.",
            "The cache replaces only get_attn_mask; all other PyTorch/MPS Swin operations are unchanged.",
            "Per-block reductions are diagnostic and should not be added as an exact full-model latency prediction.",
            "No approximation, retraining, pruning, quantization, or reduced precision is used."
        ],
    }

    out = Path("results/metalground_all_shifted_swin_mask_ablation.json")
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")

    print("\n=== Experiment 0025 ranking ===", flush=True)
    for r in ranked:
        print(
            f"{r['median_reduction_ms']:+7.3f} ms  "
            f"{r['speedup']:.3f}x  {r['module']}",
            flush=True,
        )
    print(
        f"diagnostic sum all reductions: "
        f"{total_all_median_reduction:+.3f} ms",
        flush=True,
    )
    print(
        f"diagnostic sum positive reductions: "
        f"{total_positive_median_reduction:+.3f} ms",
        flush=True,
    )
    print(f"Saved: {out}", flush=True)


if __name__ == "__main__":
    main()
