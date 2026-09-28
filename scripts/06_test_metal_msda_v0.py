#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path

import mlx.core as mx
import numpy as np
import torch

from metalground.msda_metal_v0 import msda_metal_v0


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


def compare(actual: np.ndarray, expected: np.ndarray) -> dict[str, object]:
    a = actual.astype(np.float64, copy=False)
    b = expected.astype(np.float64, copy=False)

    diff = np.abs(a - b)
    denom = np.maximum(np.abs(b), 1e-8)

    flat_a = a.reshape(-1)
    flat_b = b.reshape(-1)
    norm_a = np.linalg.norm(flat_a)
    norm_b = np.linalg.norm(flat_b)
    if norm_a == 0.0 or norm_b == 0.0:
        cosine = float("nan")
    else:
        cosine = float(np.dot(flat_a, flat_b) / (norm_a * norm_b))

    return {
        "max_abs": float(diff.max()),
        "mean_abs": float(diff.mean()),
        "rmse": float(np.sqrt(np.mean((a - b) ** 2))),
        "max_rel": float((diff / denom).max()),
        "mean_rel": float((diff / denom).mean()),
        "cosine_similarity": cosine,
        "allclose_1e-5": bool(np.allclose(a, b, rtol=1e-5, atol=1e-5)),
        "allclose_1e-4": bool(np.allclose(a, b, rtol=1e-4, atol=1e-4)),
        "finite": bool(np.isfinite(a).all()),
    }


def torch_to_mx_float32(t: torch.Tensor) -> mx.array:
    return mx.array(t.detach().cpu().numpy().astype(np.float32, copy=False))


def torch_to_mx_int32(t: torch.Tensor) -> mx.array:
    return mx.array(t.detach().cpu().numpy().astype(np.int32, copy=False))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--case",
        type=Path,
        default=Path("results/msda_cases/encoder_fp32.pt"),
    )
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--threadgroup", type=int, default=256)
    ap.add_argument(
        "--reference-ms",
        type=float,
        default=69.892125,
        help="Experiment 0004 authoritative isolated PyTorch/MPS median.",
    )
    ap.add_argument("--verbose-kernel", action="store_true")
    args = ap.parse_args()

    if not args.case.exists():
        raise SystemExit(f"Missing fixture: {args.case}")

    mx.set_default_device(mx.gpu)

    case = torch.load(args.case, map_location="cpu", weights_only=False)
    (
        value_t,
        spatial_shapes_t,
        spatial_shapes_list,
        level_start_t,
        sampling_locations_t,
        attention_weights_t,
        im2col_step,
    ) = case["args"]
    expected_t = case["output"]

    # v0 deliberately uses plain contiguous MLX arrays and int32 metadata.
    value = torch_to_mx_float32(value_t)
    spatial_shapes = torch_to_mx_int32(spatial_shapes_t)
    level_start = torch_to_mx_int32(level_start_t)
    sampling_locations = torch_to_mx_float32(sampling_locations_t)
    attention_weights = torch_to_mx_float32(attention_weights_t)

    expected = expected_t.detach().cpu().numpy().astype(np.float32, copy=False)

    print("=== MetalGround MSDA v0 ===")
    print(f"fixture: {args.case}")
    print(f"value:              {tuple(value.shape)} {value.dtype}")
    print(f"spatial_shapes:     {tuple(spatial_shapes.shape)} {spatial_shapes.dtype}")
    print(f"level_start_index:  {tuple(level_start.shape)} {level_start.dtype}")
    print(f"sampling_locations: {tuple(sampling_locations.shape)} {sampling_locations.dtype}")
    print(f"attention_weights:  {tuple(attention_weights.shape)} {attention_weights.dtype}")
    print(f"expected:           {expected.shape} {expected.dtype}")
    print(f"threadgroup:        {args.threadgroup}")

    # First invocation performs/permits Metal JIT compilation.
    print("\nCompiling / first execution...")
    out = msda_metal_v0(
        value,
        spatial_shapes,
        level_start,
        sampling_locations,
        attention_weights,
        threadgroup_size=args.threadgroup,
        verbose=args.verbose_kernel,
    )
    mx.eval(out)
    mx.synchronize()

    actual = np.array(out, copy=True)
    correctness = compare(actual, expected)

    print("\n=== Correctness ===")
    for k, v in correctness.items():
        print(f"{k:22s}: {v}")

    # Gate only catastrophic failure at this stage. We intentionally report
    # strict tolerances rather than choosing a permissive one in advance.
    if not correctness["finite"]:
        raise SystemExit("FAIL: Metal output contains NaN or Inf.")

    if actual.shape != expected.shape:
        raise SystemExit(
            f"FAIL: shape mismatch: actual={actual.shape} expected={expected.shape}"
        )

    print(f"\nWarmup: {args.warmup}")
    for _ in range(args.warmup):
        out = msda_metal_v0(
            value,
            spatial_shapes,
            level_start,
            sampling_locations,
            attention_weights,
            threadgroup_size=args.threadgroup,
        )
        mx.eval(out)
        mx.synchronize()

    print(f"Timed iterations: {args.iters}")
    samples_ms: list[float] = []
    for i in range(args.iters):
        mx.synchronize()
        t0 = time.perf_counter_ns()
        out = msda_metal_v0(
            value,
            spatial_shapes,
            level_start,
            sampling_locations,
            attention_weights,
            threadgroup_size=args.threadgroup,
        )
        mx.eval(out)
        mx.synchronize()
        dt = (time.perf_counter_ns() - t0) / 1e6
        samples_ms.append(dt)
        print(f"  {i + 1:02d}/{args.iters}: {dt:.3f} ms")

    latency = stats(samples_ms)
    speedup = args.reference_ms / latency["median_ms"]

    # Re-check the final invocation after timing.
    final_actual = np.array(out, copy=True)
    final_correctness = compare(final_actual, expected)

    print("\n=== Latency ===")
    for k, v in latency.items():
        print(f"{k:12s}: {v}")
    print(f"reference_ms: {args.reference_ms:.6f}")
    print(f"speedup:      {speedup:.3f}x")

    result = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "experiment": "0006",
        "kernel": "metal_msda_v0_fp32",
        "fixture": str(args.case),
        "method": (
            "MLX mx.fast.metal_kernel; fused bilinear sampling + attention "
            "weighting + level/point reduction; one thread per output scalar."
        ),
        "shape": {
            "value": list(value.shape),
            "spatial_shapes": list(spatial_shapes.shape),
            "level_start_index": list(level_start.shape),
            "sampling_locations": list(sampling_locations.shape),
            "attention_weights": list(attention_weights.shape),
            "output": list(expected.shape),
            "spatial_shapes_list": spatial_shapes_list,
            "im2col_step": im2col_step,
        },
        "threadgroup_size": args.threadgroup,
        "correctness_first_run": correctness,
        "correctness_after_benchmark": final_correctness,
        "latency": latency,
        "authoritative_reference_median_ms": args.reference_ms,
        "isolated_speedup_vs_experiment_0004": speedup,
        "notes": [
            "Experiment 0004 PyTorch/MPS median remains the authoritative isolated reference.",
            "JIT compilation is excluded by performing a first execution before warmup/timing.",
            "No FP16, approximation, query pruning, or architecture change is used.",
            "v0 is correctness-first; no threadgroup-memory or vectorization optimization is applied.",
        ],
    }

    out_path = Path("results/metal_msda_v0.json")
    out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(f"\nSaved: {out_path}")

    # We do not automatically declare PASS based on a hand-picked tolerance.
    # The resulting error distribution will be reviewed before freezing the
    # formal FP32 correctness threshold.
    print(
        "\nNext decision: inspect numerical error first; "
        "then freeze the FP32 acceptance tolerance and optimize v1."
    )


if __name__ == "__main__":
    main()
