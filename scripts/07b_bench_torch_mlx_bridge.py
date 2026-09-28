#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import mlx.core as mx
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


def load_case(path: Path) -> dict[str, Any]:
    case = torch.load(path, map_location="cpu", weights_only=False)
    args = list(case["args"])
    return {
        "value": args[0].to("mps"),
        "spatial_shapes": args[1].to("mps"),
        "level_start_index": args[3].to("mps"),
        "sampling_locations": args[4].to("mps"),
        "attention_weights": args[5].to("mps"),
        "expected": case["output"].to("mps"),
    }


def prepare_static_metadata(case: dict[str, Any]):
    # Metadata is tiny and invariant for the fixture/model input shape.
    spatial_mx = mx.asarray(case["spatial_shapes"], copy=False).astype(mx.int32)
    start_mx = mx.asarray(case["level_start_index"], copy=False).astype(mx.int32)
    mx.eval(spatial_mx, start_mx)
    mx.synchronize()
    return spatial_mx, start_mx


def bridge_kernel_once(
    case: dict[str, Any],
    spatial_mx: mx.array,
    start_mx: mx.array,
    threadgroup: int,
) -> torch.Tensor:
    # Explicitly synchronize the PyTorch producer before MLX consumes the
    # shared Metal buffers. No CPU tensor copy is performed.
    torch.mps.synchronize()

    value_mx = mx.asarray(case["value"], copy=False)
    loc_mx = mx.asarray(case["sampling_locations"], copy=False)
    weight_mx = mx.asarray(case["attention_weights"], copy=False)

    out_mx = msda_metal_v0(
        value_mx,
        spatial_mx,
        start_mx,
        loc_mx,
        weight_mx,
        threadgroup_size=threadgroup,
    )
    mx.eval(out_mx)
    mx.synchronize()

    out_t = torch.as_tensor(out_mx)
    if out_t.device.type != "mps":
        raise RuntimeError(f"Expected MPS output, got {out_t.device}")

    # Ensure the PyTorch consumer can safely use the shared output.
    torch.mps.synchronize()
    return out_t


def correctness_on_mps(actual: torch.Tensor, expected: torch.Tensor) -> dict[str, Any]:
    # Keep the large tensors on GPU. Only scalar reductions cross to CPU.
    diff = (actual - expected).abs()
    max_abs = float(diff.max().item())
    mean_abs = float(diff.mean().item())
    rmse = float(torch.sqrt(torch.mean((actual - expected) ** 2)).item())
    allclose_1e5 = bool(torch.allclose(actual, expected, rtol=1e-5, atol=1e-5))
    finite = bool(torch.isfinite(actual).all().item())

    a = actual.float().reshape(-1)
    b = expected.float().reshape(-1)
    cosine = float(
        (torch.dot(a, b) / (torch.linalg.vector_norm(a) * torch.linalg.vector_norm(b))).item()
    )

    torch.mps.synchronize()
    return {
        "max_abs": max_abs,
        "mean_abs": mean_abs,
        "rmse": rmse,
        "cosine_similarity": cosine,
        "allclose_1e-5": allclose_1e5,
        "finite": finite,
    }


def bench_one(
    label: str,
    path: Path,
    reference_ms: float,
    warmup: int,
    iters: int,
    threadgroup: int,
) -> dict[str, Any]:
    print(f"\n=== {label} ===", flush=True)
    print("Loading fixture...", flush=True)
    case = load_case(path)
    torch.mps.synchronize()

    print("Preparing static MLX metadata...", flush=True)
    spatial_mx, start_mx = prepare_static_metadata(case)

    print("First bridge + kernel call (JIT excluded from timing)...", flush=True)
    out = bridge_kernel_once(case, spatial_mx, start_mx, threadgroup)

    print("Checking correctness on MPS...", flush=True)
    corr = correctness_on_mps(out, case["expected"])
    for k, v in corr.items():
        print(f"  {k:20s}: {v}", flush=True)

    if not corr["finite"]:
        raise RuntimeError(f"{label}: non-finite output")
    if not corr["allclose_1e-5"]:
        raise RuntimeError(f"{label}: failed provisional FP32 correctness gate")

    print(f"Warmup x{warmup}...", flush=True)
    for i in range(warmup):
        bridge_kernel_once(case, spatial_mx, start_mx, threadgroup)
        print(f"  warmup {i+1}/{warmup}", flush=True)

    print(f"Timed iterations x{iters}...", flush=True)
    samples = []
    for i in range(iters):
        t0 = time.perf_counter_ns()
        out = bridge_kernel_once(case, spatial_mx, start_mx, threadgroup)
        dt = (time.perf_counter_ns() - t0) / 1e6
        samples.append(dt)
        print(f"  {i+1:02d}/{iters}: {dt:.3f} ms", flush=True)

    lat = stats(samples)
    speedup = reference_ms / lat["median_ms"]

    print(
        f"{label}: median={lat['median_ms']:.3f} ms, "
        f"p95={lat['p95_ms']:.3f} ms, speedup={speedup:.2f}x",
        flush=True,
    )

    return {
        "fixture": str(path),
        "correctness": corr,
        "latency_bridge_kernel_bridge": lat,
        "reference_median_ms": reference_ms,
        "speedup_vs_reference": speedup,
        "output_device": str(out.device),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--encoder-case",
        type=Path,
        default=Path("results/msda_cases/encoder_fp32.pt"),
    )
    ap.add_argument(
        "--decoder-case",
        type=Path,
        default=Path("results/msda_cases/decoder_fp32.pt"),
    )
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--threadgroup", type=int, default=256)
    args = ap.parse_args()

    if not torch.backends.mps.is_available():
        raise SystemExit("MPS unavailable.")

    for p in (args.encoder_case, args.decoder_case):
        if not p.exists():
            raise SystemExit(f"Missing fixture: {p}")

    mx.set_default_device(mx.gpu)

    encoder = bench_one(
        "encoder",
        args.encoder_case,
        reference_ms=69.892125,
        warmup=args.warmup,
        iters=args.iters,
        threadgroup=args.threadgroup,
    )
    decoder = bench_one(
        "decoder",
        args.decoder_case,
        reference_ms=3.540291,
        warmup=args.warmup,
        iters=args.iters,
        threadgroup=args.threadgroup,
    )

    result = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "experiment": "0007",
        "method": (
            "PyTorch MPS -> MLX shared Metal buffer import with copy=False -> "
            "fused MSDA v0 -> PyTorch MPS export. Timings include explicit "
            "cross-framework synchronization."
        ),
        "warmup": args.warmup,
        "iterations": args.iters,
        "threadgroup_size": args.threadgroup,
        "encoder": encoder,
        "decoder": decoder,
        "notes": [
            "Large correctness tensors remain on MPS; only scalar reductions are copied to CPU.",
            "Static shape metadata is converted to MLX int32 once outside the timed loop.",
            "JIT/first execution is excluded from timing.",
            "Timing intentionally includes synchronization required by this conservative interop implementation.",
        ],
    }

    out = Path("results/metal_msda_v0_bridge.json")
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(f"\nSaved: {out}", flush=True)


if __name__ == "__main__":
    main()
