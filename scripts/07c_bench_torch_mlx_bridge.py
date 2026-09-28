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


def load_case(path: Path) -> dict[str, Any]:
    case = torch.load(path, map_location="cpu", weights_only=False)
    args = list(case["args"])

    print("  value -> MPS", flush=True)
    value = args[0].to("mps")
    torch.mps.synchronize()

    print("  sampling_locations -> MPS", flush=True)
    sampling_locations = args[4].to("mps")
    torch.mps.synchronize()

    print("  attention_weights -> MPS", flush=True)
    attention_weights = args[5].to("mps")
    torch.mps.synchronize()

    print("  expected -> MPS", flush=True)
    expected = case["output"].to("mps")
    torch.mps.synchronize()

    # Keep tiny static metadata on CPU deliberately.
    spatial_shapes_cpu = args[1].detach().cpu()
    level_start_cpu = args[3].detach().cpu()

    return {
        "value": value,
        "sampling_locations": sampling_locations,
        "attention_weights": attention_weights,
        "expected": expected,
        "spatial_shapes_cpu": spatial_shapes_cpu,
        "level_start_cpu": level_start_cpu,
    }


def prepare_static_metadata(case: dict[str, Any]):
    # These tensors contain only a handful of integers. Do not cross the
    # PyTorch-MPS/MLX boundary for them; construct native MLX int32 arrays
    # directly from CPU metadata.
    spatial_np = case["spatial_shapes_cpu"].numpy().astype(np.int32, copy=False)
    start_np = case["level_start_cpu"].numpy().astype(np.int32, copy=False)

    spatial_mx = mx.array(spatial_np, dtype=mx.int32)
    start_mx = mx.array(start_np, dtype=mx.int32)
    mx.eval(spatial_mx, start_mx)
    mx.synchronize()
    return spatial_mx, start_mx


def bridge_kernel_once(
    case: dict[str, Any],
    spatial_mx: mx.array,
    start_mx: mx.array,
    threadgroup: int,
) -> torch.Tensor:
    # Conservative synchronization while validating cross-framework execution.
    torch.mps.synchronize()

    # Large compute tensors use the validated zero-copy Metal bridge.
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

    torch.mps.synchronize()
    return out_t


def correctness_on_mps(actual: torch.Tensor, expected: torch.Tensor) -> dict[str, Any]:
    diff = (actual - expected).abs()
    max_abs = float(diff.max().item())
    mean_abs = float(diff.mean().item())
    rmse = float(torch.sqrt(torch.mean((actual - expected) ** 2)).item())
    allclose_1e5 = bool(torch.allclose(actual, expected, rtol=1e-5, atol=1e-5))
    finite = bool(torch.isfinite(actual).all().item())

    a = actual.float().reshape(-1)
    b = expected.float().reshape(-1)
    cosine = float(
        (torch.dot(a, b) /
         (torch.linalg.vector_norm(a) * torch.linalg.vector_norm(b))).item()
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

    print("Preparing tiny static metadata directly in MLX...", flush=True)
    spatial_mx, start_mx = prepare_static_metadata(case)
    print("Static metadata ready.", flush=True)

    print("First bridge + kernel call (JIT excluded from timing)...", flush=True)
    out = bridge_kernel_once(case, spatial_mx, start_mx, threadgroup)
    print("First bridge + kernel completed.", flush=True)

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
        "experiment": "0007c",
        "method": (
            "Large PyTorch MPS compute tensors -> MLX shared Metal import "
            "with copy=False -> fused MSDA v0 -> PyTorch MPS export. "
            "Tiny static int metadata is constructed directly as native MLX int32."
        ),
        "warmup": args.warmup,
        "iterations": args.iters,
        "threadgroup_size": args.threadgroup,
        "encoder": encoder,
        "decoder": decoder,
        "notes": [
            "Large compute tensors use the validated zero-copy MPS/MLX bridge.",
            "Tiny spatial metadata intentionally avoids cross-framework MPS int64 interop.",
            "JIT/first execution is excluded from timing.",
            "Timing includes conservative explicit synchronization across frameworks.",
        ],
    }

    out = Path("results/metal_msda_v0_bridge.json")
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(f"\nSaved: {out}", flush=True)


if __name__ == "__main__":
    main()
