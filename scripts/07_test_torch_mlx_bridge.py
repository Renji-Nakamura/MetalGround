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


def compare(actual: torch.Tensor, expected: torch.Tensor) -> dict[str, Any]:
    a = actual.detach().float().cpu().numpy().astype(np.float64, copy=False)
    b = expected.detach().float().cpu().numpy().astype(np.float64, copy=False)
    diff = np.abs(a - b)
    denom = np.maximum(np.abs(b), 1e-8)

    fa = a.reshape(-1)
    fb = b.reshape(-1)
    cosine = float(np.dot(fa, fb) / (np.linalg.norm(fa) * np.linalg.norm(fb)))

    return {
        "max_abs": float(diff.max()),
        "mean_abs": float(diff.mean()),
        "rmse": float(np.sqrt(np.mean((a - b) ** 2))),
        "max_rel": float((diff / denom).max()),
        "mean_rel": float((diff / denom).mean()),
        "cosine_similarity": cosine,
        "allclose_1e-5": bool(np.allclose(a, b, rtol=1e-5, atol=1e-5)),
        "finite": bool(np.isfinite(a).all()),
    }


def load_case(path: Path, device: torch.device):
    case = torch.load(path, map_location="cpu", weights_only=False)
    args = list(case["args"])
    expected = case["output"]

    # Real compute tensors stay as PyTorch MPS tensors.
    value = args[0].to(device)
    spatial_shapes_t = args[1].to(device)
    level_start_t = args[3].to(device)
    sampling_locations = args[4].to(device)
    attention_weights = args[5].to(device)
    expected_mps = expected.to(device)

    return {
        "value": value,
        "spatial_shapes_t": spatial_shapes_t,
        "level_start_t": level_start_t,
        "sampling_locations": sampling_locations,
        "attention_weights": attention_weights,
        "expected": expected_mps,
        "spatial_shapes_list": args[2],
        "im2col_step": args[6],
    }


def bridge_once(case: dict[str, Any], threadgroup: int) -> torch.Tensor:
    # PyTorch 2.12+ ordinary MPS tensors use Metal storage that MLX can
    # generally import zero-copy. copy=False makes failure explicit.
    value_mx = mx.asarray(case["value"], copy=False)
    loc_mx = mx.asarray(case["sampling_locations"], copy=False)
    weight_mx = mx.asarray(case["attention_weights"], copy=False)

    # These metadata tensors are tiny. Import zero-copy then cast to int32
    # because v0's Metal kernel indexes them as int32.
    spatial_mx = mx.asarray(case["spatial_shapes_t"], copy=False).astype(mx.int32)
    start_mx = mx.asarray(case["level_start_t"], copy=False).astype(mx.int32)

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

    # MLX Metal arrays export to PyTorch through DLPack without a copy.
    out_torch = torch.as_tensor(out_mx)
    if out_torch.device.type != "mps":
        raise RuntimeError(
            f"Expected zero-copy Metal export to MPS, got {out_torch.device}"
        )
    return out_torch


def benchmark_case(
    label: str,
    path: Path,
    device: torch.device,
    warmup: int,
    iters: int,
    threadgroup: int,
    reference_ms: float | None,
):
    print(f"\n=== {label} ===")
    case = load_case(path, device)

    # Ensure PyTorch has finished producing/copying inputs before crossing
    # framework boundaries. DLPack conversion itself does not synchronize.
    torch.mps.synchronize()

    # First call includes any MLX/Metal specialization/JIT that has not yet run.
    first = bridge_once(case, threadgroup)
    torch.mps.synchronize()
    correctness = compare(first, case["expected"])

    print("Correctness:")
    for k, v in correctness.items():
        print(f"  {k:22s}: {v}")

    if not correctness["finite"]:
        raise RuntimeError(f"{label}: NaN/Inf")
    if not correctness["allclose_1e-5"]:
        raise RuntimeError(f"{label}: failed provisional 1e-5 correctness gate")

    for _ in range(warmup):
        # Safe interop protocol: producer is synchronized before MLX consumes.
        torch.mps.synchronize()
        out = bridge_once(case, threadgroup)
        torch.mps.synchronize()

    samples = []
    for i in range(iters):
        # This timing intentionally includes the synchronization required for
        # safe PyTorch -> MLX -> PyTorch execution.
        torch.mps.synchronize()
        t0 = time.perf_counter_ns()
        out = bridge_once(case, threadgroup)
        torch.mps.synchronize()
        dt = (time.perf_counter_ns() - t0) / 1e6
        samples.append(dt)
        print(f"  {i+1:02d}/{iters}: {dt:.3f} ms")

    latency = stats(samples)
    final_correctness = compare(out, case["expected"])

    result = {
        "fixture": str(path),
        "latency_bridge_plus_kernel": latency,
        "correctness": final_correctness,
        "reference_median_ms": reference_ms,
        "speedup_vs_reference": (
            reference_ms / latency["median_ms"] if reference_ms else None
        ),
        "torch_input_devices": {
            "value": str(case["value"].device),
            "sampling_locations": str(case["sampling_locations"].device),
            "attention_weights": str(case["attention_weights"].device),
        },
        "output_device": str(out.device),
        "notes": [
            "mx.asarray(..., copy=False) is used for MPS compute tensors; failure would raise instead of silently copying.",
            "torch.as_tensor(MLX_output) is required to return an MPS tensor.",
            "Timing includes explicit cross-framework synchronization for correctness.",
            "Tiny spatial metadata tensors are cast from int64 to int32 in MLX.",
        ],
    }

    print(
        f"{label}: median={latency['median_ms']:.3f} ms "
        f"p95={latency['p95_ms']:.3f} ms "
        + (
            f"speedup={result['speedup_vs_reference']:.2f}x"
            if reference_ms else ""
        )
    )
    return result


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
        raise SystemExit("MPS is unavailable.")

    for p in (args.encoder_case, args.decoder_case):
        if not p.exists():
            raise SystemExit(f"Missing fixture: {p}")

    device = torch.device("mps")
    mx.set_default_device(mx.gpu)

    encoder = benchmark_case(
        "encoder",
        args.encoder_case,
        device,
        args.warmup,
        args.iters,
        args.threadgroup,
        reference_ms=69.892125,
    )
    decoder = benchmark_case(
        "decoder",
        args.decoder_case,
        device,
        args.warmup,
        args.iters,
        args.threadgroup,
        reference_ms=3.540291,
    )

    record = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "experiment": "0007",
        "purpose": "PyTorch MPS <-> MLX Metal zero-copy bridge validation for fused MSDA v0",
        "threadgroup_size": args.threadgroup,
        "warmup": args.warmup,
        "iterations": args.iters,
        "encoder": encoder,
        "decoder": decoder,
    }

    out = Path("results/metal_msda_v0_bridge.json")
    out.write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n")
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
