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

import torch
import torch.nn.functional as F


def sync(device: torch.device) -> None:
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize(device)


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


def bench(fn, device: torch.device, warmup: int, iters: int):
    with torch.inference_mode():
        for _ in range(warmup):
            _ = fn()
            sync(device)

        times = []
        out = None
        for _ in range(iters):
            sync(device)
            t0 = time.perf_counter_ns()
            out = fn()
            sync(device)
            times.append((time.perf_counter_ns() - t0) / 1e6)
    return out, stats(times)


def compare(a: torch.Tensor, b: torch.Tensor) -> dict[str, float]:
    a = a.detach().float().cpu()
    b = b.detach().float().cpu()
    d = (a - b).abs()
    denom = b.abs().clamp_min(1e-8)
    return {
        "max_abs": float(d.max()),
        "mean_abs": float(d.mean()),
        "max_rel": float((d / denom).max()),
        "mean_rel": float((d / denom).mean()),
    }


def mib(n: int) -> float:
    return n / (1024 ** 2)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--case",
        type=Path,
        default=Path("results/msda_cases/encoder_fp32.pt"),
    )
    ap.add_argument("--device", choices=["mps", "cpu"], default="mps")
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--tag", default="msda_reference_parts")
    args = ap.parse_args()

    if not args.case.exists():
        raise SystemExit(f"Missing case file: {args.case}")

    device = torch.device(args.device)
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise SystemExit("MPS is unavailable.")

    case: dict[str, Any] = torch.load(
        args.case,
        map_location="cpu",
        weights_only=False,
    )

    cpu_args = case["args"]
    expected = case["output"]

    (
        value,
        value_spatial_shapes,
        value_spatial_shapes_list,
        level_start_index,
        sampling_locations,
        attention_weights,
        im2col_step,
    ) = cpu_args

    value = value.to(device)
    value_spatial_shapes = value_spatial_shapes.to(device)
    level_start_index = level_start_index.to(device)
    sampling_locations = sampling_locations.to(device)
    attention_weights = attention_weights.to(device)

    B, _, H, C = value.shape
    _, Q, _, L, P, _ = sampling_locations.shape

    # Exact reference decomposition.
    value_list = value.split(
        [height * width for height, width in value_spatial_shapes_list],
        dim=1,
    )
    sampling_grids = 2 * sampling_locations - 1

    level_values = []
    level_grids = []
    logical = {
        "value_input_bytes": value.numel() * value.element_size(),
        "sampling_locations_input_bytes": sampling_locations.numel()
        * sampling_locations.element_size(),
        "attention_weights_input_bytes": attention_weights.numel()
        * attention_weights.element_size(),
        "per_level_sample_output_bytes": [],
    }

    for level_id, (height, width) in enumerate(value_spatial_shapes_list):
        value_l = (
            value_list[level_id]
            .flatten(2)
            .transpose(1, 2)
            .reshape(B * H, C, height, width)
        )
        grid_l = (
            sampling_grids[:, :, :, level_id]
            .transpose(1, 2)
            .flatten(0, 1)
        )
        level_values.append(value_l)
        level_grids.append(grid_l)

        logical["per_level_sample_output_bytes"].append(
            (B * H * C * Q * P) * value.element_size()
        )

    # Whole reference, matching the captured source.
    def reference():
        sampled = []
        for value_l, grid_l in zip(level_values, level_grids):
            sampled.append(
                F.grid_sample(
                    value_l,
                    grid_l,
                    mode="bilinear",
                    padding_mode="zeros",
                    align_corners=False,
                )
            )

        weights = attention_weights.transpose(1, 2).reshape(
            B * H, 1, Q, L * P
        )

        output = (
            (torch.stack(sampled, dim=-2).flatten(-2) * weights)
            .sum(-1)
            .view(B, H * C, Q)
        )
        return output.transpose(1, 2).contiguous()

    print(
        f"Case: B={B}, Q={Q}, heads={H}, head_dim={C}, "
        f"levels={L}, points={P}"
    )
    print(f"Spatial shapes: {value_spatial_shapes_list}")

    ref_out, ref_time = bench(reference, device, args.warmup, args.iters)
    correctness = compare(ref_out, expected.to(device))

    print(
        f"\nWhole reference: median={ref_time['median_ms']:.3f} ms "
        f"p95={ref_time['p95_ms']:.3f} ms "
        f"max_abs={correctness['max_abs']:.3e}"
    )

    # Layout transforms are views/metadata when possible; benchmark anyway.
    level_results = []
    sampled_once = []

    for level_id, ((height, width), value_l, grid_l) in enumerate(
        zip(value_spatial_shapes_list, level_values, level_grids)
    ):
        def one_grid_sample(v=value_l, g=grid_l):
            return F.grid_sample(
                v,
                g,
                mode="bilinear",
                padding_mode="zeros",
                align_corners=False,
            )

        sample, timing = bench(one_grid_sample, device, args.warmup, args.iters)
        sampled_once.append(sample)

        out_bytes = sample.numel() * sample.element_size()
        row = {
            "level": level_id,
            "height": height,
            "width": width,
            "value_shape": list(value_l.shape),
            "grid_shape": list(grid_l.shape),
            "sample_output_shape": list(sample.shape),
            "sample_output_bytes": out_bytes,
            "grid_sample_latency": timing,
        }
        level_results.append(row)

        print(
            f"level {level_id}: {height:3d}x{width:<3d} "
            f"grid_sample median={timing['median_ms']:.3f} ms "
            f"p95={timing['p95_ms']:.3f} ms "
            f"output={mib(out_bytes):.1f} MiB"
        )

    weights = attention_weights.transpose(1, 2).reshape(
        B * H, 1, Q, L * P
    )

    def stack_only():
        return torch.stack(sampled_once, dim=-2).flatten(-2)

    stacked, stack_time = bench(stack_only, device, args.warmup, args.iters)

    def weighted_reduce():
        return (
            (stacked * weights)
            .sum(-1)
            .view(B, H * C, Q)
            .transpose(1, 2)
            .contiguous()
        )

    reduced, reduce_time = bench(
        weighted_reduce, device, args.warmup, args.iters
    )

    reduce_correctness = compare(reduced, expected.to(device))

    stack_bytes = stacked.numel() * stacked.element_size()
    product_bytes = stack_bytes  # logical size of the elementwise product
    output_bytes = reduced.numel() * reduced.element_size()

    logical.update(
        {
            "sum_per_level_sample_output_bytes": sum(
                logical["per_level_sample_output_bytes"]
            ),
            "stacked_tensor_bytes": stack_bytes,
            "logical_weighted_product_bytes": product_bytes,
            "output_bytes": output_bytes,
            "logical_materialized_write_volume_bytes": (
                sum(logical["per_level_sample_output_bytes"])
                + stack_bytes
                + product_bytes
                + output_bytes
            ),
        }
    )

    print(
        f"\nstack/flatten: median={stack_time['median_ms']:.3f} ms "
        f"tensor={mib(stack_bytes):.1f} MiB"
    )
    print(
        f"multiply+sum+final layout: median={reduce_time['median_ms']:.3f} ms "
        f"p95={reduce_time['p95_ms']:.3f} ms"
    )
    print(
        "logical materialized-write volume (not measured DRAM traffic): "
        f"{mib(logical['logical_materialized_write_volume_bytes']):.1f} MiB"
    )

    result = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "case": str(args.case),
        "device": args.device,
        "warmup": args.warmup,
        "iterations": args.iters,
        "shape": {
            "batch": B,
            "queries": Q,
            "heads": H,
            "head_dim": C,
            "levels": L,
            "points": P,
            "spatial_shapes": value_spatial_shapes_list,
        },
        "whole_reference": {
            "latency": ref_time,
            "correctness_vs_captured": correctness,
        },
        "levels": level_results,
        "stack_flatten": {
            "latency": stack_time,
            "tensor_bytes": stack_bytes,
        },
        "weighted_reduce_and_output_layout": {
            "latency": reduce_time,
            "correctness_vs_captured": reduce_correctness,
        },
        "logical_tensor_volume": logical,
        "methodology_note": (
            "MPS synchronization is inserted around each measured sub-operation. "
            "These are diagnostic isolated timings and can perturb scheduling. "
            "logical_materialized_write_volume_bytes is tensor-volume accounting, "
            "not measured physical DRAM traffic."
        ),
    }

    out = Path("results") / f"hf_{args.device}_{args.tag}.json"
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
