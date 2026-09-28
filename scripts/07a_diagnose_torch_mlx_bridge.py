#!/usr/bin/env python3
from __future__ import annotations

import platform
import sys
import time
from pathlib import Path

import mlx.core as mx
import torch

from metalground.msda_metal_v0 import msda_metal_v0


def log(msg: str) -> None:
    print(msg, flush=True)


def timed(label: str, fn):
    log(f"[BEGIN] {label}")
    t0 = time.perf_counter()
    out = fn()
    dt = time.perf_counter() - t0
    log(f"[ OK  ] {label} ({dt:.3f}s)")
    return out


def tiny_probe() -> None:
    log("\n=== Phase A: tiny PyTorch MPS <-> MLX probe ===")

    x_t = timed(
        "torch tiny tensor -> MPS",
        lambda: torch.arange(16, dtype=torch.float32, device="mps").reshape(4, 4),
    )
    timed("torch.mps.synchronize()", torch.mps.synchronize)

    x_mx = timed(
        "mx.asarray(torch_mps, copy=False)",
        lambda: mx.asarray(x_t, copy=False),
    )
    log(f"       MLX shape={x_mx.shape} dtype={x_mx.dtype}")

    y_mx = timed("create lazy MLX op y=x+1", lambda: x_mx + 1.0)
    timed("mx.eval(y)", lambda: mx.eval(y_mx))
    timed("mx.synchronize()", mx.synchronize)

    y_t = timed("torch.as_tensor(mlx_array)", lambda: torch.as_tensor(y_mx))
    log(f"       PyTorch output device={y_t.device} shape={tuple(y_t.shape)}")
    timed("torch.mps.synchronize() after export", torch.mps.synchronize)

    expected = x_t + 1.0
    ok = timed("tiny correctness check", lambda: torch.equal(y_t, expected))
    log(f"       tiny correctness={ok}")
    if not ok:
        raise RuntimeError("Tiny bridge probe returned wrong values.")

    log("Phase A PASS")


def real_probe(case_path: Path) -> None:
    log("\n=== Phase B: real encoder fixture ===")

    case = timed(
        f"torch.load({case_path})",
        lambda: torch.load(case_path, map_location="cpu", weights_only=False),
    )
    args = list(case["args"])

    value_cpu = args[0]
    spatial_cpu = args[1]
    start_cpu = args[3]
    loc_cpu = args[4]
    weight_cpu = args[5]
    expected_cpu = case["output"]

    log(f"       value CPU: {tuple(value_cpu.shape)} {value_cpu.dtype}")
    log(f"       locations: {tuple(loc_cpu.shape)} {loc_cpu.dtype}")
    log(f"       weights:   {tuple(weight_cpu.shape)} {weight_cpu.dtype}")

    value_t = timed("value CPU -> MPS", lambda: value_cpu.to("mps"))
    timed("sync after value", torch.mps.synchronize)

    loc_t = timed("sampling_locations CPU -> MPS", lambda: loc_cpu.to("mps"))
    timed("sync after locations", torch.mps.synchronize)

    weight_t = timed("attention_weights CPU -> MPS", lambda: weight_cpu.to("mps"))
    timed("sync after weights", torch.mps.synchronize)

    # Metadata is tiny. Keep it on CPU first; later we test MPS import separately.
    spatial_t = timed("spatial_shapes CPU -> MPS", lambda: spatial_cpu.to("mps"))
    timed("sync after spatial_shapes", torch.mps.synchronize)

    start_t = timed("level_start_index CPU -> MPS", lambda: start_cpu.to("mps"))
    timed("sync after level_start_index", torch.mps.synchronize)

    expected_t = timed("expected CPU -> MPS", lambda: expected_cpu.to("mps"))
    timed("sync after expected", torch.mps.synchronize)

    log("\n--- Zero-copy imports ---")
    value_mx = timed(
        "mx.asarray(value_mps, copy=False)",
        lambda: mx.asarray(value_t, copy=False),
    )
    log(f"       value_mx={value_mx.shape} {value_mx.dtype}")

    loc_mx = timed(
        "mx.asarray(locations_mps, copy=False)",
        lambda: mx.asarray(loc_t, copy=False),
    )
    log(f"       loc_mx={loc_mx.shape} {loc_mx.dtype}")

    weight_mx = timed(
        "mx.asarray(weights_mps, copy=False)",
        lambda: mx.asarray(weight_t, copy=False),
    )
    log(f"       weight_mx={weight_mx.shape} {weight_mx.dtype}")

    spatial_mx_i64 = timed(
        "mx.asarray(spatial_shapes_mps, copy=False)",
        lambda: mx.asarray(spatial_t, copy=False),
    )
    log(f"       spatial imported dtype={spatial_mx_i64.dtype}")

    start_mx_i64 = timed(
        "mx.asarray(level_start_mps, copy=False)",
        lambda: mx.asarray(start_t, copy=False),
    )
    log(f"       start imported dtype={start_mx_i64.dtype}")

    spatial_mx = timed(
        "spatial_shapes MLX int64 -> int32",
        lambda: spatial_mx_i64.astype(mx.int32),
    )
    start_mx = timed(
        "level_start MLX int64 -> int32",
        lambda: start_mx_i64.astype(mx.int32),
    )

    # Force the two tiny casts before the custom kernel.
    timed("mx.eval(metadata casts)", lambda: mx.eval(spatial_mx, start_mx))
    timed("mx.synchronize() metadata", mx.synchronize)

    log("\n--- Custom Metal kernel ---")
    out_mx = timed(
        "construct MSDA v0 lazy output",
        lambda: msda_metal_v0(
            value_mx,
            spatial_mx,
            start_mx,
            loc_mx,
            weight_mx,
            threadgroup_size=256,
        ),
    )
    log(f"       out_mx={out_mx.shape} {out_mx.dtype}")

    timed("mx.eval(MSDA v0)", lambda: mx.eval(out_mx))
    timed("mx.synchronize() after kernel", mx.synchronize)

    log("\n--- MLX -> PyTorch export ---")
    out_t = timed("torch.as_tensor(out_mx)", lambda: torch.as_tensor(out_mx))
    log(f"       output device={out_t.device} shape={tuple(out_t.shape)}")
    timed("torch.mps.synchronize() after export", torch.mps.synchronize)

    max_abs = timed(
        "MPS max absolute error",
        lambda: (out_t - expected_t).abs().max().item(),
    )
    log(f"       max_abs={max_abs:.9e}")

    log("\nPhase B PASS")
    log("Bridge and kernel both completed successfully.")


def main() -> None:
    log("=== MetalGround bridge diagnostic ===")
    log(f"Python: {platform.python_version()}")
    log(f"PyTorch: {torch.__version__}")
    log(f"MLX: {getattr(mx, '__version__', 'unknown')}")
    log(f"MPS available: {torch.backends.mps.is_available()}")

    if not torch.backends.mps.is_available():
        raise SystemExit("MPS unavailable.")

    mx.set_default_device(mx.gpu)

    tiny_probe()

    case_path = Path("results/msda_cases/encoder_fp32.pt")
    if not case_path.exists():
        raise SystemExit(f"Missing fixture: {case_path}")

    real_probe(case_path)


if __name__ == "__main__":
    main()
