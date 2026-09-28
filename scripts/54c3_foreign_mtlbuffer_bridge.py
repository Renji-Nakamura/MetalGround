#!/usr/bin/env python3
import argparse
import json
import time
from pathlib import Path

import mlx.core as mx
import numpy as np
import torch

import metal_dlpack_native as native


def stats(xs):
    a = np.asarray(xs, dtype=np.float64)
    return {
        "n": int(a.size),
        "mean": float(a.mean()),
        "median": float(np.median(a)),
        "p90": float(np.percentile(a, 90)),
        "p95": float(np.percentile(a, 95)),
        "min": float(a.min()),
        "max": float(a.max()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--output",
        default="results/metalground_foreign_mtlbuffer_bridge_0054c3.json",
    )
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--samples", type=int, default=500)
    args = ap.parse_args()

    if not torch.backends.mps.is_available():
        raise RuntimeError("PyTorch MPS unavailable")
    if not mx.metal.is_available():
        raise RuntimeError("MLX Metal unavailable")
    if not hasattr(mx, "from_dlpack"):
        raise RuntimeError("This MLX build does not expose mx.from_dlpack")

    mx.set_default_device(mx.gpu)

    shape = (1, 3, 800, 1065)
    n = int(np.prod(shape))
    foreign = native.make_zeros(shape)

    # MTLStorageModeShared == 0 on Apple platforms.
    storage_mode = int(foreign.storage_mode())
    length_bytes = int(foreign.length_bytes())
    dl_device = tuple(foreign.__dlpack_device__())

    # Seed selected values through the native shared-buffer owner.
    probes = {
        0: 1.25,
        n // 2: -2.5,
        n - 1: 3.75,
    }
    for i, v in probes.items():
        foreign.write(i, v)

    mlx_view = mx.from_dlpack(foreign, copy=False)
    mx.eval(mlx_view)
    mx.synchronize()

    torch_view = torch.from_dlpack(mlx_view, copy=False)
    torch.mps.synchronize()

    imported = {
        "mlx_shape": list(mlx_view.shape),
        "mlx_dtype": str(mlx_view.dtype),
        "torch_shape": list(torch_view.shape),
        "torch_dtype": str(torch_view.dtype),
        "torch_device": torch_view.device.type,
        "probe_values_torch": {
            str(i): float(torch_view.reshape(-1)[i].item())
            for i in probes
        },
    }

    initial_values_exact = all(
        imported["probe_values_torch"][str(i)] == float(v)
        for i, v in probes.items()
    )

    # Native -> MLX/Torch alias proof after import.
    torch.mps.synchronize()
    mx.synchronize()
    native_index = 17
    native_sentinel = 1111.25
    foreign.write(native_index, native_sentinel)
    # Trivial MPS use ensures the consumer reads after the host write.
    observed_native_to_torch = float(
        (torch_view.reshape(-1)[native_index] + 0.0).item()
    )
    torch.mps.synchronize()
    native_to_torch_alias = observed_native_to_torch == native_sentinel

    # Torch -> native alias proof.
    torch_index = 23
    torch_sentinel = -2222.5
    torch_view.reshape(-1)[torch_index] = torch_sentinel
    torch.mps.synchronize()
    observed_torch_to_native = float(foreign.read(torch_index))
    torch_to_native_alias = observed_torch_to_native == torch_sentinel

    # Also observe the Torch mutation from MLX.
    mx.synchronize()
    observed_torch_to_mlx = float(np.array(mlx_view).reshape(-1)[torch_index])
    torch_to_mlx_alias = observed_torch_to_mlx == torch_sentinel

    # DLPack import timing for the same already-allocated foreign shared buffer.
    for _ in range(args.warmup):
        a = mx.from_dlpack(foreign, copy=False)
        del a

    import_ms = []
    for _ in range(args.samples):
        t0 = time.perf_counter_ns()
        a = mx.from_dlpack(foreign, copy=False)
        t1 = time.perf_counter_ns()
        import_ms.append((t1 - t0) / 1e6)
        del a

    import_stats = stats(import_ms)

    # Full boundary creation: foreign -> MLX -> Torch, no GPU producer work.
    boundary_ms = []
    for _ in range(200):
        t0 = time.perf_counter_ns()
        a = mx.from_dlpack(foreign, copy=False)
        b = torch.from_dlpack(a, copy=False)
        t1 = time.perf_counter_ns()
        boundary_ms.append((t1 - t0) / 1e6)
        del b, a

    boundary_stats = stats(boundary_ms)

    gates = {
        "foreign_dlpack_device_is_metal_8": dl_device == (8, 0),
        "foreign_mtlbuffer_storage_mode_shared": storage_mode == 0,
        "foreign_buffer_size_sufficient": length_bytes >= n * 4,
        "mlx_shape_exact": list(mlx_view.shape) == list(shape),
        "torch_shape_exact": list(torch_view.shape) == list(shape),
        "torch_device_mps": torch_view.device.type == "mps",
        "torch_dtype_float32": str(torch_view.dtype) == "torch.float32",
        "initial_probe_values_exact": initial_values_exact,
        "native_to_torch_alias_proven": native_to_torch_alias,
        "torch_to_native_alias_proven": torch_to_native_alias,
        "torch_to_mlx_alias_proven": torch_to_mlx_alias,
        "foreign_to_mlx_import_p95_ms_le_1": import_stats["p95"] <= 1.0,
        "foreign_to_torch_boundary_p95_ms_le_1": boundary_stats["p95"] <= 1.0,
    }

    result = {
        "experiment": "0054c-3",
        "purpose": (
            "Qualify zero-copy import of a foreign shared MTLBuffer into MLX "
            "via kDLMetal DLPack, then onward to PyTorch MPS."
        ),
        "environment": {
            "torch": torch.__version__,
            "mlx": getattr(mx, "__version__", "unknown"),
            "mps_available": torch.backends.mps.is_available(),
            "mlx_metal_available": mx.metal.is_available(),
        },
        "foreign_buffer": {
            "shape": list(shape),
            "dtype": "float32",
            "dlpack_device": list(dl_device),
            "storage_mode_raw": storage_mode,
            "storage_mode_expected": "MTLStorageModeShared",
            "length_bytes": length_bytes,
            "owner": "native Objective-C++ Python extension",
        },
        "imports": imported,
        "alias_proofs": {
            "native_to_torch": {
                "index": native_index,
                "sentinel_written_via_mtlbuffer_contents": native_sentinel,
                "observed_via_torch_mps": observed_native_to_torch,
                "pass": native_to_torch_alias,
            },
            "torch_to_native": {
                "index": torch_index,
                "sentinel_written_via_torch_mps": torch_sentinel,
                "observed_via_mtlbuffer_contents": observed_torch_to_native,
                "pass": torch_to_native_alias,
            },
            "torch_to_mlx": {
                "index": torch_index,
                "sentinel_written_via_torch_mps": torch_sentinel,
                "observed_via_mlx": observed_torch_to_mlx,
                "pass": torch_to_mlx_alias,
            },
        },
        "foreign_to_mlx_import_ms": import_stats,
        "foreign_to_mlx_to_torch_boundary_ms": boundary_stats,
        "pre_registered_gates": gates,
        "all_gates_pass": all(gates.values()),
        "guardrails": [
            "The foreign buffer in 0054c-3 is a synthetic MTLStorageModeShared buffer, not yet a live camera/preprocess output buffer.",
            "0054c-3 tests the ownership/interop mechanism required to connect the native Swift/Metal producer to the already-qualified MLX->PyTorch bridge.",
            "No camera, preprocessing, or model performance claim is made from this experiment.",
            "Alias proofs are stronger evidence than API choice alone: mutations are observed across native MTLBuffer, MLX, and PyTorch MPS views."
        ],
    }

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))

    print("=== Experiment 0054c-3 — foreign MTLBuffer bridge ===")
    print("device:", dl_device, "storage mode:", storage_mode)
    print("initial probes exact:", initial_values_exact)
    print("native -> torch alias:", native_to_torch_alias)
    print("torch -> native alias:", torch_to_native_alias)
    print("torch -> mlx alias:", torch_to_mlx_alias)
    print(
        "foreign->MLX median/p95:",
        f'{import_stats["median"]:.6f}',
        f'{import_stats["p95"]:.6f}',
        "ms",
    )
    print(
        "foreign->MLX->Torch median/p95:",
        f'{boundary_stats["median"]:.6f}',
        f'{boundary_stats["p95"]:.6f}',
        "ms",
    )
    print("ALL GATES PASS:", result["all_gates_pass"])
    print("Saved:", out)


if __name__ == "__main__":
    main()
