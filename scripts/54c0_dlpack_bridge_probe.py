#!/usr/bin/env python3
import argparse
import gc
import json
import time
from pathlib import Path

import numpy as np
import mlx.core as mx
import torch


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


IDENTITY = mx.fast.metal_kernel(
    name="metalground_0054c0_identity",
    input_names=["inp"],
    output_names=["out"],
    source=r"""
        uint elem = thread_position_in_grid.x;
        out[elem] = inp[elem];
    """,
)


def import_torch_no_copy(a):
    # PyTorch 2.13 supports the modern DLPack copy= API. For this
    # experiment, a fallback that may copy would invalidate the question,
    # so failure is surfaced rather than silently degraded.
    return torch.from_dlpack(a, copy=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--metal-json",
        default="results/metalground_metal_preprocess_0054b2.json",
    )
    ap.add_argument(
        "--reference-meta",
        default="results/0054b_reference/metadata.json",
    )
    ap.add_argument(
        "--output",
        default="results/metalground_dlpack_bridge_0054c0.json",
    )
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--samples", type=int, default=500)
    args = ap.parse_args()

    if not torch.backends.mps.is_available():
        raise RuntimeError("PyTorch MPS is unavailable")
    if not mx.metal.is_available():
        raise RuntimeError("MLX Metal backend is unavailable")

    mx.set_default_device(mx.gpu)

    metal_meta = json.loads(Path(args.metal_json).read_text())
    ref_meta = json.loads(Path(args.reference_meta).read_text())
    shape = tuple(ref_meta["processor"]["output_shape_chw"])
    if shape != (3, 800, 1065):
        raise RuntimeError(f"Unexpected reference shape: {shape}")

    candidate_path = Path(metal_meta["output_file"])
    if not candidate_path.exists():
        # Keep the script portable if the JSON contains an absolute path from
        # another checkout while the canonical result filename exists locally.
        fallback = Path("results/metal_pixel_values_f32_chw_0054b2.bin")
        if fallback.exists():
            candidate_path = fallback
        else:
            raise FileNotFoundError(candidate_path)

    np_src = np.fromfile(candidate_path, dtype=np.float32).reshape(shape)

    # Input upload is intentionally outside the bridge benchmark. 0054c-0 is
    # only asking whether a Metal-produced MLX array can become a PyTorch MPS
    # tensor without a copy.
    mx_src = mx.array(np_src, dtype=mx.float32)

    produced = IDENTITY(
        inputs=[mx_src],
        grid=(int(np_src.size), 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[shape],
        output_dtypes=[mx.float32],
    )[0]

    # Critical producer ordering rule from Experiment 0036:
    # all producer GPU work must be complete before DLPack handoff.
    mx.eval(produced)
    mx.synchronize()

    dl_device = (
        list(produced.__dlpack_device__())
        if hasattr(produced, "__dlpack_device__")
        else None
    )

    # Correctness before any alias mutation.
    torch_view = import_torch_no_copy(produced)
    torch.mps.synchronize()

    imported_device = torch_view.device.type
    imported_dtype = str(torch_view.dtype)
    imported_shape = list(torch_view.shape)

    imported_cpu = torch_view.detach().cpu().numpy()
    import_diff = imported_cpu - np_src
    import_max_abs = float(np.max(np.abs(import_diff)))
    import_mean_abs = float(np.mean(np.abs(import_diff)))

    # Hard zero-copy alias proof: mutate through Torch, then observe the same
    # storage through MLX. Use a dedicated tiny buffer so the production
    # preprocessing output is not modified.
    alias_src = mx.arange(16, dtype=mx.float32)
    alias_produced = IDENTITY(
        inputs=[alias_src],
        grid=(16, 1, 1),
        threadgroup=(16, 1, 1),
        output_shapes=[(16,)],
        output_dtypes=[mx.float32],
    )[0]
    mx.eval(alias_produced)
    mx.synchronize()

    alias_torch = import_torch_no_copy(alias_produced)
    alias_before = float(np.array(alias_produced)[3])
    sentinel = 12345.25
    alias_torch[3] = sentinel
    torch.mps.synchronize()
    # DLPack conversion itself does not provide cross-framework
    # synchronization; producer/consumer transitions are explicit.
    mx.synchronize()
    alias_after = float(np.array(alias_produced)[3])
    alias_forward_pass = abs(alias_after - sentinel) == 0.0

    # Symmetric check: ordinary PyTorch 2.12+ MPS storage should be importable
    # by MLX with copy=False on Apple silicon.
    reverse_torch = torch.arange(16, dtype=torch.float32, device="mps")
    torch.mps.synchronize()
    reverse_mlx = mx.asarray(reverse_torch, copy=False)
    mx.eval(reverse_mlx)
    mx.synchronize()
    reverse_sentinel = 54321.5
    reverse_torch[5] = reverse_sentinel
    torch.mps.synchronize()
    mx.synchronize()
    reverse_after = float(np.array(reverse_mlx)[5])
    alias_reverse_pass = abs(reverse_after - reverse_sentinel) == 0.0

    del alias_torch, torch_view
    gc.collect()

    # Warm the capsule/import path using an already-synchronized producer.
    for _ in range(args.warmup):
        t = import_torch_no_copy(produced)
        del t

    bridge_ms = []
    for _ in range(args.samples):
        t0 = time.perf_counter_ns()
        t = import_torch_no_copy(produced)
        t1 = time.perf_counter_ns()
        bridge_ms.append((t1 - t0) / 1e6)
        del t

    # Consumer-touch measurement. This includes DLPack import, one trivial MPS
    # consumer op, and MPS completion. It is secondary, not the bridge-only
    # primary metric.
    consumer_ms = []
    for _ in range(100):
        t0 = time.perf_counter_ns()
        t = import_torch_no_copy(produced)
        z = t.reshape(-1)[:1024].sum()
        torch.mps.synchronize()
        _ = z
        t1 = time.perf_counter_ns()
        consumer_ms.append((t1 - t0) / 1e6)
        del t, z

    bridge_stats = stats(bridge_ms)
    consumer_stats = stats(consumer_ms)

    gates = {
        "dlpack_device_is_metal_8": dl_device is not None and dl_device[0] == 8,
        "torch_import_device_is_mps": imported_device == "mps",
        "torch_import_dtype_is_float32": imported_dtype == "torch.float32",
        "shape_exact": imported_shape == list(shape),
        "import_value_max_abs_eq_0": import_max_abs == 0.0,
        "mlx_to_torch_storage_alias_proven": alias_forward_pass,
        "torch_to_mlx_storage_alias_proven": alias_reverse_pass,
        "bridge_only_p95_ms_le_1": bridge_stats["p95"] <= 1.0,
    }

    result = {
        "experiment": "0054c-0",
        "purpose": (
            "Qualify zero-copy Metal DLPack handoff between an evaluated "
            "MLX Metal producer and PyTorch MPS before full-model integration."
        ),
        "environment": {
            "torch": torch.__version__,
            "mlx": getattr(mx, "__version__", "unknown"),
            "torch_mps_available": torch.backends.mps.is_available(),
            "mlx_metal_available": mx.metal.is_available(),
        },
        "source": {
            "shape_chw": list(shape),
            "dtype": "float32",
            "source_file": str(candidate_path),
            "producer": "MLX custom Metal identity kernel",
            "producer_sync_before_handoff": True,
        },
        "dlpack": {
            "mlx_dlpack_device": dl_device,
            "torch_device": imported_device,
            "torch_dtype": imported_dtype,
            "torch_shape": imported_shape,
            "copy_false_required": True,
        },
        "correctness": {
            "import_mean_abs": import_mean_abs,
            "import_max_abs": import_max_abs,
            "mlx_to_torch_alias": {
                "index": 3,
                "before": alias_before,
                "sentinel_written_by_torch": sentinel,
                "observed_via_mlx": alias_after,
                "pass": alias_forward_pass,
            },
            "torch_to_mlx_alias": {
                "index": 5,
                "sentinel_written_by_torch": reverse_sentinel,
                "observed_via_mlx": reverse_after,
                "pass": alias_reverse_pass,
            },
        },
        "bridge_only_ms": bridge_stats,
        "bridge_plus_trivial_torch_consumer_ms": consumer_stats,
        "pre_registered_gates": gates,
        "all_gates_pass": all(gates.values()),
        "timing_semantics": {
            "bridge_only": (
                "CPU wall time for torch.from_dlpack(mlx_array, copy=False) "
                "after MLX producer eval+synchronize; no producer work included."
            ),
            "consumer_touch": (
                "DLPack import + trivial MPS sum + torch.mps.synchronize; "
                "secondary diagnostic."
            ),
        },
        "guardrails": [
            "0054c-0 does not claim the live CVPixelBuffer-to-model path is complete.",
            "The CPU file-to-MLX upload used to seed this fixed-input probe is outside timing and is not part of the proposed live path.",
            "Zero-copy is established for the MLX Metal <-> PyTorch MPS handoff by copy=False import plus bidirectional alias observation.",
            "Cross-framework producer synchronization is explicit because DLPack itself does not synchronize pending Metal work.",
        ],
    }

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))

    print("=== Experiment 0054c-0 — Metal DLPack bridge ===")
    print("MLX DLPack device:", dl_device)
    print("Torch:", imported_device, imported_dtype, imported_shape)
    print("Import max abs:", import_max_abs)
    print("MLX -> Torch alias:", alias_forward_pass)
    print("Torch -> MLX alias:", alias_reverse_pass)
    print(
        "bridge-only median/p95:",
        f'{bridge_stats["median"]:.6f}',
        f'{bridge_stats["p95"]:.6f}',
        "ms",
    )
    print(
        "bridge+consumer median/p95:",
        f'{consumer_stats["median"]:.6f}',
        f'{consumer_stats["p95"]:.6f}',
        "ms",
    )
    print("ALL GATES PASS:", result["all_gates_pass"])
    print("Saved:", out)


if __name__ == "__main__":
    main()
