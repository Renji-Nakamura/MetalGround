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

import mlx.core as mx
import torch
import transformers

from metalground.msda_metal_v0 import msda_metal_v0


def load_module(path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load helper: {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod


def percentile(xs, p):
    ys = sorted(xs)
    if len(ys) == 1:
        return ys[0]
    k = (len(ys) - 1) * p
    f, c = math.floor(k), math.ceil(k)
    if f == c:
        return ys[f]
    return ys[f] * (c - k) + ys[c] * (k - f)


def stats(xs):
    return {
        "n": len(xs),
        "mean_ms": statistics.fmean(xs),
        "median_ms": statistics.median(xs),
        "p90_ms": percentile(xs, 0.90),
        "p95_ms": percentile(xs, 0.95),
        "min_ms": min(xs),
        "max_ms": max(xs),
    }


def compare(a: torch.Tensor, b: torch.Tensor):
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
        "finite": bool(torch.isfinite(a).all().item() and torch.isfinite(b).all().item()),
    }


def linear(x, w, b):
    y = x @ w.T
    return y if b is None else y + b


def force(name, x):
    t0 = time.perf_counter_ns()
    mx.eval(x)
    mx.synchronize()
    dt = (time.perf_counter_ns() - t0) / 1e6
    print(f"  {name:24s} {dt:9.3f} ms", flush=True)
    return x, dt


def persistent_inputs(captured, sync_fn):
    sync_fn()
    hidden = mx.asarray(captured["hidden_states"], copy=False)
    enc = mx.asarray(captured["encoder_hidden_states"], copy=False)
    ref = mx.asarray(captured["reference_points"], copy=False)
    pos = (
        None if captured.get("position_embeddings") is None
        else mx.asarray(captured["position_embeddings"], copy=False)
    )
    mask = (
        None if captured.get("attention_mask") is None
        else mx.asarray(captured["attention_mask"], copy=False)
    )
    return hidden, mask, enc, pos, ref


def exact_44a_warm(candidate, mlx_inputs):
    hidden, mask, enc, pos, ref = mlx_inputs
    records = []

    query = hidden if pos is None else hidden + pos
    query, dt = force("query_add", query)
    records.append(("query_add", dt))

    B, Q, _ = query.shape
    _, S, _ = enc.shape

    value = linear(enc, candidate.p["value_w"], candidate.p["value_b"])
    value, dt = force("value_projection", value)
    records.append(("value_projection", dt))

    if mask is not None:
        value = mx.where(mask[..., None], value, 0.0)
        value, dt = force("value_mask", value)
        records.append(("value_mask", dt))

    value = value.reshape(B, S, candidate.H, candidate.HD)
    value, dt = force("value_reshape", value)
    records.append(("value_reshape", dt))

    offsets = linear(
        query, candidate.p["sampling_w"], candidate.p["sampling_b"]
    ).reshape(B, Q, candidate.H, candidate.NL, candidate.NP, 2)
    offsets, dt = force("sampling_offsets", offsets)
    records.append(("sampling_offsets", dt))

    weight_logits = linear(
        query, candidate.p["attn_w_w"], candidate.p["attn_w_b"]
    ).reshape(B, Q, candidate.H, candidate.NL * candidate.NP)
    weight_logits, dt = force("attention_logits", weight_logits)
    records.append(("attention_logits", dt))

    weights = mx.softmax(weight_logits, axis=-1).reshape(
        B, Q, candidate.H, candidate.NL, candidate.NP
    )
    weights, dt = force("attention_softmax", weights)
    records.append(("attention_softmax", dt))

    if ref.shape[-1] == 4:
        locations = (
            ref[:, :, None, :, None, :2]
            + offsets / float(candidate.NP)
            * ref[:, :, None, :, None, 2:] * 0.5
        )
    elif ref.shape[-1] == 2:
        locations = (
            ref[:, :, None, :, None, :]
            + offsets
            / candidate.offset_normalizer[None, None, None, :, None, :]
        )
    else:
        raise RuntimeError(f"unexpected ref shape {ref.shape}")

    locations, dt = force("sampling_locations", locations)
    records.append(("sampling_locations", dt))

    core = msda_metal_v0(
        value,
        candidate.spatial_shapes,
        candidate.level_start,
        locations,
        weights,
        threadgroup_size=candidate.threadgroup,
    )
    core, dt = force("custom_metal_msda", core)
    records.append(("custom_metal_msda", dt))

    out = linear(core, candidate.p["output_w"], candidate.p["output_b"])
    out, dt = force("output_projection", out)
    records.append(("output_projection", dt))

    out_t = torch.as_tensor(out)
    return records, out_t


def bridge_diag(pair, sync_fn):
    out, weights = pair
    mx.eval(out, weights)
    mx.synchronize()
    out_t = torch.as_tensor(out)
    weights_t = torch.as_tensor(weights)
    sync_fn()
    return out_t, weights_t


def bridge_prod(pair, sync_fn):
    out, _weights = pair
    mx.eval(out)
    mx.synchronize()
    out_t = torch.as_tensor(out)
    sync_fn()
    return out_t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="IDEA-Research/grounding-dino-tiny")
    ap.add_argument("--image", type=Path, default=Path("assets/input.jpg"))
    ap.add_argument("--prompt", nargs="+", default=["a cat", "a dog"])
    ap.add_argument("--threadgroup", type=int, default=256)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--pairs", type=int, default=20)

    helpers = {
        13: "bench_fusion_algebraic_specialization.py",
        14: "full_model_fusion_algebraic.py",
        17: "bench_deformable_mlx_island.py",
        18: "full_model_deformable_islands.py",
        30: "bench_fully_folded_fusion_mlx.py",
        31: "full_model_mlx_fusion_paired.py",
        36: "wide_island_mask_sync_fix.py",
        37: "corrected_wide_island_multiprocess.py",
        38: "consolidated_runtime_ablation.py",
        44: "bench_decoder_msda_mlx_wrapper.py",
    }
    for n, fn in helpers.items():
        ap.add_argument(
            f"--exp{n}-helper",
            type=Path,
            default=Path("scripts") / f"{n:02d}_{fn}",
        )
    args = ap.parse_args()

    for n in helpers:
        p = getattr(args, f"exp{n}_helper")
        if not p.exists():
            raise SystemExit(f"Missing helper: {p}")

    if not torch.backends.mps.is_available():
        raise SystemExit("MPS unavailable.")

    h37 = load_module(args.exp37_helper, "mg44e_h37")
    h38 = load_module(args.exp38_helper, "mg44e_h38")
    h44 = load_module(args.exp44_helper, "mg44e_h44")

    print("Building current consolidated runtime...", flush=True)
    rt = h37.build_runtime(args)
    model = rt["model"]
    h14 = rt["h14"]
    inputs = rt["inputs"]

    cache = h38.ExactTextBackboneCache(model.model.text_backbone)

    target = None
    original_forward = None

    try:
        h38.set_runtime_mode(rt, cache, "wide_cache")
        cache.prime(model, inputs, h14.sync)
        cache.set_enabled(True)

        target = model.model.decoder.layers[0].encoder_attn
        original_forward = target.forward
        captured = {}

        target.forward = types.MethodType(
            lambda _self, *a, **kw: capture_wrapper(*a, **kw),
            target,
        )
        capture_wrapper = h44.bind_capture(original_forward, captured)

        print("Capturing real decoder layer0 MSDA workload...", flush=True)
        with torch.inference_mode():
            _ = model(**inputs)
            h14.sync()
        target.forward = original_forward

        shapes = h44.spatial_list_from_capture(captured)
        print(
            f"hidden={tuple(captured['hidden_states'].shape)} "
            f"memory={tuple(captured['encoder_hidden_states'].shape)} "
            f"ref={tuple(captured['reference_points'].shape)} "
            f"spatial={shapes}",
            flush=True,
        )

        ref_kwargs = dict(captured)
        with torch.inference_mode():
            ref_out = original_forward(**ref_kwargs)
            h14.sync()

        h14.sync()
        candidate = h44.MlxDecoderMsdaWrapper(
            target,
            spatial_shapes_list=shapes,
            threadgroup=args.threadgroup,
        )

        print("\nCreating ONE persistent Torch->MLX zero-copy import set...", flush=True)
        mlx_inputs = persistent_inputs(captured, h14.sync)
        print(
            "persistent imports created; these exact arrays will be reused "
            "for warmup, correctness, and timing",
            flush=True,
        )

        print("\nExact 44a-style stagewise warmup on the SAME arrays...", flush=True)
        warm_records, warm_out_t = exact_44a_warm(candidate, mlx_inputs)
        h14.sync()
        print(
            f"stagewise warm bridge PASS shape={tuple(warm_out_t.shape)}",
            flush=True,
        )

        print("\nPersistent-array full UNCOMPILED correctness preflight...", flush=True)
        t0 = time.perf_counter_ns()
        cand_diag = bridge_diag(
            candidate._forward_impl(*mlx_inputs),
            h14.sync,
        )
        preflight_ms = (time.perf_counter_ns() - t0) / 1e6
        print(
            f"persistent full preflight PASS in {preflight_ms:.3f} ms",
            flush=True,
        )

        correctness = {
            "output": compare(cand_diag[0], ref_out[0]),
            "attention_weights": compare(cand_diag[1], ref_out[1]),
        }
        print(json.dumps(correctness, indent=2), flush=True)

        correctness_pass = (
            correctness["output"]["allclose_1e-4"]
            and correctness["attention_weights"]["allclose_1e-4"]
        )

        latency = None
        derived = None

        if correctness_pass:
            print(
                "\nCorrectness passed; persistent-array steady-state "
                "paired benchmark...",
                flush=True,
            )

            _ = bridge_prod(
                candidate._forward_impl(*mlx_inputs),
                h14.sync,
            )

            for i in range(args.warmup):
                h14.sync()
                with torch.inference_mode():
                    _ = original_forward(**ref_kwargs)
                h14.sync()

                _ = bridge_prod(
                    candidate._forward_impl(*mlx_inputs),
                    h14.sync,
                )
                print(f"  warmup {i+1}/{args.warmup}", flush=True)

            current_samples = []
            persistent_samples = []
            deltas = []

            for i in range(args.pairs):
                order = (
                    ("current", "persistent")
                    if i % 2 == 0
                    else ("persistent", "current")
                )
                local = {}

                for mode in order:
                    if mode == "current":
                        h14.sync()
                        t0_ns = time.perf_counter_ns()
                        with torch.inference_mode():
                            _ = original_forward(**ref_kwargs)
                        h14.sync()
                        dt = (time.perf_counter_ns() - t0_ns) / 1e6
                        current_samples.append(dt)
                    else:
                        h14.sync()
                        t0_ns = time.perf_counter_ns()
                        _ = bridge_prod(
                            candidate._forward_impl(*mlx_inputs),
                            h14.sync,
                        )
                        dt = (time.perf_counter_ns() - t0_ns) / 1e6
                        persistent_samples.append(dt)

                    local[mode] = dt

                delta = local["current"] - local["persistent"]
                deltas.append(delta)
                print(
                    f"  pair {i+1:02d}/{args.pairs}: "
                    f"current={local['current']:.3f} ms "
                    f"persistent-mlx={local['persistent']:.3f} ms "
                    f"delta={delta:+.3f} ms",
                    flush=True,
                )

            c = stats(current_samples)
            p = stats(persistent_samples)
            d = stats(deltas)

            latency = {
                "current_pytorch_wrapper_plus_metal_core": c,
                "persistent_import_uncompiled_mlx_wrapper": p,
                "paired_delta_current_minus_persistent": d,
            }
            derived = {
                "speedup_from_medians": c["median_ms"] / p["median_ms"],
                "median_difference_of_marginals_ms":
                    c["median_ms"] - p["median_ms"],
                "paired_delta_median_ms": d["median_ms"],
                "paired_delta_mean_ms": d["mean_ms"],
                "reduction_percent_from_medians":
                    100.0 * (c["median_ms"] - p["median_ms"])
                    / c["median_ms"],
            }

        result = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "experiment": "0044e",
            "purpose": (
                "Reproduce the exact successful 44a execution condition by "
                "retaining one persistent set of zero-copy imported MLX inputs "
                "across stagewise warmup, full correctness, and steady-state "
                "benchmark."
            ),
            "model": args.model,
            "device": "mps",
            "dtype": "float32",
            "software": {
                "torch": torch.__version__,
                "transformers": transformers.__version__,
            },
            "configuration": {
                "decoder_layer": 0,
                "persistent_zero_copy_inputs": True,
                "fresh_import_each_candidate_call": False,
                "mx_compile_used": False,
                "stagewise_warmup_excluded_from_timing": True,
                "timed_final_mlx_to_torch_bridge_included": True,
                "torch_to_mlx_import_cost_in_timed_candidate": False,
                "approximation": False,
                "retraining": False,
                "reduced_precision": False,
            },
            "stagewise_warmup": [
                {"name": name, "ms": ms}
                for name, ms in warm_records
            ],
            "persistent_full_preflight_ms": preflight_ms,
            "correctness": correctness,
            "correctness_gate_passed": correctness_pass,
            "latency": latency,
            "derived": derived,
            "decision_rule": (
                "This is diagnostic, not an adoption benchmark. If persistent "
                "inputs pass while fresh-import variants stall, conclude that "
                "the local decoder-wrapper port is not safely composable with "
                "fresh Torch-produced tensors under the current interop path. "
                "Do not adopt based on persistent-input speed alone."
            ),
            "notes": [
                "This deliberately reproduces the exact input-lifetime condition of successful Experiment 0044a.",
                "The timed candidate excludes repeated Torch-to-MLX import because the same imported arrays are reused.",
                "Therefore any measured speedup is an upper-bound diagnostic, not a full-model integration result.",
                "Experiment 0039 remains the end-to-end latency authority."
            ],
        }

        out = Path("results/metalground_decoder_msda_persistent_import.json")
        out.write_text(
            json.dumps(result, indent=2, ensure_ascii=False) + "\n"
        )

        print("\n=== Experiment 0044e summary ===", flush=True)
        print(f"correctness gate: {correctness_pass}", flush=True)
        if derived is not None:
            print(
                f"current median: "
                f"{latency['current_pytorch_wrapper_plus_metal_core']['median_ms']:.3f} ms",
                flush=True,
            )
            print(
                f"persistent MLX median: "
                f"{latency['persistent_import_uncompiled_mlx_wrapper']['median_ms']:.3f} ms",
                flush=True,
            )
            print(
                f"paired saving: "
                f"{derived['paired_delta_median_ms']:+.3f} ms",
                flush=True,
            )
        print(f"Saved: {out}", flush=True)

    finally:
        if target is not None and original_forward is not None:
            try:
                target.forward = original_forward
            except Exception:
                pass
        cache.restore()
        for d in rt["layer_dispatchers"]:
            d.restore()
        for fd in rt["fusion_dispatchers"]:
            fd.restore()


if __name__ == "__main__":
    main()
