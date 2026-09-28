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
        "finite": bool(
            torch.isfinite(a).all().item()
            and torch.isfinite(b).all().item()
        ),
    }


def bridge_diag(outputs, sync_fn):
    out, weights = outputs
    mx.eval(out, weights)
    mx.synchronize()
    out_t = torch.as_tensor(out)
    weights_t = torch.as_tensor(weights)
    sync_fn()
    return out_t, weights_t


def bridge_prod(outputs, sync_fn):
    out, _weights = outputs
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
    ap.add_argument("--pairs", type=int, default=30)

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

    h37 = load_module(args.exp37_helper, "mg44c_h37")
    h38 = load_module(args.exp38_helper, "mg44c_h38")
    h44 = load_module(args.exp44_helper, "mg44c_h44")

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

        # Import parameters only after all MPS producer work has completed.
        h14.sync()
        candidate = h44.MlxDecoderMsdaWrapper(
            target,
            spatial_shapes_list=shapes,
            threadgroup=args.threadgroup,
        )

        def candidate_uncompiled():
            mlx_inputs = h44.to_mlx_inputs(captured, h14.sync)
            # IMPORTANT: bypass candidate.compiled completely.
            return candidate._forward_impl(*mlx_inputs)

        print("\nUNCOMPILED correctness preflight...", flush=True)
        t0 = time.perf_counter()
        cand_diag = bridge_diag(
            candidate_uncompiled(),
            h14.sync,
        )
        first_uncompiled_ms = (time.perf_counter() - t0) * 1000.0
        print(
            f"First uncompiled full call completed in "
            f"{first_uncompiled_ms:.3f} ms.",
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
                "\nCorrectness passed; bridge-included paired benchmark "
                "(NO mx.compile)...",
                flush=True,
            )

            # Exclude first lazy graph execution / Metal pipeline warmup.
            _ = bridge_prod(candidate_uncompiled(), h14.sync)

            for i in range(args.warmup):
                h14.sync()
                with torch.inference_mode():
                    _ = original_forward(**ref_kwargs)
                h14.sync()

                _ = bridge_prod(candidate_uncompiled(), h14.sync)
                print(f"  warmup {i+1}/{args.warmup}", flush=True)

            current_samples = []
            eager_samples = []
            deltas = []

            for i in range(args.pairs):
                order = (
                    ("current", "eager")
                    if i % 2 == 0
                    else ("eager", "current")
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
                            candidate_uncompiled(),
                            h14.sync,
                        )
                        dt = (time.perf_counter_ns() - t0_ns) / 1e6
                        eager_samples.append(dt)

                    local[mode] = dt

                delta = local["current"] - local["eager"]
                deltas.append(delta)

                print(
                    f"  pair {i+1:02d}/{args.pairs}: "
                    f"current={local['current']:.3f} ms "
                    f"mlx-eager={local['eager']:.3f} ms "
                    f"delta={delta:+.3f} ms",
                    flush=True,
                )

            c = stats(current_samples)
            e = stats(eager_samples)
            d = stats(deltas)

            latency = {
                "current_pytorch_wrapper_plus_metal_core": c,
                "uncompiled_mlx_wrapper_plus_same_metal_core": e,
                "paired_delta_current_minus_eager": d,
            }
            derived = {
                "speedup_from_medians":
                    c["median_ms"] / e["median_ms"],
                "median_difference_of_marginals_ms":
                    c["median_ms"] - e["median_ms"],
                "paired_delta_median_ms": d["median_ms"],
                "paired_delta_mean_ms": d["mean_ms"],
                "reduction_percent_from_medians":
                    100.0 * (c["median_ms"] - e["median_ms"])
                    / c["median_ms"],
            }

        result = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "experiment": "0044c",
            "purpose": (
                "Measure the decoder deformable-attention MLX wrapper without "
                "mx.compile after both whole-wrapper and split-pregraph "
                "compilation stalled on first execution."
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
                "query_shape": list(captured["hidden_states"].shape),
                "memory_shape": list(captured["encoder_hidden_states"].shape),
                "reference_points_shape":
                    list(captured["reference_points"].shape),
                "spatial_shapes_list": shapes,
                "candidate_execution": (
                    "uncompiled MLX lazy/eager wrapper graph -> same custom "
                    "Metal MSDA -> uncompiled MLX output projection"
                ),
                "mx_compile_used": False,
                "timed_bridge_included": True,
                "attention_weights_bridged_in_timing": False,
                "approximation": False,
                "retraining": False,
                "reduced_precision": False,
            },
            "diagnostic_history": {
                "0044_whole_wrapper_compile":
                    "stalled on first compiled execution",
                "0044a_uncompiled_stagewise":
                    "all stages and full uncompiled forward passed",
                "0044b_split_compile":
                    "stalled on first split-compiled preflight",
                "first_uncompiled_full_call_ms":
                    first_uncompiled_ms,
            },
            "correctness": correctness,
            "correctness_gate_passed": correctness_pass,
            "latency": latency,
            "derived": derived,
            "decision_rule": (
                "Proceed to six-decoder-layer integration only if correctness "
                "remains within 1e-4 and the uncompiled bridge-included paired "
                "median improves by at least 1.0 ms or at least 25%. If not, "
                "close the decoder-wrapper MLX port campaign rather than "
                "spending further effort on mx.compile behavior."
            ),
            "notes": [
                "The custom Metal MSDA kernel is identical between current and candidate.",
                "The candidate deliberately bypasses mx.compile.",
                "No forced intermediate eval occurs inside the uncompiled production wrapper; only final output is forced.",
                "Compile-related stalls are treated as a runtime integration constraint, not as a numerical failure.",
                "No approximation, pruning, quantization, retraining, or reduced precision is used.",
                "This is an isolated module benchmark, not a full-model authority."
            ],
        }

        out = Path(
            "results/metalground_decoder_msda_uncompiled_mlx.json"
        )
        out.write_text(
            json.dumps(result, indent=2, ensure_ascii=False) + "\n"
        )

        print("\n=== Experiment 0044c summary ===", flush=True)
        print(f"correctness gate: {correctness_pass}", flush=True)
        if derived is not None:
            print(
                f"current median: "
                f"{latency['current_pytorch_wrapper_plus_metal_core']['median_ms']:.3f} ms",
                flush=True,
            )
            print(
                f"MLX eager median: "
                f"{latency['uncompiled_mlx_wrapper_plus_same_metal_core']['median_ms']:.3f} ms",
                flush=True,
            )
            print(
                f"paired saving: "
                f"{derived['paired_delta_median_ms']:+.3f} ms",
                flush=True,
            )
            print(
                f"speedup: "
                f"{derived['speedup_from_medians']:.3f}x",
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
