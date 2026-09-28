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
import numpy as np
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
        "finite": bool(
            torch.isfinite(a).all().item()
            and torch.isfinite(b).all().item()
        ),
    }


def linear(x, w, b):
    y = x @ w.T
    return y if b is None else y + b


class SplitCompiledDecoderMsda:
    """
    Keep mx.compile away from the custom Metal kernel boundary.

        compiled MLX pregraph
            -> value projection / masking
            -> offsets
            -> attention weights / softmax
            -> sampling locations

        custom Metal MSDA core  (NOT inside mx.compile)

        compiled MLX postgraph
            -> output projection

    No mx.eval / mx.synchronize is inserted between these regions in the
    production path; MLX/Metal dependency tracking remains responsible for
    ordering. Only the final requested outputs are forced.
    """

    def __init__(self, module, spatial_shapes_list, threadgroup):
        self.threadgroup = int(threadgroup)

        self.H = int(module.n_heads)
        self.NL = int(module.n_levels)
        self.NP = int(module.n_points)
        self.DMODEL = int(module.d_model)
        self.HD = self.DMODEL // self.H

        self.p = {}

        def imp(prefix, lin):
            self.p[prefix + "_w"] = mx.asarray(lin.weight, copy=False)
            self.p[prefix + "_b"] = (
                None if lin.bias is None
                else mx.asarray(lin.bias, copy=False)
            )

        imp("sampling", module.sampling_offsets)
        imp("attn_w", module.attention_weights)
        imp("value", module.value_proj)
        imp("output", module.output_proj)

        shapes = [(int(h), int(w)) for h, w in spatial_shapes_list]
        starts = []
        acc = 0
        for h, w in shapes:
            starts.append(acc)
            acc += h * w

        self.spatial_shapes = mx.array(
            np.asarray(shapes, dtype=np.int32), dtype=mx.int32
        )
        self.level_start = mx.array(
            np.asarray(starts, dtype=np.int32), dtype=mx.int32
        )
        self.offset_normalizer = mx.array(
            np.asarray([[w, h] for h, w in shapes], dtype=np.float32),
            dtype=mx.float32,
        )

        arrays = [
            self.spatial_shapes,
            self.level_start,
            self.offset_normalizer,
        ] + [v for v in self.p.values() if v is not None]
        mx.eval(*arrays)
        mx.synchronize()

        self.pre_compiled = mx.compile(self._pre)
        self.post_compiled = mx.compile(self._post)

    def _pre(
        self,
        hidden_states,
        attention_mask,
        encoder_hidden_states,
        position_embeddings,
        reference_points,
    ):
        query = (
            hidden_states
            if position_embeddings is None
            else hidden_states + position_embeddings
        )

        B, Q, _ = query.shape
        _, S, _ = encoder_hidden_states.shape

        value = linear(
            encoder_hidden_states,
            self.p["value_w"],
            self.p["value_b"],
        )
        if attention_mask is not None:
            value = mx.where(attention_mask[..., None], value, 0.0)
        value = value.reshape(B, S, self.H, self.HD)

        offsets = linear(
            query,
            self.p["sampling_w"],
            self.p["sampling_b"],
        ).reshape(B, Q, self.H, self.NL, self.NP, 2)

        weights = linear(
            query,
            self.p["attn_w_w"],
            self.p["attn_w_b"],
        ).reshape(B, Q, self.H, self.NL * self.NP)
        weights = mx.softmax(weights, axis=-1).reshape(
            B, Q, self.H, self.NL, self.NP
        )

        if reference_points.shape[-1] == 2:
            locations = (
                reference_points[:, :, None, :, None, :]
                + offsets
                / self.offset_normalizer[
                    None, None, None, :, None, :
                ]
            )
        elif reference_points.shape[-1] == 4:
            locations = (
                reference_points[:, :, None, :, None, :2]
                + offsets / float(self.NP)
                * reference_points[
                    :, :, None, :, None, 2:
                ]
                * 0.5
            )
        else:
            raise ValueError(
                f"Unsupported reference point dim: "
                f"{reference_points.shape[-1]}"
            )

        return value, locations, weights

    def _post(self, core):
        return linear(
            core,
            self.p["output_w"],
            self.p["output_b"],
        )

    def __call__(
        self,
        hidden_states,
        attention_mask,
        encoder_hidden_states,
        position_embeddings,
        reference_points,
    ):
        value, locations, weights = self.pre_compiled(
            hidden_states,
            attention_mask,
            encoder_hidden_states,
            position_embeddings,
            reference_points,
        )

        core = msda_metal_v0(
            value,
            self.spatial_shapes,
            self.level_start,
            locations,
            weights,
            threadgroup_size=self.threadgroup,
        )
        out = self.post_compiled(core)
        return out, weights


def to_mlx_inputs(captured, sync_fn):
    # True producer sync before any zero-copy imports.
    sync_fn()
    return (
        mx.asarray(captured["hidden_states"], copy=False),
        (
            None
            if captured.get("attention_mask") is None
            else mx.asarray(captured["attention_mask"], copy=False)
        ),
        mx.asarray(captured["encoder_hidden_states"], copy=False),
        (
            None
            if captured.get("position_embeddings") is None
            else mx.asarray(captured["position_embeddings"], copy=False)
        ),
        mx.asarray(captured["reference_points"], copy=False),
    )


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

    h37 = load_module(args.exp37_helper, "mg44b_h37")
    h38 = load_module(args.exp38_helper, "mg44b_h38")
    h44 = load_module(args.exp44_helper, "mg44b_h44")

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
        capture_wrapper = h44.bind_capture(
            original_forward, captured
        )

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

        # Current exact reference.
        ref_kwargs = dict(captured)
        with torch.inference_mode():
            ref_out = original_forward(**ref_kwargs)
            h14.sync()

        # Import candidate parameters only after current MPS work is complete.
        h14.sync()
        candidate = SplitCompiledDecoderMsda(
            target,
            spatial_shapes_list=shapes,
            threadgroup=args.threadgroup,
        )

        print(
            "\nSplit-compiled correctness preflight "
            "(first pre/post compile execution)...",
            flush=True,
        )
        t0 = time.perf_counter()
        cand_diag = bridge_diag(
            candidate(*to_mlx_inputs(captured, h14.sync)),
            h14.sync,
        )
        first_call_ms = (time.perf_counter() - t0) * 1000.0
        print(
            f"First split-compiled call completed in "
            f"{first_call_ms:.3f} ms (compile included).",
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
                "\nCorrectness passed; bridge-included paired benchmark...",
                flush=True,
            )

            # One extra candidate execution to ensure both compiled regions
            # are fully warm before timed samples.
            _ = bridge_prod(
                candidate(*to_mlx_inputs(captured, h14.sync)),
                h14.sync,
            )

            for i in range(args.warmup):
                with torch.inference_mode():
                    h14.sync()
                    _ = original_forward(**ref_kwargs)
                    h14.sync()

                _ = bridge_prod(
                    candidate(*to_mlx_inputs(captured, h14.sync)),
                    h14.sync,
                )
                print(f"  warmup {i+1}/{args.warmup}", flush=True)

            current_samples = []
            split_samples = []
            deltas = []

            for i in range(args.pairs):
                order = (
                    ("current", "split")
                    if i % 2 == 0
                    else ("split", "current")
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
                            candidate(
                                *to_mlx_inputs(captured, h14.sync)
                            ),
                            h14.sync,
                        )
                        dt = (time.perf_counter_ns() - t0_ns) / 1e6
                        split_samples.append(dt)

                    local[mode] = dt

                delta = local["current"] - local["split"]
                deltas.append(delta)

                print(
                    f"  pair {i+1:02d}/{args.pairs}: "
                    f"current={local['current']:.3f} ms "
                    f"split={local['split']:.3f} ms "
                    f"delta={delta:+.3f} ms",
                    flush=True,
                )

            c = stats(current_samples)
            s = stats(split_samples)
            d = stats(deltas)

            latency = {
                "current_pytorch_wrapper_plus_metal_core": c,
                "split_compiled_mlx_plus_metal_core": s,
                "paired_delta_current_minus_split": d,
            }
            derived = {
                "speedup_from_medians":
                    c["median_ms"] / s["median_ms"],
                "median_difference_of_marginals_ms":
                    c["median_ms"] - s["median_ms"],
                "paired_delta_median_ms": d["median_ms"],
                "paired_delta_mean_ms": d["mean_ms"],
                "reduction_percent_from_medians":
                    100.0 * (c["median_ms"] - s["median_ms"])
                    / c["median_ms"],
            }

        result = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "experiment": "0044b",
            "purpose": (
                "Retry decoder deformable-attention MLX wrapper with mx.compile "
                "split around the custom Metal MSDA kernel after Experiment "
                "0044 whole-wrapper compilation stalled."
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
                    "compiled MLX pregraph -> custom Metal MSDA outside "
                    "mx.compile -> compiled MLX output projection"
                ),
                "intermediate_eval_or_sync": False,
                "timed_bridge_included": True,
                "approximation": False,
                "retraining": False,
                "reduced_precision": False,
            },
            "diagnostic_prior": {
                "whole_wrapper_mx_compile": "stalled on first execution",
                "uncompiled_stagewise_path": "all stages passed",
                "uncompiled_full_forward": "passed",
                "first_split_compiled_call_ms_compile_included":
                    first_call_ms,
            },
            "correctness": correctness,
            "correctness_gate_passed": correctness_pass,
            "latency": latency,
            "derived": derived,
            "decision_rule": (
                "Proceed to all-six-decoder integration if correctness remains "
                "within 1e-4 and bridge-included paired median improvement is "
                "at least 1.0 ms or at least 25%. Otherwise close this wrapper "
                "port path."
            ),
            "notes": [
                "The custom Metal MSDA kernel is identical to the current runtime.",
                "mx.compile does not enclose the custom Metal kernel.",
                "No forced intermediate evaluation occurs between pregraph, Metal kernel, and postgraph.",
                "Compile time is excluded from timed samples.",
                "No approximation, pruning, quantization, retraining, or reduced precision is used.",
                "This is an isolated module benchmark, not a full-model authority."
            ],
        }

        out = Path(
            "results/metalground_decoder_msda_split_compile.json"
        )
        out.write_text(
            json.dumps(result, indent=2, ensure_ascii=False) + "\n"
        )

        print("\n=== Experiment 0044b summary ===", flush=True)
        print(f"correctness gate: {correctness_pass}", flush=True)
        if derived is not None:
            print(
                f"current median: "
                f"{latency['current_pytorch_wrapper_plus_metal_core']['median_ms']:.3f} ms",
                flush=True,
            )
            print(
                f"split median:   "
                f"{latency['split_compiled_mlx_plus_metal_core']['median_ms']:.3f} ms",
                flush=True,
            )
            print(
                f"paired saving:  "
                f"{derived['paired_delta_median_ms']:+.3f} ms",
                flush=True,
            )
            print(
                f"speedup:        "
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
