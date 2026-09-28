#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import inspect
import json
import math
import statistics
import sys
import time
import types
from collections import defaultdict
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


def linear(x, w, b):
    y = x @ w.T
    return y if b is None else y + b


class MlxSwinAttention:
    """
    Exact FP32 SwinAttention specialization for a fixed captured geometry.

    Static work moved out of the measured path:
      - learned relative-position bias lookup/reshape
      - expansion and addition of the fixed shifted-window mask, when present

    Dynamic measured work:
      q/k/v projections -> scaled QK^T -> add static combined bias ->
      softmax -> AV -> output projection.
    """

    def __init__(self, module, captured_mask, sync_fn):
        self.H = int(module.num_attention_heads)
        self.HD = int(module.head_dim)
        self.scaling = float(module.scaling)

        def imp(prefix, lin):
            setattr(self, prefix + "_w", mx.asarray(lin.weight, copy=False))
            setattr(
                self,
                prefix + "_b",
                None if lin.bias is None else mx.asarray(lin.bias, copy=False),
            )

        imp("q", module.q_proj)
        imp("k", module.k_proj)
        imp("v", module.v_proj)
        imp("o", module.o_proj)

        # Produce the exact learned relative-position bias with the reference
        # module, then freeze only geometry-dependent work.
        sync_fn()
        with torch.inference_mode():
            relative_bias_t = module.relative_position_bias()
        sync_fn()

        # relative_bias_t: [1, H, N, N]
        if captured_mask is not None:
            num_windows = int(captured_mask.shape[0])
            # The captured attention input can contain multiple image batches.
            # We build the exact expanded shape used by HF forward.
            self.mask_num_windows = num_windows

            # Keep the raw captured static mask in Torch only long enough to
            # construct the combined bias on-device.
            input_batch_windows = None  # resolved in prepare_for_shape()
            self.relative_bias_t = relative_bias_t.detach()
            self.captured_mask_t = captured_mask.detach()
            self.static_combined = None
        else:
            self.mask_num_windows = None
            self.relative_bias_t = None
            self.captured_mask_t = None
            self.static_combined = mx.asarray(relative_bias_t, copy=False)

        arrays = [
            self.q_w, self.k_w, self.v_w, self.o_w
        ]
        for b in (self.q_b, self.k_b, self.v_b, self.o_b):
            if b is not None:
                arrays.append(b)
        if self.static_combined is not None:
            arrays.append(self.static_combined)
        mx.eval(*arrays)
        mx.synchronize()

        self._prepared_shape = None
        self.compiled = mx.compile(self._forward)

    def prepare_for_shape(self, input_shape, sync_fn):
        """
        Build one exact static combined bias for the captured fixed geometry.
        """
        Bwin, N, _C = [int(x) for x in input_shape]

        if self.mask_num_windows is None:
            self._prepared_shape = (Bwin, N)
            return

        if Bwin % self.mask_num_windows != 0:
            raise RuntimeError(
                f"Bwin={Bwin} not divisible by mask windows="
                f"{self.mask_num_windows}"
            )

        batch_size = Bwin // self.mask_num_windows

        sync_fn()
        rb = self.relative_bias_t
        mask = self.captured_mask_t

        # HF semantics:
        # mask [nW,N,N]
        # -> [batch,nW,1,N,N]
        # -> [Bwin,1,N,N]
        expanded_mask = (
            mask.unsqueeze(1)
            .unsqueeze(0)
            .expand(batch_size, -1, -1, -1, -1)
            .reshape(-1, 1, N, N)
        )
        combined_t = rb + expanded_mask
        sync_fn()

        self.static_combined = mx.asarray(combined_t, copy=False)
        mx.eval(self.static_combined)
        mx.synchronize()
        self._prepared_shape = (Bwin, N)

    def _forward(self, x):
        B, N, C = x.shape

        q = linear(x, self.q_w, self.q_b).reshape(
            B, N, self.H, self.HD
        )
        k = linear(x, self.k_w, self.k_b).reshape(
            B, N, self.H, self.HD
        )
        v = linear(x, self.v_w, self.v_b).reshape(
            B, N, self.H, self.HD
        )

        q = mx.transpose(q, (0, 2, 1, 3))
        k = mx.transpose(k, (0, 2, 1, 3))
        v = mx.transpose(v, (0, 2, 1, 3))

        scores = (q @ mx.transpose(k, (0, 1, 3, 2))) * self.scaling
        scores = scores + self.static_combined

        weights = mx.softmax(scores, axis=-1)
        out = weights @ v
        out = mx.transpose(out, (0, 2, 1, 3)).reshape(B, N, C)
        out = linear(out, self.o_w, self.o_b)

        return out, weights

    def __call__(self, x):
        return self.compiled(x)


def capture_attention(module, store):
    original = module.forward
    sig = inspect.signature(original)

    def wrapper(self, *args, **kwargs):
        bound = sig.bind_partial(*args, **kwargs)
        bound.apply_defaults()

        if "hidden_states" not in bound.arguments:
            raise RuntimeError(
                f"Could not capture hidden_states from {sig}"
            )

        if "hidden_states" not in store:
            store["hidden_states"] = (
                bound.arguments["hidden_states"].detach()
            )
            mask = bound.arguments.get("attention_mask")
            store["attention_mask"] = (
                None if mask is None else mask.detach()
            )

        out = original(*args, **kwargs)

        if "reference_output" not in store:
            store["reference_output"] = out[0].detach()
            store["reference_weights"] = out[1].detach()

        return out

    module.forward = types.MethodType(wrapper, module)
    return original


def run_candidate(candidate, x_torch, sync_fn, return_weights=False):
    sync_fn()
    x_mx = mx.asarray(x_torch, copy=False)
    out_mx, w_mx = candidate(x_mx)

    if return_weights:
        mx.eval(out_mx, w_mx)
        mx.synchronize()
        out_t = torch.as_tensor(out_mx)
        w_t = torch.as_tensor(w_mx)
        sync_fn()
        return out_t, w_t

    # Production-style timing path: only the attention output crosses back.
    mx.eval(out_mx)
    mx.synchronize()
    out_t = torch.as_tensor(out_mx)
    sync_fn()
    return out_t


def bench_one(name, module, store, sync_fn, warmup, pairs):
    hidden = store["hidden_states"]
    mask = store["attention_mask"]

    # Reference re-run on exactly captured inputs.
    sync_fn()
    with torch.inference_mode():
        ref = module(hidden, mask)
    sync_fn()

    # Import parameters/static geometry after all Torch producers finish.
    sync_fn()
    candidate = MlxSwinAttention(module, mask, sync_fn)
    candidate.prepare_for_shape(hidden.shape, sync_fn)

    print(
        f"\n{name}: shape={tuple(hidden.shape)} "
        f"mask={None if mask is None else tuple(mask.shape)}",
        flush=True,
    )
    print("  correctness preflight...", flush=True)

    cand_out, cand_w = run_candidate(
        candidate, hidden, sync_fn, return_weights=True
    )

    correctness = {
        "output": compare(cand_out, ref[0]),
        "attention_weights": compare(cand_w, ref[1]),
    }
    print(json.dumps(correctness, indent=2), flush=True)

    correctness_pass = (
        correctness["output"]["allclose_1e-4"]
        and correctness["attention_weights"]["allclose_1e-4"]
    )

    if not correctness_pass:
        return {
            "name": name,
            "input_shape": list(hidden.shape),
            "attention_mask_shape":
                None if mask is None else list(mask.shape),
            "correctness": correctness,
            "correctness_gate_passed": False,
            "latency": None,
            "derived": None,
        }

    # The first compiled call happened above and is excluded.
    for i in range(warmup):
        sync_fn()
        with torch.inference_mode():
            _ = module(hidden, mask)
        sync_fn()
        _ = run_candidate(candidate, hidden, sync_fn, return_weights=False)
        print(f"  warmup {i+1}/{warmup}", flush=True)

    current_samples = []
    mlx_samples = []
    deltas = []

    print(f"  paired benchmark x{pairs}...", flush=True)
    for i in range(pairs):
        order = (
            ("current", "mlx")
            if i % 2 == 0 else
            ("mlx", "current")
        )
        local = {}

        for mode in order:
            if mode == "current":
                sync_fn()
                t0 = time.perf_counter_ns()
                with torch.inference_mode():
                    _ = module(hidden, mask)
                sync_fn()
                dt = (time.perf_counter_ns() - t0) / 1e6
                current_samples.append(dt)
            else:
                sync_fn()
                t0 = time.perf_counter_ns()
                _ = run_candidate(
                    candidate, hidden, sync_fn, return_weights=False
                )
                dt = (time.perf_counter_ns() - t0) / 1e6
                mlx_samples.append(dt)

            local[mode] = dt

        delta = local["current"] - local["mlx"]
        deltas.append(delta)

        print(
            f"    {i+1:02d}/{pairs}: "
            f"current={local['current']:.3f} ms "
            f"mlx={local['mlx']:.3f} ms "
            f"delta={delta:+.3f} ms",
            flush=True,
        )

    c = stats(current_samples)
    m = stats(mlx_samples)
    d = stats(deltas)

    return {
        "name": name,
        "input_shape": list(hidden.shape),
        "attention_mask_shape":
            None if mask is None else list(mask.shape),
        "correctness": correctness,
        "correctness_gate_passed": True,
        "latency": {
            "current_pytorch_mps": c,
            "compiled_mlx_static_geometry_bridge_included": m,
            "paired_delta_current_minus_mlx": d,
        },
        "derived": {
            "speedup_from_medians": c["median_ms"] / m["median_ms"],
            "median_difference_of_marginals_ms":
                c["median_ms"] - m["median_ms"],
            "paired_delta_median_ms": d["median_ms"],
            "paired_delta_mean_ms": d["mean_ms"],
            "reduction_percent_from_medians":
                100.0 * (c["median_ms"] - m["median_ms"])
                / c["median_ms"],
        },
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="IDEA-Research/grounding-dino-tiny")
    ap.add_argument("--image", type=Path, default=Path("assets/input.jpg"))
    ap.add_argument("--prompt", nargs="+", default=["a cat", "a dog"])
    ap.add_argument("--threadgroup", type=int, default=256)
    ap.add_argument("--warmup", type=int, default=2)
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
        45: "bench_swin_stage0_mlp_mlx.py",
        46: "full_model_stage0_mlp_mlx.py",
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

    h37 = load_module(args.exp37_helper, "mg49_h37")
    h38 = load_module(args.exp38_helper, "mg49_h38")
    h45 = load_module(args.exp45_helper, "mg49_h45")
    h46 = load_module(args.exp46_helper, "mg49_h46")

    print("Building adopted runtime...", flush=True)
    rt = h37.build_runtime(args)
    model = rt["model"]
    h14 = rt["h14"]
    inputs = rt["inputs"]

    cache = h38.ExactTextBackboneCache(model.model.text_backbone)
    stage0_mlp_dispatchers = []
    capture_originals = []

    try:
        # Adopted base = 0039 + stage0 MLP optimization from 0047.
        h38.set_runtime_mode(rt, cache, "wide_cache")
        cache.prime(model, inputs, h14.sync)
        cache.set_enabled(True)

        stages = model.model.backbone.conv_encoder.model.swin.encoder.layers

        stage0_mlp_modules = [
            stages[0].blocks[0].mlp,
            stages[0].blocks[1].mlp,
        ]
        h14.sync()
        stage0_mlp_candidates = [
            h45.MlxSwinMLP(m) for m in stage0_mlp_modules
        ]
        stage0_mlp_dispatchers = [
            h46.SwinMlpDispatcher(m, c, h14.sync)
            for m, c in zip(
                stage0_mlp_modules, stage0_mlp_candidates
            )
        ]
        for d in stage0_mlp_dispatchers:
            d.set_mode("mlx")

        # Prime adopted stage0 MLP path.
        with torch.inference_mode():
            _ = model(**inputs)
        h14.sync()

        # Capture every SwinAttention input/mask under the adopted runtime.
        targets = []
        for stage_idx, stage in enumerate(stages):
            for block_idx, block in enumerate(stage.blocks):
                name = (
                    f"stage{stage_idx}.block{block_idx}.attention"
                )
                targets.append(
                    (
                        stage_idx,
                        block_idx,
                        name,
                        block.attention,
                        {},
                    )
                )

        for _si, _bi, _name, module, store in targets:
            capture_originals.append(
                (module, capture_attention(module, store))
            )

        print(
            f"Capturing {len(targets)} Swin attention workloads...",
            flush=True,
        )
        with torch.inference_mode():
            _ = model(**inputs)
        h14.sync()

        for module, original in capture_originals:
            module.forward = original
        capture_originals.clear()

        for si, bi, name, _module, store in targets:
            print(
                f"  {name}: hidden="
                f"{tuple(store['hidden_states'].shape)} "
                f"mask="
                f"{None if store['attention_mask'] is None else tuple(store['attention_mask'].shape)}",
                flush=True,
            )

        results = []
        for si, bi, name, module, store in targets:
            print(f"\n=== {name} ===", flush=True)
            r = bench_one(
                name,
                module,
                store,
                h14.sync,
                args.warmup,
                args.pairs,
            )
            r["stage_index"] = si
            r["block_index"] = bi
            results.append(r)

        by_stage = defaultdict(list)
        for r in results:
            by_stage[r["stage_index"]].append(r)

        stage_summaries = {}
        selected_stages = []

        for stage_idx in sorted(by_stage):
            rs = by_stage[stage_idx]
            correct = all(
                r["correctness_gate_passed"] for r in rs
            )
            paired = [
                r["derived"]["paired_delta_median_ms"]
                for r in rs if r["derived"] is not None
            ]
            reductions = [
                r["derived"]["reduction_percent_from_medians"]
                for r in rs if r["derived"] is not None
            ]
            all_positive = (
                len(paired) == len(rs)
                and all(x > 0.0 for x in paired)
            )
            med_red = (
                statistics.median(reductions)
                if reductions else None
            )
            med_save = (
                statistics.median(paired)
                if paired else None
            )

            selected = bool(
                correct
                and all_positive
                and med_red is not None
                and med_red >= 15.0
            )
            if selected:
                selected_stages.append(stage_idx)

            stage_summaries[str(stage_idx)] = {
                "blocks": len(rs),
                "correctness_all_passed": correct,
                "all_block_paired_medians_positive": all_positive,
                "per_block_paired_medians_ms": paired,
                "median_block_paired_saving_ms": med_save,
                "per_block_reduction_percent": reductions,
                "median_block_reduction_percent": med_red,
                "selected_for_full_model_test": selected,
            }

        result = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "experiment": "0049",
            "purpose": (
                "Exact real-workload benchmark of all Swin attention "
                "submodules under the adopted Experiment-0047 runtime, "
                "testing compiled MLX attention plus fixed-geometry "
                "relative-position/mask partial evaluation."
            ),
            "model": args.model,
            "device": "mps",
            "dtype": "float32",
            "software": {
                "torch": torch.__version__,
                "transformers": transformers.__version__,
            },
            "configuration": {
                "runtime_during_capture": (
                    "current adopted runtime including exact compiled "
                    "MLX stage0 block0/block1 MLPs"
                ),
                "targets": [r["name"] for r in results],
                "candidate": (
                    "compiled MLX q/k/v -> scaled attention -> exact "
                    "softmax -> value aggregation -> output projection"
                ),
                "static_specialization": (
                    "learned relative-position bias lookup and fixed "
                    "shift-mask expansion/addition are precomputed exactly "
                    "for the captured fixed input geometry"
                ),
                "fresh_hidden_torch_to_mlx_bridge_each_call": True,
                "attention_mask_bridge_in_timing": False,
                "final_output_mlx_to_torch_bridge_each_call": True,
                "attention_weights_bridge_in_timing": False,
                "warmup_per_target": args.warmup,
                "pairs_per_target": args.pairs,
                "approximation": False,
                "retraining": False,
                "reduced_precision": False,
            },
            "results": results,
            "stage_summaries": stage_summaries,
            "selected_stages_for_full_model_test": selected_stages,
            "stage_selection_rule": (
                "Proceed to full-model testing for a stage only if every "
                "attention block passes 1e-4 correctness, every block has "
                "positive paired median saving, and the stage median of "
                "block-level reduction percentages is at least 15%. "
                "Microbenchmark savings are selection evidence only."
            ),
            "notes": [
                "This does not replace whole Swin blocks.",
                "Window partition/reverse, LayerNorm, residuals, dropout and patch merging remain outside the candidate.",
                "Static mask/bias specialization is exact only for unchanged input geometry and model state.",
                "Experiment 0047 remains the robust runtime authority until a later full-model candidate passes fresh-process replication."
            ],
        }

        out = Path(
            "results/metalground_swin_attention_mlx.json"
        )
        out.write_text(
            json.dumps(result, indent=2, ensure_ascii=False) + "\n"
        )

        print("\n=== Experiment 0049 summary ===", flush=True)
        for stage_idx in sorted(
            stage_summaries, key=int
        ):
            s = stage_summaries[stage_idx]
            print(
                f"stage{stage_idx}: "
                f"correct={s['correctness_all_passed']} "
                f"all-positive="
                f"{s['all_block_paired_medians_positive']} "
                f"median reduction="
                f"{s['median_block_reduction_percent']:.2f}% "
                f"selected="
                f"{s['selected_for_full_model_test']}",
                flush=True,
            )
        print(
            "selected stages: "
            + (
                ", ".join(
                    f"stage{x}" for x in selected_stages
                )
                if selected_stages else "none"
            ),
            flush=True,
        )
        print(f"Saved: {out}", flush=True)

    finally:
        for module, original in capture_originals:
            try:
                module.forward = original
            except Exception:
                pass
        for d in stage0_mlp_dispatchers:
            try:
                d.restore()
            except Exception:
                pass
        cache.restore()
        for d in rt["layer_dispatchers"]:
            d.restore()
        for fd in rt["fusion_dispatchers"]:
            fd.restore()


if __name__ == "__main__":
    main()
