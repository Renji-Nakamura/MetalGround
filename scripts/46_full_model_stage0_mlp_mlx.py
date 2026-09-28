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


def tensor_error(a: torch.Tensor, b: torch.Tensor):
    af = a.detach().float()
    bf = b.detach().float()
    fa = torch.isfinite(af)
    fb = torch.isfinite(bf)
    common = fa & fb

    if bool(common.any().item()):
        d = (af[common] - bf[common]).abs()
        max_abs = float(d.max().item())
        mean_abs = float(d.mean().item())
        rmse = float(
            torch.sqrt(torch.mean((af[common] - bf[common]) ** 2)).item()
        )
    else:
        max_abs = mean_abs = rmse = 0.0

    return {
        "finite_mask_equal": bool(torch.equal(fa, fb)),
        "max_abs": max_abs,
        "mean_abs": mean_abs,
        "rmse": rmse,
        "allclose_1e-5": bool(torch.allclose(a, b, rtol=1e-5, atol=1e-5)),
        "allclose_1e-4": bool(torch.allclose(a, b, rtol=1e-4, atol=1e-4)),
    }


def audit(a, b, model, h14):
    return {
        "logits": tensor_error(a.logits, b.logits),
        "pred_boxes": tensor_error(a.pred_boxes, b.pred_boxes),
        "topk": h14.topk_audit(a, b, model.config.num_queries),
    }


class SwinMlpDispatcher:
    def __init__(self, module, candidate, sync_fn):
        self.module = module
        self.candidate = candidate
        self.sync_fn = sync_fn
        self.original_forward = module.forward
        self.mode = "current"
        self.current_calls = 0
        self.mlx_calls = 0

        def dispatched(_module_self, hidden_states, *args, **kwargs):
            if args or kwargs:
                # SwinMLP is expected to receive only hidden_states in the
                # installed Transformers 5.17 implementation.
                if self.mode == "current":
                    self.current_calls += 1
                    return self.original_forward(hidden_states, *args, **kwargs)
                raise RuntimeError(
                    f"Unexpected SwinMLP args/kwargs in MLX mode: "
                    f"args={len(args)} kwargs={sorted(kwargs.keys())}"
                )

            if self.mode == "current":
                self.current_calls += 1
                return self.original_forward(hidden_states)

            self.mlx_calls += 1

            # Synchronize at the actual PyTorch producer boundary before the
            # zero-copy shared-Metal import.
            self.sync_fn()
            x_mx = mx.asarray(hidden_states, copy=False)
            y_mx = self.candidate(x_mx)
            mx.eval(y_mx)
            mx.synchronize()

            y_t = torch.as_tensor(y_mx)
            self.sync_fn()
            return y_t

        module.forward = types.MethodType(dispatched, module)

    def set_mode(self, mode):
        if mode not in ("current", "mlx"):
            raise ValueError(mode)
        self.mode = mode

    def reset_counts(self):
        self.current_calls = 0
        self.mlx_calls = 0

    def restore(self):
        self.module.forward = self.original_forward


def timed_forward(model, inputs, sync_fn):
    sync_fn()
    t0 = time.perf_counter_ns()
    with torch.inference_mode():
        out = model(**inputs)
    sync_fn()
    return out, (time.perf_counter_ns() - t0) / 1e6


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="IDEA-Research/grounding-dino-tiny")
    ap.add_argument("--image", type=Path, default=Path("assets/input.jpg"))
    ap.add_argument("--prompt", nargs="+", default=["a cat", "a dog"])
    ap.add_argument("--threadgroup", type=int, default=256)
    ap.add_argument("--warmup-per-mode", type=int, default=3)
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

    h37 = load_module(args.exp37_helper, "mg46_h37")
    h38 = load_module(args.exp38_helper, "mg46_h38")
    h45 = load_module(args.exp45_helper, "mg46_h45")

    print("Building current consolidated runtime...", flush=True)
    rt = h37.build_runtime(args)
    model = rt["model"]
    h14 = rt["h14"]
    inputs = rt["inputs"]

    cache = h38.ExactTextBackboneCache(model.model.text_backbone)
    dispatchers = []

    try:
        # Both modes start from the adopted consolidated runtime.
        h38.set_runtime_mode(rt, cache, "wide_cache")
        cache.prime(model, inputs, h14.sync)
        cache.set_enabled(True)

        stages = model.model.backbone.conv_encoder.model.swin.encoder.layers
        stage0 = stages[0]
        mlp_modules = [
            stage0.blocks[0].mlp,
            stage0.blocks[1].mlp,
        ]

        # Synchronize before importing MPS parameters into MLX.
        h14.sync()
        candidates = [h45.MlxSwinMLP(m) for m in mlp_modules]
        dispatchers = [
            SwinMlpDispatcher(m, c, h14.sync)
            for m, c in zip(mlp_modules, candidates)
        ]

        def set_mode(mode):
            for d in dispatchers:
                d.set_mode(mode)

        # Compile candidates outside timed measurements.
        print("Priming both MLX MLP candidates...", flush=True)
        set_mode("mlx")
        with torch.inference_mode():
            _ = model(**inputs)
        h14.sync()

        # Full-model correctness references.
        print("Running full-model correctness preflight...", flush=True)

        set_mode("current")
        with torch.inference_mode():
            current_ref = model(**inputs)
        h14.sync()

        set_mode("mlx")
        with torch.inference_mode():
            candidate_ref = model(**inputs)
        h14.sync()

        correctness = {
            "candidate_vs_current": audit(
                candidate_ref, current_ref, model, h14
            ),
            "current_vs_original": audit(
                current_ref, rt["original_ref"], model, h14
            ),
            "candidate_vs_original": audit(
                candidate_ref, rt["original_ref"], model, h14
            ),
        }

        print(
            json.dumps(
                {k: v["topk"] for k, v in correctness.items()},
                indent=2,
            ),
            flush=True,
        )

        membership_pass = (
            correctness["candidate_vs_current"]["topk"]["set_overlap"] == 900
            and correctness["candidate_vs_current"]["topk"][
                "changed_membership_each_side"
            ] == 0
        )

        latency = None
        derived = None
        call_validation = None

        if membership_pass:
            print(
                "Top-k membership passed; paired full-model benchmark...",
                flush=True,
            )

            # Balanced warmup.
            for i in range(args.warmup_per_mode):
                set_mode("current")
                _ = timed_forward(model, inputs, h14.sync)

                set_mode("mlx")
                _ = timed_forward(model, inputs, h14.sync)

                print(
                    f"  warmup pair {i+1}/{args.warmup_per_mode}",
                    flush=True,
                )

            # Benchmark-only counters.
            for d in dispatchers:
                d.reset_counts()
            cache.reset_counts()
            rt["msda_state"].call_count = 0

            current_samples = []
            candidate_samples = []
            deltas = []

            for i in range(args.pairs):
                order = (
                    ("current", "mlx")
                    if i % 2 == 0
                    else ("mlx", "current")
                )
                local = {}

                for mode in order:
                    set_mode(mode)
                    _out, dt = timed_forward(model, inputs, h14.sync)
                    local[mode] = dt
                    if mode == "current":
                        current_samples.append(dt)
                    else:
                        candidate_samples.append(dt)

                delta = local["current"] - local["mlx"]
                deltas.append(delta)

                print(
                    f"  pair {i+1:02d}/{args.pairs}: "
                    f"current={local['current']:.3f} ms "
                    f"stage0-mlx={local['mlx']:.3f} ms "
                    f"delta={delta:+.3f} ms",
                    flush=True,
                )

            c = stats(current_samples)
            m = stats(candidate_samples)
            d = stats(deltas)

            latency = {
                "current_consolidated": c,
                "stage0_two_mlp_mlx": m,
                "paired_delta_current_minus_candidate": d,
            }
            derived = {
                "speedup_from_medians": c["median_ms"] / m["median_ms"],
                "median_difference_of_marginals_ms":
                    c["median_ms"] - m["median_ms"],
                "paired_delta_median_ms": d["median_ms"],
                "paired_delta_mean_ms": d["mean_ms"],
                "candidate_median_fps": 1000.0 / m["median_ms"],
                "reduction_percent_from_medians":
                    100.0 * (c["median_ms"] - m["median_ms"])
                    / c["median_ms"],
            }

            expected_each_mode_per_mlp = args.pairs
            expected_cache_hits = args.pairs * 2
            expected_decoder_msda = args.pairs * 2 * 6

            call_validation = {
                "block0_current_calls": dispatchers[0].current_calls,
                "block0_mlx_calls": dispatchers[0].mlx_calls,
                "block1_current_calls": dispatchers[1].current_calls,
                "block1_mlx_calls": dispatchers[1].mlx_calls,
                "expected_each_mode_per_mlp":
                    expected_each_mode_per_mlp,
                "cache_hits": cache.hits,
                "cache_misses": cache.misses,
                "expected_cache_hits": expected_cache_hits,
                "decoder_metal_msda_calls": rt["msda_state"].call_count,
                "expected_decoder_metal_msda_calls":
                    expected_decoder_msda,
            }

            for j, disp in enumerate(dispatchers):
                if disp.current_calls != expected_each_mode_per_mlp:
                    raise RuntimeError(
                        f"block{j} current calls {disp.current_calls} "
                        f"!= {expected_each_mode_per_mlp}"
                    )
                if disp.mlx_calls != expected_each_mode_per_mlp:
                    raise RuntimeError(
                        f"block{j} MLX calls {disp.mlx_calls} "
                        f"!= {expected_each_mode_per_mlp}"
                    )

            if cache.hits != expected_cache_hits or cache.misses != 0:
                raise RuntimeError(
                    f"cache hits/misses {cache.hits}/{cache.misses}, "
                    f"expected {expected_cache_hits}/0"
                )

            if rt["msda_state"].call_count != expected_decoder_msda:
                raise RuntimeError(
                    f"decoder MSDA calls {rt['msda_state'].call_count} "
                    f"!= {expected_decoder_msda}"
                )
        else:
            print(
                "Top-k membership FAILED; latency intentionally skipped.",
                flush=True,
            )

        result = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "experiment": "0046",
            "purpose": (
                "Paired full-model integration test replacing only the two "
                "stage-0 Swin MLPs with the exact compiled-MLX implementation "
                "validated in Experiment 0045."
            ),
            "model": args.model,
            "device": "mps",
            "dtype": "float32",
            "software": {
                "torch": torch.__version__,
                "transformers": transformers.__version__,
            },
            "configuration": {
                "baseline": (
                    "current consolidated runtime: corrected encoder wide "
                    "islands + exact fixed-prompt BERT cache + decoder Metal MSDA"
                ),
                "candidate_delta": (
                    "stage0.block0.mlp and stage0.block1.mlp replaced by "
                    "compiled MLX exact fc1->erf-GELU->fc2"
                ),
                "fresh_torch_to_mlx_bridge_each_mlp_call": True,
                "final_mlx_to_torch_bridge_each_mlp_call": True,
                "prompt_cache": "enabled identically in both modes",
                "warmup_per_mode": args.warmup_per_mode,
                "paired_iterations": args.pairs,
                "approximation": False,
                "retraining": False,
                "reduced_precision": False,
            },
            "correctness": correctness,
            "topk_membership_gate_passed": membership_pass,
            "latency": latency,
            "derived": derived,
            "call_validation": call_validation,
            "decision_rule": (
                "Proceed to fresh-process robustness replication and adopt "
                "only if top-900 membership is preserved and same-process "
                "paired median full-model saving is at least 5 ms. "
                "Otherwise keep the current runtime."
            ),
            "notes": [
                "Only the two stage-0 Swin MLP submodules are replaced; attention, shifting, masks, LayerNorm, patch merging, and all other backbone code remain unchanged.",
                "The candidate uses exact FP32 erf GELU, with no approximation, pruning, quantization, retraining, or reduced precision.",
                "The full-model timing includes both cross-runtime bridges for each replaced MLP.",
                "Experiment 0039 remains the robust latency authority until any new candidate passes fresh-process replication."
            ],
        }

        out = Path(
            "results/metalground_full_model_stage0_mlp_mlx.json"
        )
        out.write_text(
            json.dumps(result, indent=2, ensure_ascii=False) + "\n"
        )

        print("\n=== Experiment 0046 summary ===", flush=True)
        print(f"top-k membership gate: {membership_pass}", flush=True)
        if derived is not None:
            print(
                f"current median: "
                f"{latency['current_consolidated']['median_ms']:.3f} ms",
                flush=True,
            )
            print(
                f"candidate median: "
                f"{latency['stage0_two_mlp_mlx']['median_ms']:.3f} ms",
                flush=True,
            )
            print(
                f"paired saving: "
                f"{derived['paired_delta_median_ms']:+.3f} ms",
                flush=True,
            )
            print(
                f"candidate FPS: "
                f"{derived['candidate_median_fps']:.3f}",
                flush=True,
            )
        print(f"Saved: {out}", flush=True)

    finally:
        for d in dispatchers:
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
