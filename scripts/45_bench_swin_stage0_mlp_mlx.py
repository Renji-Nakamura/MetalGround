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
        "finite": bool(torch.isfinite(a).all().item() and torch.isfinite(b).all().item()),
    }


def linear(x, w, b):
    y = x @ w.T
    return y if b is None else y + b


def gelu_exact(x):
    return 0.5 * x * (1.0 + mx.erf(x / math.sqrt(2.0)))


class MlxSwinMLP:
    def __init__(self, module):
        self.fc1_w = mx.asarray(module.fc1.weight, copy=False)
        self.fc1_b = None if module.fc1.bias is None else mx.asarray(module.fc1.bias, copy=False)
        self.fc2_w = mx.asarray(module.fc2.weight, copy=False)
        self.fc2_b = None if module.fc2.bias is None else mx.asarray(module.fc2.bias, copy=False)

        arrays = [self.fc1_w, self.fc2_w]
        if self.fc1_b is not None:
            arrays.append(self.fc1_b)
        if self.fc2_b is not None:
            arrays.append(self.fc2_b)
        mx.eval(*arrays)
        mx.synchronize()

        self.compiled = mx.compile(self._forward)

    def _forward(self, x):
        x = linear(x, self.fc1_w, self.fc1_b)
        x = gelu_exact(x)
        x = linear(x, self.fc2_w, self.fc2_b)
        return x

    def __call__(self, x):
        return self.compiled(x)


def capture_once(module, store):
    original = module.forward

    def wrapper(self, hidden_states, *args, **kwargs):
        if "input" not in store:
            store["input"] = hidden_states.detach()
        out = original(hidden_states, *args, **kwargs)
        if "output" not in store:
            store["output"] = out.detach()
        return out

    module.forward = types.MethodType(wrapper, module)
    return original


def run_mlx(candidate, x_torch, sync_fn):
    sync_fn()
    x_mx = mx.asarray(x_torch, copy=False)
    y_mx = candidate(x_mx)
    mx.eval(y_mx)
    mx.synchronize()
    y_t = torch.as_tensor(y_mx)
    sync_fn()
    return y_t


def bench_one(name, module, captured, sync_fn, warmup, pairs):
    sync_fn()
    with torch.inference_mode():
        ref = module(captured["input"])
    sync_fn()

    sync_fn()
    candidate = MlxSwinMLP(module)

    print(f"\n{name}: correctness preflight...", flush=True)
    cand = run_mlx(candidate, captured["input"], sync_fn)
    correctness = compare(cand, ref)
    print(json.dumps(correctness, indent=2), flush=True)

    if not correctness["allclose_1e-4"]:
        return {
            "name": name,
            "input_shape": list(captured["input"].shape),
            "correctness": correctness,
            "correctness_gate_passed": False,
            "latency": None,
            "derived": None,
        }

    for i in range(warmup):
        sync_fn()
        with torch.inference_mode():
            _ = module(captured["input"])
        sync_fn()
        _ = run_mlx(candidate, captured["input"], sync_fn)
        print(f"  {name} warmup {i+1}/{warmup}", flush=True)

    current_samples = []
    mlx_samples = []
    deltas = []

    print(f"{name}: paired benchmark x{pairs}...", flush=True)
    for i in range(pairs):
        order = ("current", "mlx") if i % 2 == 0 else ("mlx", "current")
        local = {}

        for mode in order:
            if mode == "current":
                sync_fn()
                t0 = time.perf_counter_ns()
                with torch.inference_mode():
                    _ = module(captured["input"])
                sync_fn()
                dt = (time.perf_counter_ns() - t0) / 1e6
                current_samples.append(dt)
            else:
                sync_fn()
                t0 = time.perf_counter_ns()
                _ = run_mlx(candidate, captured["input"], sync_fn)
                dt = (time.perf_counter_ns() - t0) / 1e6
                mlx_samples.append(dt)

            local[mode] = dt

        delta = local["current"] - local["mlx"]
        deltas.append(delta)
        print(
            f"  {i+1:02d}/{pairs}: current={local['current']:.3f} ms "
            f"mlx={local['mlx']:.3f} ms delta={delta:+.3f} ms",
            flush=True,
        )

    c = stats(current_samples)
    m = stats(mlx_samples)
    d = stats(deltas)

    return {
        "name": name,
        "input_shape": list(captured["input"].shape),
        "correctness": correctness,
        "correctness_gate_passed": True,
        "latency": {
            "current_pytorch_mps": c,
            "compiled_mlx_bridge_included": m,
            "paired_delta_current_minus_mlx": d,
        },
        "derived": {
            "speedup_from_medians": c["median_ms"] / m["median_ms"],
            "median_difference_of_marginals_ms": c["median_ms"] - m["median_ms"],
            "paired_delta_median_ms": d["median_ms"],
            "paired_delta_mean_ms": d["mean_ms"],
            "reduction_percent_from_medians":
                100.0 * (c["median_ms"] - m["median_ms"]) / c["median_ms"],
        },
    }


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

    h37 = load_module(args.exp37_helper, "mg45_h37")
    h38 = load_module(args.exp38_helper, "mg45_h38")

    print("Building current consolidated runtime...", flush=True)
    rt = h37.build_runtime(args)
    model = rt["model"]
    h14 = rt["h14"]
    inputs = rt["inputs"]

    cache = h38.ExactTextBackboneCache(model.model.text_backbone)
    originals = []

    try:
        h38.set_runtime_mode(rt, cache, "wide_cache")
        cache.prime(model, inputs, h14.sync)
        cache.set_enabled(True)

        stages = model.model.backbone.conv_encoder.model.swin.encoder.layers
        stage0 = stages[0]
        if len(stage0.blocks) < 2:
            raise RuntimeError(f"Expected at least 2 stage0 blocks, got {len(stage0.blocks)}")

        targets = [
            ("stage0.block0.mlp", stage0.blocks[0].mlp, {}),
            ("stage0.block1.mlp", stage0.blocks[1].mlp, {}),
        ]

        for name, module, store in targets:
            originals.append((module, capture_once(module, store)))

        print("Capturing real stage0 MLP inputs...", flush=True)
        with torch.inference_mode():
            _ = model(**inputs)
            h14.sync()

        for module, original in originals:
            module.forward = original
        originals.clear()

        for name, _module, store in targets:
            print(
                f"  {name}: input={tuple(store['input'].shape)} "
                f"output={tuple(store['output'].shape)}",
                flush=True,
            )

        results = [
            bench_one(name, module, store, h14.sync, args.warmup, args.pairs)
            for name, module, store in targets
        ]

        both_correct = all(x["correctness_gate_passed"] for x in results)
        both_material = all(
            x["derived"] is not None
            and (
                x["derived"]["paired_delta_median_ms"] >= 1.5
                or x["derived"]["reduction_percent_from_medians"] >= 20.0
            )
            for x in results
        )

        result = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "experiment": "0045",
            "purpose": (
                "Exact real-workload microbenchmark of the two stage-0 Swin MLPs, "
                "testing compiled MLX fc1->exact GELU->fc2 against current PyTorch/MPS "
                "with fresh bridge cost included."
            ),
            "model": args.model,
            "device": "mps",
            "dtype": "float32",
            "software": {
                "torch": torch.__version__,
                "transformers": transformers.__version__,
            },
            "configuration": {
                "targets": ["stage0.block0.mlp", "stage0.block1.mlp"],
                "candidate": "compiled MLX fc1 -> exact erf GELU -> fc2",
                "fresh_torch_to_mlx_bridge_each_call": True,
                "final_mlx_to_torch_bridge_each_call": True,
                "compile_first_call_excluded": True,
                "approximation": False,
                "retraining": False,
                "reduced_precision": False,
            },
            "results": results,
            "correctness_all_passed": both_correct,
            "performance_gate_all_passed": both_material,
            "decision_rule": (
                "Proceed to a paired full-model replacement of both stage0 Swin MLPs "
                "only if both MLPs preserve outputs within 1e-4 and each shows either "
                ">=1.5 ms paired median saving or >=20% reduction bridge-included. "
                "Otherwise do not reopen the broad Swin MLX campaign."
            ),
            "notes": [
                "This targets only SwinMLP, not attention, shifting, masks, LayerNorm, or the whole block.",
                "HF Transformers SwinMLP is exactly fc1 -> activation -> fc2 in eval/inference.",
                "GELU is implemented with the exact erf formula rather than an approximate tanh form.",
                "Experiment 0039 remains the end-to-end latency authority."
            ],
        }

        out = Path("results/metalground_swin_stage0_mlp_mlx.json")
        out.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")

        print("\n=== Experiment 0045 summary ===", flush=True)
        for r in results:
            print(f"{r['name']}: correctness={r['correctness_gate_passed']}", flush=True)
            if r["derived"]:
                print(
                    f"  current={r['latency']['current_pytorch_mps']['median_ms']:.3f} ms "
                    f"mlx={r['latency']['compiled_mlx_bridge_included']['median_ms']:.3f} ms "
                    f"paired={r['derived']['paired_delta_median_ms']:+.3f} ms "
                    f"speedup={r['derived']['speedup_from_medians']:.3f}x",
                    flush=True,
                )
        print(f"performance gate all passed: {both_material}", flush=True)
        print(f"Saved: {out}", flush=True)

    finally:
        for module, original in originals:
            try:
                module.forward = original
            except Exception:
                pass
        cache.restore()
        for d in rt["layer_dispatchers"]:
            d.restore()
        for fd in rt["fusion_dispatchers"]:
            fd.restore()


if __name__ == "__main__":
    main()
