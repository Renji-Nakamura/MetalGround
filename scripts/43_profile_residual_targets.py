#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import statistics
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import torch
import transformers


def load_module(path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load helper script: {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    try:
        spec.loader.exec_module(mod)
    except Exception:
        sys.modules.pop(module_name, None)
        raise
    return mod


def stats(xs):
    ys = sorted(xs)
    n = len(ys)
    def pct(p):
        if n == 1:
            return ys[0]
        k = (n - 1) * p
        lo = int(k)
        hi = min(lo + 1, n - 1)
        f = k - lo
        return ys[lo] * (1 - f) + ys[hi] * f
    return {
        "n": n,
        "mean_ms": statistics.fmean(xs),
        "median_ms": statistics.median(xs),
        "p90_ms": pct(0.90),
        "p95_ms": pct(0.95),
        "min_ms": min(xs),
        "max_ms": max(xs),
    }


class SyncTimer:
    def __init__(self, sync_fn):
        self.sync_fn = sync_fn
        self.samples = defaultdict(list)
        self.starts = defaultdict(list)
        self.handles = []

    def add(self, name, module):
        def pre(_m, _args):
            self.sync_fn()
            self.starts[name].append(time.perf_counter_ns())

        def post(_m, _args, _out):
            self.sync_fn()
            t0 = self.starts[name].pop()
            self.samples[name].append((time.perf_counter_ns() - t0) / 1e6)

        self.handles.append(module.register_forward_pre_hook(pre))
        self.handles.append(module.register_forward_hook(post))

    def close(self):
        for h in self.handles:
            h.remove()
        self.handles.clear()


def profile_pass(model, inputs, sync_fn, modules, warmup, iters):
    for _ in range(warmup):
        sync_fn()
        with torch.inference_mode():
            _ = model(**inputs)
        sync_fn()

    timer = SyncTimer(sync_fn)
    metadata = {}

    for name, module in modules:
        timer.add(name, module)
        metadata[name] = {
            "class": module.__class__.__name__,
            "parameters": sum(p.numel() for p in module.parameters(recurse=False)),
        }

    wall = []
    for i in range(iters):
        sync_fn()
        t0 = time.perf_counter_ns()
        with torch.inference_mode():
            _ = model(**inputs)
        sync_fn()
        wall.append((time.perf_counter_ns() - t0) / 1e6)
        print(f"  iter {i+1:02d}/{iters}: {wall[-1]:.3f} ms", flush=True)

    timer.close()

    return {
        "wall": stats(wall),
        "modules": {
            name: {
                **stats(xs),
                **metadata[name],
            }
            for name, xs in sorted(timer.samples.items())
        },
    }


def direct_children(prefix, module):
    return [
        (f"{prefix}.{name}", child)
        for name, child in module.named_children()
    ]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="IDEA-Research/grounding-dino-tiny")
    ap.add_argument("--image", type=Path, default=Path("assets/input.jpg"))
    ap.add_argument("--prompt", nargs="+", default=["a cat", "a dog"])
    ap.add_argument("--threadgroup", type=int, default=256)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--iters", type=int, default=10)

    ap.add_argument("--exp13-helper", type=Path,
                    default=Path("scripts/13_bench_fusion_algebraic_specialization.py"))
    ap.add_argument("--exp14-helper", type=Path,
                    default=Path("scripts/14_full_model_fusion_algebraic.py"))
    ap.add_argument("--exp17-helper", type=Path,
                    default=Path("scripts/17_bench_deformable_mlx_island.py"))
    ap.add_argument("--exp18-helper", type=Path,
                    default=Path("scripts/18_full_model_deformable_islands.py"))
    ap.add_argument("--exp30-helper", type=Path,
                    default=Path("scripts/30_bench_fully_folded_fusion_mlx.py"))
    ap.add_argument("--exp31-helper", type=Path,
                    default=Path("scripts/31_full_model_mlx_fusion_paired.py"))
    ap.add_argument("--exp36-helper", type=Path,
                    default=Path("scripts/36_wide_island_mask_sync_fix.py"))
    ap.add_argument("--exp37-helper", type=Path,
                    default=Path("scripts/37_corrected_wide_island_multiprocess.py"))
    ap.add_argument("--exp38-helper", type=Path,
                    default=Path("scripts/38_consolidated_runtime_ablation.py"))
    args = ap.parse_args()

    for p in (
        args.exp13_helper, args.exp14_helper, args.exp17_helper,
        args.exp18_helper, args.exp30_helper, args.exp31_helper,
        args.exp36_helper, args.exp37_helper, args.exp38_helper,
    ):
        if not p.exists():
            raise SystemExit(f"Missing helper: {p}")

    if not torch.backends.mps.is_available():
        raise SystemExit("MPS unavailable.")

    h37 = load_module(args.exp37_helper, "mg43_h37")
    h38 = load_module(args.exp38_helper, "mg43_h38")

    print("Building current consolidated runtime...", flush=True)
    rt = h37.build_runtime(args)
    model = rt["model"]
    h14 = rt["h14"]
    inputs = rt["inputs"]

    cache = h38.ExactTextBackboneCache(model.model.text_backbone)

    try:
        h38.set_runtime_mode(rt, cache, "wide_cache")
        cache.prime(model, inputs, h14.sync)
        cache.set_enabled(True)

        # Uninstrumented context only; 0039 remains authority.
        uninstrumented = []
        for _ in range(args.warmup):
            h14.sync()
            with torch.inference_mode():
                _ = model(**inputs)
            h14.sync()

        for i in range(args.iters):
            h14.sync()
            t0 = time.perf_counter_ns()
            with torch.inference_mode():
                _ = model(**inputs)
            h14.sync()
            dt = (time.perf_counter_ns() - t0) / 1e6
            uninstrumented.append(dt)
            print(f"uninstrumented {i+1:02d}/{args.iters}: {dt:.3f} ms", flush=True)

        # --------------------------------------------------------------
        # Decoder direct-child pass.
        # --------------------------------------------------------------
        decoder_modules = []
        decoder_tree = {}
        for i, layer in enumerate(model.model.decoder.layers):
            names = []
            for full_name, child in direct_children(f"decoder.layer{i}", layer):
                decoder_modules.append((full_name, child))
                names.append({
                    "name": full_name,
                    "class": child.__class__.__name__,
                })
            decoder_tree[f"layer{i}"] = names

        print("\nDecoder direct-child diagnostic pass...", flush=True)
        decoder_profile = profile_pass(
            model, inputs, h14.sync,
            decoder_modules, args.warmup, args.iters
        )

        # --------------------------------------------------------------
        # Swin block direct-child pass.
        # --------------------------------------------------------------
        swin_modules = []
        swin_tree = {}
        swin_layers = model.model.backbone.conv_encoder.model.swin.encoder.layers

        for si, stage in enumerate(swin_layers):
            for bi, block in enumerate(stage.blocks):
                key = f"stage{si}.block{bi}"
                names = []
                for full_name, child in direct_children(
                    f"backbone.{key}", block
                ):
                    swin_modules.append((full_name, child))
                    names.append({
                        "name": full_name,
                        "class": child.__class__.__name__,
                    })
                swin_tree[key] = names

        print("\nSwin block direct-child diagnostic pass...", flush=True)
        swin_profile = profile_pass(
            model, inputs, h14.sync,
            swin_modules, args.warmup, args.iters
        )

        # Rankings by median diagnostic cost.
        def ranked(profile):
            rows = [
                {
                    "name": name,
                    "median_ms": rec["median_ms"],
                    "class": rec["class"],
                    "n": rec["n"],
                }
                for name, rec in profile["modules"].items()
            ]
            rows.sort(key=lambda x: x["median_ms"], reverse=True)
            return rows

        decoder_ranked = ranked(decoder_profile)
        swin_ranked = ranked(swin_profile)

        result = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "experiment": "0043",
            "purpose": (
                "Residual-target diagnostic after closing the encoder-boundary "
                "campaign: decompose decoder layers and Swin blocks into their "
                "direct child modules under the current consolidated runtime."
            ),
            "model": args.model,
            "device": "mps",
            "dtype": "float32",
            "software": {
                "torch": torch.__version__,
                "transformers": transformers.__version__,
            },
            "configuration": {
                "runtime": (
                    "corrected per-layer wide encoder islands + exact fixed-prompt "
                    "BERT cache + decoder Metal MSDA"
                ),
                "warmup_per_pass": args.warmup,
                "iterations_per_pass": args.iters,
                "profiling": "intrusive explicit synchronization around each hooked direct child",
                "approximation": False,
                "retraining": False,
                "reduced_precision": False,
            },
            "uninstrumented_context": stats(uninstrumented),
            "decoder_module_tree": decoder_tree,
            "decoder_profile": decoder_profile,
            "decoder_ranked": decoder_ranked,
            "swin_module_tree": swin_tree,
            "swin_profile": swin_profile,
            "swin_ranked": swin_ranked,
            "interpretation_rules": [
                "Experiment 0039 remains the latency authority.",
                "These timings are intrusive diagnostic measurements only.",
                "Nested/additive attribution is not valid.",
                "Use rankings to choose the next exact microbenchmark target.",
                "Do not infer a full-model speedup by summing child medians."
            ],
        }

        out = Path("results/metalground_residual_target_profile.json")
        out.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")

        print("\n=== Experiment 0043 top decoder children ===", flush=True)
        for row in decoder_ranked[:12]:
            print(
                f"{row['median_ms']:8.3f} ms  {row['name']}  [{row['class']}]",
                flush=True,
            )

        print("\n=== Experiment 0043 top Swin children ===", flush=True)
        for row in swin_ranked[:16]:
            print(
                f"{row['median_ms']:8.3f} ms  {row['name']}  [{row['class']}]",
                flush=True,
            )

        print(f"\nSaved: {out}", flush=True)

    finally:
        cache.restore()
        for d in rt["layer_dispatchers"]:
            d.restore()
        for fd in rt["fusion_dispatchers"]:
            fd.restore()


if __name__ == "__main__":
    main()
