#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import math
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
    if not xs:
        return None
    return {
        "n": len(xs),
        "mean_ms": statistics.fmean(xs),
        "median_ms": statistics.median(xs),
        "p90_ms": percentile(xs, 0.90),
        "p95_ms": percentile(xs, 0.95),
        "min_ms": min(xs),
        "max_ms": max(xs),
    }


class SyncModuleTimer:
    """
    Intrusive diagnostic timer.

    Each registered module synchronizes before and after its forward.
    Nested module timings are NOT additive and parent-minus-children must not
    be interpreted as exact exclusive GPU attribution.
    """

    def __init__(self, sync_fn):
        self.sync_fn = sync_fn
        self.samples = defaultdict(list)
        self._starts = defaultdict(list)
        self.handles = []

    def register(self, name, module):
        if module is None:
            return

        def pre(_module, _args):
            self.sync_fn()
            self._starts[name].append(time.perf_counter_ns())

        def post(_module, _args, _out):
            self.sync_fn()
            t0 = self._starts[name].pop()
            self.samples[name].append((time.perf_counter_ns() - t0) / 1e6)

        self.handles.append(module.register_forward_pre_hook(pre))
        self.handles.append(module.register_forward_hook(post))

    def close(self):
        for h in self.handles:
            h.remove()
        self.handles.clear()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="IDEA-Research/grounding-dino-tiny")
    ap.add_argument("--image", type=Path, default=Path("assets/input.jpg"))
    ap.add_argument("--prompt", nargs="+", default=["a cat", "a dog"])
    ap.add_argument("--threadgroup", type=int, default=256)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--iters", type=int, default=10)

    ap.add_argument(
        "--exp13-helper",
        type=Path,
        default=Path("scripts/13_bench_fusion_algebraic_specialization.py"),
    )
    ap.add_argument(
        "--exp14-helper",
        type=Path,
        default=Path("scripts/14_full_model_fusion_algebraic.py"),
    )
    ap.add_argument(
        "--exp17-helper",
        type=Path,
        default=Path("scripts/17_bench_deformable_mlx_island.py"),
    )
    ap.add_argument(
        "--exp18-helper",
        type=Path,
        default=Path("scripts/18_full_model_deformable_islands.py"),
    )
    ap.add_argument(
        "--exp30-helper",
        type=Path,
        default=Path("scripts/30_bench_fully_folded_fusion_mlx.py"),
    )
    ap.add_argument(
        "--exp31-helper",
        type=Path,
        default=Path("scripts/31_full_model_mlx_fusion_paired.py"),
    )
    ap.add_argument(
        "--exp36-helper",
        type=Path,
        default=Path("scripts/36_wide_island_mask_sync_fix.py"),
    )
    ap.add_argument(
        "--exp37-helper",
        type=Path,
        default=Path("scripts/37_corrected_wide_island_multiprocess.py"),
    )
    ap.add_argument(
        "--exp38-helper",
        type=Path,
        default=Path("scripts/38_consolidated_runtime_ablation.py"),
    )
    args = ap.parse_args()

    for p in (
        args.exp13_helper,
        args.exp14_helper,
        args.exp17_helper,
        args.exp18_helper,
        args.exp30_helper,
        args.exp31_helper,
        args.exp36_helper,
        args.exp37_helper,
        args.exp38_helper,
    ):
        if not p.exists():
            raise SystemExit(f"Missing helper: {p}")

    if not torch.backends.mps.is_available():
        raise SystemExit("MPS unavailable.")

    h37 = load_module(args.exp37_helper, "mg40_h37")
    h38 = load_module(args.exp38_helper, "mg40_h38")

    print("Building consolidated runtime...", flush=True)
    rt = h37.build_runtime(args)
    model = rt["model"]
    h14 = rt["h14"]
    inputs = rt["inputs"]

    cache = h38.ExactTextBackboneCache(model.model.text_backbone)

    try:
        # Final runtime only.
        h38.set_runtime_mode(rt, cache, "wide_cache")
        cache.prime(model, inputs, h14.sync)

        # Warm without instrumentation first so compile/JIT work is excluded.
        print("Uninstrumented warmup...", flush=True)
        for i in range(args.warmup):
            h38.set_runtime_mode(rt, cache, "wide_cache")
            h14.sync()
            with torch.inference_mode():
                _ = model(**inputs)
            h14.sync()
            print(f"  warmup {i+1}/{args.warmup}", flush=True)

        # One uninstrumented contextual latency sample set.
        uninstrumented = []
        for i in range(args.iters):
            h38.set_runtime_mode(rt, cache, "wide_cache")
            h14.sync()
            t0 = time.perf_counter_ns()
            with torch.inference_mode():
                _ = model(**inputs)
            h14.sync()
            dt = (time.perf_counter_ns() - t0) / 1e6
            uninstrumented.append(dt)
            print(
                f"  uninstrumented {i+1:02d}/{args.iters}: {dt:.3f} ms",
                flush=True,
            )

        # Diagnostic synchronized module profiling.
        timer = SyncModuleTimer(h14.sync)

        # Coarse model partitions.
        timer.register("backbone", getattr(model.model, "backbone", None))
        timer.register("text_backbone_cached", getattr(model.model, "text_backbone", None))
        timer.register("encoder", getattr(model.model, "encoder", None))
        timer.register("decoder", getattr(model.model, "decoder", None))

        # Encoder layers and the remaining PyTorch text-enhancer branch.
        for i, layer in enumerate(model.model.encoder.layers):
            timer.register(f"encoder.layer{i}", layer)
            timer.register(
                f"encoder.layer{i}.text_enhancer",
                getattr(layer, "text_enhancer_layer", None),
            )

        # Decoder layers.
        for i, layer in enumerate(model.model.decoder.layers):
            timer.register(f"decoder.layer{i}", layer)

        # Swin stages, if the expected HF hierarchy exists.
        try:
            swin_layers = (
                model.model.backbone.conv_encoder.model.swin.encoder.layers
            )
            for i, stage in enumerate(swin_layers):
                timer.register(f"backbone.swin_stage{i}", stage)
                blocks = getattr(stage, "blocks", None)
                if blocks is not None:
                    for j, block in enumerate(blocks):
                        timer.register(
                            f"backbone.swin_stage{i}.block{j}", block
                        )
        except Exception as exc:
            print(
                f"Warning: Swin stage hooks unavailable: {exc}",
                flush=True,
            )

        # Reset cache counters so profiling hit validation is explicit.
        cache.reset_counts()

        print("\nIntrusive synchronized profiling...", flush=True)
        profiled_wall = []
        for i in range(args.iters):
            h38.set_runtime_mode(rt, cache, "wide_cache")
            h14.sync()
            t0 = time.perf_counter_ns()
            with torch.inference_mode():
                _ = model(**inputs)
            h14.sync()
            dt = (time.perf_counter_ns() - t0) / 1e6
            profiled_wall.append(dt)
            print(
                f"  profiled {i+1:02d}/{args.iters}: {dt:.3f} ms",
                flush=True,
            )

        timer.close()

        module_stats = {
            name: stats(xs)
            for name, xs in sorted(timer.samples.items())
        }

        # Convenience summaries.
        encoder_layer_medians = {
            f"layer{i}": module_stats[f"encoder.layer{i}"]["median_ms"]
            for i in range(6)
            if f"encoder.layer{i}" in module_stats
        }
        text_enhancer_medians = {
            f"layer{i}": module_stats[
                f"encoder.layer{i}.text_enhancer"
            ]["median_ms"]
            for i in range(6)
            if f"encoder.layer{i}.text_enhancer" in module_stats
        }
        decoder_layer_medians = {
            f"layer{i}": module_stats[f"decoder.layer{i}"]["median_ms"]
            for i in range(6)
            if f"decoder.layer{i}" in module_stats
        }

        swin_stage_medians = {
            f"stage{i}": module_stats[
                f"backbone.swin_stage{i}"
            ]["median_ms"]
            for i in range(4)
            if f"backbone.swin_stage{i}" in module_stats
        }

        result = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "experiment": "0040",
            "purpose": (
                "Re-profile the consolidated MetalGround runtime after robust "
                "adoption of compiled MLX fusion, corrected wide encoder "
                "islands, and exact fixed-prompt BERT caching."
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
                    "corrected wide fusion+deformable MLX islands + "
                    "exact fixed-prompt BERT cache + decoder Metal MSDA"
                ),
                "prompt_cache": True,
                "warmup": args.warmup,
                "iterations": args.iters,
                "profiling_sync": (
                    "explicit MPS synchronization before/after every "
                    "registered module forward"
                ),
                "approximation": False,
                "retraining": False,
                "reduced_precision": False,
            },
            "uninstrumented_context": stats(uninstrumented),
            "profiled_wall": stats(profiled_wall),
            "module_stats": module_stats,
            "summaries": {
                "encoder_layer_medians_ms": encoder_layer_medians,
                "text_enhancer_medians_ms": text_enhancer_medians,
                "decoder_layer_medians_ms": decoder_layer_medians,
                "swin_stage_medians_ms": swin_stage_medians,
            },
            "cache_validation": {
                "calls": cache.calls,
                "hits": cache.hits,
                "misses": cache.misses,
                "expected_hits": args.iters,
            },
            "interpretation_rules": [
                "Use Experiment 0039, not this profiler, for headline latency.",
                "Profiler synchronization is intrusive and can perturb scheduling.",
                "Nested module medians are not additive.",
                "Parent-minus-children is not exact exclusive GPU attribution.",
                "Use this experiment only to rank the next optimization targets."
            ],
        }

        if cache.hits != args.iters or cache.misses != 0:
            raise RuntimeError(
                f"Cache profiling validation failed: "
                f"hits={cache.hits}, misses={cache.misses}, "
                f"expected hits={args.iters}, misses=0"
            )

        out = Path("results/metalground_consolidated_profile.json")
        out.write_text(
            json.dumps(result, indent=2, ensure_ascii=False) + "\n"
        )

        print("\n=== Experiment 0040 summary ===", flush=True)
        print(
            f"uninstrumented median: "
            f"{result['uninstrumented_context']['median_ms']:.3f} ms",
            flush=True,
        )
        print(
            f"profiled wall median:  "
            f"{result['profiled_wall']['median_ms']:.3f} ms",
            flush=True,
        )
        for name in ("backbone", "encoder", "decoder", "text_backbone_cached"):
            if name in module_stats:
                print(
                    f"{name:22s}: "
                    f"{module_stats[name]['median_ms']:.3f} ms",
                    flush=True,
                )
        print(f"Saved: {out}", flush=True)

    finally:
        cache.restore()
        for d in rt["layer_dispatchers"]:
            d.restore()
        for fd in rt["fusion_dispatchers"]:
            fd.restore()


if __name__ == "__main__":
    main()
