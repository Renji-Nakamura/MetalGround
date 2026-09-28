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
from typing import Any

import mlx.core as mx
import torch
import transformers
from PIL import Image
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor


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


def percentile(xs: list[float], p: float) -> float:
    ys = sorted(xs)
    if len(ys) == 1:
        return ys[0]
    k = (len(ys) - 1) * p
    f, c = math.floor(k), math.ceil(k)
    if f == c:
        return ys[f]
    return ys[f] * (c - k) + ys[c] * (k - f)


def stats(xs: list[float]) -> dict[str, float]:
    return {
        "n": len(xs),
        "mean_ms": statistics.fmean(xs),
        "median_ms": statistics.median(xs),
        "p90_ms": percentile(xs, 0.90),
        "p95_ms": percentile(xs, 0.95),
        "min_ms": min(xs),
        "max_ms": max(xs),
    }


def mask_aware_compare(a: torch.Tensor, b: torch.Tensor) -> dict[str, Any]:
    a = a.detach()
    b = b.detach()

    finite_a = torch.isfinite(a)
    finite_b = torch.isfinite(b)
    finite_mask_equal = bool(torch.equal(finite_a, finite_b))
    nan_mask_equal = bool(torch.equal(torch.isnan(a), torch.isnan(b)))
    posinf_mask_equal = bool(torch.equal(torch.isposinf(a), torch.isposinf(b)))
    neginf_mask_equal = bool(torch.equal(torch.isneginf(a), torch.isneginf(b)))

    common = finite_a & finite_b
    if bool(common.any().item()):
        af = a[common].float()
        bf = b[common].float()
        d = (af - bf).abs()
        max_abs = float(d.max().item())
        mean_abs = float(d.mean().item())
        rmse = float(torch.sqrt(torch.mean((af - bf) ** 2)).item())
    else:
        max_abs = mean_abs = rmse = 0.0

    return {
        "shape": list(a.shape),
        "finite_mask_equal": finite_mask_equal,
        "nan_mask_equal": nan_mask_equal,
        "posinf_mask_equal": posinf_mask_equal,
        "neginf_mask_equal": neginf_mask_equal,
        "finite_max_abs": max_abs,
        "finite_mean_abs": mean_abs,
        "finite_rmse": rmse,
        "allclose_1e-5": bool(torch.allclose(a, b, rtol=1e-5, atol=1e-5)),
        "allclose_1e-4": bool(torch.allclose(a, b, rtol=1e-4, atol=1e-4)),
    }


def run_timed(model, inputs, sync_fn):
    sync_fn()
    t0 = time.perf_counter_ns()
    with torch.inference_mode():
        out = model(**inputs)
    sync_fn()
    return out, (time.perf_counter_ns() - t0) / 1e6


class TextBackboneCache:
    def __init__(self, module):
        self.module = module
        self.original_forward = module.forward
        self.cached_output = None
        self.mode = "baseline"
        self.calls = 0
        self.hits = 0
        self.misses = 0

        def patched_forward(_module_self, *args, **kwargs):
            self.calls += 1

            if self.mode == "baseline":
                return self.original_forward(*args, **kwargs)

            if self.cached_output is None:
                self.misses += 1
                out = self.original_forward(*args, **kwargs)
                self.cached_output = out
                return out

            self.hits += 1
            return self.cached_output

        module.forward = types.MethodType(patched_forward, module)

    def set_baseline(self):
        self.mode = "baseline"

    def set_cached(self):
        self.mode = "cached"

    def reset_counts(self):
        self.calls = self.hits = self.misses = 0

    def prime(self, model, inputs, sync_fn):
        self.cached_output = None
        self.set_cached()
        with torch.inference_mode():
            _ = model(**inputs)
            sync_fn()
        if self.cached_output is None or self.misses != 1:
            raise RuntimeError(
                f"Expected exactly one text-cache miss while priming; "
                f"misses={self.misses}"
            )

    def restore(self):
        self.module.forward = self.original_forward


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="IDEA-Research/grounding-dino-tiny")
    ap.add_argument("--image", type=Path, default=Path("assets/input.jpg"))
    ap.add_argument("--prompt", nargs="+", default=["a cat", "a dog"])
    ap.add_argument("--warmup-per-mode", type=int, default=3)
    ap.add_argument("--pairs", type=int, default=30)
    ap.add_argument("--threadgroup", type=int, default=256)
    ap.add_argument("--v2-authoritative-ms", type=float, default=542.305167)
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
    args = ap.parse_args()

    for p in (args.exp14_helper, args.exp17_helper, args.exp18_helper):
        if not p.exists():
            raise SystemExit(f"Missing helper script: {p}")
    if not torch.backends.mps.is_available():
        raise SystemExit("MPS unavailable.")

    h14 = load_module(args.exp14_helper, "metalground_exp14_helpers_exp27")
    h17 = load_module(args.exp17_helper, "metalground_exp17_helpers_exp27")
    h18 = load_module(args.exp18_helper, "metalground_exp18_helpers_exp27")

    mx.set_default_device(mx.gpu)

    print("Loading model...", flush=True)
    processor = AutoProcessor.from_pretrained(args.model)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(args.model)
    model.eval().to("mps")
    h14.sync()

    image = Image.open(args.image).convert("RGB")
    text_labels = [args.prompt]
    cpu_inputs = processor(images=image, text=text_labels, return_tensors="pt")
    inputs = {k: v.to("mps") for k, v in cpu_inputs.items()}
    h14.sync()

    print("Building MetalGround v2...", flush=True)
    fusion_items = h14.collect_fusion_specializers(model)
    msda_state = h14.patch_msda(model, args.threadgroup)
    h14.set_fusion_mode(fusion_items, "fully_folded")

    # Capture encoder spatial shapes for the six compiled deformable islands.
    layer0 = model.model.encoder.layers[0].deformable_layer
    original_layer0_forward = layer0.forward
    enc_capture = {}

    def capture_encoder(
        self,
        hidden_states,
        attention_mask,
        position_embeddings=None,
        reference_points=None,
        spatial_shapes=None,
        spatial_shapes_list=None,
        level_start_index=None,
        output_attentions=False,
    ):
        if not enc_capture:
            enc_capture["spatial_shapes_list"] = [
                (int(h), int(w)) for h, w in spatial_shapes_list
            ]
        return original_layer0_forward(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_embeddings=position_embeddings,
            reference_points=reference_points,
            spatial_shapes=spatial_shapes,
            spatial_shapes_list=spatial_shapes_list,
            level_start_index=level_start_index,
            output_attentions=output_attentions,
        )

    layer0.forward = types.MethodType(capture_encoder, layer0)
    with torch.inference_mode():
        _ = model(**inputs)
        h14.sync()
    layer0.forward = original_layer0_forward

    if not enc_capture:
        raise RuntimeError("Failed to capture encoder spatial shapes.")

    h14.sync()
    island_state, _islands = h18.patch_encoder_deformable_islands(
        model,
        h17.MlxDeformableIsland,
        activation_name=model.config.activation_function,
        spatial_shapes_list=enc_capture["spatial_shapes_list"],
        threadgroup=args.threadgroup,
    )

    text_module = model.model.text_backbone
    cache = TextBackboneCache(text_module)

    try:
        print("Priming exact fixed-prompt BERT output cache...", flush=True)
        cache.reset_counts()
        cache.prime(model, inputs, h14.sync)
        prime_counts = {
            "calls": cache.calls,
            "hits": cache.hits,
            "misses": cache.misses,
        }

        # Same-process correctness: normal BERT vs cached BERT output.
        cache.set_baseline()
        with torch.inference_mode():
            baseline_correct = model(**inputs)
            h14.sync()

        cache.set_cached()
        with torch.inference_mode():
            cached_correct = model(**inputs)
            h14.sync()

        correctness = {
            "logits": mask_aware_compare(
                cached_correct.logits, baseline_correct.logits
            ),
            "pred_boxes": mask_aware_compare(
                cached_correct.pred_boxes, baseline_correct.pred_boxes
            ),
            "topk_cached_vs_baseline": h14.topk_audit(
                cached_correct, baseline_correct, model.config.num_queries
            ),
        }

        print("Correctness:", json.dumps(correctness, indent=2), flush=True)

        # Warm both modes.
        print("Warming both full-model modes...", flush=True)
        for i in range(args.warmup_per_mode):
            cache.set_baseline()
            _ = run_timed(model, inputs, h14.sync)
            cache.set_cached()
            _ = run_timed(model, inputs, h14.sync)
            print(
                f"  warmup pair {i+1}/{args.warmup_per_mode}",
                flush=True,
            )

        cache.reset_counts()
        baseline_samples = []
        cached_samples = []
        deltas = []

        print(f"Paired full-model benchmark x{args.pairs}...", flush=True)
        for i in range(args.pairs):
            order = (
                ("baseline", "cached")
                if i % 2 == 0
                else ("cached", "baseline")
            )
            local = {}

            for mode in order:
                if mode == "baseline":
                    cache.set_baseline()
                else:
                    cache.set_cached()

                _out, dt = run_timed(model, inputs, h14.sync)
                local[mode] = dt
                if mode == "baseline":
                    baseline_samples.append(dt)
                else:
                    cached_samples.append(dt)

            delta = local["baseline"] - local["cached"]
            deltas.append(delta)

            print(
                f"  pair {i+1:02d}/{args.pairs}: "
                f"baseline={local['baseline']:.3f} ms, "
                f"cached={local['cached']:.3f} ms, "
                f"delta={delta:+.3f} ms",
                flush=True,
            )

        baseline_stats = stats(baseline_samples)
        cached_stats = stats(cached_samples)
        delta_stats = stats(deltas)

        expected_calls = args.pairs * 2
        expected_hits = args.pairs
        if cache.calls != expected_calls:
            raise RuntimeError(
                f"Text-backbone call mismatch: got {cache.calls}, "
                f"expected {expected_calls}"
            )
        if cache.hits != expected_hits:
            raise RuntimeError(
                f"Text-cache hit mismatch: got {cache.hits}, "
                f"expected {expected_hits}"
            )
        if cache.misses != 0:
            raise RuntimeError(
                f"Timed region unexpectedly had {cache.misses} cache misses."
            )

        result = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "experiment": "0027",
            "purpose": (
                "Same-process paired full-model A/B benchmark of exact fixed-prompt "
                "BERT text-backbone caching on MetalGround v2."
            ),
            "model": args.model,
            "device": "mps",
            "dtype": "float32",
            "software": {
                "torch": torch.__version__,
                "transformers": transformers.__version__,
            },
            "configuration": {
                "base_runtime": "MetalGround v2",
                "cached_module": "model.text_backbone",
                "prompt": args.prompt,
                "warmup_per_mode": args.warmup_per_mode,
                "paired_iterations": args.pairs,
                "order": "alternating baseline->cached / cached->baseline",
                "cache_scope": "exact output for unchanged tokenized prompt",
                "approximation": False,
                "retraining": False,
                "reduced_precision": False,
            },
            "latency": {
                "baseline_v2_same_process": baseline_stats,
                "cached_text_backbone_same_process": cached_stats,
                "paired_delta_ms_baseline_minus_cached": delta_stats,
            },
            "derived": {
                "speedup_from_medians": (
                    baseline_stats["median_ms"]
                    / cached_stats["median_ms"]
                ),
                "median_reduction_ms": (
                    baseline_stats["median_ms"]
                    - cached_stats["median_ms"]
                ),
                "median_reduction_percent": (
                    100.0
                    * (
                        baseline_stats["median_ms"]
                        - cached_stats["median_ms"]
                    )
                    / baseline_stats["median_ms"]
                ),
                "authoritative_v2_median_ms": args.v2_authoritative_ms,
                "authoritative_projection_if_paired_delta_transfers_ms": (
                    args.v2_authoritative_ms - delta_stats["median_ms"]
                ),
            },
            "correctness": correctness,
            "cache_validation": {
                "prime": prime_counts,
                "timed_region": {
                    "calls": cache.calls,
                    "hits": cache.hits,
                    "misses": cache.misses,
                    "expected_hits": expected_hits,
                },
            },
            "notes": [
                "Only the initial BERT text-backbone output is cached; image-conditioned text enhancement and fusion still execute every frame.",
                "Baseline and cached full-model executions share one process and alternate order each pair.",
                "The experiment assumes the tokenized prompt is unchanged; a production cache must be keyed by prompt/tokenization/model state.",
                "The paired delta is the causal performance estimate; Experiment 0018's 542.305 ms remains the authoritative v2 headline until a fresh adopted variant is benchmarked.",
                "No approximation, quantization, pruning, retraining, or reduced precision is used."
            ],
        }

        out = Path("results/metalground_v2_prompt_cache_paired.json")
        out.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")

        print("\n=== Experiment 0027 summary ===", flush=True)
        print(
            f"baseline median: {baseline_stats['median_ms']:.3f} ms",
            flush=True,
        )
        print(
            f"cached median:   {cached_stats['median_ms']:.3f} ms",
            flush=True,
        )
        print(
            f"paired delta median: {delta_stats['median_ms']:+.3f} ms",
            flush=True,
        )
        print(
            f"speedup from medians: "
            f"{result['derived']['speedup_from_medians']:.4f}x",
            flush=True,
        )
        print(
            f"timed cache hits={cache.hits}, misses={cache.misses}",
            flush=True,
        )
        print(f"Saved: {out}", flush=True)

    finally:
        cache.restore()


if __name__ == "__main__":
    main()
