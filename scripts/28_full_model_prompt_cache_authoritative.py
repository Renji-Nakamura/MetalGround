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
        "p99_ms": percentile(xs, 0.99),
        "min_ms": min(xs),
        "max_ms": max(xs),
    }


def mask_aware_error(a: torch.Tensor, b: torch.Tensor) -> dict[str, Any]:
    a = a.detach()
    b = b.detach()

    finite_a = torch.isfinite(a)
    finite_b = torch.isfinite(b)
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
        "finite_mask_equal": bool(torch.equal(finite_a, finite_b)),
        "nan_mask_equal": bool(torch.equal(torch.isnan(a), torch.isnan(b))),
        "posinf_mask_equal": bool(torch.equal(torch.isposinf(a), torch.isposinf(b))),
        "neginf_mask_equal": bool(torch.equal(torch.isneginf(a), torch.isneginf(b))),
        "finite_max_abs": max_abs,
        "finite_mean_abs": mean_abs,
        "finite_rmse": rmse,
        "allclose_1e-5": bool(torch.allclose(a, b, rtol=1e-5, atol=1e-5)),
        "allclose_1e-4": bool(torch.allclose(a, b, rtol=1e-4, atol=1e-4)),
    }


class FixedPromptTextCache:
    def __init__(self, module):
        self.module = module
        self.original_forward = module.forward
        self.cached_output = None
        self.enabled = False
        self.calls = 0
        self.hits = 0
        self.misses = 0

        def patched_forward(_module_self, *args, **kwargs):
            self.calls += 1
            if not self.enabled:
                return self.original_forward(*args, **kwargs)

            if self.cached_output is None:
                self.misses += 1
                out = self.original_forward(*args, **kwargs)
                self.cached_output = out
                return out

            self.hits += 1
            return self.cached_output

        module.forward = types.MethodType(patched_forward, module)

    def set_enabled(self, enabled: bool):
        self.enabled = bool(enabled)

    def reset_counts(self):
        self.calls = self.hits = self.misses = 0

    def prime(self, model, inputs, sync_fn):
        self.cached_output = None
        self.reset_counts()
        self.set_enabled(True)
        with torch.inference_mode():
            _ = model(**inputs)
            sync_fn()
        if self.misses != 1 or self.cached_output is None:
            raise RuntimeError(
                f"Expected exactly one cache miss while priming; "
                f"misses={self.misses}"
            )

    def restore(self):
        self.module.forward = self.original_forward


def benchmark(model, inputs, sync_fn, warmup: int, iters: int):
    with torch.inference_mode():
        for i in range(warmup):
            _ = model(**inputs)
            sync_fn()
            print(f"  warmup {i+1}/{warmup}", flush=True)

        samples = []
        final = None
        for i in range(iters):
            sync_fn()
            t0 = time.perf_counter_ns()
            final = model(**inputs)
            sync_fn()
            dt = (time.perf_counter_ns() - t0) / 1e6
            samples.append(dt)
            print(f"  {i+1:02d}/{iters}: {dt:.3f} ms", flush=True)

    return final, stats(samples)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="IDEA-Research/grounding-dino-tiny")
    ap.add_argument("--image", type=Path, default=Path("assets/input.jpg"))
    ap.add_argument("--prompt", nargs="+", default=["a cat", "a dog"])
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--threadgroup", type=int, default=256)
    ap.add_argument("--original-baseline-ms", type=float, default=999.116584)
    ap.add_argument("--v2-baseline-ms", type=float, default=542.305167)
    ap.add_argument("--box-threshold", type=float, default=0.3)
    ap.add_argument("--text-threshold", type=float, default=0.25)
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

    h14 = load_module(args.exp14_helper, "metalground_exp14_helpers_exp28")
    h17 = load_module(args.exp17_helper, "metalground_exp17_helpers_exp28")
    h18 = load_module(args.exp18_helper, "metalground_exp18_helpers_exp28")

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

    print("Original PyTorch/MPS correctness oracle...", flush=True)
    with torch.inference_mode():
        original_reference = model(**inputs)
        h14.sync()

    original_detections = h14.detection_summary(
        processor,
        original_reference,
        inputs["input_ids"],
        text_labels,
        image.size,
        args.box_threshold,
        args.text_threshold,
    )

    print("Building MetalGround v2...", flush=True)
    fusion_items = h14.collect_fusion_specializers(model)
    msda_state = h14.patch_msda(model, args.threadgroup)
    h14.set_fusion_mode(fusion_items, "fully_folded")

    # Capture the runtime multiscale geometry needed by the six compiled
    # encoder deformable execution islands.
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

    cache = FixedPromptTextCache(model.model.text_backbone)

    try:
        # First fully-built v2 forward (untimed) for v2-vs-cache exactness.
        cache.set_enabled(False)
        with torch.inference_mode():
            v2_reference = model(**inputs)
            h14.sync()

        print("Priming fixed-prompt BERT cache (excluded from timing)...", flush=True)
        cache.prime(model, inputs, h14.sync)
        prime_counts = {
            "calls": cache.calls,
            "hits": cache.hits,
            "misses": cache.misses,
        }

        # Exact cache-specific correctness: same v2 model with/without BERT cache.
        cache.set_enabled(True)
        with torch.inference_mode():
            cached_correct = model(**inputs)
            h14.sync()

        cache_specific_correctness = {
            "logits": mask_aware_error(
                cached_correct.logits, v2_reference.logits
            ),
            "pred_boxes": mask_aware_error(
                cached_correct.pred_boxes, v2_reference.pred_boxes
            ),
            "topk": h14.topk_audit(
                cached_correct, v2_reference, model.config.num_queries
            ),
        }

        # End-to-end correctness against original PyTorch/MPS reference.
        end_to_end_correctness = {
            "logits": mask_aware_error(
                cached_correct.logits, original_reference.logits
            ),
            "pred_boxes": mask_aware_error(
                cached_correct.pred_boxes, original_reference.pred_boxes
            ),
            "topk": h14.topk_audit(
                cached_correct, original_reference, model.config.num_queries
            ),
            "detections": h14.detection_summary(
                processor,
                cached_correct,
                inputs["input_ids"],
                text_labels,
                image.size,
                args.box_threshold,
                args.text_threshold,
            ),
        }

        print(
            "Cache-specific correctness:",
            json.dumps(cache_specific_correctness, indent=2),
            flush=True,
        )

        # Reset counters after all untimed correctness work. The timed region
        # must be cache-hit-only.
        cache.reset_counts()
        cache.set_enabled(True)

        print(
            f"Authoritative single-variant benchmark: "
            f"warmup={args.warmup}, iters={args.iters}",
            flush=True,
        )
        final, latency = benchmark(
            model, inputs, h14.sync, args.warmup, args.iters
        )

        total_forwards = args.warmup + args.iters
        if cache.calls != total_forwards:
            raise RuntimeError(
                f"Text-backbone call mismatch: got {cache.calls}, "
                f"expected {total_forwards}"
            )
        if cache.hits != total_forwards or cache.misses != 0:
            raise RuntimeError(
                f"Expected hit-only timed region: hits={cache.hits}, "
                f"misses={cache.misses}, expected hits={total_forwards}"
            )

        expected_encoder_island_calls = total_forwards * 6
        expected_decoder_msda_calls = total_forwards * 6

        # island/msda state may include prior untimed forwards, so only report
        # these counters rather than asserting an exact benchmark-only total
        # unless helper state was explicitly resettable.
        final_cache_specific = {
            "logits": mask_aware_error(final.logits, v2_reference.logits),
            "pred_boxes": mask_aware_error(
                final.pred_boxes, v2_reference.pred_boxes
            ),
            "topk": h14.topk_audit(
                final, v2_reference, model.config.num_queries
            ),
        }

        final_end_to_end = {
            "logits": mask_aware_error(
                final.logits, original_reference.logits
            ),
            "pred_boxes": mask_aware_error(
                final.pred_boxes, original_reference.pred_boxes
            ),
            "topk": h14.topk_audit(
                final, original_reference, model.config.num_queries
            ),
            "detections": h14.detection_summary(
                processor,
                final,
                inputs["input_ids"],
                text_labels,
                image.size,
                args.box_threshold,
                args.text_threshold,
            ),
        }

        median = latency["median_ms"]

        result = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "experiment": "0028",
            "purpose": (
                "Fresh-process authoritative benchmark of MetalGround v3: "
                "MetalGround v2 plus exact fixed-prompt BERT text-backbone caching."
            ),
            "model": args.model,
            "device": "mps",
            "dtype": "float32",
            "software": {
                "torch": torch.__version__,
                "transformers": transformers.__version__,
            },
            "configuration": {
                "runtime_name": "MetalGround v3",
                "base_runtime": "MetalGround v2",
                "prompt": args.prompt,
                "prompt_cache": (
                    "exact cached output of model.text_backbone for unchanged "
                    "tokenized prompt"
                ),
                "image_conditioned_text_layers_cached": False,
                "warmup": args.warmup,
                "timed_iterations": args.iters,
                "threadgroup": args.threadgroup,
                "approximation": False,
                "retraining": False,
                "reduced_precision": False,
            },
            "baselines": {
                "original_pytorch_mps_median_ms": args.original_baseline_ms,
                "metalground_v2_median_ms": args.v2_baseline_ms,
            },
            "latency": latency,
            "derived": {
                "speedup_vs_original": args.original_baseline_ms / median,
                "latency_reduction_vs_original_ms":
                    args.original_baseline_ms - median,
                "latency_reduction_vs_original_percent":
                    100.0
                    * (args.original_baseline_ms - median)
                    / args.original_baseline_ms,
                "speedup_vs_v2": args.v2_baseline_ms / median,
                "latency_reduction_vs_v2_ms":
                    args.v2_baseline_ms - median,
                "latency_reduction_vs_v2_percent":
                    100.0
                    * (args.v2_baseline_ms - median)
                    / args.v2_baseline_ms,
                "fps_from_median": 1000.0 / median,
            },
            "correctness": {
                "original_reference_detections": original_detections,
                "cache_specific_first": cache_specific_correctness,
                "end_to_end_first": end_to_end_correctness,
                "cache_specific_final": final_cache_specific,
                "end_to_end_final": final_end_to_end,
            },
            "cache_validation": {
                "prime": prime_counts,
                "benchmark_region": {
                    "calls": cache.calls,
                    "hits": cache.hits,
                    "misses": cache.misses,
                    "expected_hits": total_forwards,
                },
            },
            "notes": [
                "The prompt cache is exact only while the tokenized prompt and model state are unchanged.",
                "Only the initial BERT text-backbone output is cached; image-conditioned text enhancement and cross-modal fusion still execute each frame.",
                "The cache prime and all correctness-only forwards are excluded from the timed benchmark.",
                "The benchmark region is validated to contain text-cache hits only.",
                "No approximation, quantization, pruning, retraining, or reduced precision is used.",
                "Dataset-level accuracy equivalence remains unmeasured."
            ],
        }

        out = Path("results/metalground_v3_prompt_cache_authoritative.json")
        out.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")

        print("\n=== Experiment 0028 / MetalGround v3 ===", flush=True)
        print(f"median: {median:.3f} ms", flush=True)
        print(f"p95:   {latency['p95_ms']:.3f} ms", flush=True)
        print(f"p99:   {latency['p99_ms']:.3f} ms", flush=True)
        print(
            f"vs original: {result['derived']['speedup_vs_original']:.3f}x, "
            f"{result['derived']['latency_reduction_vs_original_percent']:.2f}% "
            "latency reduction",
            flush=True,
        )
        print(
            f"vs v2:       {result['derived']['speedup_vs_v2']:.3f}x, "
            f"{result['derived']['latency_reduction_vs_v2_ms']:+.3f} ms",
            flush=True,
        )
        print(
            f"FPS: {result['derived']['fps_from_median']:.3f}",
            flush=True,
        )
        print(
            f"text cache benchmark hits={cache.hits}, misses={cache.misses}",
            flush=True,
        )
        print(f"Saved: {out}", flush=True)

    finally:
        cache.restore()


if __name__ == "__main__":
    main()
