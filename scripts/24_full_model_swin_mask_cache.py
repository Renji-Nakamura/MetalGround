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
from dataclasses import dataclass
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


@dataclass
class MaskCacheState:
    calls: int = 0
    hits: int = 0
    misses: int = 0


def patch_shifted_swin_mask_cache(model):
    """
    Cache get_attn_mask() on each shifted Swin block, keyed by the exact
    padded geometry + dtype + device. This preserves behavior for shape changes:
    unseen geometries simply populate a new cache entry.
    """
    state = MaskCacheState()
    patched = []

    swin = model.model.backbone.conv_encoder.model.swin
    for stage_i, stage in enumerate(swin.encoder.layers):
        for block_i, block in enumerate(stage.blocks):
            # Swin alternates non-shifted / shifted blocks. Use the configured
            # block value as the selection criterion. Runtime may still clamp
            # shift to 0 for tiny feature maps; in that case original behavior
            # remains correct.
            configured_shift = int(block.shift_size)
            if configured_shift <= 0:
                continue

            original = block.get_attn_mask
            cache: dict[tuple[Any, ...], torch.Tensor | None] = {}

            def make_cached(original_fn, cache_dict):
                def cached_get_attn_mask(self, height, width, dtype, device):
                    state.calls += 1
                    key = (
                        int(height),
                        int(width),
                        str(dtype),
                        str(device),
                        int(self.window_size),
                        int(self.shift_size),
                    )
                    if key in cache_dict:
                        state.hits += 1
                        return cache_dict[key]

                    state.misses += 1
                    out = original_fn(height, width, dtype, device)
                    if out is not None:
                        out = out.detach()
                    cache_dict[key] = out
                    return out

                return cached_get_attn_mask

            block.get_attn_mask = types.MethodType(
                make_cached(original, cache), block
            )
            patched.append(
                {
                    "stage": stage_i,
                    "block": block_i,
                    "module": (
                        f"model.backbone.conv_encoder.model.swin.encoder."
                        f"layers.{stage_i}.blocks.{block_i}"
                    ),
                    "configured_shift": configured_shift,
                    "cache": cache,
                }
            )

    return state, patched


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

    h14 = load_module(args.exp14_helper, "metalground_exp14_helpers_exp24")
    h17 = load_module(args.exp17_helper, "metalground_exp17_helpers_exp24")
    h18 = load_module(args.exp18_helper, "metalground_exp18_helpers_exp24")

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

    print("Reference PyTorch/MPS forward for correctness oracle...", flush=True)
    with torch.inference_mode():
        reference = model(**inputs)
        h14.sync()

    reference_detections = h14.detection_summary(
        processor,
        reference,
        inputs["input_ids"],
        text_labels,
        image.size,
        args.box_threshold,
        args.text_threshold,
    )

    print("Building MetalGround v2 state...", flush=True)
    fusion_items = h14.collect_fusion_specializers(model)
    msda_state = h14.patch_msda(model, args.threadgroup)
    h14.set_fusion_mode(fusion_items, "fully_folded")

    # Capture runtime spatial shapes for encoder deformable islands.
    layer0 = model.model.encoder.layers[0].deformable_layer
    original_layer0_forward = layer0.forward
    captured = {}

    def capture_forward(
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
        if not captured:
            captured["spatial_shapes_list"] = [
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

    layer0.forward = types.MethodType(capture_forward, layer0)
    with torch.inference_mode():
        _ = model(**inputs)
        h14.sync()
    layer0.forward = original_layer0_forward

    if not captured:
        raise RuntimeError("Failed to capture encoder spatial shapes.")

    h14.sync()
    island_state, islands = h18.patch_encoder_deformable_islands(
        model,
        h17.MlxDeformableIsland,
        activation_name=model.config.activation_function,
        spatial_shapes_list=captured["spatial_shapes_list"],
        threadgroup=args.threadgroup,
    )

    print("Patching shifted-Swin attention-mask caches...", flush=True)
    mask_state, mask_patches = patch_shifted_swin_mask_cache(model)
    print(
        "  patched shifted blocks:",
        [x["module"] for x in mask_patches],
        flush=True,
    )

    # First fully patched forward: populate each fixed-resolution mask cache and
    # exclude all compilation/cache construction from timed measurements.
    print("First v2+Swin-cache forward (excluded from timing)...", flush=True)
    mask_state.calls = mask_state.hits = mask_state.misses = 0
    island_state.calls = 0
    msda_state.call_count = 0

    with torch.inference_mode():
        first = model(**inputs)
        h14.sync()

    first_cache_counts = {
        "calls": mask_state.calls,
        "hits": mask_state.hits,
        "misses": mask_state.misses,
    }

    first_correctness = {
        "logits": h14.mask_aware_error(first.logits, reference.logits),
        "pred_boxes": h14.mask_aware_error(
            first.pred_boxes, reference.pred_boxes
        ),
        "topk": h14.topk_audit(first, reference, model.config.num_queries),
        "detections": h14.detection_summary(
            processor,
            first,
            inputs["input_ids"],
            text_labels,
            image.size,
            args.box_threshold,
            args.text_threshold,
        ),
        "encoder_island_calls": island_state.calls,
        "decoder_patched_msda_calls": msda_state.call_count,
        "mask_cache_counts": first_cache_counts,
    }

    expected_shifted_blocks = len(mask_patches)
    if mask_state.misses != expected_shifted_blocks:
        raise RuntimeError(
            f"Expected {expected_shifted_blocks} first-use cache misses, "
            f"got {mask_state.misses}"
        )

    print("Top-k:", json.dumps(first_correctness["topk"], indent=2), flush=True)
    print(
        "Mask cache first-forward counts:",
        json.dumps(first_cache_counts),
        flush=True,
    )

    # Benchmark only cache-hit steady state.
    print(
        f"\nBenchmarking full model steady-state: "
        f"warmup={args.warmup}, iters={args.iters}",
        flush=True,
    )
    mask_state.calls = mask_state.hits = mask_state.misses = 0
    island_state.calls = 0
    msda_state.call_count = 0

    final, latency = benchmark(
        model, inputs, h14.sync, args.warmup, args.iters
    )

    total_forwards = args.warmup + args.iters
    expected_mask_calls = total_forwards * expected_shifted_blocks
    expected_encoder_islands = total_forwards * 6
    expected_decoder_msda = total_forwards * 6

    if mask_state.calls != expected_mask_calls:
        raise RuntimeError(
            f"Mask-cache call mismatch: got {mask_state.calls}, "
            f"expected {expected_mask_calls}"
        )
    if mask_state.misses != 0:
        raise RuntimeError(
            f"Steady-state benchmark unexpectedly had {mask_state.misses} "
            "mask-cache misses."
        )
    if mask_state.hits != expected_mask_calls:
        raise RuntimeError(
            f"Mask-cache hit mismatch: got {mask_state.hits}, "
            f"expected {expected_mask_calls}"
        )
    if island_state.calls != expected_encoder_islands:
        raise RuntimeError(
            f"Encoder island mismatch: got {island_state.calls}, "
            f"expected {expected_encoder_islands}"
        )
    if msda_state.call_count != expected_decoder_msda:
        raise RuntimeError(
            f"Decoder MSDA mismatch: got {msda_state.call_count}, "
            f"expected {expected_decoder_msda}"
        )

    final_correctness = {
        "logits": h14.mask_aware_error(final.logits, reference.logits),
        "pred_boxes": h14.mask_aware_error(
            final.pred_boxes, reference.pred_boxes
        ),
        "topk": h14.topk_audit(final, reference, model.config.num_queries),
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
        "experiment": "0024",
        "purpose": (
            "Full-model integration of exact shifted-Swin attention-mask caching "
            "on top of authoritative MetalGround v2."
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
            "swin_specialization": (
                "per-block get_attn_mask cache keyed by padded geometry, "
                "dtype, device, window size, and runtime shift size"
            ),
            "patched_shifted_blocks": [
                {
                    "module": x["module"],
                    "stage": x["stage"],
                    "block": x["block"],
                    "configured_shift": x["configured_shift"],
                }
                for x in mask_patches
            ],
            "warmup": args.warmup,
            "timed_iterations": args.iters,
            "approximation": False,
            "retraining": False,
            "reduced_precision": False,
        },
        "baselines": {
            "experiment_0001_original_pytorch_mps_median_ms":
                args.original_baseline_ms,
            "experiment_0018_metalground_v2_median_ms":
                args.v2_baseline_ms,
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
            "latency_reduction_vs_v2_ms": args.v2_baseline_ms - median,
            "latency_reduction_vs_v2_percent":
                100.0
                * (args.v2_baseline_ms - median)
                / args.v2_baseline_ms,
            "fps_from_median": 1000.0 / median,
        },
        "correctness": {
            "reference_detections": reference_detections,
            "first_patched_forward": first_correctness,
            "final_timed_forward": final_correctness,
        },
        "cache_validation": {
            "first_forward": first_cache_counts,
            "steady_state": {
                "calls": mask_state.calls,
                "hits": mask_state.hits,
                "misses": mask_state.misses,
                "expected_calls": expected_mask_calls,
            },
        },
        "call_counts": {
            "encoder_island_calls": island_state.calls,
            "expected_encoder_island_calls": expected_encoder_islands,
            "decoder_patched_msda_calls": msda_state.call_count,
            "expected_decoder_patched_msda_calls": expected_decoder_msda,
        },
        "notes": [
            "Only exact shifted-window attention masks are cached; relative-position bias remains on the original path because Experiment 0023 showed no robust benefit.",
            "Cache keys include geometry, dtype, device, window size, and runtime shift size, so unseen shapes fall back to exact mask construction and populate a new entry.",
            "The first patched forward populates caches and is excluded from timing.",
            "The timed region is validated to contain cache hits only.",
            "No approximation, quantization, pruning, retraining, or reduced precision is used.",
            "Dataset-level accuracy equivalence remains unmeasured."
        ],
    }

    out = Path("results/metalground_v2_swin_mask_cache_full_model.json")
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")

    print("\n=== Experiment 0024 summary ===", flush=True)
    print(f"median: {median:.3f} ms", flush=True)
    print(f"p95:   {latency['p95_ms']:.3f} ms", flush=True)
    print(f"p99:   {latency['p99_ms']:.3f} ms", flush=True)
    print(
        f"speedup vs v2:       {result['derived']['speedup_vs_v2']:.3f}x",
        flush=True,
    )
    print(
        f"speedup vs original: {result['derived']['speedup_vs_original']:.3f}x",
        flush=True,
    )
    print(
        f"steady-state mask cache: hits={mask_state.hits}, "
        f"misses={mask_state.misses}",
        flush=True,
    )
    print(f"Saved: {out}", flush=True)


if __name__ == "__main__":
    main()
