#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path

import mlx.core as mx
import torch
from PIL import Image
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor


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


def load_exp14_helpers(path: Path):
    import sys

    module_name = "metalground_exp14_helpers"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load helper script: {path}")

    mod = importlib.util.module_from_spec(spec)

    # Python 3.12 dataclasses resolves cls.__module__ through sys.modules
    # while the module body is being executed. Register it before exec_module().
    sys.modules[module_name] = mod
    try:
        spec.loader.exec_module(mod)
    except Exception:
        sys.modules.pop(module_name, None)
        raise

    return mod


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="IDEA-Research/grounding-dino-tiny")
    ap.add_argument("--image", type=Path, default=Path("assets/input.jpg"))
    ap.add_argument("--prompt", nargs="+", default=["a cat", "a dog"])
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--threadgroup", type=int, default=256)
    ap.add_argument("--original-baseline-ms", type=float, default=999.116584)
    ap.add_argument("--msda-v0-baseline-ms", type=float, default=632.6745625)
    ap.add_argument("--box-threshold", type=float, default=0.3)
    ap.add_argument("--text-threshold", type=float, default=0.25)
    ap.add_argument(
        "--helper-script",
        type=Path,
        default=Path("scripts/14_full_model_fusion_algebraic.py"),
        help="Experiment 0014 implementation used as the canonical helper source.",
    )
    args = ap.parse_args()

    if not torch.backends.mps.is_available():
        raise SystemExit("MPS unavailable.")
    if not args.helper_script.exists():
        raise SystemExit(f"Missing helper script: {args.helper_script}")

    helper_bytes = args.helper_script.read_bytes()
    helper_sha256 = hashlib.sha256(helper_bytes).hexdigest()
    h = load_exp14_helpers(args.helper_script)

    mx.set_default_device(mx.gpu)

    print("Loading model...", flush=True)
    processor = AutoProcessor.from_pretrained(args.model)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(args.model)
    model.eval().to("mps")
    h.sync()

    image = Image.open(args.image).convert("RGB")
    text_labels = [args.prompt]
    cpu_inputs = processor(images=image, text=text_labels, return_tensors="pt")
    inputs = {k: v.to("mps") for k, v in cpu_inputs.items()}
    h.sync()

    print("Reference PyTorch/MPS forward for correctness oracle...", flush=True)
    with torch.inference_mode():
        reference = model(**inputs)
        h.sync()

    reference_detections = h.detection_summary(
        processor,
        reference,
        inputs["input_ids"],
        text_labels,
        image.size,
        args.box_threshold,
        args.text_threshold,
    )

    print("Building six fully-folded fusion specializers...", flush=True)
    fusion_items = h.collect_fusion_specializers(model)

    print("Patching twelve MSDA cores with MetalGround v0...", flush=True)
    msda_state = h.patch_msda(model, args.threadgroup)

    print("Enabling fully-folded fusion in all six encoder layers...", flush=True)
    h.set_fusion_mode(fusion_items, "fully_folded")

    # First execution after all patches: JIT/cache establishment + correctness.
    print("First MetalGround v1 forward (excluded from benchmark)...", flush=True)
    msda_state.call_count = 0
    with torch.inference_mode():
        first = model(**inputs)
        h.sync()

    first_correctness = {
        "logits": h.mask_aware_error(first.logits, reference.logits),
        "pred_boxes": h.mask_aware_error(first.pred_boxes, reference.pred_boxes),
        "topk": h.topk_audit(first, reference, model.config.num_queries),
        "detections": h.detection_summary(
            processor,
            first,
            inputs["input_ids"],
            text_labels,
            image.size,
            args.box_threshold,
            args.text_threshold,
        ),
        "msda_calls": msda_state.call_count,
    }

    print("Top-k audit:", json.dumps(first_correctness["topk"], indent=2), flush=True)
    print("Detections:", json.dumps(first_correctness["detections"], indent=2), flush=True)

    print(
        f"\nAuthoritative MetalGround v1 benchmark: "
        f"warmup={args.warmup}, iters={args.iters}",
        flush=True,
    )

    msda_state.call_count = 0
    with torch.inference_mode():
        for i in range(args.warmup):
            _ = model(**inputs)
            h.sync()
            print(f"  warmup {i+1}/{args.warmup}", flush=True)

        samples: list[float] = []
        final = None
        for i in range(args.iters):
            h.sync()
            t0 = time.perf_counter_ns()
            final = model(**inputs)
            h.sync()
            dt_ms = (time.perf_counter_ns() - t0) / 1e6
            samples.append(dt_ms)
            print(f"  {i+1:02d}/{args.iters}: {dt_ms:.3f} ms", flush=True)

    latency = stats(samples)

    expected_calls = (args.warmup + args.iters) * 12
    if msda_state.call_count != expected_calls:
        raise RuntimeError(
            f"MSDA call-count mismatch: got {msda_state.call_count}, "
            f"expected {expected_calls}"
        )

    assert final is not None
    final_correctness = {
        "logits": h.mask_aware_error(final.logits, reference.logits),
        "pred_boxes": h.mask_aware_error(final.pred_boxes, reference.pred_boxes),
        "topk": h.topk_audit(final, reference, model.config.num_queries),
        "detections": h.detection_summary(
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
        "experiment": "0015",
        "purpose": (
            "Fresh-process authoritative benchmark of MetalGround v1: "
            "twelve fused Metal MSDA cores plus six fully-folded fusion attentions."
        ),
        "model": args.model,
        "device": "mps",
        "dtype": "float32",
        "image": str(args.image),
        "prompt": args.prompt,
        "helper_source": {
            "path": str(args.helper_script),
            "sha256": helper_sha256,
        },
        "configuration": {
            "msda": "metalground_msda_v0_fp32",
            "fusion": "fully_folded_algebraic_bimha",
            "fusion_layers": 6,
            "batch_specialization": 1,
            "threadgroup_size": args.threadgroup,
            "warmup": args.warmup,
            "timed_iterations": args.iters,
            "approximation": False,
            "retraining": False,
            "reduced_precision": False,
        },
        "baselines": {
            "experiment_0001_original_pytorch_mps_median_ms": args.original_baseline_ms,
            "experiment_0009_msda_v0_median_ms": args.msda_v0_baseline_ms,
        },
        "latency": latency,
        "derived": {
            "speedup_vs_original_pytorch_mps": args.original_baseline_ms / median,
            "latency_reduction_vs_original_ms": args.original_baseline_ms - median,
            "latency_reduction_vs_original_percent": (
                100.0 * (args.original_baseline_ms - median) / args.original_baseline_ms
            ),
            "speedup_vs_msda_v0": args.msda_v0_baseline_ms / median,
            "latency_reduction_vs_msda_v0_ms": args.msda_v0_baseline_ms - median,
            "latency_reduction_vs_msda_v0_percent": (
                100.0 * (args.msda_v0_baseline_ms - median) / args.msda_v0_baseline_ms
            ),
            "fps_from_median": 1000.0 / median,
        },
        "correctness": {
            "reference_detections": reference_detections,
            "first_patched_forward": first_correctness,
            "final_timed_forward": final_correctness,
        },
        "call_counts": {
            "msda_timed_region_including_warmup": msda_state.call_count,
            "expected": expected_calls,
        },
        "notes": [
            "This is a fresh-process, single-variant benchmark intended to become the authoritative MetalGround v1 latency.",
            "The first patched execution is excluded from timing.",
            "Experiment 0001 and Experiment 0009 remain the authoritative historical baselines.",
            "Raw query-wise tensors are rank-sensitive; top-k set overlap/order and postprocessed detections are recorded separately.",
            "Dataset-level accuracy equivalence remains to be established."
        ],
    }

    out = Path("results/metalground_v1_authoritative_bench.json")
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")

    print("\n=== Experiment 0015 summary ===", flush=True)
    print(f"median:  {median:.3f} ms", flush=True)
    print(f"p95:     {latency['p95_ms']:.3f} ms", flush=True)
    print(f"p99:     {latency['p99_ms']:.3f} ms", flush=True)
    print(
        f"speedup vs original: "
        f"{result['derived']['speedup_vs_original_pytorch_mps']:.3f}x",
        flush=True,
    )
    print(
        f"speedup vs MSDA-v0:  "
        f"{result['derived']['speedup_vs_msda_v0']:.3f}x",
        flush=True,
    )
    print(f"FPS: {result['derived']['fps_from_median']:.3f}", flush=True)
    print(f"Saved: {out}", flush=True)


if __name__ == "__main__":
    main()
