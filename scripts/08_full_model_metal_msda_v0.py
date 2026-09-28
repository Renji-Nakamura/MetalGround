#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import statistics
import time
import types
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import mlx.core as mx
import numpy as np
import torch
import transformers
from PIL import Image
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

from metalground.msda_metal_v0 import msda_metal_v0


def sync_mps() -> None:
    torch.mps.synchronize()


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


def tensor_error(actual: torch.Tensor, expected: torch.Tensor) -> dict[str, Any]:
    a = actual.detach().float()
    b = expected.detach().float()
    d = (a - b).abs()

    result = {
        "shape": list(a.shape),
        "max_abs": float(d.max().item()),
        "mean_abs": float(d.mean().item()),
        "rmse": float(torch.sqrt(torch.mean((a - b) ** 2)).item()),
        "allclose_1e-5": bool(torch.allclose(a, b, rtol=1e-5, atol=1e-5)),
        "allclose_1e-4": bool(torch.allclose(a, b, rtol=1e-4, atol=1e-4)),
        "finite": bool(torch.isfinite(a).all().item()),
    }
    sync_mps()
    return result


def detection_summary(
    processor,
    outputs,
    input_ids,
    text_labels,
    image_size_wh,
    box_threshold: float,
    text_threshold: float,
):
    target_sizes = [image_size_wh[::-1]]
    det = processor.post_process_grounded_object_detection(
        outputs,
        input_ids,
        text_labels=text_labels,
        target_sizes=target_sizes,
        threshold=box_threshold,
        text_threshold=text_threshold,
    )[0]

    boxes = det["boxes"].detach().float().cpu().tolist()
    scores = det["scores"].detach().float().cpu().tolist()
    labels = list(det["text_labels"])

    return [
        {
            "label": label,
            "score": score,
            "box": box,
        }
        for label, score, box in zip(labels, scores, boxes)
    ]


@dataclass
class PatchState:
    threadgroup_size: int
    module_names: list[str] = field(default_factory=list)
    call_count: int = 0
    encoder_calls: int = 0
    decoder_calls: int = 0
    metadata_cache: dict[tuple[tuple[int, int], ...], tuple[mx.array, mx.array]] = field(
        default_factory=dict
    )

    def metadata(
        self,
        spatial_shapes_list: list[tuple[int, int]],
    ) -> tuple[mx.array, mx.array]:
        key = tuple((int(h), int(w)) for h, w in spatial_shapes_list)
        if key in self.metadata_cache:
            return self.metadata_cache[key]

        shapes_np = np.asarray(key, dtype=np.int32)
        starts = []
        acc = 0
        for h, w in key:
            starts.append(acc)
            acc += h * w
        starts_np = np.asarray(starts, dtype=np.int32)

        shapes_mx = mx.array(shapes_np, dtype=mx.int32)
        starts_mx = mx.array(starts_np, dtype=mx.int32)
        mx.eval(shapes_mx, starts_mx)
        mx.synchronize()

        self.metadata_cache[key] = (shapes_mx, starts_mx)
        return shapes_mx, starts_mx


def make_metal_forward(state: PatchState, module_name: str):
    is_encoder = ".encoder.layers." in module_name

    def metal_forward(
        self,
        value: torch.Tensor,
        value_spatial_shapes: torch.Tensor,
        value_spatial_shapes_list: list[tuple],
        level_start_index: torch.Tensor,
        sampling_locations: torch.Tensor,
        attention_weights: torch.Tensor,
        im2col_step: int,
    ):
        # Inference-only experimental backend.
        if value.device.type != "mps":
            raise RuntimeError(f"MetalGround v0 expects MPS value tensor, got {value.device}")
        if value.dtype != torch.float32:
            raise RuntimeError(f"MetalGround v0 is FP32-only, got {value.dtype}")

        # The validated bridge requires ordinary contiguous MPS tensors.
        # If upstream changes layout, make that copy explicit and measurable.
        if not value.is_contiguous():
            value = value.contiguous()
        if not sampling_locations.is_contiguous():
            sampling_locations = sampling_locations.contiguous()
        if not attention_weights.is_contiguous():
            attention_weights = attention_weights.contiguous()

        # Conservative synchronization protocol validated in Experiment 0007.
        # Future versions may reduce these barriers.
        torch.mps.synchronize()

        value_mx = mx.asarray(value, copy=False)
        locations_mx = mx.asarray(sampling_locations, copy=False)
        weights_mx = mx.asarray(attention_weights, copy=False)
        spatial_mx, start_mx = state.metadata(value_spatial_shapes_list)

        out_mx = msda_metal_v0(
            value_mx,
            spatial_mx,
            start_mx,
            locations_mx,
            weights_mx,
            threadgroup_size=state.threadgroup_size,
        )
        mx.eval(out_mx)
        mx.synchronize()

        out_t = torch.as_tensor(out_mx)
        if out_t.device.type != "mps":
            raise RuntimeError(f"MetalGround output unexpectedly on {out_t.device}")

        state.call_count += 1
        if is_encoder:
            state.encoder_calls += 1
        else:
            state.decoder_calls += 1

        return out_t

    return metal_forward


def patch_msda_cores(model: torch.nn.Module, threadgroup_size: int) -> PatchState:
    state = PatchState(threadgroup_size=threadgroup_size)

    for name, module in model.named_modules():
        if module.__class__.__name__ != "MultiScaleDeformableAttention":
            continue

        module.forward = types.MethodType(make_metal_forward(state, name), module)
        state.module_names.append(name)

    if len(state.module_names) != 12:
        raise RuntimeError(
            f"Expected 12 MultiScaleDeformableAttention cores, found {len(state.module_names)}"
        )

    return state


def benchmark_model(model, inputs, warmup: int, iters: int) -> tuple[Any, dict[str, float]]:
    with torch.inference_mode():
        for i in range(warmup):
            _ = model(**inputs)
            sync_mps()
            print(f"  warmup {i+1}/{warmup}", flush=True)

        samples = []
        out = None
        for i in range(iters):
            sync_mps()
            t0 = time.perf_counter_ns()
            out = model(**inputs)
            sync_mps()
            dt = (time.perf_counter_ns() - t0) / 1e6
            samples.append(dt)
            print(f"  {i+1:02d}/{iters}: {dt:.3f} ms", flush=True)

    return out, stats(samples)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="IDEA-Research/grounding-dino-tiny")
    ap.add_argument("--image", type=Path, default=Path("assets/input.jpg"))
    ap.add_argument("--prompt", nargs="+", default=["a cat", "a dog"])
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--threadgroup", type=int, default=256)
    ap.add_argument(
        "--baseline-ms",
        type=float,
        default=999.116584,
        help="Experiment 0001 authoritative uninstrumented MPS forward median.",
    )
    ap.add_argument("--box-threshold", type=float, default=0.3)
    ap.add_argument("--text-threshold", type=float, default=0.25)
    args = ap.parse_args()

    if not torch.backends.mps.is_available():
        raise SystemExit("MPS unavailable.")
    if not args.image.exists():
        raise SystemExit(f"Missing image: {args.image}")

    mx.set_default_device(mx.gpu)
    device = torch.device("mps")

    print(f"Loading {args.model}...", flush=True)
    processor = AutoProcessor.from_pretrained(args.model)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(args.model)
    model.eval().to(device)
    sync_mps()

    image = Image.open(args.image).convert("RGB")
    text_labels = [args.prompt]  # Preserve Experiment 0001 workload semantics.
    cpu_inputs = processor(
        images=image,
        text=text_labels,
        return_tensors="pt",
    )
    inputs = {k: v.to(device) for k, v in cpu_inputs.items()}
    sync_mps()

    print("Running one unpatched reference forward for correctness oracle...", flush=True)
    with torch.inference_mode():
        reference = model(**inputs)
        sync_mps()

    # Detach only the final tensors needed for comparison; keep them on MPS.
    ref_logits = reference.logits.detach().clone()
    ref_boxes = reference.pred_boxes.detach().clone()
    sync_mps()

    ref_detections = detection_summary(
        processor,
        reference,
        inputs["input_ids"],
        text_labels,
        image.size,
        args.box_threshold,
        args.text_threshold,
    )

    print("Patching 12 MSDA cores with MetalGround v0...", flush=True)
    state = patch_msda_cores(model, args.threadgroup)
    for name in state.module_names:
        print(f"  patched: {name}", flush=True)

    print("\nFirst patched forward (JIT + correctness, excluded from benchmark)...", flush=True)
    state.call_count = state.encoder_calls = state.decoder_calls = 0
    with torch.inference_mode():
        patched_first = model(**inputs)
        sync_mps()

    first_call_counts = {
        "total": state.call_count,
        "encoder": state.encoder_calls,
        "decoder": state.decoder_calls,
    }

    print("Comparing final logits / boxes...", flush=True)
    correctness = {
        "logits": tensor_error(patched_first.logits, ref_logits),
        "pred_boxes": tensor_error(patched_first.pred_boxes, ref_boxes),
    }

    patched_detections = detection_summary(
        processor,
        patched_first,
        inputs["input_ids"],
        text_labels,
        image.size,
        args.box_threshold,
        args.text_threshold,
    )

    print(json.dumps(correctness, indent=2), flush=True)
    print("Reference detections:", json.dumps(ref_detections, indent=2), flush=True)
    print("Patched detections:", json.dumps(patched_detections, indent=2), flush=True)
    print(f"MSDA calls in first patched forward: {first_call_counts}", flush=True)

    if not correctness["logits"]["finite"] or not correctness["pred_boxes"]["finite"]:
        raise RuntimeError("Non-finite final model outputs after Metal patch.")

    print(
        f"\nBenchmarking patched full model: warmup={args.warmup}, iters={args.iters}",
        flush=True,
    )
    state.call_count = state.encoder_calls = state.decoder_calls = 0
    final_output, latency = benchmark_model(model, inputs, args.warmup, args.iters)

    total_forward_passes = args.warmup + args.iters
    expected_calls = total_forward_passes * 12
    call_counts = {
        "total": state.call_count,
        "encoder": state.encoder_calls,
        "decoder": state.decoder_calls,
        "expected_total": expected_calls,
    }

    speedup = args.baseline_ms / latency["median_ms"]
    fps = 1000.0 / latency["median_ms"]

    final_correctness = {
        "logits": tensor_error(final_output.logits, ref_logits),
        "pred_boxes": tensor_error(final_output.pred_boxes, ref_boxes),
    }

    result = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "experiment": "0008",
        "model": args.model,
        "device": "mps",
        "dtype": "float32",
        "image": str(args.image),
        "image_size_wh": list(image.size),
        "prompt": args.prompt,
        "software": {
            "torch": torch.__version__,
            "transformers": transformers.__version__,
        },
        "metalground": {
            "kernel": "metal_msda_v0_fp32",
            "threadgroup_size": args.threadgroup,
            "patched_modules": state.module_names,
            "num_patched_modules": len(state.module_names),
            "interop": (
                "large MPS tensors imported into MLX with copy=False; "
                "static spatial metadata constructed natively in MLX int32; "
                "conservative explicit synchronization around each custom core"
            ),
        },
        "correctness_first_patched_forward": correctness,
        "first_patched_forward_call_counts": first_call_counts,
        "reference_detections": ref_detections,
        "patched_detections": patched_detections,
        "patched_forward_latency": latency,
        "experiment_0001_baseline_median_ms": args.baseline_ms,
        "full_model_speedup_vs_experiment_0001": speedup,
        "patched_fps_from_median": fps,
        "benchmark_call_counts": call_counts,
        "final_correctness": final_correctness,
        "notes": [
            "The Experiment 0001 uninstrumented PyTorch/MPS median remains the external baseline.",
            "The unpatched reference forward in this script is used only as a numerical oracle, not as the performance baseline.",
            "JIT/first patched execution is excluded from timed iterations.",
            "This v0 integration uses conservative per-MSDA cross-framework synchronization.",
            "Prompt construction is intentionally preserved from Experiment 0001 for workload comparability.",
        ],
    }

    out = Path("results/metalground_v0_full_model.json")
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")

    print("\n=== Experiment 0008 summary ===", flush=True)
    print(f"patched median: {latency['median_ms']:.3f} ms", flush=True)
    print(f"patched p95:    {latency['p95_ms']:.3f} ms", flush=True)
    print(f"speedup:        {speedup:.3f}x", flush=True)
    print(f"FPS:            {fps:.3f}", flush=True)
    print(f"MSDA calls:     {call_counts}", flush=True)
    print(f"Saved: {out}", flush=True)


if __name__ == "__main__":
    main()
