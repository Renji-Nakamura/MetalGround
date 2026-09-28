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


def sync() -> None:
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


def mask_aware_error(a: torch.Tensor, b: torch.Tensor) -> dict[str, Any]:
    a = a.detach().float()
    b = b.detach().float()

    af = torch.isfinite(a)
    bf = torch.isfinite(b)
    both = af & bf

    out = {
        "shape": list(a.shape),
        "finite_mask_equal": bool(torch.equal(af, bf)),
        "nan_mask_equal": bool(torch.equal(torch.isnan(a), torch.isnan(b))),
        "posinf_mask_equal": bool(torch.equal(torch.isposinf(a), torch.isposinf(b))),
        "neginf_mask_equal": bool(torch.equal(torch.isneginf(a), torch.isneginf(b))),
    }

    if bool(both.any()):
        aa, bb = a[both], b[both]
        d = (aa - bb).abs()
        out.update(
            {
                "finite_max_abs": float(d.max().item()),
                "finite_mean_abs": float(d.mean().item()),
                "finite_rmse": float(torch.sqrt(torch.mean((aa - bb) ** 2)).item()),
                "finite_allclose_1e-5": bool(
                    torch.allclose(aa, bb, rtol=1e-5, atol=1e-5)
                ),
                "finite_allclose_1e-4": bool(
                    torch.allclose(aa, bb, rtol=1e-4, atol=1e-4)
                ),
            }
        )
    sync()
    return out


def detection_summary(
    processor,
    outputs,
    input_ids,
    text_labels,
    image_size_wh,
    box_threshold: float,
    text_threshold: float,
):
    det = processor.post_process_grounded_object_detection(
        outputs,
        input_ids,
        text_labels=text_labels,
        target_sizes=[image_size_wh[::-1]],
        threshold=box_threshold,
        text_threshold=text_threshold,
    )[0]

    return [
        {
            "label": label,
            "score": float(score),
            "box": [float(x) for x in box],
        }
        for label, score, box in zip(
            det["text_labels"],
            det["scores"].detach().float().cpu().tolist(),
            det["boxes"].detach().float().cpu().tolist(),
        )
    ]


@dataclass
class PatchState:
    threadgroup: int
    module_names: list[str] = field(default_factory=list)
    metadata_cache: dict = field(default_factory=dict)
    call_count: int = 0
    encoder_calls: int = 0
    decoder_calls: int = 0

    def metadata(self, shapes):
        key = tuple((int(h), int(w)) for h, w in shapes)
        if key not in self.metadata_cache:
            starts, acc = [], 0
            for h, w in key:
                starts.append(acc)
                acc += h * w
            sm = mx.array(np.asarray(key, dtype=np.int32), dtype=mx.int32)
            st = mx.array(np.asarray(starts, dtype=np.int32), dtype=mx.int32)
            mx.eval(sm, st)
            mx.synchronize()
            self.metadata_cache[key] = (sm, st)
        return self.metadata_cache[key]


def make_forward(state: PatchState, name: str):
    is_encoder = ".encoder.layers." in name

    def forward(
        self,
        value,
        value_spatial_shapes,
        value_spatial_shapes_list,
        level_start_index,
        sampling_locations,
        attention_weights,
        im2col_step,
    ):
        if value.dtype != torch.float32 or value.device.type != "mps":
            raise RuntimeError(
                f"MetalGround v0 expects FP32 MPS tensor, got {value.dtype} {value.device}"
            )

        if not value.is_contiguous():
            value = value.contiguous()
        if not sampling_locations.is_contiguous():
            sampling_locations = sampling_locations.contiguous()
        if not attention_weights.is_contiguous():
            attention_weights = attention_weights.contiguous()

        # Conservative Experiment 0007 protocol.
        sync()

        vm = mx.asarray(value, copy=False)
        lm = mx.asarray(sampling_locations, copy=False)
        wm = mx.asarray(attention_weights, copy=False)
        sm, st = state.metadata(value_spatial_shapes_list)

        out_mx = msda_metal_v0(
            vm,
            sm,
            st,
            lm,
            wm,
            threadgroup_size=state.threadgroup,
        )
        mx.eval(out_mx)
        mx.synchronize()

        out = torch.as_tensor(out_mx)
        if out.device.type != "mps":
            raise RuntimeError(f"Unexpected output device: {out.device}")

        state.call_count += 1
        if is_encoder:
            state.encoder_calls += 1
        else:
            state.decoder_calls += 1
        return out

    return forward


def patch_all_msda(model, threadgroup: int) -> PatchState:
    state = PatchState(threadgroup=threadgroup)
    for name, module in model.named_modules():
        if module.__class__.__name__ != "MultiScaleDeformableAttention":
            continue
        module.forward = types.MethodType(make_forward(state, name), module)
        state.module_names.append(name)

    if len(state.module_names) != 12:
        raise RuntimeError(f"Expected 12 MSDA cores, got {len(state.module_names)}")
    return state


def benchmark(model, inputs, warmup: int, iters: int):
    with torch.inference_mode():
        for i in range(warmup):
            _ = model(**inputs)
            sync()
            print(f"  warmup {i+1}/{warmup}", flush=True)

        samples = []
        out = None
        for i in range(iters):
            sync()
            t0 = time.perf_counter_ns()
            out = model(**inputs)
            sync()
            dt = (time.perf_counter_ns() - t0) / 1e6
            samples.append(dt)
            print(f"  {i+1:02d}/{iters}: {dt:.3f} ms", flush=True)

    return out, stats(samples)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="IDEA-Research/grounding-dino-tiny")
    ap.add_argument("--image", type=Path, default=Path("assets/input.jpg"))
    ap.add_argument("--prompt", nargs="+", default=["a cat", "a dog"])
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--threadgroup", type=int, default=256)
    ap.add_argument("--baseline-ms", type=float, default=999.116584)
    ap.add_argument("--box-threshold", type=float, default=0.3)
    ap.add_argument("--text-threshold", type=float, default=0.25)
    args = ap.parse_args()

    if not torch.backends.mps.is_available():
        raise SystemExit("MPS unavailable.")

    mx.set_default_device(mx.gpu)

    print("Loading model...", flush=True)
    processor = AutoProcessor.from_pretrained(args.model)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(args.model)
    model.eval().to("mps")
    sync()

    image = Image.open(args.image).convert("RGB")
    text_labels = [args.prompt]
    cpu_inputs = processor(images=image, text=text_labels, return_tensors="pt")
    inputs = {k: v.to("mps") for k, v in cpu_inputs.items()}
    sync()

    print("Reference forward for correctness oracle...", flush=True)
    with torch.inference_mode():
        ref = model(**inputs)
        sync()

    ref_logits = ref.logits.detach().clone()
    ref_boxes = ref.pred_boxes.detach().clone()
    ref_detections = detection_summary(
        processor,
        ref,
        inputs["input_ids"],
        text_labels,
        image.size,
        args.box_threshold,
        args.text_threshold,
    )

    print("Patching all 12 MSDA cores...", flush=True)
    state = patch_all_msda(model, args.threadgroup)
    for name in state.module_names:
        print(f"  {name}", flush=True)

    print("\nFirst patched forward (JIT excluded from timing)...", flush=True)
    state.call_count = state.encoder_calls = state.decoder_calls = 0
    with torch.inference_mode():
        first = model(**inputs)
        sync()

    first_counts = {
        "total": state.call_count,
        "encoder": state.encoder_calls,
        "decoder": state.decoder_calls,
    }

    correctness = {
        "logits_mask_aware": mask_aware_error(first.logits, ref_logits),
        "pred_boxes": mask_aware_error(first.pred_boxes, ref_boxes),
    }
    patched_detections = detection_summary(
        processor,
        first,
        inputs["input_ids"],
        text_labels,
        image.size,
        args.box_threshold,
        args.text_threshold,
    )

    print("\nRaw tensor comparison (rank-sensitive):", flush=True)
    print(json.dumps(correctness, indent=2), flush=True)
    print("Reference detections:", json.dumps(ref_detections, indent=2), flush=True)
    print("Patched detections:", json.dumps(patched_detections, indent=2), flush=True)
    print(f"MSDA calls in first patched forward: {first_counts}", flush=True)

    print(
        f"\nBenchmarking MetalGround v0 full model: warmup={args.warmup}, "
        f"iters={args.iters}",
        flush=True,
    )
    state.call_count = state.encoder_calls = state.decoder_calls = 0
    final, lat = benchmark(model, inputs, args.warmup, args.iters)

    expected_calls = (args.warmup + args.iters) * 12
    call_counts = {
        "total": state.call_count,
        "encoder": state.encoder_calls,
        "decoder": state.decoder_calls,
        "expected_total": expected_calls,
    }

    speedup = args.baseline_ms / lat["median_ms"]
    fps = 1000.0 / lat["median_ms"]

    final_detections = detection_summary(
        processor,
        final,
        inputs["input_ids"],
        text_labels,
        image.size,
        args.box_threshold,
        args.text_threshold,
    )

    result = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "experiment": "0009",
        "model": args.model,
        "device": "mps",
        "dtype": "float32",
        "image": str(args.image),
        "prompt": args.prompt,
        "software": {
            "torch": torch.__version__,
            "transformers": transformers.__version__,
        },
        "metalground": {
            "kernel": "metal_msda_v0_fp32",
            "patched_modules": state.module_names,
            "num_patched_modules": len(state.module_names),
            "threadgroup_size": args.threadgroup,
            "interop": (
                "large compute tensors zero-copy MPS<->MLX; tiny static metadata "
                "native MLX int32; conservative explicit synchronization per MSDA"
            ),
        },
        "correctness_context": {
            "experiment_0008d_conclusion": (
                "large rank-wise raw-output differences are causally dominated by "
                "three adjacent proposal rank swaps inside an unchanged top-900 set"
            ),
            "first_patched_forward_rank_sensitive_comparison": correctness,
            "reference_detections": ref_detections,
            "patched_detections": patched_detections,
            "final_benchmark_detections": final_detections,
            "first_forward_call_counts": first_counts,
        },
        "performance": {
            "patched_forward_latency": lat,
            "experiment_0001_authoritative_median_ms": args.baseline_ms,
            "speedup_vs_experiment_0001": speedup,
            "fps_from_median": fps,
            "benchmark_call_counts": call_counts,
        },
        "notes": [
            "Experiment 0001's uninstrumented PyTorch/MPS median remains the authoritative external baseline.",
            "The reference forward in this script is a correctness oracle, not a timing baseline.",
            "First patched execution/JIT is excluded from timing.",
            "This v0 hybrid integration intentionally retains conservative synchronization at every MSDA boundary.",
            "Raw query-wise tensors are rank-sensitive; Experiment 0008d established that three adjacent top-k rank swaps dominate the large raw max error.",
            "Dataset-level accuracy equivalence has not yet been established."
        ],
    }

    out_path = Path("results/metalground_v0_full_model_bench.json")
    out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")

    print("\n=== Experiment 0009 summary ===", flush=True)
    print(f"median:  {lat['median_ms']:.3f} ms", flush=True)
    print(f"p95:     {lat['p95_ms']:.3f} ms", flush=True)
    print(f"speedup: {speedup:.3f}x", flush=True)
    print(f"FPS:     {fps:.3f}", flush=True)
    print(f"calls:   {call_counts}", flush=True)
    print(f"Saved:   {out_path}", flush=True)


if __name__ == "__main__":
    main()
