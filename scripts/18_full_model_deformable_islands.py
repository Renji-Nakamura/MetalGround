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
class IslandPatchState:
    calls: int = 0


def patch_encoder_deformable_islands(
    model,
    island_cls,
    *,
    activation_name: str,
    spatial_shapes_list: list[tuple[int, int]],
    threadgroup: int,
):
    state = IslandPatchState()
    islands = []

    for i in range(6):
        layer = model.model.encoder.layers[i].deformable_layer
        island = island_cls(
            layer,
            activation_name=activation_name,
            spatial_shapes_list=spatial_shapes_list,
            threadgroup=threadgroup,
        )
        if island.compiled is None:
            raise RuntimeError(
                f"mx.compile unavailable for encoder layer {i}: "
                f"{island.compiled_error}"
            )

        def make_forward(bound_island):
            def forward(
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
                # DLPack/Metal sharing does not itself synchronize pending work.
                torch.mps.synchronize()

                hidden_mx = mx.asarray(hidden_states, copy=False)
                mask_mx = mx.asarray(attention_mask, copy=False)
                pos_mx = mx.asarray(position_embeddings, copy=False)
                ref_mx = mx.asarray(reference_points, copy=False)

                hidden_out_mx, attn_weights_mx = bound_island.compiled_call(
                    hidden_mx, mask_mx, pos_mx, ref_mx
                )
                mx.eval(hidden_out_mx, attn_weights_mx)
                mx.synchronize()

                hidden_out = torch.as_tensor(hidden_out_mx)
                attn_weights = torch.as_tensor(attn_weights_mx)
                torch.mps.synchronize()

                state.calls += 1
                return hidden_out, attn_weights

            return forward

        layer.forward = types.MethodType(make_forward(island), layer)
        islands.append(island)

    return state, islands


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
    ap.add_argument("--v1-baseline-ms", type=float, default=574.3521465)
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
    args = ap.parse_args()

    for p in (args.exp14_helper, args.exp17_helper):
        if not p.exists():
            raise SystemExit(f"Missing helper script: {p}")
    if not torch.backends.mps.is_available():
        raise SystemExit("MPS unavailable.")

    h14 = load_module(args.exp14_helper, "metalground_exp14_helpers_exp18")
    h17 = load_module(args.exp17_helper, "metalground_exp17_helpers_exp18")
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

    print("Building six fully-folded fusion specializers...", flush=True)
    fusion_items = h14.collect_fusion_specializers(model)

    print("Patching all 12 MSDA cores (encoder patches will later be bypassed by islands)...", flush=True)
    msda_state = h14.patch_msda(model, args.threadgroup)

    print("Enabling fully-folded fusion...", flush=True)
    h14.set_fusion_mode(fusion_items, "fully_folded")

    # Capture the runtime spatial metadata from the actual v1 trajectory.
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
    print("Capturing v1 spatial metadata...", flush=True)
    with torch.inference_mode():
        _ = model(**inputs)
        h14.sync()
    layer0.forward = original_layer0_forward

    if not captured:
        raise RuntimeError("Failed to capture spatial metadata.")

    print(f"spatial_shapes={captured['spatial_shapes_list']}", flush=True)

    # Import parameters only after producer synchronization.
    h14.sync()
    print("Building + compiling six encoder deformable MLX islands...", flush=True)
    island_state, islands = patch_encoder_deformable_islands(
        model,
        h17.MlxDeformableIsland,
        activation_name=model.config.activation_function,
        spatial_shapes_list=captured["spatial_shapes_list"],
        threadgroup=args.threadgroup,
    )

    print("First MetalGround v2-candidate forward (compile/JIT excluded)...", flush=True)
    msda_state.call_count = 0
    island_state.calls = 0
    with torch.inference_mode():
        first = model(**inputs)
        h14.sync()

    first_correctness = {
        "logits": h14.mask_aware_error(first.logits, reference.logits),
        "pred_boxes": h14.mask_aware_error(first.pred_boxes, reference.pred_boxes),
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
        "patched_msda_calls": msda_state.call_count,
    }

    print("Top-k:", json.dumps(first_correctness["topk"], indent=2), flush=True)
    print("Detections:", json.dumps(first_correctness["detections"], indent=2), flush=True)
    print(
        f"first-forward calls: islands={island_state.calls}, "
        f"patched_msda={msda_state.call_count}",
        flush=True,
    )

    if island_state.calls != 6:
        raise RuntimeError(
            f"Expected 6 encoder island calls, got {island_state.calls}"
        )
    # Encoder self_attn forwards are bypassed; only six decoder MSDA cores remain.
    if msda_state.call_count != 6:
        raise RuntimeError(
            f"Expected 6 decoder MSDA calls, got {msda_state.call_count}"
        )

    print(
        f"\nBenchmarking full model: warmup={args.warmup}, iters={args.iters}",
        flush=True,
    )
    msda_state.call_count = 0
    island_state.calls = 0

    final, latency = benchmark(
        model, inputs, h14.sync, args.warmup, args.iters
    )

    total_forwards = args.warmup + args.iters
    expected_islands = total_forwards * 6
    expected_decoder_msda = total_forwards * 6

    if island_state.calls != expected_islands:
        raise RuntimeError(
            f"Island call mismatch: got {island_state.calls}, "
            f"expected {expected_islands}"
        )
    if msda_state.call_count != expected_decoder_msda:
        raise RuntimeError(
            f"Decoder MSDA call mismatch: got {msda_state.call_count}, "
            f"expected {expected_decoder_msda}"
        )

    final_correctness = {
        "logits": h14.mask_aware_error(final.logits, reference.logits),
        "pred_boxes": h14.mask_aware_error(final.pred_boxes, reference.pred_boxes),
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
        "experiment": "0018",
        "purpose": (
            "Full-model integration of six compiled encoder deformable MLX islands "
            "on top of MetalGround v1."
        ),
        "model": args.model,
        "device": "mps",
        "dtype": "float32",
        "software": {
            "torch": torch.__version__,
            "transformers": transformers.__version__,
        },
        "configuration": {
            "fusion": "six fully_folded algebraic BiMHA layers",
            "encoder_deformable": "six compiled MLX execution islands",
            "encoder_msda_core": "MetalGround MSDA v0 called inside MLX islands",
            "decoder_msda_core": "MetalGround MSDA v0 via PyTorch-MPS/MLX bridge",
            "threadgroup_size": args.threadgroup,
            "warmup": args.warmup,
            "timed_iterations": args.iters,
            "approximation": False,
            "retraining": False,
            "reduced_precision": False,
        },
        "baselines": {
            "experiment_0001_original_pytorch_mps_median_ms": args.original_baseline_ms,
            "experiment_0015_metalground_v1_median_ms": args.v1_baseline_ms,
        },
        "latency": latency,
        "derived": {
            "speedup_vs_original": args.original_baseline_ms / median,
            "latency_reduction_vs_original_ms": args.original_baseline_ms - median,
            "latency_reduction_vs_original_percent": (
                100.0 * (args.original_baseline_ms - median)
                / args.original_baseline_ms
            ),
            "speedup_vs_v1": args.v1_baseline_ms / median,
            "latency_reduction_vs_v1_ms": args.v1_baseline_ms - median,
            "latency_reduction_vs_v1_percent": (
                100.0 * (args.v1_baseline_ms - median) / args.v1_baseline_ms
            ),
            "fps_from_median": 1000.0 / median,
        },
        "correctness": {
            "reference_detections": reference_detections,
            "first_patched_forward": first_correctness,
            "final_timed_forward": final_correctness,
        },
        "call_counts": {
            "encoder_island_calls": island_state.calls,
            "expected_encoder_island_calls": expected_islands,
            "decoder_patched_msda_calls": msda_state.call_count,
            "expected_decoder_patched_msda_calls": expected_decoder_msda,
        },
        "notes": [
            "Encoder deformable-layer self-attention modules are bypassed by the islands, so the patched MSDA call counter covers decoder cores only.",
            "Each encoder island maintains explicit synchronization at its PyTorch/MLX entry and exit.",
            "The first fully patched forward is excluded from timing.",
            "No retraining, approximation, pruning, or reduced precision is used.",
            "Dataset-level accuracy equivalence remains unmeasured."
        ],
    }

    out = Path("results/metalground_v2_candidate_full_model.json")
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")

    print("\n=== Experiment 0018 summary ===", flush=True)
    print(f"median: {median:.3f} ms", flush=True)
    print(f"p95:   {latency['p95_ms']:.3f} ms", flush=True)
    print(f"p99:   {latency['p99_ms']:.3f} ms", flush=True)
    print(
        f"speedup vs v1:       {result['derived']['speedup_vs_v1']:.3f}x",
        flush=True,
    )
    print(
        f"speedup vs original: {result['derived']['speedup_vs_original']:.3f}x",
        flush=True,
    )
    print(f"FPS: {result['derived']['fps_from_median']:.3f}", flush=True)
    print(f"Saved: {out}", flush=True)


if __name__ == "__main__":
    main()
