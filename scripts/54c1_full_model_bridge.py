#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import statistics
import sys
import time
from pathlib import Path

import mlx.core as mx
import numpy as np
import torch
from PIL import Image


def load_module(path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load helper: {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod


def percentile(xs, p):
    ys = sorted(float(x) for x in xs)
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


IDENTITY = mx.fast.metal_kernel(
    name="metalground_0054c1_identity",
    input_names=["inp"],
    output_names=["out"],
    source=r"""
        uint elem = thread_position_in_grid.x;
        out[elem] = inp[elem];
    """,
)


def torch_from_mlx_no_copy(a):
    return torch.from_dlpack(a, copy=False)


def mask_aware_error(a: torch.Tensor, b: torch.Tensor):
    a = a.detach()
    b = b.detach()
    fa = torch.isfinite(a)
    fb = torch.isfinite(b)
    common = fa & fb
    if bool(common.any().item()):
        da = a[common].float()
        db = b[common].float()
        diff = (da - db).abs()
        max_abs = float(diff.max().item())
        mean_abs = float(diff.mean().item())
        rmse = float(torch.sqrt(torch.mean((da - db) ** 2)).item())
    else:
        max_abs = mean_abs = rmse = 0.0
    return {
        "finite_mask_equal": bool(torch.equal(fa, fb)),
        "max_abs": max_abs,
        "mean_abs": mean_abs,
        "rmse": rmse,
        "allclose_1e5": bool(torch.allclose(a, b, rtol=1e-5, atol=1e-5)),
    }


def label_multiset(result):
    labels = result.get("text_labels", result.get("labels", []))
    return sorted(str(x) for x in labels)


def result_count(result):
    return int(len(result["scores"]))


def same_label_greedy_iou_min(a, b):
    labels_a = list(a.get("text_labels", a.get("labels", [])))
    labels_b = list(b.get("text_labels", b.get("labels", [])))
    boxes_a = a["boxes"].detach().float().cpu()
    boxes_b = b["boxes"].detach().float().cpu()

    if len(labels_a) == 0 and len(labels_b) == 0:
        return 1.0
    if sorted(map(str, labels_a)) != sorted(map(str, labels_b)):
        return 0.0

    used = set()
    ious = []
    for i, la in enumerate(labels_a):
        candidates = [j for j, lb in enumerate(labels_b)
                      if j not in used and str(lb) == str(la)]
        if not candidates:
            return 0.0
        ba = boxes_a[i]
        best_j = None
        best_iou = -1.0
        for j in candidates:
            bb = boxes_b[j]
            x1 = max(float(ba[0]), float(bb[0]))
            y1 = max(float(ba[1]), float(bb[1]))
            x2 = min(float(ba[2]), float(bb[2]))
            y2 = min(float(ba[3]), float(bb[3]))
            inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
            aa = max(0.0, float(ba[2] - ba[0])) * max(0.0, float(ba[3] - ba[1]))
            ab = max(0.0, float(bb[2] - bb[0])) * max(0.0, float(bb[3] - bb[1]))
            union = aa + ab - inter
            iou = inter / union if union > 0 else 1.0
            if iou > best_iou:
                best_iou = iou
                best_j = j
        used.add(best_j)
        ious.append(best_iou)
    return float(min(ious)) if ious else 1.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="IDEA-Research/grounding-dino-tiny")
    ap.add_argument("--image", type=Path, default=Path("assets/input.jpg"))
    ap.add_argument("--prompt", nargs="+", default=["a cat", "a dog"])
    ap.add_argument("--box-threshold", type=float, default=0.30)
    ap.add_argument("--text-threshold", type=float, default=0.25)
    ap.add_argument("--threadgroup", type=int, default=256)
    ap.add_argument("--warmup-per-mode", type=int, default=3)
    ap.add_argument("--pairs", type=int, default=20)
    ap.add_argument(
        "--output",
        type=Path,
        default=Path("results/metalground_full_model_bridge_0054c1.json"),
    )

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
        45: "bench_swin_stage0_mlp_mlx.py",
        46: "full_model_stage0_mlp_mlx.py",
    }
    for n, fn in helpers.items():
        ap.add_argument(
            f"--exp{n}-helper",
            type=Path,
            default=Path("scripts") / f"{n:02d}_{fn}",
        )
    args = ap.parse_args()

    if not args.image.exists():
        raise SystemExit(f"Missing image: {args.image}")
    for n in helpers:
        p = getattr(args, f"exp{n}_helper")
        if not p.exists():
            raise SystemExit(f"Missing helper: {p}")
    if not torch.backends.mps.is_available():
        raise SystemExit("PyTorch MPS unavailable.")
    if not mx.metal.is_available():
        raise SystemExit("MLX Metal unavailable.")

    mx.set_default_device(mx.gpu)

    h37 = load_module(args.exp37_helper, "mg54c1_h37")
    h38 = load_module(args.exp38_helper, "mg54c1_h38")
    h45 = load_module(args.exp45_helper, "mg54c1_h45")
    h46 = load_module(args.exp46_helper, "mg54c1_h46")

    print("Building adopted MetalGround runtime...", flush=True)
    rt = h37.build_runtime(args)
    model = rt["model"]
    processor = rt["processor"]
    h14 = rt["h14"]

    # Adopt prompt cache.
    cache = h38.ExactTextBackboneCache(model.model.text_backbone)
    h38.set_runtime_mode(rt, cache, "wide_cache")

    # Adopt stage-0 compiled MLX MLPs.
    stages = model.model.backbone.conv_encoder.model.swin.encoder.layers
    stage0_modules = [stages[0].blocks[0].mlp, stages[0].blocks[1].mlp]
    h14.sync()
    stage0_candidates = [h45.MlxSwinMLP(m) for m in stage0_modules]
    stage0_dispatchers = [
        h46.SwinMlpDispatcher(m, c, h14.sync)
        for m, c in zip(stage0_modules, stage0_candidates)
    ]
    for d in stage0_dispatchers:
        d.set_mode("mlx")

    image = Image.open(args.image).convert("RGB")
    cpu_inputs = processor(
        images=image,
        text=[args.prompt],
        return_tensors="pt",
    )
    expected_shape = (1, 3, 800, 1065)
    if tuple(cpu_inputs["pixel_values"].shape) != expected_shape:
        raise RuntimeError(
            f"Unexpected pixel_values shape {tuple(cpu_inputs['pixel_values'].shape)}; "
            f"expected {expected_shape}"
        )

    reference_inputs = {k: v.to("mps") for k, v in cpu_inputs.items()}
    h14.sync()

    # Prime text cache with the reference input before correctness/timing.
    cache.cached_output = None
    cache.prime(model, reference_inputs, h14.sync)
    cache.set_enabled(True)

    # Create a Metal-produced MLX array with values EXACTLY equal to the
    # reference processor tensor. CPU->MLX seed upload is outside timing and is
    # deliberately not part of the scientific question in 0054c-1.
    pixel_np = cpu_inputs["pixel_values"].numpy().astype(np.float32, copy=False)
    mx_seed = mx.array(pixel_np, dtype=mx.float32)
    produced = IDENTITY(
        inputs=[mx_seed],
        grid=(int(pixel_np.size), 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[expected_shape],
        output_dtypes=[mx.float32],
    )[0]
    mx.eval(produced)
    mx.synchronize()

    bridged_once = torch_from_mlx_no_copy(produced)
    torch.mps.synchronize()
    bridge_device = bridged_once.device.type
    bridge_shape = list(bridged_once.shape)
    bridge_dtype = str(bridged_once.dtype)
    pixel_diff = (
        bridged_once.detach().cpu()
        - cpu_inputs["pixel_values"]
    ).abs()
    pixel_max_abs = float(pixel_diff.max().item())
    pixel_mean_abs = float(pixel_diff.mean().item())
    del bridged_once

    def candidate_inputs_new_view():
        # The MLX owner `produced` remains alive for the whole experiment.
        pix = torch_from_mlx_no_copy(produced)
        return {
            **reference_inputs,
            "pixel_values": pix,
        }

    # One correctness pair.
    h14.sync()
    with torch.inference_mode():
        reference_out = model(**reference_inputs)
    h14.sync()

    candidate_inputs = candidate_inputs_new_view()
    with torch.inference_mode():
        candidate_out = model(**candidate_inputs)
    h14.sync()

    raw = {
        "logits": mask_aware_error(reference_out.logits, candidate_out.logits),
        "pred_boxes": mask_aware_error(
            reference_out.pred_boxes, candidate_out.pred_boxes
        ),
        "topk": h14.topk_audit(
            reference_out, candidate_out, model.config.num_queries
        ),
    }

    target_size = [(image.height, image.width)]
    ref_post = processor.post_process_grounded_object_detection(
        reference_out,
        reference_inputs["input_ids"],
        threshold=args.box_threshold,
        text_threshold=args.text_threshold,
        target_sizes=target_size,
    )[0]
    cand_post = processor.post_process_grounded_object_detection(
        candidate_out,
        candidate_inputs["input_ids"],
        threshold=args.box_threshold,
        text_threshold=args.text_threshold,
        target_sizes=target_size,
    )[0]
    h14.sync()

    post = {
        "reference_count": result_count(ref_post),
        "candidate_count": result_count(cand_post),
        "count_equal": result_count(ref_post) == result_count(cand_post),
        "label_multiset_equal": label_multiset(ref_post) == label_multiset(cand_post),
        "same_label_greedy_min_iou": same_label_greedy_iou_min(ref_post, cand_post),
    }

    # Warm both storage modes.
    for i in range(args.warmup_per_mode):
        with torch.inference_mode():
            _ = model(**reference_inputs)
        h14.sync()
        ci = candidate_inputs_new_view()
        with torch.inference_mode():
            _ = model(**ci)
        h14.sync()
        print(f"warmup {i+1}/{args.warmup_per_mode}", flush=True)

    reference_ms = []
    candidate_bridge_plus_model_ms = []
    paired_delta_ms = []

    for i in range(args.pairs):
        reference_first = (i % 2 == 0)
        order = ("reference", "candidate") if reference_first else ("candidate", "reference")
        local = {}

        for mode in order:
            h14.sync()
            t0 = time.perf_counter_ns()
            if mode == "reference":
                inputs = reference_inputs
            else:
                # Candidate timing intentionally includes the DLPack import.
                inputs = candidate_inputs_new_view()
            with torch.inference_mode():
                _ = model(**inputs)
            h14.sync()
            local[mode] = (time.perf_counter_ns() - t0) / 1e6

        reference_ms.append(local["reference"])
        candidate_bridge_plus_model_ms.append(local["candidate"])
        paired_delta_ms.append(local["candidate"] - local["reference"])
        print(
            f"pair {i+1:02d}/{args.pairs}: "
            f"ref={local['reference']:.3f} ms "
            f"candidate={local['candidate']:.3f} ms "
            f"delta={paired_delta_ms[-1]:+.3f} ms",
            flush=True,
        )

    ref_stats = stats(reference_ms)
    cand_stats = stats(candidate_bridge_plus_model_ms)
    delta_stats = stats(paired_delta_ms)

    topk = raw["topk"]
    required_topk_keys = {"set_overlap", "rankwise_identical"}
    missing_topk_keys = sorted(required_topk_keys - set(topk))
    if missing_topk_keys:
        raise RuntimeError(
            f"Unexpected topk_audit schema; missing={missing_topk_keys}, "
            f"available={sorted(topk.keys())}"
        )

    gates = {
        "bridge_device_mps": bridge_device == "mps",
        "bridge_dtype_float32": bridge_dtype == "torch.float32",
        "bridge_shape_exact": bridge_shape == list(expected_shape),
        "pixel_values_max_abs_eq_0": pixel_max_abs == 0.0,
        "raw_logits_allclose_1e5": raw["logits"]["allclose_1e5"],
        "raw_boxes_allclose_1e5": raw["pred_boxes"]["allclose_1e5"],
        "topk_membership_900_of_900": int(topk["set_overlap"]) == 900,
        "topk_rankwise_900_of_900": int(topk["rankwise_identical"]) == 900,
        "post_count_equal": post["count_equal"],
        "post_label_multiset_equal": post["label_multiset_equal"],
        "post_same_label_min_iou_ge_0_9999": post["same_label_greedy_min_iou"] >= 0.9999,
        "paired_median_regression_ms_le_10": delta_stats["median_ms"] <= 10.0,
    }

    result = {
        "experiment": "0054c-1",
        "purpose": (
            "Verify that the adopted MetalGround full model can consume an "
            "exact-value MLX-Metal -> PyTorch-MPS DLPack view without "
            "behavioral divergence or material latency regression."
        ),
        "model": args.model,
        "image": str(args.image),
        "prompt": args.prompt,
        "thresholds": {
            "box": args.box_threshold,
            "text": args.text_threshold,
        },
        "bridge": {
            "producer": "MLX custom Metal identity kernel",
            "copy_false": True,
            "producer_eval_and_sync_before_handoff": True,
            "torch_device": bridge_device,
            "torch_dtype": bridge_dtype,
            "torch_shape": bridge_shape,
            "pixel_values_mean_abs": pixel_mean_abs,
            "pixel_values_max_abs": pixel_max_abs,
            "mlx_owner_kept_alive_through_model_use": True,
        },
        "correctness": {
            "raw": raw,
            "postprocess": post,
        },
        "timing": {
            "warmup_per_mode": args.warmup_per_mode,
            "pairs": args.pairs,
            "alternating_order": True,
            "reference_model_only_ms": ref_stats,
            "candidate_dlpack_import_plus_model_ms": cand_stats,
            "paired_candidate_minus_reference_ms": delta_stats,
            "candidate_timing_includes_dlpack_import": True,
            "producer_work_excluded": True,
        },
        "pre_registered_gates": gates,
        "all_gates_pass": all(gates.values()),
        "guardrails": [
            "0054c-1 isolates full-model consumption of bridged Metal storage; it does not yet compose the live camera and 0054b preprocessing producer in one process.",
            "The candidate producer values are intentionally exact copies of the HF reference pixel_values so preprocessing numerical error cannot confound the bridge integration result.",
            "CPU->MLX seeding occurs before timing and is not part of the proposed final live path.",
            "A positive candidate-reference timing delta above 10 ms median is a pre-registered integration regression failure even though 0054c-0 bridge creation itself is sub-microsecond.",
        ],
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2))

    print("\n=== Experiment 0054c-1 ===")
    print("pixel max abs:", pixel_max_abs)
    print("raw logits:", raw["logits"])
    print("raw boxes:", raw["pred_boxes"])
    print("topk:", topk)
    print("post:", post)
    print("reference median:", ref_stats["median_ms"], "ms")
    print("candidate median:", cand_stats["median_ms"], "ms")
    print("paired delta median:", delta_stats["median_ms"], "ms")
    print("ALL GATES PASS:", result["all_gates_pass"])
    print("Saved:", args.output)


if __name__ == "__main__":
    main()
