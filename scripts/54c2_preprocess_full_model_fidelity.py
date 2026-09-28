#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
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


IDENTITY = mx.fast.metal_kernel(
    name="metalground_0054c2_identity",
    input_names=["inp"],
    output_names=["out"],
    source=r"""
        uint elem = thread_position_in_grid.x;
        out[elem] = inp[elem];
    """,
)


def finite_error(a: torch.Tensor, b: torch.Tensor):
    a = a.detach()
    b = b.detach()
    fa = torch.isfinite(a)
    fb = torch.isfinite(b)
    common = fa & fb
    if bool(common.any().item()):
        x = a[common].float()
        y = b[common].float()
        d = (x - y).abs()
        return {
            "finite_mask_equal": bool(torch.equal(fa, fb)),
            "mean_abs": float(d.mean().item()),
            "max_abs": float(d.max().item()),
            "rmse": float(torch.sqrt(torch.mean((x - y) ** 2)).item()),
        }
    return {
        "finite_mask_equal": bool(torch.equal(fa, fb)),
        "mean_abs": 0.0,
        "max_abs": 0.0,
        "rmse": 0.0,
    }


def label_multiset(result):
    labels = result.get("text_labels", result.get("labels", []))
    return sorted(str(x) for x in labels)


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
        candidates = [
            j for j, lb in enumerate(labels_b)
            if j not in used and str(lb) == str(la)
        ]
        if not candidates:
            return 0.0
        ba = boxes_a[i]
        best_j, best_iou = None, -1.0
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
    ap.add_argument(
        "--reference-meta",
        type=Path,
        default=Path("results/0054b_reference/metadata.json"),
    )
    ap.add_argument(
        "--letterbox-image",
        type=Path,
        default=Path("results/0054b_reference/letterbox_reference.png"),
    )
    ap.add_argument(
        "--metal-json",
        type=Path,
        default=Path("results/metalground_metal_preprocess_0054b2.json"),
    )
    ap.add_argument(
        "--output",
        type=Path,
        default=Path("results/metalground_preprocess_full_model_0054c2.json"),
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

    for p in [args.image, args.reference_meta, args.letterbox_image, args.metal_json]:
        if not p.exists():
            raise SystemExit(f"Missing required artifact: {p}")
    for n in helpers:
        p = getattr(args, f"exp{n}_helper")
        if not p.exists():
            raise SystemExit(f"Missing helper: {p}")
    if not torch.backends.mps.is_available():
        raise SystemExit("PyTorch MPS unavailable")
    if not mx.metal.is_available():
        raise SystemExit("MLX Metal unavailable")

    mx.set_default_device(mx.gpu)

    h37 = load_module(args.exp37_helper, "mg54c2_h37")
    h38 = load_module(args.exp38_helper, "mg54c2_h38")
    h45 = load_module(args.exp45_helper, "mg54c2_h45")
    h46 = load_module(args.exp46_helper, "mg54c2_h46")

    print("Building adopted MetalGround runtime...", flush=True)
    rt = h37.build_runtime(args)
    model = rt["model"]
    processor = rt["processor"]
    h14 = rt["h14"]

    cache = h38.ExactTextBackboneCache(model.model.text_backbone)
    h38.set_runtime_mode(rt, cache, "wide_cache")

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

    # Reference path is exactly the 0054b CPU/PIL/HF semantics: the saved
    # 1200x901 PIL letterbox is handed to GroundingDinoImageProcessor.
    image = Image.open(args.letterbox_image).convert("RGB")
    ref_cpu = processor(
        images=image,
        text=[args.prompt],
        return_tensors="pt",
    )
    expected = (1, 3, 800, 1065)
    if tuple(ref_cpu["pixel_values"].shape) != expected:
        raise RuntimeError(
            f"Unexpected reference shape {tuple(ref_cpu['pixel_values'].shape)}"
        )
    reference_inputs = {k: v.to("mps") for k, v in ref_cpu.items()}
    h14.sync()

    cache.cached_output = None
    cache.prime(model, reference_inputs, h14.sync)
    cache.set_enabled(True)

    # Candidate values are the actual output produced by Experiment 0054b-2.
    metal_meta = json.loads(args.metal_json.read_text())
    candidate_path = Path(metal_meta["output_file"])
    if not candidate_path.exists():
        fallback = Path("results/metal_pixel_values_f32_chw_0054b2.bin")
        if fallback.exists():
            candidate_path = fallback
        else:
            raise FileNotFoundError(candidate_path)

    candidate_np = np.fromfile(candidate_path, dtype=np.float32).reshape(expected)
    ref_np = ref_cpu["pixel_values"].numpy()
    input_abs = np.abs(candidate_np - ref_np)
    input_diff = {
        "mean_abs": float(input_abs.mean()),
        "median_abs": float(np.median(input_abs)),
        "p95_abs": float(np.percentile(input_abs, 95)),
        "p99_abs": float(np.percentile(input_abs, 99)),
        "max_abs": float(input_abs.max()),
    }

    # Put the actual 0054b-2 values into a fresh Metal-produced MLX array,
    # then use the qualified 0054c-0 zero-copy handoff.
    seed = mx.array(candidate_np, dtype=mx.float32)
    produced = IDENTITY(
        inputs=[seed],
        grid=(int(candidate_np.size), 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[expected],
        output_dtypes=[mx.float32],
    )[0]
    mx.eval(produced)
    mx.synchronize()
    candidate_pixel_values = torch.from_dlpack(produced, copy=False)
    candidate_inputs = {
        **reference_inputs,
        "pixel_values": candidate_pixel_values,
    }

    h14.sync()
    with torch.inference_mode():
        ref_out = model(**reference_inputs)
    h14.sync()

    with torch.inference_mode():
        cand_out = model(**candidate_inputs)
    h14.sync()

    topk = h14.topk_audit(ref_out, cand_out, model.config.num_queries)
    required = {"set_overlap", "rankwise_identical"}
    missing = sorted(required - set(topk))
    if missing:
        raise RuntimeError(
            f"Unexpected topk schema missing={missing}, available={sorted(topk)}"
        )

    raw = {
        "logits": finite_error(ref_out.logits, cand_out.logits),
        "pred_boxes": finite_error(ref_out.pred_boxes, cand_out.pred_boxes),
        "topk": topk,
    }

    target_sizes = [(image.height, image.width)]
    ref_post = processor.post_process_grounded_object_detection(
        ref_out,
        reference_inputs["input_ids"],
        threshold=args.box_threshold,
        text_threshold=args.text_threshold,
        target_sizes=target_sizes,
    )[0]
    cand_post = processor.post_process_grounded_object_detection(
        cand_out,
        candidate_inputs["input_ids"],
        threshold=args.box_threshold,
        text_threshold=args.text_threshold,
        target_sizes=target_sizes,
    )[0]
    h14.sync()

    post = {
        "reference_count": int(len(ref_post["scores"])),
        "candidate_count": int(len(cand_post["scores"])),
        "count_equal": int(len(ref_post["scores"])) == int(len(cand_post["scores"])),
        "label_multiset_equal": label_multiset(ref_post) == label_multiset(cand_post),
        "same_label_greedy_min_iou": same_label_greedy_iou_min(ref_post, cand_post),
    }

    # Reuse the already pre-registered 0054b local input gates and the
    # trajectory/behavior criteria already used throughout MetalGround.
    gates = {
        "0054b_input_mean_abs_le_0_010": input_diff["mean_abs"] <= 0.010,
        "0054b_input_p99_abs_le_0_050": input_diff["p99_abs"] <= 0.050,
        "0054b_input_max_abs_le_0_250": input_diff["max_abs"] <= 0.250,
        "topk_membership_900_of_900": int(topk["set_overlap"]) == 900,
        "post_count_equal": post["count_equal"],
        "post_label_multiset_equal": post["label_multiset_equal"],
        "post_same_label_min_iou_ge_0_99": post["same_label_greedy_min_iou"] >= 0.99,
    }

    result = {
        "experiment": "0054c-2",
        "purpose": (
            "Measure downstream full-model behavior when the actual 0054b-2 "
            "Pillow-compatible Metal preprocess tensor is delivered through "
            "the qualified 0054c-0 zero-copy DLPack bridge."
        ),
        "model": args.model,
        "prompt": args.prompt,
        "reference": {
            "image": str(args.letterbox_image),
            "semantics": "saved PIL letterbox -> GroundingDinoImageProcessor",
        },
        "candidate": {
            "source_file": str(candidate_path),
            "semantics": "actual 0054b-2 Metal preprocess output",
            "handoff": "MLX Metal producer -> explicit sync -> DLPack copy=False -> PyTorch MPS",
            "cpu_file_seed_is_outside_final_live_path": True,
        },
        "input_difference": input_diff,
        "correctness": {
            "raw": raw,
            "postprocess": post,
        },
        "pre_registered_gates": gates,
        "all_gates_pass": all(gates.values()),
        "interpretation_guardrails": [
            "Raw logits/box differences are reported descriptively; the gate emphasizes proposal-set and downstream behavior because MetalGround has already documented near-tie trajectory sensitivity.",
            "Top-k rankwise equality is reported but not gated; membership is the pre-registered proposal criterion.",
            "0054c-2 does not establish the final live CVPixelBuffer-to-destination-buffer ownership bridge because the successful 0054b-2 tensor is seeded from its saved file for this fixed-input composition test.",
            "No performance claim is made from 0054c-2; 0054b-2 and 0054c-0 already isolate preprocess and handoff costs."
        ],
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2))

    print("=== Experiment 0054c-2 ===")
    print("input diff:", input_diff)
    print("raw logits:", raw["logits"])
    print("raw boxes:", raw["pred_boxes"])
    print("topk:", topk)
    print("postprocess:", post)
    print("ALL GATES PASS:", result["all_gates_pass"])
    print("Saved:", args.output)


if __name__ == "__main__":
    main()
