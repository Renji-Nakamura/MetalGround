#!/usr/bin/env python3
from __future__ import annotations

import argparse
import collections
import importlib.util
import json
import math
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoModelForZeroShotObjectDetection


THRESHOLD_PROFILES = [
    {"id": "permissive", "box_threshold": 0.20, "text_threshold": 0.20},
    {"id": "default", "box_threshold": 0.30, "text_threshold": 0.25},
    {"id": "strict", "box_threshold": 0.40, "text_threshold": 0.30},
]


def load_module(path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load helper: {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod


def median_or_none(xs):
    return None if not xs else statistics.median(xs)


def box_iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    aa = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    ba = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = aa + ba - inter
    return 0.0 if union <= 0.0 else inter / union


def match_detections(ref, cand):
    ref_labels = collections.Counter(d["label"] for d in ref)
    cand_labels = collections.Counter(d["label"] for d in cand)

    # Global greedy matching by IoU, constrained to equal labels.
    pairs = []
    for i, r in enumerate(ref):
        for j, c in enumerate(cand):
            if r["label"] == c["label"]:
                pairs.append((box_iou(r["box"], c["box"]), i, j))
    pairs.sort(reverse=True)

    used_r = set()
    used_c = set()
    matches = []
    for iou, i, j in pairs:
        if i in used_r or j in used_c:
            continue
        used_r.add(i)
        used_c.add(j)
        r = ref[i]
        c = cand[j]
        matches.append(
            {
                "ref_index": i,
                "cand_index": j,
                "label": r["label"],
                "iou": iou,
                "score_abs_diff": abs(r["score"] - c["score"]),
                "box_max_abs_diff_px": max(
                    abs(a - b) for a, b in zip(r["box"], c["box"])
                ),
            }
        )

    ious = [m["iou"] for m in matches]
    score_diffs = [m["score_abs_diff"] for m in matches]
    box_diffs = [m["box_max_abs_diff_px"] for m in matches]

    same_count = len(ref) == len(cand)
    label_multiset_equal = ref_labels == cand_labels
    all_label_matched = (
        len(matches) == len(ref) == len(cand)
    )

    return {
        "original_count": len(ref),
        "final_count": len(cand),
        "count_equal": same_count,
        "label_multiset_equal": label_multiset_equal,
        "matched_same_label": len(matches),
        "unmatched_original": len(ref) - len(used_r),
        "unmatched_final": len(cand) - len(used_c),
        "all_label_matched": all_label_matched,
        "all_matched_iou_ge_0_50": bool(
            all_label_matched and all(x >= 0.50 for x in ious)
        ),
        "all_matched_iou_ge_0_99": bool(
            all_label_matched and all(x >= 0.99 for x in ious)
        ),
        "min_matched_iou": min(ious) if ious else (
            1.0 if len(ref) == len(cand) == 0 else None
        ),
        "median_matched_iou": median_or_none(ious),
        "max_score_abs_diff": max(score_diffs) if score_diffs else 0.0,
        "max_box_abs_diff_px": max(box_diffs) if box_diffs else 0.0,
        "matches": matches,
    }


def raw_error(a: torch.Tensor, b: torch.Tensor):
    af = a.detach().float()
    bf = b.detach().float()
    fa = torch.isfinite(af)
    fb = torch.isfinite(bf)
    common = fa & fb
    if bool(common.any().item()):
        d = (af[common] - bf[common]).abs()
        return {
            "finite_mask_equal": bool(torch.equal(fa, fb)),
            "max_abs": float(d.max().item()),
            "mean_abs": float(d.mean().item()),
            "rmse": float(
                torch.sqrt(torch.mean((af[common] - bf[common]) ** 2)).item()
            ),
        }
    return {
        "finite_mask_equal": bool(torch.equal(fa, fb)),
        "max_abs": 0.0,
        "mean_abs": 0.0,
        "rmse": 0.0,
    }


def topk_detail(final_out, original_out, k):
    rs = original_out.enc_outputs_class.detach().float().max(-1).values
    cs = final_out.enc_outputs_class.detach().float().max(-1).values
    _, ri = torch.topk(rs, k, dim=1)
    _, ci = torch.topk(cs, k, dim=1)

    r = ri[0].cpu().tolist()
    c = ci[0].cpu().tolist()
    mismatches = [i for i, (a, b) in enumerate(zip(r, c)) if a != b]

    swap_pairs = []
    covered = set()
    p = 0
    while p + 1 < len(mismatches):
        i = mismatches[p]
        j = mismatches[p + 1]
        if (
            j == i + 1
            and r[i] == c[j]
            and r[j] == c[i]
        ):
            a = r[i]
            b = r[j]
            swap_pairs.append(
                {
                    "ranks_zero_based": [i, j],
                    "proposal_indices_original_order": [a, b],
                    "original_score_gap_abs": float(
                        abs(rs[0, a] - rs[0, b]).item()
                    ),
                    "final_score_gap_abs": float(
                        abs(cs[0, a] - cs[0, b]).item()
                    ),
                }
            )
            covered.update((i, j))
            p += 2
        else:
            p += 1

    rset = set(r)
    cset = set(c)
    score_diff = (cs - rs).abs()

    return {
        "k": k,
        "set_overlap": len(rset & cset),
        "changed_membership_each_side": k - len(rset & cset),
        "rankwise_identical": k - len(mismatches),
        "mismatch_count": len(mismatches),
        "mismatched_ranks_zero_based": mismatches,
        "all_rank_mismatches_are_adjacent_swaps": (
            len(covered) == len(mismatches)
        ),
        "adjacent_swap_pairs": swap_pairs,
        "score_vector_max_abs": float(score_diff.max().item()),
        "score_vector_mean_abs": float(score_diff.mean().item()),
    }


def make_inputs(processor, image, prompts):
    text_labels = [prompts]
    cpu = processor(
        images=image,
        text=text_labels,
        return_tensors="pt",
    )
    return {k: v.to("mps") for k, v in cpu.items()}, text_labels


def aggregate(cases, num_queries):
    agg = {
        "total_cases": len(cases),
        "topk": {},
        "threshold_profiles": {},
        "by_prompt": {},
    }

    topks = [c["topk"] for c in cases]
    agg["topk"] = {
        "membership_preserved_cases": sum(
            t["set_overlap"] == num_queries for t in topks
        ),
        "membership_preserved_rate": (
            sum(t["set_overlap"] == num_queries for t in topks) / len(topks)
            if topks else 0.0
        ),
        "rankwise_exact_cases": sum(
            t["rankwise_identical"] == num_queries for t in topks
        ),
        "rankwise_exact_rate": (
            sum(t["rankwise_identical"] == num_queries for t in topks) / len(topks)
            if topks else 0.0
        ),
        "adjacent_swap_only_cases": sum(
            t["mismatch_count"] > 0
            and t["all_rank_mismatches_are_adjacent_swaps"]
            for t in topks
        ),
        "max_changed_membership_each_side": max(
            (t["changed_membership_each_side"] for t in topks),
            default=0,
        ),
        "max_mismatch_count": max(
            (t["mismatch_count"] for t in topks),
            default=0,
        ),
        "median_rankwise_identical": median_or_none(
            [t["rankwise_identical"] for t in topks]
        ),
        "max_score_vector_abs_diff": max(
            (t["score_vector_max_abs"] for t in topks),
            default=0.0,
        ),
    }

    mismatch_hist = collections.Counter()
    for t in topks:
        mismatch_hist.update(t["mismatched_ranks_zero_based"])
    agg["topk"]["mismatch_rank_histogram"] = {
        str(k): v for k, v in sorted(mismatch_hist.items())
    }

    for profile in THRESHOLD_PROFILES:
        pid = profile["id"]
        vals = [c["postprocess"][pid] for c in cases]
        agg["threshold_profiles"][pid] = {
            "box_threshold": profile["box_threshold"],
            "text_threshold": profile["text_threshold"],
            "count_equal_cases": sum(v["count_equal"] for v in vals),
            "label_multiset_equal_cases": sum(
                v["label_multiset_equal"] for v in vals
            ),
            "all_label_matched_cases": sum(
                v["all_label_matched"] for v in vals
            ),
            "all_matched_iou_ge_0_99_cases": sum(
                v["all_matched_iou_ge_0_99"] for v in vals
            ),
            "count_equal_rate": (
                sum(v["count_equal"] for v in vals) / len(vals)
                if vals else 0.0
            ),
            "label_multiset_equal_rate": (
                sum(v["label_multiset_equal"] for v in vals) / len(vals)
                if vals else 0.0
            ),
            "high_iou_agreement_rate": (
                sum(v["all_matched_iou_ge_0_99"] for v in vals) / len(vals)
                if vals else 0.0
            ),
            "max_score_abs_diff": max(
                (v["max_score_abs_diff"] for v in vals),
                default=0.0,
            ),
            "max_box_abs_diff_px": max(
                (v["max_box_abs_diff_px"] for v in vals),
                default=0.0,
            ),
        }

    prompt_ids = sorted({c["prompt_id"] for c in cases})
    for pid in prompt_ids:
        subset = [c for c in cases if c["prompt_id"] == pid]
        agg["by_prompt"][pid] = {
            "cases": len(subset),
            "topk_membership_preserved": sum(
                c["topk"]["set_overlap"] == num_queries for c in subset
            ),
            "rankwise_exact": sum(
                c["topk"]["rankwise_identical"] == num_queries
                for c in subset
            ),
            "default_detection_high_iou_agreement": sum(
                c["postprocess"]["default"]["all_matched_iou_ge_0_99"]
                for c in subset
            ),
        }

    agg["raw_output"] = {
        "max_logit_abs_diff": max(
            (c["raw"]["logits"]["max_abs"] for c in cases),
            default=0.0,
        ),
        "median_logit_max_abs_diff": median_or_none(
            [c["raw"]["logits"]["max_abs"] for c in cases]
        ),
        "max_box_abs_diff": max(
            (c["raw"]["pred_boxes"]["max_abs"] for c in cases),
            default=0.0,
        ),
        "median_box_max_abs_diff": median_or_none(
            [c["raw"]["pred_boxes"]["max_abs"] for c in cases]
        ),
    }

    return agg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--manifest",
        type=Path,
        default=Path("assets/faithfulness_0051/manifest.json"),
    )
    ap.add_argument(
        "--prompt-manifest",
        type=Path,
        default=Path("assets/faithfulness_0051/prompts.json"),
    )
    ap.add_argument("--limit-images", type=int)
    ap.add_argument("--limit-prompts", type=int)
    ap.add_argument(
        "--output",
        type=Path,
        default=Path("results/metalground_faithfulness_sweep_0051.json"),
    )
    ap.add_argument(
        "--model",
        default="IDEA-Research/grounding-dino-tiny",
    )
    ap.add_argument("--threadgroup", type=int, default=256)

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

    for p in [args.manifest, args.prompt_manifest]:
        if not p.exists():
            raise SystemExit(
                f"Missing {p}. Run scripts/51_prepare_faithfulness_set.py first."
            )
    for n in helpers:
        p = getattr(args, f"exp{n}_helper")
        if not p.exists():
            raise SystemExit(f"Missing helper: {p}")

    if not torch.backends.mps.is_available():
        raise SystemExit("MPS unavailable.")

    manifest = json.loads(args.manifest.read_text())
    prompt_manifest = json.loads(args.prompt_manifest.read_text())

    images = manifest["images"]
    prompts = prompt_manifest["prompt_sets"]
    if args.limit_images is not None:
        images = images[: args.limit_images]
    if args.limit_prompts is not None:
        prompts = prompts[: args.limit_prompts]
    if not images or not prompts:
        raise SystemExit("No images/prompts selected.")

    first_image = Path(images[0]["path"])
    first_prompts = list(prompts[0]["labels"])

    # build_runtime expects these experiment-era fields on args.
    args.image = first_image
    args.prompt = first_prompts

    h37 = load_module(args.exp37_helper, "mg51_h37")
    h38 = load_module(args.exp38_helper, "mg51_h38")
    h45 = load_module(args.exp45_helper, "mg51_h45")
    h46 = load_module(args.exp46_helper, "mg51_h46")

    print("Building adopted MetalGround runtime...", flush=True)
    rt = h37.build_runtime(args)
    final_model = rt["model"]
    processor = rt["processor"]
    h14 = rt["h14"]

    cache = h38.ExactTextBackboneCache(
        final_model.model.text_backbone
    )
    stage0_dispatchers = []

    try:
        h38.set_runtime_mode(rt, cache, "wide_cache")

        stages = (
            final_model.model.backbone.conv_encoder.model.swin.encoder.layers
        )
        stage0_modules = [
            stages[0].blocks[0].mlp,
            stages[0].blocks[1].mlp,
        ]
        h14.sync()
        stage0_candidates = [
            h45.MlxSwinMLP(m) for m in stage0_modules
        ]
        stage0_dispatchers = [
            h46.SwinMlpDispatcher(m, c, h14.sync)
            for m, c in zip(stage0_modules, stage0_candidates)
        ]
        for d in stage0_dispatchers:
            d.set_mode("mlx")

        print("Loading untouched original model...", flush=True)
        original_model = (
            AutoModelForZeroShotObjectDetection
            .from_pretrained(args.model)
            .eval()
            .to("mps")
        )
        h14.sync()

        base_pixel_shape = tuple(rt["inputs"]["pixel_values"].shape)
        cases = []
        total = len(images) * len(prompts)
        case_idx = 0

        for prompt_idx, prompt_spec in enumerate(prompts):
            prompt_id = prompt_spec["id"]
            labels = list(prompt_spec["labels"])

            # Prime exact text cache once per prompt set. Text backbone output
            # is image-independent, and the cache is invalidated between prompts.
            prime_image = Image.open(images[0]["path"]).convert("RGB")
            prime_inputs, _ = make_inputs(
                processor, prime_image, labels
            )
            if tuple(prime_inputs["pixel_values"].shape) != base_pixel_shape:
                raise RuntimeError(
                    f"Fixed-geometry invariant failed for prompt {prompt_id}: "
                    f"{tuple(prime_inputs['pixel_values'].shape)} != "
                    f"{base_pixel_shape}"
                )

            cache.cached_output = None
            cache.prime(final_model, prime_inputs, h14.sync)
            cache.set_enabled(True)

            for image_idx, image_spec in enumerate(images):
                case_idx += 1
                image_path = Path(image_spec["path"])
                image = Image.open(image_path).convert("RGB")
                inputs, text_labels = make_inputs(
                    processor, image, labels
                )

                if tuple(inputs["pixel_values"].shape) != base_pixel_shape:
                    raise RuntimeError(
                        f"Fixed-geometry invariant failed at {image_path}: "
                        f"{tuple(inputs['pixel_values'].shape)} != "
                        f"{base_pixel_shape}"
                    )

                print(
                    f"[{case_idx:03d}/{total}] "
                    f"image={image_spec['id']} prompt={prompt_id}",
                    flush=True,
                )

                with torch.inference_mode():
                    original_out = original_model(**inputs)
                h14.sync()

                with torch.inference_mode():
                    final_out = final_model(**inputs)
                h14.sync()

                topk = topk_detail(
                    final_out,
                    original_out,
                    final_model.config.num_queries,
                )

                raw = {
                    "logits": raw_error(
                        final_out.logits,
                        original_out.logits,
                    ),
                    "pred_boxes": raw_error(
                        final_out.pred_boxes,
                        original_out.pred_boxes,
                    ),
                }

                post = {}
                for profile in THRESHOLD_PROFILES:
                    ref_det = h14.detection_summary(
                        processor,
                        original_out,
                        inputs["input_ids"],
                        text_labels,
                        image.size,
                        profile["box_threshold"],
                        profile["text_threshold"],
                    )
                    final_det = h14.detection_summary(
                        processor,
                        final_out,
                        inputs["input_ids"],
                        text_labels,
                        image.size,
                        profile["box_threshold"],
                        profile["text_threshold"],
                    )
                    comp = match_detections(ref_det, final_det)
                    # Preserve full detections only when there is a visible
                    # set disagreement; keeps the main result compact.
                    if not comp["all_matched_iou_ge_0_99"]:
                        comp["original_detections"] = ref_det
                        comp["final_detections"] = final_det
                    post[profile["id"]] = comp

                cases.append(
                    {
                        "image_id": image_spec["id"],
                        "image_path": str(image_path),
                        "sentinel": bool(image_spec.get("sentinel", False)),
                        "prompt_id": prompt_id,
                        "labels": labels,
                        "raw": raw,
                        "topk": topk,
                        "postprocess": post,
                    }
                )

        summary = aggregate(
            cases,
            final_model.config.num_queries,
        )

        result = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "experiment": "0051",
            "purpose": (
                "Multi-image, multi-prompt behavioral-faithfulness sweep "
                "comparing untouched PyTorch/MPS Grounding DINO against the "
                "adopted MetalGround runtime at fixed webcam-style geometry."
            ),
            "model": args.model,
            "device": "mps",
            "dtype": "float32",
            "source_manifest": str(args.manifest),
            "prompt_manifest": str(args.prompt_manifest),
            "configuration": {
                "images": len(images),
                "prompt_sets": len(prompts),
                "cases": len(cases),
                "fixed_pixel_values_shape": list(base_pixel_shape),
                "threshold_profiles": THRESHOLD_PROFILES,
                "original": "untouched transformers/PyTorch MPS FP32",
                "final": (
                    "adopted MetalGround: corrected wide encoder MLX islands "
                    "+ exact prompt cache + decoder custom Metal MSDA + exact "
                    "compiled MLX stage0 block0/block1 Swin MLPs"
                ),
                "performance_timing": False,
                "approximation": False,
                "retraining": False,
                "reduced_precision": False,
            },
            "summary": summary,
            "cases": cases,
            "interpretation_guardrails": [
                "Top-k membership preservation measures proposal-set stability, not dataset-level detection accuracy.",
                "Raw logits/boxes may diverge after near-tie ordered proposal swaps even when proposal membership is unchanged.",
                "Postprocessed agreement is therefore reported separately at three fixed threshold profiles.",
                "All images are letterboxed to one fixed geometry because the current runtime is intentionally shape-specialized for the webcam use case."
            ],
        }

        out = args.output
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            json.dumps(result, indent=2, ensure_ascii=False) + "\n"
        )

        print("\n=== Experiment 0051 summary ===", flush=True)
        print(
            f"cases: {summary['total_cases']}",
            flush=True,
        )
        print(
            "top-k membership preserved: "
            f"{summary['topk']['membership_preserved_cases']}/"
            f"{summary['total_cases']}",
            flush=True,
        )
        print(
            "rankwise exact: "
            f"{summary['topk']['rankwise_exact_cases']}/"
            f"{summary['total_cases']}",
            flush=True,
        )
        print(
            "adjacent-swap-only nonexact cases: "
            f"{summary['topk']['adjacent_swap_only_cases']}",
            flush=True,
        )
        for pid, s in summary["threshold_profiles"].items():
            print(
                f"{pid}: label-multiset="
                f"{s['label_multiset_equal_cases']}/"
                f"{summary['total_cases']} "
                f"high-IoU="
                f"{s['all_matched_iou_ge_0_99_cases']}/"
                f"{summary['total_cases']}",
                flush=True,
            )
        print(f"Saved: {out}", flush=True)

    finally:
        for d in stage0_dispatchers:
            try:
                d.restore()
            except Exception:
                pass
        cache.restore()
        for d in rt["layer_dispatchers"]:
            d.restore()
        for fd in rt["fusion_dispatchers"]:
            fd.restore()


if __name__ == "__main__":
    main()
