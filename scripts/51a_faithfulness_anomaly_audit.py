#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoModelForZeroShotObjectDetection


def load_module(path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load helper: {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod


def center_to_xyxy(boxes: torch.Tensor, image_size_wh: tuple[int, int]):
    w, h = image_size_wh
    cx, cy, bw, bh = boxes.unbind(-1)
    out = torch.stack(
        [
            cx - 0.5 * bw,
            cy - 0.5 * bh,
            cx + 0.5 * bw,
            cy + 0.5 * bh,
        ],
        dim=-1,
    )
    scale = torch.tensor(
        [w, h, w, h],
        device=out.device,
        dtype=out.dtype,
    )
    return out * scale


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


def contiguous_groups(indices):
    if not indices:
        return []
    indices = sorted(indices)
    groups = [[indices[0]]]
    for x in indices[1:]:
        if x == groups[-1][-1] + 1:
            groups[-1].append(x)
        else:
            groups.append([x])
    return groups


def topk_full_detail(final_out, original_out, k, expected_mismatches):
    rs = original_out.enc_outputs_class.detach().float().max(-1).values[0]
    fs = final_out.enc_outputs_class.detach().float().max(-1).values[0]
    rv, ri = torch.topk(rs, k)
    fv, fi = torch.topk(fs, k)

    r_idx = ri.cpu().tolist()
    f_idx = fi.cpu().tolist()
    r_val = rv.cpu().tolist()
    f_val = fv.cpu().tolist()

    mismatches = [
        rank
        for rank, (a, b) in enumerate(zip(r_idx, f_idx))
        if a != b
    ]

    groups = []
    for ranks in contiguous_groups(mismatches):
        lo = max(0, ranks[0] - 2)
        hi = min(k, ranks[-1] + 3)
        window = []
        for rank in range(lo, hi):
            oq = r_idx[rank]
            fq = f_idx[rank]
            window.append(
                {
                    "rank_zero_based": rank,
                    "original_query_index": oq,
                    "final_query_index": fq,
                    "same_query_at_rank": oq == fq,
                    "original_rank_score": r_val[rank],
                    "final_rank_score": f_val[rank],
                    "original_score_of_final_query": float(rs[fq].item()),
                    "final_score_of_original_query": float(fs[oq].item()),
                }
            )

        original_order = [r_idx[r] for r in ranks]
        final_order = [f_idx[r] for r in ranks]
        same_members = set(original_order) == set(final_order)

        groups.append(
            {
                "mismatch_ranks_zero_based": ranks,
                "same_query_members_within_group": same_members,
                "original_query_order": original_order,
                "final_query_order": final_order,
                "original_scores_in_original_order": [
                    float(rs[q].item()) for q in original_order
                ],
                "final_scores_in_original_order": [
                    float(fs[q].item()) for q in original_order
                ],
                "original_scores_in_final_order": [
                    float(rs[q].item()) for q in final_order
                ],
                "final_scores_in_final_order": [
                    float(fs[q].item()) for q in final_order
                ],
                "context_window": window,
            }
        )

    return {
        "k": k,
        "set_overlap": len(set(r_idx) & set(f_idx)),
        "rankwise_identical": k - len(mismatches),
        "mismatched_ranks_zero_based": mismatches,
        "expected_mismatched_ranks_from_0051": expected_mismatches,
        "exact_mismatch_pattern_reproduced": mismatches == expected_mismatches,
        "groups": groups,
        "score_vector_max_abs": float((fs - rs).abs().max().item()),
        "score_vector_mean_abs": float((fs - rs).abs().mean().item()),
    }


def phrase_from_mask(processor, input_ids_1d, mask_1d):
    mask = mask_1d.clone()
    if mask.numel():
        mask[0] = False
        mask[-1] = False
    ids = input_ids_1d[mask].detach().cpu().tolist()
    return processor.batch_decode([ids])[0]


def query_token_detail(
    processor,
    input_ids_1d,
    original_probs,
    final_probs,
    query_index,
    text_threshold,
):
    seq_len = int(input_ids_1d.numel())
    op = original_probs[query_index, :seq_len].detach().float()
    fp = final_probs[query_index, :seq_len].detach().float()
    token_ids = input_ids_1d.detach().cpu().tolist()
    tokens = processor.tokenizer.convert_ids_to_tokens(token_ids)

    om = op > text_threshold
    fm = fp > text_threshold

    rows = []
    threshold_crossings = []
    for pos, (tid, tok) in enumerate(zip(token_ids, tokens)):
        row = {
            "position": pos,
            "token_id": int(tid),
            "token": tok,
            "original_prob": float(op[pos].item()),
            "final_prob": float(fp[pos].item()),
            "abs_diff": float(abs(op[pos] - fp[pos]).item()),
            "original_above_text_threshold": bool(om[pos].item()),
            "final_above_text_threshold": bool(fm[pos].item()),
        }
        rows.append(row)
        if row["original_above_text_threshold"] != row["final_above_text_threshold"]:
            threshold_crossings.append(row)

    return {
        "query_index": int(query_index),
        "text_threshold": text_threshold,
        "original_phrase": phrase_from_mask(
            processor, input_ids_1d, om
        ),
        "final_phrase": phrase_from_mask(
            processor, input_ids_1d, fm
        ),
        "threshold_crossings": threshold_crossings,
        "tokens": rows,
    }


def postprocess_query_records(
    processor,
    outputs,
    input_ids_1d,
    image_size_wh,
    box_threshold,
    text_threshold,
):
    probs = torch.sigmoid(outputs.logits.detach().float())[0]
    scores = probs.max(-1).values
    boxes = center_to_xyxy(
        outputs.pred_boxes.detach().float()[0],
        image_size_wh,
    )

    keep = (scores > box_threshold).nonzero(as_tuple=True)[0]
    records = []
    for q in keep.detach().cpu().tolist():
        mask = probs[q, : input_ids_1d.numel()] > text_threshold
        records.append(
            {
                "query_index": int(q),
                "score": float(scores[q].item()),
                "phrase": phrase_from_mask(
                    processor,
                    input_ids_1d,
                    mask,
                ),
                "box": boxes[q].detach().cpu().tolist(),
            }
        )
    return records, probs


def greedy_box_pairs(original_records, final_records):
    candidates = []
    for i, r in enumerate(original_records):
        for j, f in enumerate(final_records):
            candidates.append((box_iou(r["box"], f["box"]), i, j))
    candidates.sort(reverse=True)

    used_r = set()
    used_f = set()
    pairs = []
    for iou, i, j in candidates:
        if i in used_r or j in used_f:
            continue
        used_r.add(i)
        used_f.add(j)
        pairs.append((iou, i, j))
    return pairs


def label_threshold_audit(
    processor,
    original_out,
    final_out,
    input_ids,
    image_size_wh,
    box_threshold,
    text_threshold,
):
    input_ids_1d = input_ids[0]
    original_records, original_probs = postprocess_query_records(
        processor,
        original_out,
        input_ids_1d,
        image_size_wh,
        box_threshold,
        text_threshold,
    )
    final_records, final_probs = postprocess_query_records(
        processor,
        final_out,
        input_ids_1d,
        image_size_wh,
        box_threshold,
        text_threshold,
    )

    pairs = greedy_box_pairs(original_records, final_records)
    differing = []
    paired_summary = []

    for iou, i, j in pairs:
        r = original_records[i]
        f = final_records[j]
        entry = {
            "iou": iou,
            "original_query_index": r["query_index"],
            "final_query_index": f["query_index"],
            "same_query_index": r["query_index"] == f["query_index"],
            "original_phrase": r["phrase"],
            "final_phrase": f["phrase"],
            "phrase_equal": r["phrase"] == f["phrase"],
            "original_score": r["score"],
            "final_score": f["score"],
            "original_box": r["box"],
            "final_box": f["box"],
        }
        paired_summary.append(entry)

        if r["phrase"] != f["phrase"]:
            oq = r["query_index"]
            fq = f["query_index"]
            detail = {
                **entry,
                "original_query_token_view": query_token_detail(
                    processor,
                    input_ids_1d,
                    original_probs,
                    final_probs,
                    oq,
                    text_threshold,
                ),
            }
            if fq != oq:
                detail["final_query_token_view"] = query_token_detail(
                    processor,
                    input_ids_1d,
                    original_probs,
                    final_probs,
                    fq,
                    text_threshold,
                )
            differing.append(detail)

    return {
        "box_threshold": box_threshold,
        "text_threshold": text_threshold,
        "original_kept_query_indices": [
            x["query_index"] for x in original_records
        ],
        "final_kept_query_indices": [
            x["query_index"] for x in final_records
        ],
        "same_kept_query_set": (
            set(x["query_index"] for x in original_records)
            == set(x["query_index"] for x in final_records)
        ),
        "paired_by_box": paired_summary,
        "phrase_differences": differing,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--full-result",
        type=Path,
        default=Path("results/metalground_faithfulness_sweep_0051.json"),
    )
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
    ap.add_argument(
        "--output",
        type=Path,
        default=Path("results/metalground_faithfulness_anomaly_audit_0051a.json"),
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

    required = [
        args.full_result,
        args.manifest,
        args.prompt_manifest,
    ]
    required += [getattr(args, f"exp{n}_helper") for n in helpers]
    for p in required:
        if not p.exists():
            raise SystemExit(f"Missing required file: {p}")

    if not torch.backends.mps.is_available():
        raise SystemExit("MPS unavailable.")

    previous = json.loads(args.full_result.read_text())
    manifest = json.loads(args.manifest.read_text())
    prompt_manifest = json.loads(args.prompt_manifest.read_text())

    image_by_id = {x["id"]: x for x in manifest["images"]}
    prompt_by_id = {
        x["id"]: x for x in prompt_manifest["prompt_sets"]
    }

    targets = []
    for case in previous["cases"]:
        reasons = []
        if (
            case["topk"]["mismatch_count"] > 0
            and not case["topk"]["all_rank_mismatches_are_adjacent_swaps"]
        ):
            reasons.append("non_adjacent_rank_permutation")

        bad_profiles = [
            pid
            for pid, v in case["postprocess"].items()
            if not v["label_multiset_equal"]
        ]
        for pid in bad_profiles:
            reasons.append(f"label_multiset_mismatch:{pid}")

        if reasons:
            targets.append(
                {
                    "image_id": case["image_id"],
                    "prompt_id": case["prompt_id"],
                    "reasons": reasons,
                    "expected_mismatched_ranks": case["topk"][
                        "mismatched_ranks_zero_based"
                    ],
                    "bad_profiles": bad_profiles,
                }
            )

    if not targets:
        raise SystemExit("No anomaly targets found in the 0051 result.")

    print(f"0051a targets: {len(targets)}", flush=True)
    for t in targets:
        print(
            f"  {t['image_id']} + {t['prompt_id']}: "
            f"{', '.join(t['reasons'])}",
            flush=True,
        )

    first = targets[0]
    args.image = Path(image_by_id[first["image_id"]]["path"])
    args.prompt = list(prompt_by_id[first["prompt_id"]]["labels"])

    h37 = load_module(args.exp37_helper, "mg51a_h37")
    h38 = load_module(args.exp38_helper, "mg51a_h38")
    h45 = load_module(args.exp45_helper, "mg51a_h45")
    h46 = load_module(args.exp46_helper, "mg51a_h46")

    print("Building adopted MetalGround runtime...", flush=True)
    rt = h37.build_runtime(args)
    final_model = rt["model"]
    processor = rt["processor"]
    h14 = rt["h14"]

    cache = h38.ExactTextBackboneCache(final_model.model.text_backbone)
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
        stage0_candidates = [h45.MlxSwinMLP(m) for m in stage0_modules]
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
        audited = []

        # Group by prompt so the exact text cache is primed only when needed.
        ordered_prompt_ids = []
        for t in targets:
            if t["prompt_id"] not in ordered_prompt_ids:
                ordered_prompt_ids.append(t["prompt_id"])

        completed = 0
        for prompt_id in ordered_prompt_ids:
            prompt_spec = prompt_by_id[prompt_id]
            labels = list(prompt_spec["labels"])
            prompt_targets = [
                t for t in targets if t["prompt_id"] == prompt_id
            ]

            prime_image = Image.open(
                image_by_id[prompt_targets[0]["image_id"]]["path"]
            ).convert("RGB")
            prime_inputs = processor(
                images=prime_image,
                text=[labels],
                return_tensors="pt",
            )
            prime_inputs = {k: v.to("mps") for k, v in prime_inputs.items()}

            if tuple(prime_inputs["pixel_values"].shape) != base_pixel_shape:
                raise RuntimeError(
                    f"Fixed-geometry invariant failed for {prompt_id}: "
                    f"{tuple(prime_inputs['pixel_values'].shape)} vs "
                    f"{base_pixel_shape}"
                )

            cache.cached_output = None
            cache.prime(final_model, prime_inputs, h14.sync)
            cache.set_enabled(True)

            for target in prompt_targets:
                completed += 1
                image_spec = image_by_id[target["image_id"]]
                image = Image.open(image_spec["path"]).convert("RGB")
                text_labels = [labels]
                inputs = processor(
                    images=image,
                    text=text_labels,
                    return_tensors="pt",
                )
                inputs = {k: v.to("mps") for k, v in inputs.items()}

                if tuple(inputs["pixel_values"].shape) != base_pixel_shape:
                    raise RuntimeError(
                        f"Fixed-geometry invariant failed at "
                        f"{image_spec['path']}"
                    )

                print(
                    f"[{completed}/{len(targets)}] "
                    f"{target['image_id']} + {prompt_id}",
                    flush=True,
                )

                with torch.inference_mode():
                    original_out = original_model(**inputs)
                h14.sync()
                with torch.inference_mode():
                    final_out = final_model(**inputs)
                h14.sync()

                topk = topk_full_detail(
                    final_out,
                    original_out,
                    final_model.config.num_queries,
                    target["expected_mismatched_ranks"],
                )

                profile_audits = {}
                for pid in target["bad_profiles"]:
                    previous_profile = next(
                        p
                        for p in previous["configuration"]["threshold_profiles"]
                        if p["id"] == pid
                    )
                    profile_audits[pid] = label_threshold_audit(
                        processor,
                        original_out,
                        final_out,
                        inputs["input_ids"],
                        image.size,
                        previous_profile["box_threshold"],
                        previous_profile["text_threshold"],
                    )

                audited.append(
                    {
                        "image_id": target["image_id"],
                        "image_path": image_spec["path"],
                        "prompt_id": prompt_id,
                        "labels": labels,
                        "reasons": target["reasons"],
                        "topk": topk,
                        "label_threshold_audits": profile_audits,
                    }
                )

        non_adjacent_targets = [
            c for c in audited
            if "non_adjacent_rank_permutation" in c["reasons"]
        ]
        label_targets = [
            c for c in audited
            if any(r.startswith("label_multiset_mismatch:") for r in c["reasons"])
        ]

        all_label_crossings = []
        for c in label_targets:
            for pid, audit in c["label_threshold_audits"].items():
                for d in audit["phrase_differences"]:
                    qv = d["original_query_token_view"]
                    for x in qv["threshold_crossings"]:
                        all_label_crossings.append(
                            {
                                "image_id": c["image_id"],
                                "prompt_id": c["prompt_id"],
                                "profile": pid,
                                "original_phrase": d["original_phrase"],
                                "final_phrase": d["final_phrase"],
                                "query_index": qv["query_index"],
                                **x,
                            }
                        )

        result = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "experiment": "0051a",
            "purpose": (
                "Targeted audit of the six Experiment 0051 anomaly cases: "
                "one non-adjacent-only top-k rank permutation and five "
                "permissive-threshold phrase-label disagreements."
            ),
            "model": args.model,
            "device": "mps",
            "dtype": "float32",
            "source_0051_result": str(args.full_result),
            "target_selection": (
                "Automatically selected from 0051: any case where "
                "all_rank_mismatches_are_adjacent_swaps is false, or any "
                "postprocess profile where label_multiset_equal is false."
            ),
            "targets": len(targets),
            "summary": {
                "non_adjacent_rank_targets": len(non_adjacent_targets),
                "label_mismatch_targets": len(label_targets),
                "all_topk_membership_preserved_on_rerun": all(
                    c["topk"]["set_overlap"] == final_model.config.num_queries
                    for c in audited
                ),
                "exact_topk_mismatch_patterns_reproduced": sum(
                    c["topk"]["exact_mismatch_pattern_reproduced"]
                    for c in audited
                ),
                "total_targets": len(audited),
                "token_threshold_crossing_events_in_label_mismatches": len(
                    all_label_crossings
                ),
            },
            "token_threshold_crossing_events": all_label_crossings,
            "cases": audited,
            "notes": [
                "This is a targeted diagnostic rerun, not a new population-level faithfulness estimate.",
                "Proposal rank windows save both proposal indices and proposal scores, allowing multi-way local permutations to be reconstructed.",
                "For phrase-label disagreements, detections are paired by box IoU without requiring label equality, then token probabilities around the text threshold are saved.",
                "No performance timing is performed."
            ],
        }

        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(result, indent=2, ensure_ascii=False) + "\n"
        )

        print("\n=== Experiment 0051a summary ===", flush=True)
        print(f"targets: {len(audited)}", flush=True)
        print(
            "top-k membership preserved on rerun: "
            f"{result['summary']['all_topk_membership_preserved_on_rerun']}",
            flush=True,
        )
        print(
            "exact prior mismatch patterns reproduced: "
            f"{result['summary']['exact_topk_mismatch_patterns_reproduced']}/"
            f"{len(audited)}",
            flush=True,
        )
        print(
            "token threshold-crossing events: "
            f"{len(all_label_crossings)}",
            flush=True,
        )
        print(f"Saved: {args.output}", flush=True)

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
