#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import types
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import mlx.core as mx
import numpy as np
import torch
from PIL import Image
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

from metalground.msda_metal_v0 import msda_metal_v0


def sync():
    torch.mps.synchronize()


@dataclass
class MetalContext:
    threadgroup: int = 256
    metadata_cache: dict = field(default_factory=dict)

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

    def run(self, value, shapes, locations, weights):
        if not value.is_contiguous():
            value = value.contiguous()
        if not locations.is_contiguous():
            locations = locations.contiguous()
        if not weights.is_contiguous():
            weights = weights.contiguous()

        sync()
        vm = mx.asarray(value, copy=False)
        lm = mx.asarray(locations, copy=False)
        wm = mx.asarray(weights, copy=False)
        sm, st = self.metadata(shapes)

        out = msda_metal_v0(
            vm, sm, st, lm, wm, threadgroup_size=self.threadgroup
        )
        mx.eval(out)
        mx.synchronize()
        result = torch.as_tensor(out)
        sync()
        return result


def patch_forward(ctx):
    def f(
        self,
        value,
        value_spatial_shapes,
        value_spatial_shapes_list,
        level_start_index,
        sampling_locations,
        attention_weights,
        im2col_step,
    ):
        return ctx.run(
            value, value_spatial_shapes_list, sampling_locations, attention_weights
        )
    return f


def selection(out, k: int):
    full_logits = out.enc_outputs_class.detach().float()
    full_coords = out.enc_outputs_coord_logits.detach().float()

    scores = full_logits.max(-1).values
    vals, idx = torch.topk(scores, k, dim=1)

    gathered = torch.gather(
        full_coords, 1, idx.unsqueeze(-1).expand(-1, -1, 4)
    ).sigmoid()

    # Verify our reconstruction matches the actual model output.
    reconstruction_max_abs = float(
        (gathered - out.init_reference_points.detach().float()).abs().max().item()
    )
    sync()

    return {
        "scores": scores,
        "topk_values": vals,
        "topk_indices": idx,
        "gathered_reference_points": gathered,
        "reconstruction_max_abs": reconstruction_max_abs,
    }


def summarize(ref, cur, k: int):
    rs = ref["scores"][0]
    cs = cur["scores"][0]
    ri = ref["topk_indices"][0]
    ci = cur["topk_indices"][0]

    rset = set(ri.cpu().tolist())
    cset = set(ci.cpu().tolist())
    overlap = rset & cset
    ref_only = sorted(rset - cset)
    cur_only = sorted(cset - rset)

    rankwise_equal = int((ri == ci).sum().item())

    # Full score perturbation.
    sd = (cs - rs).abs()
    score_max_abs = float(sd.max().item())
    score_mean_abs = float(sd.mean().item())

    # Cutoff and margin. topk values are descending.
    sorted_ref = torch.sort(rs, descending=True).values
    sorted_cur = torch.sort(cs, descending=True).values

    ref_cutoff = float(sorted_ref[k - 1].item())
    ref_next = float(sorted_ref[k].item())
    cur_cutoff = float(sorted_cur[k - 1].item())
    cur_next = float(sorted_cur[k].item())

    eps_counts = {}
    for eps in [1e-7, 5e-7, 1e-6, 2e-6, 5e-6, 1e-5, 2e-5, 5e-5, 1e-4]:
        eps_counts[f"{eps:.0e}"] = {
            "reference_candidates_within_cutoff": int(
                ((rs - ref_cutoff).abs() <= eps).sum().item()
            ),
            "patched_candidates_within_reference_cutoff": int(
                ((cs - ref_cutoff).abs() <= eps).sum().item()
            ),
        }

    def candidate_rows(indices):
        rows = []
        for i in indices[:50]:
            rows.append(
                {
                    "index": i,
                    "reference_score": float(rs[i].item()),
                    "patched_score": float(cs[i].item()),
                    "delta": float((cs[i] - rs[i]).item()),
                    "reference_distance_to_cutoff": float((rs[i] - ref_cutoff).item()),
                    "patched_distance_to_reference_cutoff": float(
                        (cs[i] - ref_cutoff).item()
                    ),
                }
            )
        return rows

    refpts_diff = (
        cur["gathered_reference_points"]
        - ref["gathered_reference_points"]
    ).abs()

    result = {
        "topk_k": k,
        "overlap_count": len(overlap),
        "changed_membership_count_each_side": k - len(overlap),
        "jaccard": len(overlap) / len(rset | cset),
        "rankwise_identical_count": rankwise_equal,
        "score_vector_max_abs": score_max_abs,
        "score_vector_mean_abs": score_mean_abs,
        "reference_cutoff_score_rank_k": ref_cutoff,
        "reference_rank_k_plus_1_score": ref_next,
        "reference_cutoff_margin": ref_cutoff - ref_next,
        "patched_cutoff_score_rank_k": cur_cutoff,
        "patched_rank_k_plus_1_score": cur_next,
        "patched_cutoff_margin": cur_cutoff - cur_next,
        "candidates_near_cutoff": eps_counts,
        "reference_only_candidates": candidate_rows(ref_only),
        "patched_only_candidates": candidate_rows(cur_only),
        "gathered_reference_points_max_abs": float(refpts_diff.max().item()),
        "gathered_reference_points_mean_abs": float(refpts_diff.mean().item()),
        "selection_reconstruction_max_abs_reference": ref[
            "reconstruction_max_abs"
        ],
        "selection_reconstruction_max_abs_patched": cur[
            "reconstruction_max_abs"
        ],
    }
    sync()
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="IDEA-Research/grounding-dino-tiny")
    ap.add_argument("--image", type=Path, default=Path("assets/input.jpg"))
    ap.add_argument("--prompt", nargs="+", default=["a cat", "a dog"])
    ap.add_argument("--threadgroup", type=int, default=256)
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

    cores = []
    for name, module in model.named_modules():
        if module.__class__.__name__ == "MultiScaleDeformableAttention":
            cores.append((name, module, module.forward))

    encoder = [x for x in cores if ".encoder.layers." in x[0]]
    if len(encoder) != 6:
        raise RuntimeError(f"Expected 6 encoder MSDA cores, got {len(encoder)}")

    def restore():
        for _, module, orig in cores:
            module.forward = orig

    ctx = MetalContext(threadgroup=args.threadgroup)
    metal = patch_forward(ctx)

    print("Reference forward...", flush=True)
    with torch.inference_mode():
        ref_out = model(**inputs)
        sync()
    ref_sel = selection(ref_out, model.config.num_queries)
    print(
        f"Reference selection reconstruction max abs: "
        f"{ref_sel['reconstruction_max_abs']:.3e}",
        flush=True,
    )

    variants = [("encoder_all6", set(n for n, _, _ in encoder))]
    for i, (name, _, _) in enumerate(encoder):
        variants.append((f"encoder_only_layer_{i}", {name}))

    results = {}

    for label, names in variants:
        restore()
        for name, module, _ in cores:
            if name in names:
                module.forward = types.MethodType(metal, module)

        print(f"\nRunning {label}...", flush=True)
        with torch.inference_mode():
            out = model(**inputs)
            sync()

        cur_sel = selection(out, model.config.num_queries)
        summary = summarize(ref_sel, cur_sel, model.config.num_queries)
        results[label] = {
            "patched_modules": sorted(names),
            "selection": summary,
        }

        print(
            f"  overlap={summary['overlap_count']}/900 "
            f"changed={summary['changed_membership_count_each_side']} "
            f"rankwise_equal={summary['rankwise_identical_count']}/900",
            flush=True,
        )
        print(
            f"  score max abs={summary['score_vector_max_abs']:.3e} "
            f"cutoff margin(ref)={summary['reference_cutoff_margin']:.3e}",
            flush=True,
        )
        print(
            f"  init-ref max abs="
            f"{summary['gathered_reference_points_max_abs']:.6f}",
            flush=True,
        )

    restore()

    record = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "experiment": "0008c",
        "purpose": (
            "Directly audit two-stage top-k proposal selection under tiny "
            "encoder perturbations from Metal MSDA v0."
        ),
        "model": args.model,
        "num_queries": model.config.num_queries,
        "reference_selection_reconstruction_max_abs": ref_sel[
            "reconstruction_max_abs"
        ],
        "variants": results,
        "notes": [
            "Selection is reconstructed from model outputs using enc_outputs_class.max(-1) followed by torch.topk(k=900), then gathering enc_outputs_coord_logits and sigmoid.",
            "The reconstructed reference points are checked against the model's actual init_reference_points.",
            "This experiment directly tests whether small encoder score perturbations change the top-k proposal membership/order."
        ],
    }

    out_path = Path("results/metalground_v0_topk_audit.json")
    out_path.write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n")
    print(f"\nSaved: {out_path}", flush=True)


if __name__ == "__main__":
    main()
