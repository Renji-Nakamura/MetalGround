#!/usr/bin/env python3
from __future__ import annotations

import argparse
import contextlib
import json
import types
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import mlx.core as mx
import numpy as np
import torch
from PIL import Image
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

from metalground.msda_metal_v0 import msda_metal_v0


def sync() -> None:
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


def make_patch(ctx: MetalContext):
    def patched(
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
            value,
            value_spatial_shapes_list,
            sampling_locations,
            attention_weights,
        )
    return patched


def topk_indices(out, k: int):
    scores = out.enc_outputs_class.detach().float().max(-1).values
    vals, idx = torch.topk(scores, k, dim=1)
    return scores, vals, idx


def mask_aware_error(a: torch.Tensor, b: torch.Tensor) -> dict[str, Any]:
    a = a.detach().float()
    b = b.detach().float()
    af = torch.isfinite(a)
    bf = torch.isfinite(b)
    both = af & bf

    result = {
        "shape": list(a.shape),
        "finite_mask_equal": bool(torch.equal(af, bf)),
        "neginf_mask_equal": bool(torch.equal(torch.isneginf(a), torch.isneginf(b))),
        "nan_mask_equal": bool(torch.equal(torch.isnan(a), torch.isnan(b))),
    }
    if bool(both.any()):
        aa, bb = a[both], b[both]
        d = (aa - bb).abs()
        result.update(
            {
                "max_abs": float(d.max().item()),
                "mean_abs": float(d.mean().item()),
                "rmse": float(torch.sqrt(torch.mean((aa - bb) ** 2)).item()),
                "allclose_1e-5": bool(torch.allclose(aa, bb, rtol=1e-5, atol=1e-5)),
                "allclose_1e-4": bool(torch.allclose(aa, bb, rtol=1e-4, atol=1e-4)),
            }
        )
    sync()
    return result


def compare_outputs(out, ref) -> dict[str, Any]:
    return {
        "init_reference_points": mask_aware_error(
            out.init_reference_points, ref.init_reference_points
        ),
        "last_hidden_state": mask_aware_error(
            out.last_hidden_state, ref.last_hidden_state
        ),
        "logits": mask_aware_error(out.logits, ref.logits),
        "pred_boxes": mask_aware_error(out.pred_boxes, ref.pred_boxes),
    }


def mismatch_details(ref_scores, metal_scores, ref_idx, metal_idx, coord_ref, coord_metal):
    r = ref_idx[0].detach().cpu().tolist()
    m = metal_idx[0].detach().cpu().tolist()
    ref_rank = {proposal: rank for rank, proposal in enumerate(r)}
    metal_rank = {proposal: rank for rank, proposal in enumerate(m)}

    mismatched_ranks = [i for i, (a, b) in enumerate(zip(r, m)) if a != b]
    details = []
    for rank in mismatched_ranks:
        rp = r[rank]
        mp = m[rank]
        details.append(
            {
                "rank_zero_based": rank,
                "rank_one_based": rank + 1,
                "reference_proposal_index": rp,
                "metal_proposal_index": mp,
                "reference_proposal_rank_in_metal": metal_rank[rp],
                "metal_proposal_rank_in_reference": ref_rank[mp],
                "reference_score_of_reference_proposal": float(ref_scores[0, rp].item()),
                "metal_score_of_reference_proposal": float(metal_scores[0, rp].item()),
                "reference_score_of_metal_proposal": float(ref_scores[0, mp].item()),
                "metal_score_of_metal_proposal": float(metal_scores[0, mp].item()),
                "reference_coord_of_reference_proposal": coord_ref[0, rp].detach().float().cpu().tolist(),
                "metal_coord_of_reference_proposal": coord_metal[0, rp].detach().float().cpu().tolist(),
                "reference_coord_of_metal_proposal": coord_ref[0, mp].detach().float().cpu().tolist(),
                "metal_coord_of_metal_proposal": coord_metal[0, mp].detach().float().cpu().tolist(),
            }
        )

    # Proposal-identity-aligned coordinate comparison across the common 900-set.
    ref_coords_by_id = torch.gather(
        coord_ref, 1, ref_idx.unsqueeze(-1).expand(-1, -1, 4)
    ).sigmoid()
    metal_coords_same_ids = torch.gather(
        coord_metal, 1, ref_idx.unsqueeze(-1).expand(-1, -1, 4)
    ).sigmoid()
    identity_err = mask_aware_error(metal_coords_same_ids, ref_coords_by_id)

    return {
        "mismatched_rank_count": len(mismatched_ranks),
        "mismatched_ranks_zero_based": mismatched_ranks,
        "details": details,
        "proposal_identity_aligned_reference_points": identity_err,
    }


@contextlib.contextmanager
def force_reference_topk(reference_indices: torch.Tensor, sequence_length: int, k: int):
    original = torch.topk
    hits = {"count": 0}

    def wrapped(input, k_arg, dim=None, largest=True, sorted=True, *, out=None):
        normalized_dim = (input.ndim - 1) if dim is None else dim
        if (
            out is None
            and input.ndim == 2
            and input.shape[0] == reference_indices.shape[0]
            and input.shape[1] == sequence_length
            and int(k_arg) == k
            and normalized_dim == 1
            and largest
            and sorted
        ):
            idx = reference_indices.to(input.device)
            vals = torch.gather(input, 1, idx)
            hits["count"] += 1
            return torch.return_types.topk((vals, idx))
        return original(input, k_arg, dim=dim, largest=largest, sorted=sorted, out=out)

    torch.topk = wrapped
    try:
        yield hits
    finally:
        torch.topk = original


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
        raise RuntimeError(f"Expected 6 encoder cores, got {len(encoder)}")

    print(f"embedding_init_target={getattr(model.config, 'embedding_init_target', None)}", flush=True)

    print("Reference forward...", flush=True)
    with torch.inference_mode():
        ref = model(**inputs)
        sync()

    k = model.config.num_queries
    ref_scores, _, ref_idx = topk_indices(ref, k)
    seq_len = ref_scores.shape[1]

    ctx = MetalContext(threadgroup=args.threadgroup)
    patch = make_patch(ctx)
    for name, module, _ in encoder:
        module.forward = types.MethodType(patch, module)

    print("Metal encoder forward with normal top-k...", flush=True)
    with torch.inference_mode():
        metal = model(**inputs)
        sync()

    metal_scores, _, metal_idx = topk_indices(metal, k)

    print("Metal encoder forward with REFERENCE top-k order forced...", flush=True)
    with force_reference_topk(ref_idx, seq_len, k) as hits:
        with torch.inference_mode():
            forced = model(**inputs)
            sync()
    print(f"  intercepted torch.topk calls: {hits['count']}", flush=True)
    if hits["count"] != 1:
        raise RuntimeError(
            f"Expected exactly one proposal top-k interception, got {hits['count']}"
        )

    normal_cmp = compare_outputs(metal, ref)
    forced_cmp = compare_outputs(forced, ref)

    audit = mismatch_details(
        ref_scores,
        metal_scores,
        ref_idx,
        metal_idx,
        ref.enc_outputs_coord_logits.detach().float(),
        metal.enc_outputs_coord_logits.detach().float(),
    )

    overlap = len(set(ref_idx[0].cpu().tolist()) & set(metal_idx[0].cpu().tolist()))
    rank_equal = int((ref_idx == metal_idx).sum().item())

    result = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "experiment": "0008d",
        "purpose": (
            "Causal intervention on two-stage proposal ordering: compare normal "
            "Metal encoder inference against Metal encoder inference with the "
            "reference top-k proposal order forcibly restored."
        ),
        "model": args.model,
        "embedding_init_target": getattr(model.config, "embedding_init_target", None),
        "topk": {
            "k": k,
            "set_overlap": overlap,
            "rankwise_identical": rank_equal,
            "normal_rank_audit": audit,
        },
        "normal_metal_vs_reference": normal_cmp,
        "forced_reference_order_metal_vs_reference": forced_cmp,
        "topk_interception_count": hits["count"],
        "notes": [
            "Only encoder MSDA cores are replaced by MetalGround v0; decoder MSDA remains the PyTorch/MPS reference.",
            "The forced-order run uses the same Metal encoder outputs and coordinate logits; only the proposal top-k order is replaced by the reference order.",
            "If forced order collapses final-output error, rank instability is causally responsible for the large rank-wise divergence.",
            "Proposal-identity-aligned coordinate error separates real coordinate drift from simple rank permutation.",
        ],
    }

    out_path = Path("results/metalground_v0_rank_intervention.json")
    out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")

    print("\n=== Summary ===", flush=True)
    print(f"top-k overlap: {overlap}/{k}", flush=True)
    print(f"rankwise identical: {rank_equal}/{k}", flush=True)
    print(f"mismatched ranks: {audit['mismatched_rank_count']}", flush=True)
    print(
        "identity-aligned init-ref max abs: "
        f"{audit['proposal_identity_aligned_reference_points'].get('max_abs')}",
        flush=True,
    )
    print(
        "normal final box max abs: "
        f"{normal_cmp['pred_boxes'].get('max_abs')}",
        flush=True,
    )
    print(
        "forced-order final box max abs: "
        f"{forced_cmp['pred_boxes'].get('max_abs')}",
        flush=True,
    )
    print(
        "normal finite-logit max abs: "
        f"{normal_cmp['logits'].get('max_abs')}",
        flush=True,
    )
    print(
        "forced-order finite-logit max abs: "
        f"{forced_cmp['logits'].get('max_abs')}",
        flush=True,
    )
    print(f"Saved: {out_path}", flush=True)


if __name__ == "__main__":
    main()
