#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
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


def finite_tensor_error(a: torch.Tensor, b: torch.Tensor) -> dict[str, Any]:
    """Compare tensors while explicitly accounting for +/-inf and NaN masks."""
    a = a.detach().float()
    b = b.detach().float()

    a_nan = torch.isnan(a)
    b_nan = torch.isnan(b)
    a_pos = torch.isposinf(a)
    b_pos = torch.isposinf(b)
    a_neg = torch.isneginf(a)
    b_neg = torch.isneginf(b)
    a_fin = torch.isfinite(a)
    b_fin = torch.isfinite(b)

    finite_mask = a_fin & b_fin
    finite_count = int(finite_mask.sum().item())
    total = a.numel()

    if finite_count:
        d = (a[finite_mask] - b[finite_mask]).abs()
        max_abs = float(d.max().item())
        mean_abs = float(d.mean().item())
        rmse = float(
            torch.sqrt(torch.mean((a[finite_mask] - b[finite_mask]) ** 2)).item()
        )
        allclose_1e5 = bool(
            torch.allclose(
                a[finite_mask], b[finite_mask], rtol=1e-5, atol=1e-5
            )
        )
        allclose_1e4 = bool(
            torch.allclose(
                a[finite_mask], b[finite_mask], rtol=1e-4, atol=1e-4
            )
        )
    else:
        max_abs = mean_abs = rmse = float("nan")
        allclose_1e5 = allclose_1e4 = True

    result = {
        "shape": list(a.shape),
        "numel": total,
        "finite_count_actual": int(a_fin.sum().item()),
        "finite_count_reference": int(b_fin.sum().item()),
        "finite_mask_equal": bool(torch.equal(a_fin, b_fin)),
        "nan_count_actual": int(a_nan.sum().item()),
        "nan_count_reference": int(b_nan.sum().item()),
        "nan_mask_equal": bool(torch.equal(a_nan, b_nan)),
        "posinf_count_actual": int(a_pos.sum().item()),
        "posinf_count_reference": int(b_pos.sum().item()),
        "posinf_mask_equal": bool(torch.equal(a_pos, b_pos)),
        "neginf_count_actual": int(a_neg.sum().item()),
        "neginf_count_reference": int(b_neg.sum().item()),
        "neginf_mask_equal": bool(torch.equal(a_neg, b_neg)),
        "finite_max_abs": max_abs,
        "finite_mean_abs": mean_abs,
        "finite_rmse": rmse,
        "finite_allclose_1e-5": allclose_1e5,
        "finite_allclose_1e-4": allclose_1e4,
    }
    sync()
    return result


def box_error(a: torch.Tensor, b: torch.Tensor) -> dict[str, Any]:
    a = a.detach().float()
    b = b.detach().float()
    d = (a - b).abs().reshape(-1)

    qs = torch.quantile(
        d,
        torch.tensor([0.5, 0.9, 0.95, 0.99, 0.999], device=d.device),
    )
    result = {
        "shape": list(a.shape),
        "max_abs": float(d.max().item()),
        "mean_abs": float(d.mean().item()),
        "rmse": float(torch.sqrt(torch.mean(d**2)).item()),
        "q50_abs": float(qs[0].item()),
        "q90_abs": float(qs[1].item()),
        "q95_abs": float(qs[2].item()),
        "q99_abs": float(qs[3].item()),
        "q999_abs": float(qs[4].item()),
        "count_gt_1e-4": int((d > 1e-4).sum().item()),
        "count_gt_1e-3": int((d > 1e-3).sum().item()),
        "count_gt_1e-2": int((d > 1e-2).sum().item()),
        "count_gt_1e-1": int((d > 1e-1).sum().item()),
        "allclose_1e-5": bool(torch.allclose(a, b, rtol=1e-5, atol=1e-5)),
        "allclose_1e-4": bool(torch.allclose(a, b, rtol=1e-4, atol=1e-4)),
        "finite": bool(torch.isfinite(a).all().item()),
    }
    sync()
    return result


@dataclass
class MetalContext:
    threadgroup: int = 256
    metadata_cache: dict[
        tuple[tuple[int, int], ...], tuple[mx.array, mx.array]
    ] = field(default_factory=dict)

    def metadata(self, shapes_list):
        key = tuple((int(h), int(w)) for h, w in shapes_list)
        if key not in self.metadata_cache:
            starts = []
            acc = 0
            for h, w in key:
                starts.append(acc)
                acc += h * w

            sm = mx.array(np.asarray(key, dtype=np.int32), dtype=mx.int32)
            st = mx.array(np.asarray(starts, dtype=np.int32), dtype=mx.int32)
            mx.eval(sm, st)
            mx.synchronize()
            self.metadata_cache[key] = (sm, st)
        return self.metadata_cache[key]

    def run(self, value, shapes_list, locations, weights):
        if not value.is_contiguous():
            value = value.contiguous()
        if not locations.is_contiguous():
            locations = locations.contiguous()
        if not weights.is_contiguous():
            weights = weights.contiguous()

        sync()
        value_mx = mx.asarray(value, copy=False)
        loc_mx = mx.asarray(locations, copy=False)
        weights_mx = mx.asarray(weights, copy=False)
        sm, st = self.metadata(shapes_list)

        out_mx = msda_metal_v0(
            value_mx,
            sm,
            st,
            loc_mx,
            weights_mx,
            threadgroup_size=self.threadgroup,
        )
        mx.eval(out_mx)
        mx.synchronize()
        out = torch.as_tensor(out_mx)
        sync()
        return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="IDEA-Research/grounding-dino-tiny")
    ap.add_argument("--image", type=Path, default=Path("assets/input.jpg"))
    ap.add_argument("--prompt", nargs="+", default=["a cat", "a dog"])
    ap.add_argument("--threadgroup", type=int, default=256)
    args = ap.parse_args()

    if not torch.backends.mps.is_available():
        raise SystemExit("MPS unavailable.")

    mx.set_default_device(mx.gpu)
    device = torch.device("mps")

    print("Loading model...", flush=True)
    processor = AutoProcessor.from_pretrained(args.model)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(args.model)
    model.eval().to(device)
    sync()

    image = Image.open(args.image).convert("RGB")
    text_labels = [args.prompt]
    cpu_inputs = processor(images=image, text=text_labels, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in cpu_inputs.items()}
    sync()

    ctx = MetalContext(threadgroup=args.threadgroup)

    cores: list[tuple[str, torch.nn.Module, Any]] = []
    for name, module in model.named_modules():
        if module.__class__.__name__ == "MultiScaleDeformableAttention":
            cores.append((name, module, module.forward))

    if len(cores) != 12:
        raise RuntimeError(f"Expected 12 MSDA cores, got {len(cores)}")

    per_core: dict[str, Any] = {}

    print("\nPhase A: audit Metal output against original core on same inputs.", flush=True)
    for name, module, original_forward in cores:
        def make_audit(n, orig):
            def audited_forward(
                self,
                value,
                value_spatial_shapes,
                value_spatial_shapes_list,
                level_start_index,
                sampling_locations,
                attention_weights,
                im2col_step,
            ):
                ref = orig(
                    value,
                    value_spatial_shapes,
                    value_spatial_shapes_list,
                    level_start_index,
                    sampling_locations,
                    attention_weights,
                    im2col_step,
                )
                metal = ctx.run(
                    value,
                    value_spatial_shapes_list,
                    sampling_locations,
                    attention_weights,
                )
                err = finite_tensor_error(metal, ref)
                per_core[n] = err
                print(
                    f"  {n}: max_abs={err['finite_max_abs']:.3e} "
                    f"mean_abs={err['finite_mean_abs']:.3e} "
                    f"allclose1e-5={err['finite_allclose_1e-5']}",
                    flush=True,
                )
                return ref
            return audited_forward

        module.forward = types.MethodType(make_audit(name, original_forward), module)

    with torch.inference_mode():
        reference = model(**inputs)
        sync()

    # Restore original forwards.
    for _, module, original_forward in cores:
        module.forward = original_forward

    ref_logits = reference.logits.detach().clone()
    ref_boxes = reference.pred_boxes.detach().clone()
    sync()

    print("\nPhase B: patch all 12 cores and run one full forward.", flush=True)

    for name, module, original_forward in cores:
        def make_patch(n):
            def patched_forward(
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
            return patched_forward
        module.forward = types.MethodType(make_patch(name), module)

    with torch.inference_mode():
        patched = model(**inputs)
        sync()

    logits_cmp = finite_tensor_error(patched.logits, ref_logits)
    boxes_cmp = box_error(patched.pred_boxes, ref_boxes)

    print("\nFinal logits mask-aware comparison:", flush=True)
    print(json.dumps(logits_cmp, indent=2), flush=True)
    print("\nFinal box comparison:", flush=True)
    print(json.dumps(boxes_cmp, indent=2), flush=True)

    result = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "experiment": "0008a",
        "purpose": (
            "Correctness audit before timing: compare Metal v0 with original "
            "MSDA on each of the 12 real invocations, then compare full-model "
            "outputs using mask-aware logits handling."
        ),
        "per_core": per_core,
        "final_logits_mask_aware": logits_cmp,
        "final_pred_boxes": boxes_cmp,
        "notes": [
            "Phase A returns the original MSDA output downstream, so each Metal core is compared on exactly the same input as the reference core.",
            "Final logits are compared only on jointly finite elements while +/-inf/NaN masks are compared separately.",
            "No performance claim is made by this diagnostic experiment.",
        ],
    }

    out = Path("results/metalground_v0_correctness_audit.json")
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(f"\nSaved: {out}", flush=True)


if __name__ == "__main__":
    main()
