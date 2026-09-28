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


def sync():
    torch.mps.synchronize()


def compare_finite(a: torch.Tensor | None, b: torch.Tensor | None) -> dict[str, Any] | None:
    if a is None or b is None:
        return None

    a = a.detach().float()
    b = b.detach().float()

    afin = torch.isfinite(a)
    bfin = torch.isfinite(b)
    both = afin & bfin

    out = {
        "shape": list(a.shape),
        "finite_mask_equal": bool(torch.equal(afin, bfin)),
        "nan_mask_equal": bool(torch.equal(torch.isnan(a), torch.isnan(b))),
        "posinf_mask_equal": bool(torch.equal(torch.isposinf(a), torch.isposinf(b))),
        "neginf_mask_equal": bool(torch.equal(torch.isneginf(a), torch.isneginf(b))),
    }

    if bool(both.any()):
        da = a[both]
        db = b[both]
        d = (da - db).abs()
        out.update(
            {
                "max_abs": float(d.max().item()),
                "mean_abs": float(d.mean().item()),
                "rmse": float(torch.sqrt(torch.mean((da - db) ** 2)).item()),
                "allclose_1e-5": bool(torch.allclose(da, db, rtol=1e-5, atol=1e-5)),
                "allclose_1e-4": bool(torch.allclose(da, db, rtol=1e-4, atol=1e-4)),
            }
        )
    sync()
    return out


def per_decoder_layer_compare(a: torch.Tensor | None, b: torch.Tensor | None):
    if a is None or b is None:
        return None
    if a.ndim < 2 or a.shape[1] != 6:
        return {"whole": compare_finite(a, b)}
    return {
        f"layer_{i}": compare_finite(a[:, i], b[:, i])
        for i in range(a.shape[1])
    }


@dataclass
class MetalContext:
    threadgroup: int = 256
    metadata_cache: dict[
        tuple[tuple[int, int], ...], tuple[mx.array, mx.array]
    ] = field(default_factory=dict)

    def metadata(self, shapes):
        key = tuple((int(h), int(w)) for h, w in shapes)
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


def snapshot(out):
    fields = [
        "encoder_last_hidden_state_vision",
        "encoder_last_hidden_state_text",
        "enc_outputs_class",
        "enc_outputs_coord_logits",
        "encoder_logits",
        "encoder_pred_boxes",
        "init_reference_points",
        "intermediate_hidden_states",
        "intermediate_reference_points",
        "last_hidden_state",
        "logits",
        "pred_boxes",
    ]
    snap = {}
    for f in fields:
        x = getattr(out, f, None)
        snap[f] = None if x is None else x.detach().clone()
    sync()
    return snap


def compare_snap(actual, ref):
    result = {}
    simple = [
        "encoder_last_hidden_state_vision",
        "encoder_last_hidden_state_text",
        "enc_outputs_class",
        "enc_outputs_coord_logits",
        "encoder_logits",
        "encoder_pred_boxes",
        "init_reference_points",
        "last_hidden_state",
        "logits",
        "pred_boxes",
    ]
    for f in simple:
        result[f] = compare_finite(actual.get(f), ref.get(f))

    result["intermediate_hidden_states"] = per_decoder_layer_compare(
        actual.get("intermediate_hidden_states"),
        ref.get("intermediate_hidden_states"),
    )
    result["intermediate_reference_points"] = per_decoder_layer_compare(
        actual.get("intermediate_reference_points"),
        ref.get("intermediate_reference_points"),
    )
    return result


def compact_print(name, cmp):
    print(f"\n=== {name} ===", flush=True)
    for field in [
        "encoder_last_hidden_state_vision",
        "enc_outputs_class",
        "init_reference_points",
        "last_hidden_state",
        "logits",
        "pred_boxes",
    ]:
        x = cmp.get(field)
        if not x:
            continue
        print(
            f"{field:34s} max_abs={x.get('max_abs')} "
            f"mean_abs={x.get('mean_abs')} "
            f"allclose1e-5={x.get('allclose_1e-5')} "
            f"finite_mask_equal={x.get('finite_mask_equal')}",
            flush=True,
        )

    ir = cmp.get("intermediate_reference_points")
    if isinstance(ir, dict):
        for layer, x in ir.items():
            if x:
                print(
                    f"  refpts {layer}: max_abs={x.get('max_abs')} "
                    f"mean_abs={x.get('mean_abs')}",
                    flush=True,
                )


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
    if len(cores) != 12:
        raise RuntimeError(f"Expected 12 cores, got {len(cores)}")

    encoder_names = [x[0] for x in cores if ".encoder.layers." in x[0]]
    decoder_names = [x[0] for x in cores if ".decoder.layers." in x[0]]

    print("Reference forward...", flush=True)
    with torch.inference_mode():
        ref_out = model(**inputs)
        sync()
    ref = snapshot(ref_out)

    ctx = MetalContext(threadgroup=args.threadgroup)
    patch_fn = make_patch(ctx)

    def restore():
        for _, module, original in cores:
            module.forward = original

    def patch_selected(names: set[str]):
        restore()
        for name, module, _ in cores:
            if name in names:
                module.forward = types.MethodType(patch_fn, module)

    variants: list[tuple[str, set[str]]] = []

    # Single broad cuts.
    variants.append(("decoder_only_all6", set(decoder_names)))
    variants.append(("encoder_only_all6", set(encoder_names)))
    variants.append(("all12", set(encoder_names + decoder_names)))

    # Encoder cumulative sweep: locates where perturbation grows.
    for n in range(1, 7):
        variants.append(
            (f"encoder_prefix_{n}", set(encoder_names[:n]))
        )

    # Encoder single-layer sweep: detects an anomalously sensitive layer.
    for i, name in enumerate(encoder_names):
        variants.append((f"encoder_only_layer_{i}", {name}))

    results = {}

    for label, selected in variants:
        print(f"\nRunning {label} ({len(selected)} patched cores)...", flush=True)
        patch_selected(selected)
        with torch.inference_mode():
            out = model(**inputs)
            sync()
        cmp = compare_snap(snapshot(out), ref)
        results[label] = {
            "patched_modules": sorted(selected),
            "comparison": cmp,
        }
        compact_print(label, cmp)

    restore()

    record = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "experiment": "0008b",
        "purpose": (
            "Localize numerical amplification by patching decoder-only, "
            "encoder-only, cumulative encoder prefixes, and individual encoder layers."
        ),
        "encoder_modules": encoder_names,
        "decoder_modules": decoder_names,
        "variants": results,
        "interpretation_hint": (
            "A sharp jump between encoder output and init_reference_points suggests "
            "discontinuous two-stage query/proposal selection. Decoder-only staying "
            "small would further isolate the issue to encoder-side perturbation."
        ),
    }

    out_path = Path("results/metalground_v0_amplification_sweep.json")
    out_path.write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n")
    print(f"\nSaved: {out_path}", flush=True)


if __name__ == "__main__":
    main()
