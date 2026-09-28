#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import statistics
import time
import types
from collections import defaultdict
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


def sync_hybrid() -> None:
    torch.mps.synchronize()
    mx.synchronize()


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
        "min_ms": min(xs),
        "max_ms": max(xs),
    }


@dataclass
class MetalPatchState:
    threadgroup: int
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


def make_metal_msda_forward(state: MetalPatchState):
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
        if not value.is_contiguous():
            value = value.contiguous()
        if not sampling_locations.is_contiguous():
            sampling_locations = sampling_locations.contiguous()
        if not attention_weights.is_contiguous():
            attention_weights = attention_weights.contiguous()

        torch.mps.synchronize()

        vm = mx.asarray(value, copy=False)
        lm = mx.asarray(sampling_locations, copy=False)
        wm = mx.asarray(attention_weights, copy=False)
        sm, st = state.metadata(value_spatial_shapes_list)

        out_mx = msda_metal_v0(
            vm, sm, st, lm, wm, threadgroup_size=state.threadgroup
        )
        mx.eval(out_mx)
        mx.synchronize()

        out = torch.as_tensor(out_mx)
        if out.device.type != "mps":
            raise RuntimeError(f"Unexpected output device: {out.device}")
        return out

    return forward


def patch_msda(model: torch.nn.Module, threadgroup: int) -> None:
    state = MetalPatchState(threadgroup=threadgroup)
    fwd = make_metal_msda_forward(state)
    n = 0
    for _, module in model.named_modules():
        if module.__class__.__name__ == "MultiScaleDeformableAttention":
            module.forward = types.MethodType(fwd, module)
            n += 1
    if n != 12:
        raise RuntimeError(f"Expected 12 MSDA cores, got {n}")


class HookProfiler:
    def __init__(self, named: dict[str, torch.nn.Module], names: list[str]):
        self.named = named
        self.names = names
        self.starts: dict[str, list[int]] = defaultdict(list)
        self.samples: dict[str, list[float]] = defaultdict(list)
        self.handles = []

    def _pre(self, name):
        def hook(module, inputs):
            sync_hybrid()
            self.starts[name].append(time.perf_counter_ns())
        return hook

    def _post(self, name):
        def hook(module, inputs, output):
            sync_hybrid()
            t1 = time.perf_counter_ns()
            t0 = self.starts[name].pop()
            self.samples[name].append((t1 - t0) / 1e6)
        return hook

    def __enter__(self):
        for name in self.names:
            if name not in self.named:
                raise KeyError(name)
            m = self.named[name]
            self.handles.append(m.register_forward_pre_hook(self._pre(name)))
            self.handles.append(m.register_forward_hook(self._post(name)))
        return self

    def __exit__(self, exc_type, exc, tb):
        for h in self.handles:
            h.remove()

    def report(self):
        return {
            name: {
                "class": type(self.named[name]).__name__,
                **stats(self.samples[name]),
            }
            for name in self.names
            if self.samples[name]
        }


def names_by_suffix(named, suffixes):
    out = []
    for name in named:
        if any(name.endswith(s) for s in suffixes):
            out.append(name)
    return out


def encoder_fusion_names(named):
    return [
        f"model.encoder.layers.{i}.fusion_layer"
        for i in range(6)
        if f"model.encoder.layers.{i}.fusion_layer" in named
    ]


def encoder_fusion_attn_names(named):
    return [
        f"model.encoder.layers.{i}.fusion_layer.attn"
        for i in range(6)
        if f"model.encoder.layers.{i}.fusion_layer.attn" in named
    ]


def encoder_fusion_children(named):
    suffixes = (
        ".fusion_layer.layer_norm_vision",
        ".fusion_layer.layer_norm_text",
        ".fusion_layer.attn.vision_proj",
        ".fusion_layer.attn.text_proj",
        ".fusion_layer.attn.values_vision_proj",
        ".fusion_layer.attn.values_text_proj",
        ".fusion_layer.attn.out_vision_proj",
        ".fusion_layer.attn.out_text_proj",
    )
    return [
        n for n in named
        if n.startswith("model.encoder.layers.") and n.endswith(suffixes)
    ]


def encoder_deformable_layers(named):
    return [
        f"model.encoder.layers.{i}.deformable_layer"
        for i in range(6)
        if f"model.encoder.layers.{i}.deformable_layer" in named
    ]


def encoder_deformable_wrappers(named):
    return [
        f"model.encoder.layers.{i}.deformable_layer.self_attn"
        for i in range(6)
        if f"model.encoder.layers.{i}.deformable_layer.self_attn" in named
    ]


def encoder_deformable_children(named):
    suffixes = (
        ".deformable_layer.self_attn.sampling_offsets",
        ".deformable_layer.self_attn.attention_weights",
        ".deformable_layer.self_attn.value_proj",
        ".deformable_layer.self_attn.output_proj",
        ".deformable_layer.self_attn_layer_norm",
        ".deformable_layer.fc1",
        ".deformable_layer.fc2",
        ".deformable_layer.final_layer_norm",
    )
    return [
        n for n in named
        if n.startswith("model.encoder.layers.") and n.endswith(suffixes)
    ]


def category_for(name: str) -> str:
    cats = [
        "layer_norm_vision",
        "layer_norm_text",
        "vision_proj",
        "text_proj",
        "values_vision_proj",
        "values_text_proj",
        "out_vision_proj",
        "out_text_proj",
        "sampling_offsets",
        "attention_weights",
        "value_proj",
        "output_proj",
        "self_attn_layer_norm",
        "fc1",
        "fc2",
        "final_layer_norm",
    ]
    for c in cats:
        if name.endswith("." + c):
            return c
    return "other"


def rollup(rows):
    by_cat: dict[str, list[float]] = defaultdict(list)
    for name, row in rows.items():
        by_cat[category_for(name)].append(float(row["median_ms"]))

    result = {}
    for cat, vals in by_cat.items():
        result[cat] = {
            "count": len(vals),
            "sum_median_ms": sum(vals),
            "median_per_module_ms": statistics.median(vals),
        }
    return dict(sorted(result.items(), key=lambda kv: kv[1]["sum_median_ms"], reverse=True))


def profile_group(model, named, name, modules, iters):
    print(f"\n=== {name}: {len(modules)} modules ===", flush=True)
    full = []
    with HookProfiler(named, modules) as hp:
        with torch.inference_mode():
            for i in range(iters):
                sync_hybrid()
                t0 = time.perf_counter_ns()
                _ = model(**profile_group.inputs)
                sync_hybrid()
                dt = (time.perf_counter_ns() - t0) / 1e6
                full.append(dt)
                print(f"  pass {i+1}/{iters}: {dt:.2f} ms", flush=True)

        rows = hp.report()

    ranked = sorted(rows.items(), key=lambda kv: kv[1]["median_ms"], reverse=True)
    for module_name, row in ranked[:30]:
        print(
            f"  {row['median_ms']:8.3f} ms  {row['class']:<32} {module_name}",
            flush=True,
        )

    return {
        "selected_modules": modules,
        "instrumented_full_forward": stats(full),
        "modules": rows,
        "sum_module_medians_ms": sum(r["median_ms"] for r in rows.values()),
        "rollup": rollup(rows),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="IDEA-Research/grounding-dino-tiny")
    ap.add_argument("--image", type=Path, default=Path("assets/input.jpg"))
    ap.add_argument("--prompt", nargs="+", default=["a cat", "a dog"])
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--iters", type=int, default=3)
    ap.add_argument("--threadgroup", type=int, default=256)
    ap.add_argument("--baseline-ms", type=float, default=632.6745625)
    args = ap.parse_args()

    if not torch.backends.mps.is_available():
        raise SystemExit("MPS unavailable.")

    mx.set_default_device(mx.gpu)

    print("Loading model...", flush=True)
    processor = AutoProcessor.from_pretrained(args.model)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(args.model)
    model.eval().to("mps")
    sync_hybrid()

    image = Image.open(args.image).convert("RGB")
    text_labels = [args.prompt]
    cpu_inputs = processor(images=image, text=text_labels, return_tensors="pt")
    inputs = {k: v.to("mps") for k, v in cpu_inputs.items()}
    sync_hybrid()
    profile_group.inputs = inputs

    patch_msda(model, args.threadgroup)
    named = dict(model.named_modules())

    groups = {
        "fusion_layers": encoder_fusion_names(named),
        "fusion_attn": encoder_fusion_attn_names(named),
        "fusion_children": encoder_fusion_children(named),
        "deformable_layers": encoder_deformable_layers(named),
        "deformable_wrappers": encoder_deformable_wrappers(named),
        "deformable_children": encoder_deformable_children(named),
    }

    print(f"Warmup x{args.warmup}...", flush=True)
    with torch.inference_mode():
        for i in range(args.warmup):
            _ = model(**inputs)
            sync_hybrid()
            print(f"  warmup {i+1}/{args.warmup}", flush=True)

    results = {}
    for group_name, modules in groups.items():
        results[group_name] = profile_group(
            model, named, group_name, modules, args.iters
        )

    fusion_parent = results["fusion_layers"]["sum_module_medians_ms"]
    fusion_attn = results["fusion_attn"]["sum_module_medians_ms"]
    fusion_children_sum = results["fusion_children"]["sum_module_medians_ms"]

    deform_parent = results["deformable_layers"]["sum_module_medians_ms"]
    deform_wrapper = results["deformable_wrappers"]["sum_module_medians_ms"]
    deform_children_sum = results["deformable_children"]["sum_module_medians_ms"]

    derived = {
        "fusion_parent_sum_median_ms": fusion_parent,
        "fusion_attn_sum_median_ms": fusion_attn,
        "fusion_children_sum_median_ms": fusion_children_sum,
        "fusion_parent_minus_attn_ms": fusion_parent - fusion_attn,
        "fusion_attn_minus_profiled_child_modules_ms": fusion_attn - fusion_children_sum,
        "deformable_parent_sum_median_ms": deform_parent,
        "deformable_wrapper_sum_median_ms": deform_wrapper,
        "deformable_children_sum_median_ms": deform_children_sum,
        "deformable_parent_minus_wrapper_ms": deform_parent - deform_wrapper,
        "baseline_median_ms": args.baseline_ms,
    }

    record = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "experiment": "0011",
        "purpose": (
            "Decompose the post-MSDA encoder bottlenecks: fusion internals and "
            "remaining deformable-layer projections/FFN."
        ),
        "model": args.model,
        "device": "mps",
        "dtype": "float32",
        "image": str(args.image),
        "prompt": args.prompt,
        "warmup": args.warmup,
        "iterations_per_group": args.iters,
        "baseline_median_ms": args.baseline_ms,
        "groups": results,
        "derived": derived,
        "methodology_note": (
            "Each group is profiled in a separate forward pass set. Hooks synchronize "
            "PyTorch MPS and MLX at module boundaries and therefore perturb scheduling. "
            "Sums of medians are diagnostic only and are not additive GPU timelines. "
            "Experiment 0009 remains the authoritative uninstrumented latency."
        ),
    }

    out = Path("results/metalground_v0_encoder_subparts.json")
    out.write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n")

    print("\n=== Derived summary ===", flush=True)
    print(json.dumps(derived, indent=2), flush=True)
    print("\nFusion child rollup:", flush=True)
    print(json.dumps(results["fusion_children"]["rollup"], indent=2), flush=True)
    print("\nDeformable child rollup:", flush=True)
    print(json.dumps(results["deformable_children"]["rollup"], indent=2), flush=True)
    print(f"\nSaved: {out}", flush=True)


if __name__ == "__main__":
    main()
