#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import statistics
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import mlx.core as mx
import torch
from PIL import Image
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor


def load_helpers(path: Path):
    name = "metalground_exp14_helpers_profile_v1"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load helper script: {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    try:
        spec.loader.exec_module(mod)
    except Exception:
        sys.modules.pop(name, None)
        raise
    return mod


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
        self.handles.clear()

    def report(self):
        return {
            name: {
                "class": type(self.named[name]).__name__,
                **stats(self.samples[name]),
            }
            for name in self.names
            if self.samples[name]
        }


def existing(named, names):
    return [n for n in names if n in named]


def build_groups(named):
    coarse = existing(
        named,
        [
            "model.backbone",
            "model.text_backbone",
            "model.encoder",
            "model.decoder",
            "model.enc_output",
            "model.enc_output_norm",
            "model.encoder_output_bbox_embed",
            "model.encoder_output_class_embed",
        ],
    )

    encoder_layers = existing(
        named, [f"model.encoder.layers.{i}" for i in range(6)]
    )

    encoder_parts = []
    for i in range(6):
        for suffix in ("text_enhancer_layer", "fusion_layer", "deformable_layer"):
            n = f"model.encoder.layers.{i}.{suffix}"
            if n in named:
                encoder_parts.append(n)

    fusion_attn = existing(
        named,
        [f"model.encoder.layers.{i}.fusion_layer.attn" for i in range(6)],
    )

    deformable_wrappers = existing(
        named,
        [f"model.encoder.layers.{i}.deformable_layer.self_attn" for i in range(6)],
    )

    msda_cores = [
        name
        for name, module in named.items()
        if name and type(module).__name__ == "MultiScaleDeformableAttention"
    ]

    ffn = []
    for i in range(6):
        for suffix in ("fc1", "fc2", "self_attn_layer_norm", "final_layer_norm"):
            n = f"model.encoder.layers.{i}.deformable_layer.{suffix}"
            if n in named:
                ffn.append(n)

    msda_projection = []
    for i in range(6):
        for suffix in (
            "sampling_offsets",
            "attention_weights",
            "value_proj",
            "output_proj",
        ):
            n = f"model.encoder.layers.{i}.deformable_layer.self_attn.{suffix}"
            if n in named:
                msda_projection.append(n)

    decoder_layers = existing(
        named, [f"model.decoder.layers.{i}" for i in range(6)]
    )

    return {
        "coarse": coarse,
        "encoder_layers": encoder_layers,
        "encoder_parts": encoder_parts,
        "fusion_attn": fusion_attn,
        "encoder_ffn_norm": ffn,
        "encoder_msda_wrappers": deformable_wrappers,
        "encoder_msda_projection": msda_projection,
        "msda_cores_all12": msda_cores,
        "decoder_layers": decoder_layers,
    }


def rollup(rows):
    total = sum(r["median_ms"] for r in rows.values())
    return {"sum_module_medians_ms": total}


def profile_group(model, inputs, named, group_name, module_names, iters):
    print(f"\n=== {group_name}: {len(module_names)} modules ===", flush=True)
    full = []

    with HookProfiler(named, module_names) as hp:
        with torch.inference_mode():
            for i in range(iters):
                sync_hybrid()
                t0 = time.perf_counter_ns()
                _ = model(**inputs)
                sync_hybrid()
                dt = (time.perf_counter_ns() - t0) / 1e6
                full.append(dt)
                print(
                    f"  pass {i+1}/{iters}: {dt:.3f} ms instrumented full forward",
                    flush=True,
                )

        rows = hp.report()

    ranked = sorted(rows.items(), key=lambda kv: kv[1]["median_ms"], reverse=True)
    for name, row in ranked:
        print(
            f"  {row['median_ms']:8.3f} ms  {row['class']:<42} {name}",
            flush=True,
        )

    return {
        "selected_modules": module_names,
        "instrumented_full_forward": stats(full),
        "modules": rows,
        "aggregate": rollup(rows),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="IDEA-Research/grounding-dino-tiny")
    ap.add_argument("--image", type=Path, default=Path("assets/input.jpg"))
    ap.add_argument("--prompt", nargs="+", default=["a cat", "a dog"])
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--iters", type=int, default=3)
    ap.add_argument("--threadgroup", type=int, default=256)
    ap.add_argument("--baseline-ms", type=float, default=574.3521465)
    ap.add_argument(
        "--helper-script",
        type=Path,
        default=Path("scripts/14_full_model_fusion_algebraic.py"),
    )
    args = ap.parse_args()

    if not torch.backends.mps.is_available():
        raise SystemExit("MPS unavailable.")
    if not args.helper_script.exists():
        raise SystemExit(f"Missing helper script: {args.helper_script}")

    h = load_helpers(args.helper_script)
    mx.set_default_device(mx.gpu)

    print("Loading model...", flush=True)
    processor = AutoProcessor.from_pretrained(args.model)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(args.model)
    model.eval().to("mps")
    h.sync()

    image = Image.open(args.image).convert("RGB")
    text_labels = [args.prompt]
    cpu_inputs = processor(images=image, text=text_labels, return_tensors="pt")
    inputs = {k: v.to("mps") for k, v in cpu_inputs.items()}
    h.sync()

    print("Building six fully-folded fusion specializers...", flush=True)
    fusion_items = h.collect_fusion_specializers(model)

    print("Patching twelve MSDA cores...", flush=True)
    _msda_state = h.patch_msda(model, args.threadgroup)

    print("Enabling fully-folded fusion...", flush=True)
    h.set_fusion_mode(fusion_items, "fully_folded")

    print(f"Warmup x{args.warmup}...", flush=True)
    with torch.inference_mode():
        for i in range(args.warmup):
            _ = model(**inputs)
            sync_hybrid()
            print(f"  warmup {i+1}/{args.warmup}", flush=True)

    named = dict(model.named_modules())
    groups = build_groups(named)

    results = {}
    for group_name, module_names in groups.items():
        if module_names:
            results[group_name] = profile_group(
                model, inputs, named, group_name, module_names, args.iters
            )

    derived = {
        "authoritative_v1_median_ms": args.baseline_ms,
        "coarse_ranked": [],
    }

    if "coarse" in results:
        coarse_rows = results["coarse"]["modules"]
        derived["coarse_ranked"] = [
            {
                "name": name,
                "median_ms": row["median_ms"],
                "share_of_v1_percent": 100.0 * row["median_ms"] / args.baseline_ms,
            }
            for name, row in sorted(
                coarse_rows.items(),
                key=lambda kv: kv[1]["median_ms"],
                reverse=True,
            )
        ]

    result = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "experiment": "0016",
        "purpose": (
            "Re-profile MetalGround v1 after both MSDA and fusion specialization "
            "to determine the next measured optimization target."
        ),
        "model": args.model,
        "device": "mps",
        "dtype": "float32",
        "configuration": {
            "msda": "metalground_msda_v0_fp32",
            "fusion": "fully_folded_algebraic_bimha",
            "threadgroup_size": args.threadgroup,
            "baseline_median_ms": args.baseline_ms,
            "warmup": args.warmup,
            "iterations_per_group": args.iters,
        },
        "groups": results,
        "derived": derived,
        "methodology_note": (
            "Hooks synchronize both PyTorch MPS and MLX Metal at measured module "
            "boundaries, perturbing scheduling. Experiment 0015's fresh-process "
            "574.352 ms median remains the authoritative MetalGround v1 latency."
        ),
    }

    out = Path("results/metalground_v1_stage_profile.json")
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")

    print("\n=== Experiment 0016 coarse ranking ===", flush=True)
    for row in derived["coarse_ranked"]:
        print(
            f"{row['median_ms']:8.3f} ms  "
            f"{row['share_of_v1_percent']:5.1f}%  {row['name']}",
            flush=True,
        )
    print(f"\nSaved: {out}", flush=True)


if __name__ == "__main__":
    main()
