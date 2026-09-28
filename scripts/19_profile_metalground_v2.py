#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import statistics
import sys
import time
import types
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import mlx.core as mx
import torch
from PIL import Image
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor


def load_module(path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load helper script: {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    try:
        spec.loader.exec_module(mod)
    except Exception:
        sys.modules.pop(module_name, None)
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
            module = self.named[name]
            self.handles.append(module.register_forward_pre_hook(self._pre(name)))
            self.handles.append(module.register_forward_hook(self._post(name)))
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


def existing(named, candidates):
    return [x for x in candidates if x in named]


def by_class(named, wanted):
    wanted = set(wanted)
    return [
        name
        for name, module in named.items()
        if name and type(module).__name__ in wanted
    ]


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

    deformable_islands = existing(
        named,
        [f"model.encoder.layers.{i}.deformable_layer" for i in range(6)],
    )

    decoder_layers = existing(
        named, [f"model.decoder.layers.{i}" for i in range(6)]
    )

    # Robust across small Transformers naming shifts: class-based discovery.
    swin_stages = by_class(
        named,
        {
            "SwinStage",
            "GroundingDinoSwinStage",
        },
    )
    swin_layers = by_class(
        named,
        {
            "SwinLayer",
            "GroundingDinoSwinLayer",
        },
    )
    swin_attn = by_class(
        named,
        {
            "SwinAttention",
            "SwinSelfAttention",
            "GroundingDinoSwinAttention",
            "GroundingDinoSwinSelfAttention",
        },
    )
    swin_embeddings = by_class(
        named,
        {
            "SwinEmbeddings",
            "SwinPatchEmbeddings",
            "GroundingDinoSwinEmbeddings",
            "GroundingDinoSwinPatchEmbeddings",
        },
    )

    # Fallback / additional backbone landmarks if the exact classes differ.
    backbone_major_candidates = [
        "model.backbone.conv_encoder",
        "model.backbone.conv_encoder.model",
        "model.backbone.conv_encoder.model.embeddings",
        "model.backbone.conv_encoder.model.encoder",
        "model.backbone.position_embedding",
    ]
    backbone_major = existing(named, backbone_major_candidates)

    return {
        "coarse": coarse,
        "encoder_layers": encoder_layers,
        "encoder_parts": encoder_parts,
        "fusion_attn": fusion_attn,
        "encoder_deformable_islands": deformable_islands,
        "decoder_layers": decoder_layers,
        "backbone_major": backbone_major,
        "swin_stages": swin_stages,
        "swin_layers": swin_layers,
        "swin_attention": swin_attn,
        "swin_embeddings": swin_embeddings,
    }


def profile_group(model, inputs, named, group_name, names, iters):
    print(f"\n=== {group_name}: {len(names)} modules ===", flush=True)
    full = []
    with HookProfiler(named, names) as hp:
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
            f"  {row['median_ms']:8.3f} ms  "
            f"{row['class']:<40} {name}",
            flush=True,
        )

    return {
        "selected_modules": names,
        "instrumented_full_forward": stats(full),
        "modules": rows,
        "aggregate": {
            "sum_module_medians_ms": sum(r["median_ms"] for r in rows.values())
        },
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="IDEA-Research/grounding-dino-tiny")
    ap.add_argument("--image", type=Path, default=Path("assets/input.jpg"))
    ap.add_argument("--prompt", nargs="+", default=["a cat", "a dog"])
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--iters", type=int, default=3)
    ap.add_argument("--threadgroup", type=int, default=256)
    ap.add_argument("--baseline-ms", type=float, default=542.305167)
    ap.add_argument(
        "--exp14-helper",
        type=Path,
        default=Path("scripts/14_full_model_fusion_algebraic.py"),
    )
    ap.add_argument(
        "--exp17-helper",
        type=Path,
        default=Path("scripts/17_bench_deformable_mlx_island.py"),
    )
    ap.add_argument(
        "--exp18-helper",
        type=Path,
        default=Path("scripts/18_full_model_deformable_islands.py"),
    )
    args = ap.parse_args()

    for p in (args.exp14_helper, args.exp17_helper, args.exp18_helper):
        if not p.exists():
            raise SystemExit(f"Missing helper script: {p}")
    if not torch.backends.mps.is_available():
        raise SystemExit("MPS unavailable.")

    h14 = load_module(args.exp14_helper, "metalground_exp14_helpers_exp19")
    h17 = load_module(args.exp17_helper, "metalground_exp17_helpers_exp19")
    h18 = load_module(args.exp18_helper, "metalground_exp18_helpers_exp19")

    mx.set_default_device(mx.gpu)

    print("Loading model...", flush=True)
    processor = AutoProcessor.from_pretrained(args.model)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(args.model)
    model.eval().to("mps")
    h14.sync()

    image = Image.open(args.image).convert("RGB")
    text_labels = [args.prompt]
    cpu_inputs = processor(images=image, text=text_labels, return_tensors="pt")
    inputs = {k: v.to("mps") for k, v in cpu_inputs.items()}
    h14.sync()

    print("Building MetalGround v2 state...", flush=True)
    fusion_items = h14.collect_fusion_specializers(model)
    _msda_state = h14.patch_msda(model, args.threadgroup)
    h14.set_fusion_mode(fusion_items, "fully_folded")

    # Capture runtime spatial metadata before replacing encoder deformable layers.
    layer0 = model.model.encoder.layers[0].deformable_layer
    original_layer0_forward = layer0.forward
    captured = {}

    def capture_forward(
        self,
        hidden_states,
        attention_mask,
        position_embeddings=None,
        reference_points=None,
        spatial_shapes=None,
        spatial_shapes_list=None,
        level_start_index=None,
        output_attentions=False,
    ):
        if not captured:
            captured["spatial_shapes_list"] = [
                (int(h), int(w)) for h, w in spatial_shapes_list
            ]
        return original_layer0_forward(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_embeddings=position_embeddings,
            reference_points=reference_points,
            spatial_shapes=spatial_shapes,
            spatial_shapes_list=spatial_shapes_list,
            level_start_index=level_start_index,
            output_attentions=output_attentions,
        )

    layer0.forward = types.MethodType(capture_forward, layer0)
    with torch.inference_mode():
        _ = model(**inputs)
        h14.sync()
    layer0.forward = original_layer0_forward

    if not captured:
        raise RuntimeError("Failed to capture spatial shapes.")

    h14.sync()
    _island_state, _islands = h18.patch_encoder_deformable_islands(
        model,
        h17.MlxDeformableIsland,
        activation_name=model.config.activation_function,
        spatial_shapes_list=captured["spatial_shapes_list"],
        threadgroup=args.threadgroup,
    )

    print(f"Warmup x{args.warmup}...", flush=True)
    with torch.inference_mode():
        for i in range(args.warmup):
            _ = model(**inputs)
            sync_hybrid()
            print(f"  warmup {i+1}/{args.warmup}", flush=True)

    named = dict(model.named_modules())
    groups = build_groups(named)

    results = {}
    for group_name, names in groups.items():
        if not names:
            print(f"\nSkipping empty group: {group_name}", flush=True)
            continue
        results[group_name] = profile_group(
            model, inputs, named, group_name, names, args.iters
        )

    derived = {"authoritative_v2_median_ms": args.baseline_ms}

    if "coarse" in results:
        derived["coarse_ranked"] = [
            {
                "name": name,
                "class": row["class"],
                "median_ms": row["median_ms"],
                "share_of_v2_percent": 100.0
                * row["median_ms"]
                / args.baseline_ms,
            }
            for name, row in sorted(
                results["coarse"]["modules"].items(),
                key=lambda kv: kv[1]["median_ms"],
                reverse=True,
            )
        ]

    result = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "experiment": "0019",
        "purpose": (
            "Re-profile authoritative MetalGround v2 and decompose the Swin "
            "backbone to select the next optimization target."
        ),
        "model": args.model,
        "device": "mps",
        "dtype": "float32",
        "configuration": {
            "baseline_median_ms": args.baseline_ms,
            "fusion": "six fully_folded algebraic BiMHA layers",
            "encoder_deformable": "six compiled MLX execution islands",
            "decoder_msda": "MetalGround MSDA v0 bridge",
            "threadgroup_size": args.threadgroup,
            "warmup": args.warmup,
            "iterations_per_group": args.iters,
        },
        "groups": results,
        "derived": derived,
        "methodology_note": (
            "All hook measurements synchronize PyTorch MPS and MLX Metal at "
            "module boundaries and are diagnostic only. Experiment 0018's "
            "542.305 ms median remains the authoritative MetalGround v2 latency."
        ),
    }

    out = Path("results/metalground_v2_stage_profile.json")
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")

    print("\n=== Experiment 0019 coarse ranking ===", flush=True)
    for row in derived.get("coarse_ranked", []):
        print(
            f"{row['median_ms']:8.3f} ms "
            f"{row['share_of_v2_percent']:5.1f}% "
            f"{row['name']}",
            flush=True,
        )

    if "swin_stages" in results:
        print("\n=== Swin stage sum ===", flush=True)
        print(
            f"{results['swin_stages']['aggregate']['sum_module_medians_ms']:.3f} ms",
            flush=True,
        )

    print(f"\nSaved: {out}", flush=True)


if __name__ == "__main__":
    main()
