#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import platform
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
import transformers
from PIL import Image
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

from metalground.msda_metal_v0 import msda_metal_v0


def sync_hybrid() -> None:
    # Diagnostic profiling boundary for a hybrid PyTorch-MPS / MLX-Metal model.
    torch.mps.synchronize()
    mx.synchronize()


def percentile(xs: list[float], p: float) -> float:
    if not xs:
        return math.nan
    ys = sorted(xs)
    if len(ys) == 1:
        return ys[0]
    k = (len(ys) - 1) * p
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return ys[f]
    return ys[f] * (c - k) + ys[c] * (k - f)


def summary_ms(xs: list[float]) -> dict[str, float]:
    return {
        "n": len(xs),
        "mean_ms": statistics.fmean(xs),
        "median_ms": statistics.median(xs),
        "p90_ms": percentile(xs, 0.90),
        "p95_ms": percentile(xs, 0.95),
        "min_ms": min(xs),
        "max_ms": max(xs),
    }


def tensor_summary(x: Any, depth: int = 0) -> Any:
    if depth > 2:
        return type(x).__name__
    if torch.is_tensor(x):
        return {
            "shape": list(x.shape),
            "dtype": str(x.dtype).replace("torch.", ""),
            "device": str(x.device),
        }
    if isinstance(x, dict):
        return {str(k): tensor_summary(v, depth + 1) for k, v in list(x.items())[:12]}
    if isinstance(x, (list, tuple)):
        return [tensor_summary(v, depth + 1) for v in list(x)[:12]]
    to_tuple = getattr(x, "to_tuple", None)
    if callable(to_tuple):
        try:
            return tensor_summary(to_tuple(), depth + 1)
        except Exception:
            pass
    return type(x).__name__


@dataclass
class PatchState:
    threadgroup: int
    metadata_cache: dict[
        tuple[tuple[int, int], ...], tuple[mx.array, mx.array]
    ] = field(default_factory=dict)
    module_names: list[str] = field(default_factory=list)

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


def make_metal_forward(state: PatchState):
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

        # Same conservative protocol as Experiment 0009.
        torch.mps.synchronize()

        value_mx = mx.asarray(value, copy=False)
        loc_mx = mx.asarray(sampling_locations, copy=False)
        weight_mx = mx.asarray(attention_weights, copy=False)
        spatial_mx, start_mx = state.metadata(value_spatial_shapes_list)

        out_mx = msda_metal_v0(
            value_mx,
            spatial_mx,
            start_mx,
            loc_mx,
            weight_mx,
            threadgroup_size=state.threadgroup,
        )
        mx.eval(out_mx)
        mx.synchronize()

        out = torch.as_tensor(out_mx)
        if out.device.type != "mps":
            raise RuntimeError(f"Unexpected MetalGround output device: {out.device}")
        return out

    return forward


def patch_all_msda(model: torch.nn.Module, threadgroup: int) -> PatchState:
    state = PatchState(threadgroup=threadgroup)
    patched_forward = make_metal_forward(state)

    for name, module in model.named_modules():
        if module.__class__.__name__ != "MultiScaleDeformableAttention":
            continue
        module.forward = types.MethodType(patched_forward, module)
        state.module_names.append(name)

    if len(state.module_names) != 12:
        raise RuntimeError(f"Expected 12 MSDA cores, got {len(state.module_names)}")
    return state


class ModuleProfiler:
    def __init__(
        self,
        named_modules: dict[str, torch.nn.Module],
        names: list[str],
    ) -> None:
        self.named_modules = named_modules
        self.names = names
        self.samples_ns: dict[str, list[int]] = defaultdict(list)
        self.starts: dict[str, list[int]] = defaultdict(list)
        self.io_examples: dict[str, dict[str, Any]] = {}
        self.handles: list[Any] = []

    def _pre(self, name: str):
        def hook(module: torch.nn.Module, inputs: tuple[Any, ...]) -> None:
            sync_hybrid()
            self.starts[name].append(time.perf_counter_ns())
            if name not in self.io_examples:
                self.io_examples[name] = {"input": tensor_summary(inputs)}
        return hook

    def _post(self, name: str):
        def hook(module: torch.nn.Module, inputs: tuple[Any, ...], output: Any) -> None:
            sync_hybrid()
            t1 = time.perf_counter_ns()
            t0 = self.starts[name].pop()
            self.samples_ns[name].append(t1 - t0)
            if "output" not in self.io_examples.get(name, {}):
                self.io_examples.setdefault(name, {})["output"] = tensor_summary(output)
        return hook

    def __enter__(self):
        missing = [n for n in self.names if n not in self.named_modules]
        if missing:
            raise KeyError(f"Missing modules: {missing}")

        for name in self.names:
            module = self.named_modules[name]
            self.handles.append(module.register_forward_pre_hook(self._pre(name)))
            self.handles.append(module.register_forward_hook(self._post(name)))
        return self

    def __exit__(self, exc_type, exc, tb):
        for h in self.handles:
            h.remove()
        self.handles.clear()

    def report(self) -> dict[str, Any]:
        rows = {}
        for name in self.names:
            xs_ms = [x / 1e6 for x in self.samples_ns.get(name, [])]
            if xs_ms:
                rows[name] = {
                    "class": type(self.named_modules[name]).__name__,
                    **summary_ms(xs_ms),
                    "io_example": self.io_examples.get(name, {}),
                }
        return rows


def find_by_class(named_modules: dict[str, torch.nn.Module], class_name: str) -> list[str]:
    return [
        name
        for name, module in named_modules.items()
        if name and type(module).__name__ == class_name
    ]


def build_profile_groups(named_modules: dict[str, torch.nn.Module]) -> dict[str, list[str]]:
    names = set(named_modules)

    coarse_candidates = [
        "model.backbone",
        "model.text_backbone",
        "model.text_projection",
        "model.input_proj_vision.0",
        "model.input_proj_vision.1",
        "model.input_proj_vision.2",
        "model.input_proj_vision.3",
        "model.encoder",
        "model.enc_output",
        "model.enc_output_norm",
        "model.encoder_output_bbox_embed",
        "model.encoder_output_class_embed",
        "model.decoder",
    ]
    coarse = [n for n in coarse_candidates if n in names]

    encoder_layers = [
        f"model.encoder.layers.{i}"
        for i in range(6)
        if f"model.encoder.layers.{i}" in names
    ]

    encoder_parts = []
    for i in range(6):
        for suffix in ("text_enhancer_layer", "fusion_layer", "deformable_layer"):
            n = f"model.encoder.layers.{i}.{suffix}"
            if n in names:
                encoder_parts.append(n)

    decoder_layers = [
        f"model.decoder.layers.{i}"
        for i in range(6)
        if f"model.decoder.layers.{i}" in names
    ]

    decoder_parts = []
    for i in range(6):
        for suffix in ("self_attn", "encoder_attn_text", "encoder_attn"):
            n = f"model.decoder.layers.{i}.{suffix}"
            if n in names:
                decoder_parts.append(n)

    msda_wrappers = find_by_class(
        named_modules, "GroundingDinoMultiscaleDeformableAttention"
    )
    msda_cores = find_by_class(
        named_modules, "MultiScaleDeformableAttention"
    )

    return {
        "coarse": coarse,
        "encoder_layers": encoder_layers,
        "encoder_parts": encoder_parts,
        "decoder_layers": decoder_layers,
        "decoder_parts": decoder_parts,
        "msda_wrappers": msda_wrappers,
        "msda_cores": msda_cores,
    }


def load_metalground_baseline(path: Path | None) -> float | None:
    if path is None or not path.exists():
        return None
    data = json.loads(path.read_text())
    try:
        return float(data["performance"]["patched_forward_latency"]["median_ms"])
    except Exception:
        return None


def group_aggregate(rows: dict[str, Any]) -> dict[str, float]:
    medians = [float(v["median_ms"]) for v in rows.values()]
    means = [float(v["mean_ms"]) for v in rows.values()]
    return {
        "sum_module_medians_ms": sum(medians),
        "sum_module_means_ms": sum(means),
    }


def print_ranked(group_name: str, rows: dict[str, Any], baseline_ms: float | None) -> None:
    print(f"\n=== {group_name} ===")
    ranked = sorted(rows.items(), key=lambda kv: kv[1]["median_ms"], reverse=True)
    for name, row in ranked:
        share = ""
        if baseline_ms and baseline_ms > 0:
            share = f"  ~{100.0 * row['median_ms'] / baseline_ms:5.1f}% MetalGround-v0"
        print(
            f"{row['median_ms']:9.2f} ms  "
            f"{row['class']:<48}  {name}{share}"
        )

    total = sum(row["median_ms"] for row in rows.values())
    if baseline_ms and baseline_ms > 0:
        print(
            f"sum of module medians: {total:.2f} ms "
            f"(~{100.0 * total / baseline_ms:.1f}% of uninstrumented MetalGround-v0; "
            f"diagnostic only)"
        )
    else:
        print(f"sum of module medians: {total:.2f} ms")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", type=Path, default=Path("assets/input.jpg"))
    ap.add_argument("--model", default="IDEA-Research/grounding-dino-tiny")
    ap.add_argument("--prompt", nargs="+", default=["a cat", "a dog"])
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--iters", type=int, default=3)
    ap.add_argument("--threadgroup", type=int, default=256)
    ap.add_argument(
        "--groups",
        nargs="+",
        default=[
            "coarse",
            "encoder_layers",
            "encoder_parts",
            "decoder_layers",
            "decoder_parts",
            "msda_wrappers",
            "msda_cores",
        ],
    )
    ap.add_argument(
        "--baseline",
        type=Path,
        default=Path("results/metalground_v0_full_model_bench.json"),
    )
    ap.add_argument("--tag", default="metalground_v0_stage_profile")
    args = ap.parse_args()

    if not args.image.exists():
        raise SystemExit(f"Missing image: {args.image}")
    if not torch.backends.mps.is_available():
        raise SystemExit("MPS is unavailable.")

    mx.set_default_device(mx.gpu)

    print(f"Loading {args.model} on MPS...", flush=True)
    processor = AutoProcessor.from_pretrained(args.model)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(args.model)
    model.eval().to("mps")
    sync_hybrid()

    image = Image.open(args.image).convert("RGB")
    text_labels = [args.prompt]
    cpu_inputs = processor(images=image, text=text_labels, return_tensors="pt")
    inputs = {k: v.to("mps") for k, v in cpu_inputs.items()}
    sync_hybrid()

    print("Patching 12 MSDA cores with MetalGround v0...", flush=True)
    patch_state = patch_all_msda(model, args.threadgroup)

    named_modules = dict(model.named_modules())
    profile_groups = build_profile_groups(named_modules)

    unknown_groups = [g for g in args.groups if g not in profile_groups]
    if unknown_groups:
        raise SystemExit(
            f"Unknown groups: {unknown_groups}. Available: {list(profile_groups)}"
        )

    baseline_ms = load_metalground_baseline(args.baseline)
    if baseline_ms is not None:
        print(f"Uninstrumented MetalGround-v0 median: {baseline_ms:.3f} ms", flush=True)
    else:
        print("No MetalGround-v0 baseline loaded.", flush=True)

    print(f"\nGlobal warmup: {args.warmup}", flush=True)
    with torch.inference_mode():
        for i in range(args.warmup):
            _ = model(**inputs)
            sync_hybrid()
            print(f"  warmup {i+1}/{args.warmup}", flush=True)

    results: dict[str, Any] = {}

    for group_name in args.groups:
        selected = profile_groups[group_name]
        if not selected:
            print(f"\nSkipping empty group: {group_name}", flush=True)
            continue

        print(
            f"\nProfiling {group_name}: {len(selected)} modules x {args.iters} passes",
            flush=True,
        )

        full_forward_ms = []

        with ModuleProfiler(named_modules, selected) as prof:
            with torch.inference_mode():
                for i in range(args.iters):
                    sync_hybrid()
                    t0 = time.perf_counter_ns()
                    _ = model(**inputs)
                    sync_hybrid()
                    dt_ms = (time.perf_counter_ns() - t0) / 1e6
                    full_forward_ms.append(dt_ms)
                    print(
                        f"  {group_name} pass {i+1:02d}/{args.iters}: "
                        f"{dt_ms:.2f} ms instrumented full forward",
                        flush=True,
                    )

            rows = prof.report()

        results[group_name] = {
            "selected_modules": selected,
            "instrumented_full_forward": summary_ms(full_forward_ms),
            "modules": rows,
            "aggregate": group_aggregate(rows),
        }
        print_ranked(group_name, rows, baseline_ms)

    msda_summary = {}
    for source_group in ("msda_wrappers", "msda_cores"):
        if source_group not in results:
            continue
        rows = results[source_group]["modules"]
        encoder = [
            r["median_ms"]
            for n, r in rows.items()
            if n.startswith("model.encoder.")
        ]
        decoder = [
            r["median_ms"]
            for n, r in rows.items()
            if n.startswith("model.decoder.")
        ]
        msda_summary[source_group] = {
            "encoder_count": len(encoder),
            "encoder_sum_median_ms": sum(encoder),
            "decoder_count": len(decoder),
            "decoder_sum_median_ms": sum(decoder),
            "total_sum_median_ms": sum(encoder) + sum(decoder),
        }

    record = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "experiment": "0010",
        "tag": args.tag,
        "model": args.model,
        "device": "mps",
        "dtype": "float32",
        "image": str(args.image),
        "image_size_wh": list(image.size),
        "prompt": args.prompt,
        "warmup": args.warmup,
        "iterations_per_group": args.iters,
        "metalground": {
            "kernel": "metal_msda_v0_fp32",
            "threadgroup_size": args.threadgroup,
            "patched_modules": patch_state.module_names,
        },
        "methodology_note": (
            "Hybrid execution is asynchronous across PyTorch MPS and MLX Metal. "
            "Hooks synchronize both runtimes at measured boundaries. These timings "
            "are intentionally diagnostic and perturb scheduling. Experiment 0009's "
            "uninstrumented median remains the authoritative MetalGround-v0 latency."
        ),
        "baseline_file": str(args.baseline),
        "baseline_forward_median_ms": baseline_ms,
        "software": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
        },
        "groups": results,
        "msda_summary": msda_summary,
    }

    out_dir = Path("results")
    out_dir.mkdir(exist_ok=True)
    out = out_dir / f"{args.tag}.json"
    out.write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n")

    print("\n=== MetalGround-v0 MSDA aggregate ===", flush=True)
    print(json.dumps(msda_summary, indent=2), flush=True)
    print(f"\nSaved: {out}", flush=True)
    print(
        "\nIMPORTANT: Experiment 0009 uninstrumented latency is the headline number. "
        "Experiment 0010 is only for new bottleneck attribution.",
        flush=True,
    )


if __name__ == "__main__":
    main()
