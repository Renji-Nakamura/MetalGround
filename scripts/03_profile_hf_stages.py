#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import platform
import statistics
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import transformers
from PIL import Image
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor


def sync(device: torch.device) -> None:
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize(device)


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
    # ModelOutput and similar objects generally expose to_tuple().
    to_tuple = getattr(x, "to_tuple", None)
    if callable(to_tuple):
        try:
            return tensor_summary(to_tuple(), depth + 1)
        except Exception:
            pass
    return type(x).__name__


class ModuleProfiler:
    def __init__(
        self,
        device: torch.device,
        named_modules: dict[str, torch.nn.Module],
        names: list[str],
    ) -> None:
        self.device = device
        self.named_modules = named_modules
        self.names = names
        self.samples_ns: dict[str, list[int]] = defaultdict(list)
        self.starts: dict[str, list[int]] = defaultdict(list)
        self.io_examples: dict[str, dict[str, Any]] = {}
        self.handles: list[Any] = []

    def _pre(self, name: str):
        def hook(module: torch.nn.Module, inputs: tuple[Any, ...]) -> None:
            sync(self.device)
            self.starts[name].append(time.perf_counter_ns())
            if name not in self.io_examples:
                self.io_examples[name] = {"input": tensor_summary(inputs)}
        return hook

    def _post(self, name: str):
        def hook(module: torch.nn.Module, inputs: tuple[Any, ...], output: Any) -> None:
            sync(self.device)
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
        name for name, module in named_modules.items()
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

    encoder_parts: list[str] = []
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

    decoder_parts: list[str] = []
    for i in range(6):
        # These are disjoint major attention blocks within each decoder layer.
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


def load_baseline(path: Path | None) -> dict[str, Any] | None:
    if path is None or not path.exists():
        return None
    with path.open() as f:
        return json.load(f)


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
            share = f"  ~{100.0 * row['median_ms'] / baseline_ms:5.1f}% baseline"
        print(
            f"{row['median_ms']:9.2f} ms  "
            f"{row['class']:<48}  {name}{share}"
        )

    total = sum(row["median_ms"] for row in rows.values())
    if baseline_ms and baseline_ms > 0:
        print(
            f"sum of module medians: {total:.2f} ms "
            f"(~{100.0 * total / baseline_ms:.1f}% of uninstrumented baseline; "
            f"diagnostic only)"
        )
    else:
        print(f"sum of module medians: {total:.2f} ms")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", choices=["mps", "cpu"], default="mps")
    ap.add_argument("--image", type=Path, default=Path("assets/input.jpg"))
    ap.add_argument("--model", default="IDEA-Research/grounding-dino-tiny")
    ap.add_argument("--prompt", nargs="+", default=["a cat", "a dog"])
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--iters", type=int, default=3)
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
        default=Path("results/hf_mps_fp32_catdog.json"),
    )
    ap.add_argument("--tag", default="stage_profile")
    args = ap.parse_args()

    if not args.image.exists():
        raise SystemExit(f"Missing image: {args.image}")
    if args.device == "mps" and not torch.backends.mps.is_available():
        raise SystemExit("MPS is unavailable in this environment.")

    device = torch.device(args.device)

    print(f"Loading {args.model} on {device} ...")
    processor = AutoProcessor.from_pretrained(args.model)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(args.model)
    model.eval().to(device)
    sync(device)

    image = Image.open(args.image).convert("RGB")
    # Keep exactly the same prompt construction as Experiment 0001.
    text_labels = [args.prompt]
    cpu_inputs = processor(images=image, text=text_labels, return_tensors="pt")
    inputs = {
        k: v.to(device) if hasattr(v, "to") else v
        for k, v in cpu_inputs.items()
    }

    input_shapes = {
        k: tensor_summary(v)
        for k, v in inputs.items()
    }

    named_modules = dict(model.named_modules())
    profile_groups = build_profile_groups(named_modules)

    unknown_groups = [g for g in args.groups if g not in profile_groups]
    if unknown_groups:
        raise SystemExit(
            f"Unknown groups: {unknown_groups}. Available: {list(profile_groups)}"
        )

    baseline = load_baseline(args.baseline)
    baseline_ms = None
    if baseline is not None:
        try:
            if baseline.get("device") == args.device:
                baseline_ms = float(baseline["forward"]["median_ms"])
        except Exception:
            baseline_ms = None

    print("\nProcessed inputs:")
    print(json.dumps(input_shapes, indent=2))
    if baseline_ms is not None:
        print(f"\nUninstrumented baseline median: {baseline_ms:.2f} ms")
    else:
        print("\nNo matching uninstrumented baseline loaded.")

    # Global warmup before any hooks.
    print(f"\nGlobal warmup: {args.warmup}")
    with torch.inference_mode():
        for _ in range(args.warmup):
            _ = model(**inputs)
            sync(device)

    results: dict[str, Any] = {}

    for group_name in args.groups:
        selected = profile_groups[group_name]
        if not selected:
            print(f"\nSkipping empty group: {group_name}")
            continue

        print(
            f"\nProfiling {group_name}: "
            f"{len(selected)} modules × {args.iters} forward passes"
        )

        full_forward_ms: list[float] = []

        with ModuleProfiler(device, named_modules, selected) as prof:
            with torch.inference_mode():
                for i in range(args.iters):
                    sync(device)
                    t0 = time.perf_counter_ns()
                    _ = model(**inputs)
                    sync(device)
                    dt_ms = (time.perf_counter_ns() - t0) / 1e6
                    full_forward_ms.append(dt_ms)
                    print(
                        f"  {group_name} pass {i + 1:02d}/{args.iters}: "
                        f"{dt_ms:.2f} ms instrumented full forward"
                    )

            rows = prof.report()

        results[group_name] = {
            "selected_modules": selected,
            "instrumented_full_forward": summary_ms(full_forward_ms),
            "modules": rows,
            "aggregate": group_aggregate(rows),
        }
        print_ranked(group_name, rows, baseline_ms)

    # Useful aggregate view of the twelve MSDA instances.
    msda_summary: dict[str, Any] = {}
    for source_group in ("msda_wrappers", "msda_cores"):
        if source_group not in results:
            continue
        rows = results[source_group]["modules"]
        encoder = [
            r["median_ms"] for n, r in rows.items()
            if n.startswith("model.encoder.")
        ]
        decoder = [
            r["median_ms"] for n, r in rows.items()
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
        "tag": args.tag,
        "model": args.model,
        "device": args.device,
        "dtype": "float32",
        "image": str(args.image),
        "image_size_wh": list(image.size),
        "prompt": args.prompt,
        "warmup": args.warmup,
        "iterations_per_group": args.iters,
        "methodology_note": (
            "MPS is asynchronous. Module hooks synchronize at measured boundaries. "
            "These measurements are diagnostic and can perturb scheduling; "
            "the uninstrumented baseline remains the authoritative end-to-end latency."
        ),
        "baseline_file": str(args.baseline) if args.baseline else None,
        "baseline_forward_median_ms": baseline_ms,
        "processed_inputs": input_shapes,
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
    out = out_dir / f"hf_{args.device}_{args.tag}.json"
    out.write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n")

    print("\n=== MSDA aggregate ===")
    print(json.dumps(msda_summary, indent=2))

    print(f"\nSaved: {out}")
    print(
        "\nIMPORTANT: use the original uninstrumented baseline for headline latency. "
        "This profiler is for bottleneck attribution."
    )


if __name__ == "__main__":
    main()
