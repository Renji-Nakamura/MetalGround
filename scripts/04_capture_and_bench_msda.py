#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import math
import platform
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import transformers
from PIL import Image
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor


ENCODER_CORE = "model.encoder.layers.0.deformable_layer.self_attn.attn"
DECODER_CORE = "model.decoder.layers.0.encoder_attn.attn"


def sync(device: torch.device) -> None:
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize(device)


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


def tree_map(x: Any, fn) -> Any:
    if torch.is_tensor(x):
        return fn(x)
    if isinstance(x, tuple):
        return tuple(tree_map(v, fn) for v in x)
    if isinstance(x, list):
        return [tree_map(v, fn) for v in x]
    if isinstance(x, dict):
        return {k: tree_map(v, fn) for k, v in x.items()}
    return x


def to_cpu_clone(x: Any) -> Any:
    return tree_map(x, lambda t: t.detach().cpu().clone())


def to_device(x: Any, device: torch.device) -> Any:
    return tree_map(x, lambda t: t.to(device))


def tensor_tree_summary(x: Any) -> Any:
    if torch.is_tensor(x):
        return {
            "shape": list(x.shape),
            "dtype": str(x.dtype).replace("torch.", ""),
            "numel": x.numel(),
            "bytes": x.numel() * x.element_size(),
        }
    if isinstance(x, tuple):
        return [tensor_tree_summary(v) for v in x]
    if isinstance(x, list):
        return [tensor_tree_summary(v) for v in x]
    if isinstance(x, dict):
        return {str(k): tensor_tree_summary(v) for k, v in x.items()}
    return x if isinstance(x, (str, int, float, bool, type(None))) else type(x).__name__


def tensor_bytes(x: Any) -> int:
    if torch.is_tensor(x):
        return x.numel() * x.element_size()
    if isinstance(x, (tuple, list)):
        return sum(tensor_bytes(v) for v in x)
    if isinstance(x, dict):
        return sum(tensor_bytes(v) for v in x.values())
    return 0


class CaseCapture:
    def __init__(self, module: torch.nn.Module):
        self.args = None
        self.kwargs = None
        self.output = None
        self.handles = []

        def pre_hook(mod, args, kwargs):
            if self.args is None:
                self.args = to_cpu_clone(args)
                self.kwargs = to_cpu_clone(kwargs)

        def post_hook(mod, args, kwargs, output):
            if self.output is None:
                self.output = to_cpu_clone(output)

        self.handles.append(
            module.register_forward_pre_hook(pre_hook, with_kwargs=True)
        )
        self.handles.append(
            module.register_forward_hook(post_hook, with_kwargs=True)
        )

    def close(self):
        for h in self.handles:
            h.remove()
        self.handles.clear()


def compare_output(actual: Any, expected: Any) -> dict[str, float]:
    if not torch.is_tensor(actual) or not torch.is_tensor(expected):
        raise TypeError("Expected tensor outputs for MSDA core.")
    a = actual.detach().float().cpu()
    b = expected.detach().float().cpu()
    diff = (a - b).abs()
    denom = b.abs().clamp_min(1e-8)
    return {
        "max_abs": float(diff.max()),
        "mean_abs": float(diff.mean()),
        "max_rel": float((diff / denom).max()),
        "mean_rel": float((diff / denom).mean()),
    }


def bench_case(
    module: torch.nn.Module,
    case: dict[str, Any],
    device: torch.device,
    warmup: int,
    iters: int,
) -> dict[str, Any]:
    args = to_device(case["args"], device)
    kwargs = to_device(case["kwargs"], device)
    expected = case["output"]

    with torch.inference_mode():
        for _ in range(warmup):
            _ = module(*args, **kwargs)
            sync(device)

        times = []
        out = None
        for _ in range(iters):
            sync(device)
            t0 = time.perf_counter_ns()
            out = module(*args, **kwargs)
            sync(device)
            times.append((time.perf_counter_ns() - t0) / 1e6)

    assert out is not None
    return {
        "latency": stats(times),
        "correctness_vs_captured": compare_output(out, expected),
        "args_summary": tensor_tree_summary(case["args"]),
        "input_tensor_bytes": tensor_bytes(case["args"]) + tensor_bytes(case["kwargs"]),
        "output_summary": tensor_tree_summary(expected),
        "output_tensor_bytes": tensor_bytes(expected),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", choices=["mps", "cpu"], default="mps")
    ap.add_argument("--model", default="IDEA-Research/grounding-dino-tiny")
    ap.add_argument("--image", type=Path, default=Path("assets/input.jpg"))
    ap.add_argument("--prompt", nargs="+", default=["a cat", "a dog"])
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--iters", type=int, default=20)
    args = ap.parse_args()

    device = torch.device(args.device)
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise SystemExit("MPS is unavailable.")

    if not args.image.exists():
        raise SystemExit(f"Missing image: {args.image}")

    print(f"Loading {args.model} on {device} ...")
    processor = AutoProcessor.from_pretrained(args.model)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(args.model)
    model.eval().to(device)
    sync(device)

    named = dict(model.named_modules())
    for target in (ENCODER_CORE, DECODER_CORE):
        if target not in named:
            raise SystemExit(f"Missing target module: {target}")

    image = Image.open(args.image).convert("RGB")
    # Same construction used by previous experiments, so this remains comparable.
    cpu_inputs = processor(
        images=image,
        text=[args.prompt],
        return_tensors="pt",
    )
    inputs = {
        k: v.to(device) if hasattr(v, "to") else v
        for k, v in cpu_inputs.items()
    }

    captures = {
        "encoder": CaseCapture(named[ENCODER_CORE]),
        "decoder": CaseCapture(named[DECODER_CORE]),
    }

    print("Capturing one real encoder and decoder MSDA invocation...")
    with torch.inference_mode():
        _ = model(**inputs)
        sync(device)

    for c in captures.values():
        c.close()

    cases: dict[str, dict[str, Any]] = {}
    for key, capture in captures.items():
        if capture.args is None or capture.output is None:
            raise RuntimeError(f"Failed to capture {key} case.")
        cases[key] = {
            "args": capture.args,
            "kwargs": capture.kwargs or {},
            "output": capture.output,
            "module_name": ENCODER_CORE if key == "encoder" else DECODER_CORE,
        }

    case_dir = Path("results/msda_cases")
    case_dir.mkdir(parents=True, exist_ok=True)

    for key, case in cases.items():
        out = case_dir / f"{key}_fp32.pt"
        torch.save(case, out)
        print(f"Saved real {key} case: {out} ({out.stat().st_size / 1024**2:.1f} MiB)")

    # Save the exact implementation source used by this local Transformers install.
    core_class = type(named[ENCODER_CORE])
    source_text = None
    source_file = None
    try:
        source_text = inspect.getsource(core_class)
        source_file = inspect.getsourcefile(core_class)
        source_out = case_dir / "multiscale_deformable_attention_source.py"
        source_out.write_text(source_text)
        print(f"Saved implementation source: {source_out}")
    except Exception as e:
        print(f"Could not extract source with inspect: {e}")

    print(f"\nBenchmarking isolated MSDA core ({args.warmup} warmup, {args.iters} iterations)...")
    results = {}
    for key, case in cases.items():
        module = named[case["module_name"]]
        result = bench_case(module, case, device, args.warmup, args.iters)
        results[key] = result
        lat = result["latency"]
        err = result["correctness_vs_captured"]
        print(
            f"{key:7s}: median={lat['median_ms']:.3f} ms "
            f"p95={lat['p95_ms']:.3f} ms "
            f"max_abs_err={err['max_abs']:.3e}"
        )

    metadata = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "model": args.model,
        "device": args.device,
        "dtype": "float32",
        "image": str(args.image),
        "image_size_wh": list(image.size),
        "prompt": args.prompt,
        "targets": {
            "encoder": ENCODER_CORE,
            "decoder": DECODER_CORE,
        },
        "warmup": args.warmup,
        "iterations": args.iters,
        "software": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
        },
        "implementation_source": {
            "class": core_class.__name__,
            "source_file": source_file,
            "sha256": (
                hashlib.sha256(source_text.encode()).hexdigest()
                if source_text is not None else None
            ),
        },
        "results": results,
        "methodology_note": (
            "Cases are captured from a real end-to-end Grounding DINO forward pass, "
            "then replayed directly through the same MultiScaleDeformableAttention "
            "module with MPS synchronization around each isolated invocation."
        ),
    }

    out_json = Path("results/hf_mps_msda_microbench.json")
    out_json.write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n")
    print(f"\nSaved benchmark: {out_json}")


if __name__ == "__main__":
    main()
