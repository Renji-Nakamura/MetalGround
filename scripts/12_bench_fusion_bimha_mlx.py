#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import math
import statistics
import time
import types
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import mlx.core as mx
import numpy as np
import torch
import transformers
from PIL import Image
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor


def sync_torch() -> None:
    torch.mps.synchronize()


def sync_mlx() -> None:
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


def compare_tensors(actual: torch.Tensor, expected: torch.Tensor) -> dict[str, Any]:
    a = actual.detach().float()
    b = expected.detach().float()
    d = (a - b).abs()

    return {
        "shape": list(a.shape),
        "max_abs": float(d.max().item()),
        "mean_abs": float(d.mean().item()),
        "rmse": float(torch.sqrt(torch.mean((a - b) ** 2)).item()),
        "allclose_1e-5": bool(torch.allclose(a, b, rtol=1e-5, atol=1e-5)),
        "allclose_1e-4": bool(torch.allclose(a, b, rtol=1e-4, atol=1e-4)),
        "finite": bool(torch.isfinite(a).all().item()),
    }


def benchmark_torch(module, captured, warmup: int, iters: int):
    args = (
        captured["vision_features"],
        captured["text_features"],
    )
    kwargs = {
        "vision_attention_mask": captured["vision_attention_mask"],
        "text_attention_mask": captured["text_attention_mask"],
    }

    with torch.inference_mode():
        for _ in range(warmup):
            sync_torch()
            out = module(*args, **kwargs)
            sync_torch()

        samples = []
        out = None
        for i in range(iters):
            sync_torch()
            t0 = time.perf_counter_ns()
            out = module(*args, **kwargs)
            sync_torch()
            dt = (time.perf_counter_ns() - t0) / 1e6
            samples.append(dt)
            print(f"  torch {i+1:02d}/{iters}: {dt:.3f} ms", flush=True)

    return out, stats(samples)


def linear_mlx(x, weight, bias):
    y = x @ mx.transpose(weight)
    if bias is not None:
        y = y + bias
    return y


class MlxBiMHA:
    def __init__(self, torch_module):
        self.num_heads = int(torch_module.num_heads)
        self.head_dim = int(torch_module.head_dim)
        self.embed_dim = int(torch_module.embed_dim)
        self.scale = float(torch_module.scale)

        sync_torch()

        names = (
            "vision_proj",
            "text_proj",
            "values_vision_proj",
            "values_text_proj",
            "out_vision_proj",
            "out_text_proj",
        )
        self.params = {}
        for name in names:
            layer = getattr(torch_module, name)
            self.params[name] = {
                "weight": mx.asarray(layer.weight, copy=False),
                "bias": None if layer.bias is None else mx.asarray(layer.bias, copy=False),
            }

        # Ensure all shared parameter arrays are ready before compilation/benchmarking.
        arrays = []
        for p in self.params.values():
            arrays.append(p["weight"])
            if p["bias"] is not None:
                arrays.append(p["bias"])
        mx.eval(*arrays)
        sync_mlx()

        self.compiled = mx.compile(self._forward_impl)

    def _proj(self, name, x):
        p = self.params[name]
        return linear_mlx(x, p["weight"], p["bias"])

    def _reshape_heads(self, x, batch_size: int, seq_len: int):
        # PyTorch reference:
        # [B,L,E] -> [B,L,H,D] -> [B,H,L,D] -> [B*H,L,D]
        return (
            x.reshape(batch_size, seq_len, self.num_heads, self.head_dim)
            .transpose(0, 2, 1, 3)
            .reshape(batch_size * self.num_heads, seq_len, self.head_dim)
        )

    def _forward_impl(self, vision_features, text_features, vision_mask, text_mask):
        batch_size = vision_features.shape[0]
        tgt_len = vision_features.shape[1]
        src_len = text_features.shape[1]

        q = self._proj("vision_proj", vision_features) * self.scale
        k = self._proj("text_proj", text_features)
        vv = self._proj("values_vision_proj", vision_features)
        vt = self._proj("values_text_proj", text_features)

        q = self._reshape_heads(q, batch_size, tgt_len)
        k = self._reshape_heads(k, batch_size, src_len)
        vv = self._reshape_heads(vv, batch_size, tgt_len)
        vt = self._reshape_heads(vt, batch_size, src_len)

        # [B*H,V,D] x [B*H,D,T] -> [B*H,V,T]
        attn = q @ k.transpose(0, 2, 1)

        # Match HF GroundingDinoBiMultiHeadAttention exactly:
        # global max subtraction, then clamp.
        attn = attn - mx.max(attn)
        attn = mx.clip(attn, -50000.0, 50000.0)

        text_attn = attn.transpose(0, 2, 1)
        text_attn = text_attn - mx.max(text_attn, axis=-1, keepdims=True)
        text_attn = mx.clip(text_attn, -50000.0, 50000.0)

        # Masks are bool arrays. True means masked/padding.
        if vision_mask.ndim == 2:
            vm = vision_mask.reshape(batch_size, 1, 1, tgt_len)
            vm = mx.broadcast_to(vm, (batch_size, self.num_heads, 1, tgt_len))
            vm = vm.reshape(batch_size * self.num_heads, 1, tgt_len)
            text_attn = mx.where(vm, -float("inf"), text_attn)

        text_attn_weights = mx.softmax(text_attn, axis=-1)

        if text_mask.ndim == 2:
            tm = text_mask.reshape(batch_size, 1, 1, src_len)
            tm = mx.broadcast_to(tm, (batch_size, self.num_heads, 1, src_len))
            tm = tm.reshape(batch_size * self.num_heads, 1, src_len)
            attn = mx.where(tm, -float("inf"), attn)

        vision_attn_weights = mx.softmax(attn, axis=-1)

        # fusion_dropout == 0.0 for the target Grounding DINO Tiny config.
        vision_out = vision_attn_weights @ vt
        text_out = text_attn_weights @ vv

        vision_out = (
            vision_out.reshape(batch_size, self.num_heads, tgt_len, self.head_dim)
            .transpose(0, 2, 1, 3)
            .reshape(batch_size, tgt_len, self.embed_dim)
        )
        text_out = (
            text_out.reshape(batch_size, self.num_heads, src_len, self.head_dim)
            .transpose(0, 2, 1, 3)
            .reshape(batch_size, src_len, self.embed_dim)
        )

        vision_out = self._proj("out_vision_proj", vision_out)
        text_out = self._proj("out_text_proj", text_out)

        return vision_out, vision_attn_weights, text_out, text_attn_weights

    def call_eager(self, vision, text, vision_mask, text_mask):
        return self._forward_impl(vision, text, vision_mask, text_mask)

    def call_compiled(self, vision, text, vision_mask, text_mask):
        return self.compiled(vision, text, vision_mask, text_mask)


def to_mlx_inputs(captured):
    sync_torch()

    vision = mx.asarray(captured["vision_features"], copy=False)
    text = mx.asarray(captured["text_features"], copy=False)

    # bool MPS tensors use the validated shared Metal bridge.
    vm = captured["vision_attention_mask"]
    tm = captured["text_attention_mask"]

    if vm is None:
        vm_mx = mx.zeros((vision.shape[0], vision.shape[1]), dtype=mx.bool_)
    else:
        vm_mx = mx.asarray(vm, copy=False)

    if tm is None:
        tm_mx = mx.zeros((text.shape[0], text.shape[1]), dtype=mx.bool_)
    else:
        tm_mx = mx.asarray(tm, copy=False)

    return vision, text, vm_mx, tm_mx


def mlx_outputs_to_torch(outputs):
    vision_out, vision_attn, text_out, text_attn = outputs
    mx.eval(vision_out, vision_attn, text_out, text_attn)
    sync_mlx()

    result = (
        torch.as_tensor(vision_out),
        torch.as_tensor(vision_attn),
        torch.as_tensor(text_out),
        torch.as_tensor(text_attn),
    )
    sync_torch()
    return result


def benchmark_mlx(
    impl: MlxBiMHA,
    captured,
    compiled: bool,
    warmup: int,
    iters: int,
):
    label = "mlx_compiled" if compiled else "mlx_eager"
    call = impl.call_compiled if compiled else impl.call_eager

    # JIT/graph build first call is always excluded.
    vision, text, vm, tm = to_mlx_inputs(captured)
    first = call(vision, text, vm, tm)
    _ = mlx_outputs_to_torch(first)

    for _ in range(warmup):
        sync_torch()
        vision, text, vm, tm = to_mlx_inputs(captured)
        out = call(vision, text, vm, tm)
        _ = mlx_outputs_to_torch(out)

    samples = []
    final_torch = None
    for i in range(iters):
        sync_torch()
        t0 = time.perf_counter_ns()

        vision, text, vm, tm = to_mlx_inputs(captured)
        out = call(vision, text, vm, tm)
        final_torch = mlx_outputs_to_torch(out)

        dt = (time.perf_counter_ns() - t0) / 1e6
        samples.append(dt)
        print(f"  {label} {i+1:02d}/{iters}: {dt:.3f} ms", flush=True)

    return final_torch, stats(samples)


def flatten_reference_outputs(out):
    (vision_out, vision_attn), (text_out, text_attn) = out
    return vision_out, vision_attn, text_out, text_attn


def correctness(actual_tuple, expected_tuple):
    names = (
        "vision_output",
        "vision_attention_weights",
        "text_output",
        "text_attention_weights",
    )
    result = {}
    for name, actual, expected in zip(names, actual_tuple, expected_tuple):
        result[name] = compare_tensors(actual, expected)
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="IDEA-Research/grounding-dino-tiny")
    ap.add_argument("--image", type=Path, default=Path("assets/input.jpg"))
    ap.add_argument("--prompt", nargs="+", default=["a cat", "a dog"])
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--iters", type=int, default=20)
    args = ap.parse_args()

    if not torch.backends.mps.is_available():
        raise SystemExit("MPS unavailable.")

    mx.set_default_device(mx.gpu)

    print("Loading model...", flush=True)
    processor = AutoProcessor.from_pretrained(args.model)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(args.model)
    model.eval().to("mps")
    sync_torch()

    image = Image.open(args.image).convert("RGB")
    text_labels = [args.prompt]
    cpu_inputs = processor(images=image, text=text_labels, return_tensors="pt")
    inputs = {k: v.to("mps") for k, v in cpu_inputs.items()}
    sync_torch()

    attn = model.model.encoder.layers[0].fusion_layer.attn
    original_forward = attn.forward
    captured: dict[str, Any] = {}

    def capture_forward(self, vision_features, text_features, vision_attention_mask=None, text_attention_mask=None):
        if not captured:
            captured["vision_features"] = vision_features.detach()
            captured["text_features"] = text_features.detach()
            captured["vision_attention_mask"] = (
                None if vision_attention_mask is None else vision_attention_mask.detach()
            )
            captured["text_attention_mask"] = (
                None if text_attention_mask is None else text_attention_mask.detach()
            )
        return original_forward(
            vision_features,
            text_features,
            vision_attention_mask=vision_attention_mask,
            text_attention_mask=text_attention_mask,
        )

    attn.forward = types.MethodType(capture_forward, attn)

    print("Capturing real layer-0 fusion attention inputs...", flush=True)
    with torch.inference_mode():
        _ = model(**inputs)
        sync_torch()

    attn.forward = original_forward

    if not captured:
        raise RuntimeError("Failed to capture fusion attention inputs.")

    print(
        f"vision={tuple(captured['vision_features'].shape)} "
        f"text={tuple(captured['text_features'].shape)}",
        flush=True,
    )
    print(
        f"embed_dim={attn.embed_dim} heads={attn.num_heads} head_dim={attn.head_dim}",
        flush=True,
    )

    # Save a frozen reproducibility fixture.
    fixture_dir = Path("results/fusion_cases")
    fixture_dir.mkdir(parents=True, exist_ok=True)

    print("Running one PyTorch reference output for fixture...", flush=True)
    with torch.inference_mode():
        ref_output = attn(
            captured["vision_features"],
            captured["text_features"],
            vision_attention_mask=captured["vision_attention_mask"],
            text_attention_mask=captured["text_attention_mask"],
        )
        sync_torch()

    source = inspect.getsource(type(attn))
    source_hash = hashlib.sha256(source.encode("utf-8")).hexdigest()
    (fixture_dir / "grounding_dino_bimha_source.py").write_text(source)

    fixture = {
        "vision_features": captured["vision_features"].detach().cpu(),
        "text_features": captured["text_features"].detach().cpu(),
        "vision_attention_mask": (
            None
            if captured["vision_attention_mask"] is None
            else captured["vision_attention_mask"].detach().cpu()
        ),
        "text_attention_mask": (
            None
            if captured["text_attention_mask"] is None
            else captured["text_attention_mask"].detach().cpu()
        ),
        "state_dict": {k: v.detach().cpu() for k, v in attn.state_dict().items()},
        "output": tuple(
            tuple(x.detach().cpu() for x in pair)
            for pair in ref_output
        ),
        "metadata": {
            "module_class": type(attn).__name__,
            "embed_dim": int(attn.embed_dim),
            "num_heads": int(attn.num_heads),
            "head_dim": int(attn.head_dim),
            "scale": float(attn.scale),
            "dropout": float(attn.dropout),
            "transformers_version": transformers.__version__,
            "source_sha256": source_hash,
        },
    }
    torch.save(fixture, fixture_dir / "layer0_fp32.pt")

    print("\nBenchmarking isolated PyTorch/MPS BiMHA...", flush=True)
    torch_out, torch_lat = benchmark_torch(attn, captured, args.warmup, args.iters)
    torch_flat = flatten_reference_outputs(torch_out)

    print("\nPreparing MLX faithful BiMHA...", flush=True)
    mlx_impl = MlxBiMHA(attn)

    print("\nBenchmarking MLX eager...", flush=True)
    mlx_eager_out, mlx_eager_lat = benchmark_mlx(
        mlx_impl, captured, compiled=False, warmup=args.warmup, iters=args.iters
    )
    eager_corr = correctness(mlx_eager_out, torch_flat)

    print("\nBenchmarking MLX compiled...", flush=True)
    mlx_compiled_out, mlx_compiled_lat = benchmark_mlx(
        mlx_impl, captured, compiled=True, warmup=args.warmup, iters=args.iters
    )
    compiled_corr = correctness(mlx_compiled_out, torch_flat)

    result = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "experiment": "0012",
        "purpose": (
            "Feasibility test for replacing GroundingDinoBiMultiHeadAttention "
            "with a faithful MLX implementation on the real layer-0 fusion workload."
        ),
        "model": args.model,
        "device": "mps",
        "dtype": "float32",
        "image": str(args.image),
        "prompt": args.prompt,
        "software": {
            "torch": torch.__version__,
            "transformers": transformers.__version__,
        },
        "fixture": {
            "path": "results/fusion_cases/layer0_fp32.pt",
            "source_path": "results/fusion_cases/grounding_dino_bimha_source.py",
            "source_sha256": source_hash,
            "vision_shape": list(captured["vision_features"].shape),
            "text_shape": list(captured["text_features"].shape),
            "embed_dim": int(attn.embed_dim),
            "num_heads": int(attn.num_heads),
            "head_dim": int(attn.head_dim),
            "dropout": float(attn.dropout),
        },
        "latency": {
            "pytorch_mps": torch_lat,
            "mlx_eager_bridge_included": mlx_eager_lat,
            "mlx_compiled_bridge_included": mlx_compiled_lat,
            "speedup_mlx_eager_vs_pytorch": (
                torch_lat["median_ms"] / mlx_eager_lat["median_ms"]
            ),
            "speedup_mlx_compiled_vs_pytorch": (
                torch_lat["median_ms"] / mlx_compiled_lat["median_ms"]
            ),
        },
        "correctness": {
            "mlx_eager_vs_pytorch": eager_corr,
            "mlx_compiled_vs_pytorch": compiled_corr,
        },
        "notes": [
            "PyTorch and MLX use the same captured real fusion inputs and the same trained weights.",
            "MLX projection weights are shared from PyTorch MPS with copy=False and cached outside timing.",
            "MLX timings include input zero-copy bridge, evaluation/synchronization, output export, and conservative synchronization.",
            "First mx.compile/JIT execution is excluded from timing.",
            "No approximation or reduced precision is used.",
            "This is an isolated operator feasibility benchmark; full-model integration is a later experiment."
        ],
    }

    out_path = Path("results/metalground_fusion_bimha_microbench.json")
    out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")

    print("\n=== Experiment 0012 summary ===", flush=True)
    print(
        f"PyTorch median:      {torch_lat['median_ms']:.3f} ms",
        flush=True,
    )
    print(
        f"MLX eager median:    {mlx_eager_lat['median_ms']:.3f} ms "
        f"({result['latency']['speedup_mlx_eager_vs_pytorch']:.2f}x)",
        flush=True,
    )
    print(
        f"MLX compiled median: {mlx_compiled_lat['median_ms']:.3f} ms "
        f"({result['latency']['speedup_mlx_compiled_vs_pytorch']:.2f}x)",
        flush=True,
    )
    print(
        "compiled vision output correctness:",
        compiled_corr["vision_output"],
        flush=True,
    )
    print(f"Saved: {out_path}", flush=True)


if __name__ == "__main__":
    main()
