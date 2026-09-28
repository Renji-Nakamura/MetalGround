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
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

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


def sync_torch():
    torch.mps.synchronize()


def sync_mlx():
    mx.synchronize()


def percentile(xs, p):
    ys = sorted(xs)
    if len(ys) == 1:
        return ys[0]
    k = (len(ys) - 1) * p
    f, c = math.floor(k), math.ceil(k)
    if f == c:
        return ys[f]
    return ys[f] * (c - k) + ys[c] * (k - f)


def stats(xs):
    return {
        "n": len(xs),
        "mean_ms": statistics.fmean(xs),
        "median_ms": statistics.median(xs),
        "p90_ms": percentile(xs, 0.90),
        "p95_ms": percentile(xs, 0.95),
        "min_ms": min(xs),
        "max_ms": max(xs),
    }


def compare(a: torch.Tensor, b: torch.Tensor) -> dict[str, Any]:
    af = a.detach().float()
    bf = b.detach().float()
    d = (af - bf).abs()
    return {
        "shape": list(a.shape),
        "max_abs": float(d.max().item()),
        "mean_abs": float(d.mean().item()),
        "rmse": float(torch.sqrt(torch.mean((af - bf) ** 2)).item()),
        "allclose_1e-5": bool(torch.allclose(a, b, rtol=1e-5, atol=1e-5)),
        "allclose_1e-4": bool(torch.allclose(a, b, rtol=1e-4, atol=1e-4)),
        "finite": bool(torch.isfinite(a).all().item()),
    }


def flatten_output(out):
    (vision_out, vision_attn), (text_out, text_attn) = out
    return vision_out, vision_attn, text_out, text_attn


def compare_output(actual, expected):
    names = (
        "vision_output",
        "vision_attention_weights",
        "text_output",
        "text_attention_weights",
    )
    aa = flatten_output(actual)
    bb = flatten_output(expected)
    return {n: compare(a, b) for n, a, b in zip(names, aa, bb)}


class MlxFullyFoldedFusion:
    def __init__(self, torch_specialized):
        self.scale = float(torch_specialized.scale)

        def imp(t):
            return mx.asarray(t, copy=False)

        self.score_M = imp(torch_specialized.score_M)
        self.score_cv = imp(torch_specialized.score_cv)
        self.score_ct = imp(torch_specialized.score_ct)
        self.score_c0 = imp(torch_specialized.score_c0)
        self.C_v = imp(torch_specialized.C_v)
        self.c_v = imp(torch_specialized.c_v)
        self.C_t = imp(torch_specialized.C_t)
        self.c_t = imp(torch_specialized.c_t)
        self.out_vision_bias = imp(torch_specialized.out_vision_bias)
        self.out_text_bias = imp(torch_specialized.out_text_bias)

        mx.eval(
            self.score_M,
            self.score_cv,
            self.score_ct,
            self.score_c0,
            self.C_v,
            self.c_v,
            self.C_t,
            self.c_t,
            self.out_vision_bias,
            self.out_text_bias,
        )
        sync_mlx()

        self.compiled = mx.compile(self._forward_impl)

    def _normalize_scores(self, score, vision_mask, text_mask):
        score = score - mx.max(score)
        score = mx.clip(score, -50000.0, 50000.0)

        text_score = score.transpose(0, 2, 1)
        text_score = text_score - mx.max(text_score, axis=-1, keepdims=True)
        text_score = mx.clip(text_score, -50000.0, 50000.0)

        if vision_mask is not None:
            # PyTorch reference uses masked_fill(mask, -inf).
            vm = vision_mask[0][None, None, :]
            text_score = mx.where(vm, -mx.inf, text_score)

        text_attn = mx.softmax(text_score, axis=-1)

        if text_mask is not None:
            tm = text_mask[0][None, None, :]
            score = mx.where(tm, -mx.inf, score)

        vision_attn = mx.softmax(score, axis=-1)
        return vision_attn, text_attn

    def _forward_impl(self, vision, text, vision_mask, text_mask):
        # Batch-1 specialization, identical algebra to Experiment 0013.
        v = vision[0]  # [V,d]
        t = text[0]    # [T,d]

        R = self.score_M @ t.T
        R = R + self.score_cv[:, :, None]

        score = v[None, :, :] @ R

        text_bias = (t @ self.score_ct.T).T
        text_bias = text_bias + self.score_c0[:, None]
        score = (score + text_bias[:, None, :]) * self.scale

        vision_attn, text_attn = self._normalize_scores(
            score, vision_mask, text_mask
        )

        z_text = t[None, :, :] @ self.C_v
        z_text = z_text + self.c_v[:, None, :]

        delta_v = vision_attn @ z_text
        delta_v = mx.sum(delta_v, axis=0)
        delta_v = delta_v + self.out_vision_bias
        delta_v = delta_v[None, :, :]

        reduced_v = text_attn @ v
        row_sum = mx.sum(text_attn, axis=-1, keepdims=True)
        delta_t_heads = reduced_v @ self.C_t
        delta_t_heads = delta_t_heads + row_sum * self.c_t[:, None, :]
        delta_t = mx.sum(delta_t_heads, axis=0) + self.out_text_bias
        delta_t = delta_t[None, :, :]

        return delta_v, vision_attn, delta_t, text_attn

    def compiled_call(self, vision, text, vision_mask, text_mask):
        return self.compiled(vision, text, vision_mask, text_mask)


def torch_to_mlx_inputs(vision, text, vm, tm):
    sync_torch()
    vision_mx = mx.asarray(vision, copy=False)
    text_mx = mx.asarray(text, copy=False)
    vm_mx = None if vm is None else mx.asarray(vm, copy=False)
    tm_mx = None if tm is None else mx.asarray(tm, copy=False)
    return vision_mx, text_mx, vm_mx, tm_mx


def mlx_flat_to_torch(out):
    mx.eval(*out)
    sync_mlx()
    tv = torch.as_tensor(out[0])
    va = torch.as_tensor(out[1])
    tt = torch.as_tensor(out[2])
    ta = torch.as_tensor(out[3])
    sync_torch()
    return (tv, va), (tt, ta)


def bench_torch(label, fn, call_args, call_kwargs, warmup, iters):
    with torch.inference_mode():
        for _ in range(warmup):
            sync_torch()
            out = fn(*call_args, **call_kwargs)
            sync_torch()

        samples = []
        final = None
        for i in range(iters):
            sync_torch()
            t0 = time.perf_counter_ns()
            final = fn(*call_args, **call_kwargs)
            sync_torch()
            dt = (time.perf_counter_ns() - t0) / 1e6
            samples.append(dt)
            print(f"  {label} {i+1:02d}/{iters}: {dt:.3f} ms", flush=True)
    return final, stats(samples)


def bench_mlx_internal(label, fn, args, warmup, iters):
    first = fn(*args)
    mx.eval(*first)
    sync_mlx()

    for _ in range(warmup):
        out = fn(*args)
        mx.eval(*out)
        sync_mlx()

    samples = []
    final = None
    for i in range(iters):
        sync_mlx()
        t0 = time.perf_counter_ns()
        final = fn(*args)
        mx.eval(*final)
        sync_mlx()
        dt = (time.perf_counter_ns() - t0) / 1e6
        samples.append(dt)
        print(f"  {label} {i+1:02d}/{iters}: {dt:.3f} ms", flush=True)

    return final, stats(samples)


def bench_mlx_bridge(label, fn, torch_args, warmup, iters):
    vision, text, vm, tm = torch_args

    for _ in range(warmup + 1):
        args_mx = torch_to_mlx_inputs(vision, text, vm, tm)
        out = fn(*args_mx)
        _ = mlx_flat_to_torch(out)

    samples = []
    final = None
    for i in range(iters):
        sync_torch()
        t0 = time.perf_counter_ns()
        args_mx = torch_to_mlx_inputs(vision, text, vm, tm)
        out = fn(*args_mx)
        final = mlx_flat_to_torch(out)
        dt = (time.perf_counter_ns() - t0) / 1e6
        samples.append(dt)
        print(f"  {label} {i+1:02d}/{iters}: {dt:.3f} ms", flush=True)

    return final, stats(samples)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="IDEA-Research/grounding-dino-tiny")
    ap.add_argument("--image", type=Path, default=Path("assets/input.jpg"))
    ap.add_argument("--prompt", nargs="+", default=["a cat", "a dog"])
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument(
        "--exp13-helper",
        type=Path,
        default=Path("scripts/13_bench_fusion_algebraic_specialization.py"),
    )
    args = ap.parse_args()

    if not args.exp13_helper.exists():
        raise SystemExit(f"Missing helper: {args.exp13_helper}")
    if not torch.backends.mps.is_available():
        raise SystemExit("MPS unavailable.")

    h13 = load_module(args.exp13_helper, "metalground_exp13_helpers_exp30")
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
    captured = {}

    def capture_forward(
        self,
        vision_features,
        text_features,
        vision_attention_mask=None,
        text_attention_mask=None,
    ):
        if not captured:
            captured["vision"] = vision_features.detach()
            captured["text"] = text_features.detach()
            captured["vision_mask"] = (
                None if vision_attention_mask is None
                else vision_attention_mask.detach()
            )
            captured["text_mask"] = (
                None if text_attention_mask is None
                else text_attention_mask.detach()
            )
        return original_forward(
            vision_features,
            text_features,
            vision_attention_mask=vision_attention_mask,
            text_attention_mask=text_attention_mask,
        )

    attn.forward = types.MethodType(capture_forward, attn)
    print("Capturing real layer-0 fusion workload...", flush=True)
    with torch.inference_mode():
        _ = model(**inputs)
        sync_torch()
    attn.forward = original_forward

    vision = captured["vision"]
    text = captured["text"]
    vm = captured["vision_mask"]
    tm = captured["text_mask"]

    print(
        f"vision={tuple(vision.shape)} text={tuple(text.shape)} "
        f"embed={attn.embed_dim} heads={attn.num_heads} head_dim={attn.head_dim}",
        flush=True,
    )

    specialized = h13.AlgebraicBiMHA(attn)

    call_args = (vision, text)
    call_kwargs = {
        "vision_attention_mask": vm,
        "text_attention_mask": tm,
    }

    print("PyTorch fully-folded correctness reference...", flush=True)
    with torch.inference_mode():
        torch_ref = specialized.fully_folded(*call_args, **call_kwargs)
        sync_torch()

    # Producer sync before importing folded weights into MLX.
    sync_torch()
    mlx_spec = MlxFullyFoldedFusion(specialized)
    mlx_args = torch_to_mlx_inputs(vision, text, vm, tm)

    print("MLX correctness preflight...", flush=True)
    mlx_pre_flat = mlx_spec.compiled_call(*mlx_args)
    mlx_pre = mlx_flat_to_torch(mlx_pre_flat)
    corr_vs_torch_folded = compare_output(mlx_pre, torch_ref)

    with torch.inference_mode():
        original_ref = original_forward(*call_args, **call_kwargs)
        sync_torch()
    corr_vs_original = compare_output(mlx_pre, original_ref)

    print(
        "MLX vs PyTorch fully-folded:",
        json.dumps(corr_vs_torch_folded, indent=2),
        flush=True,
    )

    print("\nBenchmark PyTorch/MPS fully-folded...", flush=True)
    torch_final, torch_lat = bench_torch(
        "torch_fully_folded",
        specialized.fully_folded,
        call_args,
        call_kwargs,
        args.warmup,
        args.iters,
    )

    print("\nBenchmark MLX compiled fully-folded internal...", flush=True)
    mlx_internal_final_flat, mlx_internal_lat = bench_mlx_internal(
        "mlx_internal",
        mlx_spec.compiled_call,
        mlx_args,
        args.warmup,
        args.iters,
    )
    mlx_internal_final = mlx_flat_to_torch(mlx_internal_final_flat)

    print("\nBenchmark MLX compiled fully-folded bridge-included...", flush=True)
    mlx_bridge_final, mlx_bridge_lat = bench_mlx_bridge(
        "mlx_bridge",
        mlx_spec.compiled_call,
        (vision, text, vm, tm),
        args.warmup,
        args.iters,
    )

    final_correctness = {
        "mlx_internal_vs_torch_fully_folded": compare_output(
            mlx_internal_final, torch_final
        ),
        "mlx_bridge_vs_torch_fully_folded": compare_output(
            mlx_bridge_final, torch_final
        ),
        "mlx_bridge_vs_original_attention": compare_output(
            mlx_bridge_final, original_ref
        ),
    }

    result = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "experiment": "0030",
        "purpose": (
            "Test whether the exact algebraically fully-folded Grounding DINO "
            "fusion specialization benefits from MLX compilation enough to "
            "justify a wider encoder execution island."
        ),
        "model": args.model,
        "device": "mps",
        "dtype": "float32",
        "shape": {
            "vision": list(vision.shape),
            "text": list(text.shape),
            "heads": int(attn.num_heads),
            "head_dim": int(attn.head_dim),
            "embed_dim": int(attn.embed_dim),
        },
        "latency": {
            "pytorch_mps_fully_folded": torch_lat,
            "mlx_compiled_internal": mlx_internal_lat,
            "mlx_compiled_bridge_included": mlx_bridge_lat,
        },
        "derived": {
            "speedup_mlx_internal_vs_torch": (
                torch_lat["median_ms"] / mlx_internal_lat["median_ms"]
            ),
            "speedup_mlx_bridge_vs_torch": (
                torch_lat["median_ms"] / mlx_bridge_lat["median_ms"]
            ),
            "bridge_overhead_estimate_ms": (
                mlx_bridge_lat["median_ms"] - mlx_internal_lat["median_ms"]
            ),
        },
        "correctness": {
            "preflight_mlx_vs_torch_fully_folded": corr_vs_torch_folded,
            "preflight_mlx_vs_original_attention": corr_vs_original,
            "final": final_correctness,
        },
        "decision_rule": (
            "Proceed to a wider encoder-layer MLX execution island only if "
            "bridge-included MLX fully-folded fusion is at least ~5% faster "
            "than the PyTorch/MPS fully-folded specialization while preserving "
            "the established FP32 correctness envelope. Otherwise keep fusion "
            "in PyTorch and avoid the larger port."
        ),
        "notes": [
            "The MLX implementation uses the same folded tensors and same algebra as Experiment 0013.",
            "Folded weights are constructed once outside timed regions.",
            "MLX-internal timing excludes PyTorch<->MLX conversion/synchronization; bridge-included timing includes conservative explicit synchronization.",
            "No approximation, pruning, retraining, quantization, or reduced precision is used.",
            "Floating-point association differs across backends, so bitwise equality is not required."
        ],
    }

    out = Path("results/metalground_fully_folded_fusion_mlx.json")
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")

    print("\n=== Experiment 0030 summary ===", flush=True)
    print(
        f"PyTorch fully-folded: {torch_lat['median_ms']:.3f} ms",
        flush=True,
    )
    print(
        f"MLX internal:         {mlx_internal_lat['median_ms']:.3f} ms "
        f"({result['derived']['speedup_mlx_internal_vs_torch']:.3f}x)",
        flush=True,
    )
    print(
        f"MLX bridge:           {mlx_bridge_lat['median_ms']:.3f} ms "
        f"({result['derived']['speedup_mlx_bridge_vs_torch']:.3f}x)",
        flush=True,
    )
    print(f"Saved: {out}", flush=True)


if __name__ == "__main__":
    main()
