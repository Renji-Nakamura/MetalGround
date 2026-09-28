#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import statistics
import time
import types
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import mlx.core as mx
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


def compare_torch(a: torch.Tensor, b: torch.Tensor) -> dict[str, Any]:
    a = a.detach().float()
    b = b.detach().float()
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


def linear(x, w, b):
    y = x @ w.T
    if b is not None:
        y = y + b
    return y


class MlxStaticSwinBlock:
    """
    Fixed-resolution, batch-1 Swin block specialization.

    Specialization opportunities:
      * precompute the shifted-window mask for the captured resolution
      * precompute the learned relative-position bias lookup
      * optionally concatenate Q/K/V weights into one GEMM
      * compile the whole LN -> window attention -> residual -> MLP block
    """

    def __init__(self, block, input_dimensions: tuple[int, int]):
        self.block = block
        self.height = int(input_dimensions[0])
        self.width = int(input_dimensions[1])

        # Match HF's dynamic clamp once, then specialize to the resulting values.
        block.set_shift_and_window_size(input_dimensions)
        self.ws = int(block.window_size)
        self.shift = int(block.shift_size)

        self.C = int(block.layernorm_before.normalized_shape[0])
        self.H = int(block.attention.num_attention_heads)
        self.HD = int(block.attention.head_dim)
        self.scale = float(block.attention.scaling)
        self.ln1_eps = float(block.layernorm_before.eps)
        self.ln2_eps = float(block.layernorm_after.eps)

        # SwinLayer itself does not retain `config`, but SwinAttention does.
        # HF SwinMLP is constructed from the same config and uses
        # ACT2FN[config.hidden_act], so this is the authoritative activation name.
        hidden_act = getattr(block.attention.config, "hidden_act", None)
        if not isinstance(hidden_act, str):
            raise RuntimeError(
                f"Expected string Swin hidden_act, got {hidden_act!r}"
            )
        self.activation_name = hidden_act.lower()

        self.pad_right = (self.ws - self.width % self.ws) % self.ws
        self.pad_bottom = (self.ws - self.height % self.ws) % self.ws
        self.hp = self.height + self.pad_bottom
        self.wp = self.width + self.pad_right
        self.num_windows = (self.hp // self.ws) * (self.wp // self.ws)
        self.window_area = self.ws * self.ws

        # Parameter import happens after explicit producer synchronization.
        def imp(t):
            return mx.asarray(t, copy=False)

        self.p = {
            "ln1_w": imp(block.layernorm_before.weight),
            "ln1_b": imp(block.layernorm_before.bias),
            "ln2_w": imp(block.layernorm_after.weight),
            "ln2_b": imp(block.layernorm_after.bias),
            "q_w": imp(block.attention.q_proj.weight),
            "q_b": imp(block.attention.q_proj.bias),
            "k_w": imp(block.attention.k_proj.weight),
            "k_b": imp(block.attention.k_proj.bias),
            "v_w": imp(block.attention.v_proj.weight),
            "v_b": imp(block.attention.v_proj.bias),
            "o_w": imp(block.attention.o_proj.weight),
            "o_b": imp(block.attention.o_proj.bias),
            "fc1_w": imp(block.mlp.fc1.weight),
            "fc1_b": imp(block.mlp.fc1.bias),
            "fc2_w": imp(block.mlp.fc2.weight),
            "fc2_b": imp(block.mlp.fc2.bias),
        }

        # Exact algebraic Q/K/V packing. Static weight construction is outside timing.
        self.qkv_w = mx.concatenate(
            [self.p["q_w"], self.p["k_w"], self.p["v_w"]], axis=0
        )
        self.qkv_b = mx.concatenate(
            [self.p["q_b"], self.p["k_b"], self.p["v_b"]], axis=0
        )

        # Partial-evaluate relative position lookup and shift mask at this resolution.
        with torch.inference_mode():
            rel = block.attention.relative_position_bias().detach()
            mask = block.get_attn_mask(
                self.hp,
                self.wp,
                dtype=rel.dtype,
                device=rel.device,
            )
            if mask is None:
                combined = rel
            else:
                combined = rel + mask[:, None, :, :]
            sync_torch()

        self.combined_mask = mx.asarray(combined, copy=False)

        arrays = list(self.p.values()) + [
            self.qkv_w,
            self.qkv_b,
            self.combined_mask,
        ]
        mx.eval(*arrays)
        sync_mlx()

        self.separate_compiled = mx.compile(self._forward_separate_qkv)
        self.fused_compiled = mx.compile(self._forward_fused_qkv)

    def _activation(self, x):
        name = self.activation_name
        if name == "gelu":
            return 0.5 * x * (1.0 + mx.erf(x / math.sqrt(2.0)))
        if name in ("gelu_new", "gelu_fast"):
            c = math.sqrt(2.0 / math.pi)
            return 0.5 * x * (
                1.0 + mx.tanh(c * (x + 0.044715 * x * x * x))
            )
        if name == "relu":
            return mx.maximum(x, 0.0)
        if name in ("silu", "swish"):
            return x * mx.sigmoid(x)
        raise ValueError(f"Unsupported Swin activation: {name}")

    def _partition(self, x):
        B = x.shape[0]
        x = x.reshape(
            B,
            self.hp // self.ws,
            self.ws,
            self.wp // self.ws,
            self.ws,
            self.C,
        )
        x = x.transpose(0, 1, 3, 2, 4, 5)
        return x.reshape(-1, self.window_area, self.C)

    def _reverse(self, windows):
        B = windows.shape[0] // self.num_windows
        x = windows.reshape(
            B,
            self.hp // self.ws,
            self.wp // self.ws,
            self.ws,
            self.ws,
            self.C,
        )
        x = x.transpose(0, 1, 3, 2, 4, 5)
        return x.reshape(B, self.hp, self.wp, self.C)

    def _attention_from_qkv(self, q, k, v):
        # [nW, 49, C] -> [nW, heads, 49, head_dim]
        q = q.reshape(-1, self.window_area, self.H, self.HD).transpose(
            0, 2, 1, 3
        )
        k = k.reshape(-1, self.window_area, self.H, self.HD).transpose(
            0, 2, 1, 3
        )
        v = v.reshape(-1, self.window_area, self.H, self.HD).transpose(
            0, 2, 1, 3
        )

        scores = (q @ k.transpose(0, 1, 3, 2)) * self.scale
        scores = scores + self.combined_mask
        weights = mx.softmax(scores, axis=-1)

        out = weights @ v
        out = out.transpose(0, 2, 1, 3).reshape(
            -1, self.window_area, self.C
        )
        out = linear(out, self.p["o_w"], self.p["o_b"])
        return out, weights

    def _forward_common_pre(self, hidden):
        B = hidden.shape[0]
        shortcut = hidden
        x = mx.fast.layer_norm(
            hidden, self.p["ln1_w"], self.p["ln1_b"], self.ln1_eps
        )
        x = x.reshape(B, self.height, self.width, self.C)

        if self.pad_bottom or self.pad_right:
            x = mx.pad(
                x,
                [
                    (0, 0),
                    (0, self.pad_bottom),
                    (0, self.pad_right),
                    (0, 0),
                ],
            )

        if self.shift > 0:
            x = mx.roll(
                x,
                (-self.shift, -self.shift),
                (1, 2),
            )

        windows = self._partition(x)
        return shortcut, windows

    def _forward_common_post(self, shortcut, attn_windows):
        B = shortcut.shape[0]
        x = self._reverse(
            attn_windows.reshape(-1, self.ws, self.ws, self.C)
        )

        if self.shift > 0:
            x = mx.roll(
                x,
                (self.shift, self.shift),
                (1, 2),
            )

        if self.pad_bottom or self.pad_right:
            x = x[:, : self.height, : self.width, :]

        x = x.reshape(B, self.height * self.width, self.C)
        x = shortcut + x

        residual = x
        x = mx.fast.layer_norm(
            x, self.p["ln2_w"], self.p["ln2_b"], self.ln2_eps
        )
        x = linear(x, self.p["fc1_w"], self.p["fc1_b"])
        x = self._activation(x)
        x = linear(x, self.p["fc2_w"], self.p["fc2_b"])
        x = x + residual
        return x

    def _forward_separate_qkv(self, hidden):
        shortcut, windows = self._forward_common_pre(hidden)

        q = linear(windows, self.p["q_w"], self.p["q_b"])
        k = linear(windows, self.p["k_w"], self.p["k_b"])
        v = linear(windows, self.p["v_w"], self.p["v_b"])

        attn_out, weights = self._attention_from_qkv(q, k, v)
        hidden_out = self._forward_common_post(shortcut, attn_out)
        return hidden_out, weights

    def _forward_fused_qkv(self, hidden):
        shortcut, windows = self._forward_common_pre(hidden)

        qkv = linear(windows, self.qkv_w, self.qkv_b)
        q, k, v = mx.split(qkv, 3, axis=-1)

        attn_out, weights = self._attention_from_qkv(q, k, v)
        hidden_out = self._forward_common_post(shortcut, attn_out)
        return hidden_out, weights


def benchmark_torch(block, hidden, input_dimensions, warmup, iters):
    kwargs = {"always_partition": False}

    with torch.inference_mode():
        for _ in range(warmup):
            sync_torch()
            out = block(hidden, input_dimensions, **kwargs)
            sync_torch()

        samples = []
        final = None
        for i in range(iters):
            sync_torch()
            t0 = time.perf_counter_ns()
            final = block(hidden, input_dimensions, **kwargs)
            sync_torch()
            dt = (time.perf_counter_ns() - t0) / 1e6
            samples.append(dt)
            print(f"    torch {i+1:02d}/{iters}: {dt:.3f} ms", flush=True)

    return final, stats(samples)


def benchmark_mlx_internal(label, fn, hidden_mx, warmup, iters):
    first = fn(hidden_mx)
    mx.eval(*first)
    sync_mlx()

    for _ in range(warmup):
        out = fn(hidden_mx)
        mx.eval(*out)
        sync_mlx()

    samples = []
    final = None
    for i in range(iters):
        sync_mlx()
        t0 = time.perf_counter_ns()
        final = fn(hidden_mx)
        mx.eval(*final)
        sync_mlx()
        dt = (time.perf_counter_ns() - t0) / 1e6
        samples.append(dt)
        print(f"    {label} {i+1:02d}/{iters}: {dt:.3f} ms", flush=True)

    return final, stats(samples)


def benchmark_mlx_bridge(label, fn, hidden_torch, warmup, iters):
    # Conservative explicit producer/consumer synchronization.
    for _ in range(warmup + 1):
        sync_torch()
        x = mx.asarray(hidden_torch, copy=False)
        out = fn(x)
        mx.eval(*out)
        sync_mlx()
        y = torch.as_tensor(out[0])
        sync_torch()
        _ = y

    samples = []
    final = None
    for i in range(iters):
        sync_torch()
        t0 = time.perf_counter_ns()

        x = mx.asarray(hidden_torch, copy=False)
        out = fn(x)
        mx.eval(*out)
        sync_mlx()
        final = torch.as_tensor(out[0])
        sync_torch()

        dt = (time.perf_counter_ns() - t0) / 1e6
        samples.append(dt)
        print(f"    {label} {i+1:02d}/{iters}: {dt:.3f} ms", flush=True)

    return final, stats(samples)


def mlx_outputs_to_torch(out):
    mx.eval(*out)
    sync_mlx()
    hidden = torch.as_tensor(out[0])
    weights = torch.as_tensor(out[1])
    sync_torch()
    return hidden, weights


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

    stage0 = model.model.backbone.conv_encoder.model.swin.encoder.layers[0]
    targets = [
        ("stage0_block0", stage0.blocks[0]),
        ("stage0_block1", stage0.blocks[1]),
    ]

    captured: dict[str, dict[str, Any]] = {}
    originals = {}

    for name, block in targets:
        originals[name] = block.forward
        orig = block.forward

        def make_capture(tag, original_forward):
            def capture(
                self,
                hidden_states,
                input_dimensions,
                always_partition=False,
                **kwargs,
            ):
                if tag not in captured:
                    captured[tag] = {
                        "hidden_states": hidden_states.detach(),
                        "input_dimensions": (
                            int(input_dimensions[0]),
                            int(input_dimensions[1]),
                        ),
                    }
                return original_forward(
                    hidden_states,
                    input_dimensions,
                    always_partition=always_partition,
                    **kwargs,
                )
            return capture

        block.forward = types.MethodType(make_capture(name, orig), block)

    print("Capturing real stage-0 Swin block inputs...", flush=True)
    with torch.inference_mode():
        _ = model(**inputs)
        sync_torch()

    for name, block in targets:
        block.forward = originals[name]

    if set(captured) != {name for name, _ in targets}:
        raise RuntimeError(f"Capture failure: got keys {sorted(captured)}")

    result_blocks = {}

    for name, block in targets:
        print(f"\n=== {name} ===", flush=True)
        cap = captured[name]
        hidden = cap["hidden_states"]
        dims = cap["input_dimensions"]

        block.set_shift_and_window_size(dims)

        print(
            f"  hidden={tuple(hidden.shape)} dims={dims} "
            f"window={block.window_size} shift={block.shift_size} "
            f"heads={block.attention.num_attention_heads} "
            f"activation={block.attention.config.hidden_act}",
            flush=True,
        )

        print("  PyTorch/MPS reference...", flush=True)
        # Timed reference matches normal inference: no attention tensor is
        # requested/materialized as an output.
        ref_out, torch_lat = benchmark_torch(
            block, hidden, dims, args.warmup, args.iters
        )

        # Correctness-only untimed eager-attention oracle.
        #
        # HF SwinLayer always returns (hidden, attn_weights), but optimized
        # attention backends may return attn_weights=None. The eager Swin
        # attention implementation explicitly materializes and returns weights.
        # Keep the timed reference on the model's normal backend, and switch
        # only this untimed oracle pass to eager.
        original_attn_impl = block.attention.config._attn_implementation
        try:
            block.attention.config._attn_implementation = "eager"
            with torch.inference_mode():
                sync_torch()
                ref_eager_out = block(
                    hidden,
                    dims,
                    always_partition=False,
                )
                sync_torch()
        finally:
            block.attention.config._attn_implementation = original_attn_impl

        if (
            not isinstance(ref_eager_out, tuple)
            or len(ref_eager_out) < 2
            or ref_eager_out[1] is None
        ):
            raise RuntimeError(
                "Eager Swin attention did not return attention weights; "
                "cannot run the intended attention correctness audit."
            )

        sync_torch()
        mlx_block = MlxStaticSwinBlock(block, dims)
        hidden_mx = mx.asarray(hidden, copy=False)
        mx.eval(hidden_mx)
        sync_mlx()

        print("  MLX separate-QKV compiled internal...", flush=True)
        sep_out, sep_internal = benchmark_mlx_internal(
            "mlx_sep_internal",
            mlx_block.separate_compiled,
            hidden_mx,
            args.warmup,
            args.iters,
        )

        print("  MLX fused-QKV compiled internal...", flush=True)
        fused_out, fused_internal = benchmark_mlx_internal(
            "mlx_fused_internal",
            mlx_block.fused_compiled,
            hidden_mx,
            args.warmup,
            args.iters,
        )

        sep_t = mlx_outputs_to_torch(sep_out)
        fused_t = mlx_outputs_to_torch(fused_out)

        correctness = {
            "separate_qkv": {
                "hidden": compare_torch(sep_t[0], ref_out[0]),
                "attention_weights": compare_torch(
                    sep_t[1], ref_eager_out[1]
                ),
            },
            "fused_qkv": {
                "hidden": compare_torch(fused_t[0], ref_out[0]),
                "attention_weights": compare_torch(
                    fused_t[1], ref_eager_out[1]
                ),
            },
            "fused_vs_separate": {
                "hidden": compare_torch(fused_t[0], sep_t[0]),
                "attention_weights": compare_torch(fused_t[1], sep_t[1]),
            },
        }

        print("  MLX fused-QKV bridge-included...", flush=True)
        _, fused_bridge = benchmark_mlx_bridge(
            "mlx_fused_bridge",
            mlx_block.fused_compiled,
            hidden,
            args.warmup,
            args.iters,
        )

        result_blocks[name] = {
            "input_shape": list(hidden.shape),
            "input_dimensions": list(dims),
            "window_size": mlx_block.ws,
            "shift_size": mlx_block.shift,
            "padded_dimensions": [mlx_block.hp, mlx_block.wp],
            "num_windows": mlx_block.num_windows,
            "channels": mlx_block.C,
            "heads": mlx_block.H,
            "head_dim": mlx_block.HD,
            "reference_attention_backend": str(original_attn_impl),
            "attention_weight_oracle_backend": "eager",
            "latency": {
                "pytorch_mps": torch_lat,
                "mlx_separate_qkv_internal": sep_internal,
                "mlx_fused_qkv_internal": fused_internal,
                "mlx_fused_qkv_bridge_included": fused_bridge,
                "speedup_fused_internal_vs_pytorch": (
                    torch_lat["median_ms"] / fused_internal["median_ms"]
                ),
                "speedup_fused_bridge_vs_pytorch": (
                    torch_lat["median_ms"] / fused_bridge["median_ms"]
                ),
                "fused_qkv_vs_separate_internal": (
                    sep_internal["median_ms"] / fused_internal["median_ms"]
                ),
            },
            "correctness": correctness,
        }

        print(
            f"  summary: torch={torch_lat['median_ms']:.3f} ms, "
            f"fused internal={fused_internal['median_ms']:.3f} ms, "
            f"fused bridge={fused_bridge['median_ms']:.3f} ms",
            flush=True,
        )

    result = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "experiment": "0022",
        "purpose": (
            "Feasibility benchmark for fixed-resolution whole-Swin-block "
            "specialization on the two expensive stage-0 blocks."
        ),
        "model": args.model,
        "device": "mps",
        "dtype": "float32",
        "software": {
            "torch": torch.__version__,
            "transformers": transformers.__version__,
        },
        "specialization": {
            "fixed_batch": 1,
            "fixed_input_resolution": True,
            "precomputed_shift_mask": True,
            "precomputed_relative_position_bias_lookup": True,
            "whole_block_mx_compile": True,
            "fused_qkv_gemm_variant": True,
            "approximation": False,
            "retraining": False,
            "reduced_precision": False,
        },
        "blocks": result_blocks,
        "notes": [
            "The reference is the exact real stage-0 Swin block workload from Grounding DINO Tiny.",
            "Static shift masks and relative-position bias lookups are partial-evaluated because resolution and trained weights are fixed at inference.",
            "The fused-QKV variant concatenates the three trained projection matrices without changing model parameters.",
            "MLX internal timings exclude PyTorch<->MLX bridge overhead; bridge-included timings conservatively synchronize producer and consumer.",
            "A production backbone implementation would amortize the bridge across multiple consecutive Swin blocks/stages rather than bridge every block.",
            "No approximation, quantization, pruning, retraining, or reduced precision is used."
        ],
    }

    out_path = Path("results/metalground_swin_stage0_block_microbench.json")
    out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")

    print("\n=== Experiment 0022 summary ===", flush=True)
    for name, row in result_blocks.items():
        lat = row["latency"]
        print(
            f"{name:14s}: "
            f"torch {lat['pytorch_mps']['median_ms']:.3f} ms | "
            f"MLX fused internal {lat['mlx_fused_qkv_internal']['median_ms']:.3f} ms "
            f"({lat['speedup_fused_internal_vs_pytorch']:.3f}x) | "
            f"bridge {lat['mlx_fused_qkv_bridge_included']['median_ms']:.3f} ms "
            f"({lat['speedup_fused_bridge_vs_pytorch']:.3f}x)",
            flush=True,
        )
    print(f"Saved: {out_path}", flush=True)


if __name__ == "__main__":
    main()
