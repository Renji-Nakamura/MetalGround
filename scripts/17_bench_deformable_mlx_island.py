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
import numpy as np
import torch
import transformers
from PIL import Image
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

from metalground.msda_metal_v0 import msda_metal_v0


def sync_torch() -> None:
    torch.mps.synchronize()


def sync_mlx() -> None:
    mx.synchronize()


def load_helpers(path: Path):
    name = "metalground_exp14_helpers_exp17"
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


def compare(a: torch.Tensor, b: torch.Tensor) -> dict[str, Any]:
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


def linear(x: mx.array, weight: mx.array, bias: mx.array | None) -> mx.array:
    y = x @ weight.T
    if bias is not None:
        y = y + bias
    return y


class MlxDeformableIsland:
    """
    Correctness-first MLX island for one GroundingDinoDeformableLayer.

    The island contains:
      - MSDA sampling-offset / attention-weight / value projections
      - MetalGround fused MSDA v0 core
      - MSDA output projection
      - self-attention residual + LayerNorm
      - encoder FFN fc1 -> activation -> fc2
      - FFN residual + final LayerNorm

    PyTorch->MLX synchronization occurs once at island entry and MLX->PyTorch
    synchronization once at island exit.
    """

    def __init__(
        self,
        layer,
        *,
        activation_name: str,
        spatial_shapes_list: list[tuple[int, int]],
        threadgroup: int,
    ):
        self.layer = layer
        self.activation_name = activation_name
        self.threadgroup = int(threadgroup)

        sa = layer.self_attn
        self.H = int(sa.n_heads)
        self.NL = int(sa.n_levels)
        self.NP = int(sa.n_points)
        self.DMODEL = int(sa.d_model)
        self.HD = self.DMODEL // self.H

        self.self_ln_eps = float(layer.self_attn_layer_norm.eps)
        self.final_ln_eps = float(layer.final_layer_norm.eps)

        # PyTorch 2.12+ ordinary MPS tensors are shared-storage DLPack inputs.
        # Producer synchronization is performed before constructing this object.
        self.p = {}

        def import_linear(prefix: str, lin):
            self.p[prefix + "_w"] = mx.asarray(lin.weight, copy=False)
            self.p[prefix + "_b"] = (
                None if lin.bias is None else mx.asarray(lin.bias, copy=False)
            )

        import_linear("sampling", sa.sampling_offsets)
        import_linear("attn_w", sa.attention_weights)
        import_linear("value", sa.value_proj)
        import_linear("output", sa.output_proj)
        import_linear("fc1", layer.fc1)
        import_linear("fc2", layer.fc2)

        self.p["self_ln_w"] = mx.asarray(
            layer.self_attn_layer_norm.weight, copy=False
        )
        self.p["self_ln_b"] = mx.asarray(
            layer.self_attn_layer_norm.bias, copy=False
        )
        self.p["final_ln_w"] = mx.asarray(
            layer.final_layer_norm.weight, copy=False
        )
        self.p["final_ln_b"] = mx.asarray(
            layer.final_layer_norm.bias, copy=False
        )

        shapes = [(int(h), int(w)) for h, w in spatial_shapes_list]
        starts = []
        acc = 0
        for h, w in shapes:
            starts.append(acc)
            acc += h * w

        self.spatial_shapes = mx.array(
            np.asarray(shapes, dtype=np.int32), dtype=mx.int32
        )
        self.level_start = mx.array(
            np.asarray(starts, dtype=np.int32), dtype=mx.int32
        )
        self.offset_normalizer = mx.array(
            np.asarray([[w, h] for h, w in shapes], dtype=np.float32),
            dtype=mx.float32,
        )

        arrays = [self.spatial_shapes, self.level_start, self.offset_normalizer]
        for v in self.p.values():
            if v is not None:
                arrays.append(v)
        mx.eval(*arrays)
        sync_mlx()

        self.compiled = None
        self.compiled_error = None
        try:
            self.compiled = mx.compile(self._forward_impl)
        except Exception as exc:
            self.compiled_error = f"{type(exc).__name__}: {exc}"

    def _activation(self, x: mx.array) -> mx.array:
        name = self.activation_name.lower()
        if name == "relu":
            return mx.maximum(x, 0.0)
        if name in ("silu", "swish"):
            return x * mx.sigmoid(x)
        if name == "gelu":
            return 0.5 * x * (1.0 + mx.erf(x / math.sqrt(2.0)))
        if name in ("gelu_new", "gelu_fast"):
            c = math.sqrt(2.0 / math.pi)
            return 0.5 * x * (
                1.0 + mx.tanh(c * (x + 0.044715 * x * x * x))
            )
        raise ValueError(f"Unsupported activation: {self.activation_name}")

    def _forward_impl(
        self,
        hidden: mx.array,
        attention_mask: mx.array,
        position: mx.array,
        reference_points: mx.array,
    ):
        B, S, _ = hidden.shape

        # GroundingDinoMultiscaleDeformableAttention:
        # offsets/weights use hidden + position, values use the original hidden.
        query = hidden + position

        value = linear(hidden, self.p["value_w"], self.p["value_b"])
        value = mx.where(attention_mask[..., None], value, 0.0)
        value = value.reshape(B, S, self.H, self.HD)

        offsets = linear(query, self.p["sampling_w"], self.p["sampling_b"])
        offsets = offsets.reshape(B, S, self.H, self.NL, self.NP, 2)

        weights = linear(query, self.p["attn_w_w"], self.p["attn_w_b"])
        weights = weights.reshape(B, S, self.H, self.NL * self.NP)
        weights = mx.softmax(weights, axis=-1)
        weights = weights.reshape(B, S, self.H, self.NL, self.NP)

        if reference_points.shape[-1] == 2:
            locations = (
                reference_points[:, :, None, :, None, :]
                + offsets
                / self.offset_normalizer[None, None, None, :, None, :]
            )
        elif reference_points.shape[-1] == 4:
            locations = (
                reference_points[:, :, None, :, None, :2]
                + offsets
                / float(self.NP)
                * reference_points[:, :, None, :, None, 2:]
                * 0.5
            )
        else:
            raise ValueError(
                f"Unsupported reference-point shape: {reference_points.shape}"
            )

        attn_out = msda_metal_v0(
            value,
            self.spatial_shapes,
            self.level_start,
            locations,
            weights,
            threadgroup_size=self.threadgroup,
        )

        attn_out = linear(
            attn_out, self.p["output_w"], self.p["output_b"]
        )

        x = hidden + attn_out
        x = mx.fast.layer_norm(
            x,
            self.p["self_ln_w"],
            self.p["self_ln_b"],
            self.self_ln_eps,
        )

        residual = x
        x = linear(x, self.p["fc1_w"], self.p["fc1_b"])
        x = self._activation(x)
        x = linear(x, self.p["fc2_w"], self.p["fc2_b"])
        x = residual + x
        x = mx.fast.layer_norm(
            x,
            self.p["final_ln_w"],
            self.p["final_ln_b"],
            self.final_ln_eps,
        )

        return x, weights

    def eager(self, hidden, attention_mask, position, reference_points):
        return self._forward_impl(
            hidden, attention_mask, position, reference_points
        )

    def compiled_call(
        self, hidden, attention_mask, position, reference_points
    ):
        if self.compiled is None:
            raise RuntimeError(self.compiled_error or "mx.compile unavailable")
        return self.compiled(
            hidden, attention_mask, position, reference_points
        )


def to_mlx_inputs(captured):
    # DLPack conversion does not synchronize pending Metal work.
    sync_torch()

    hidden = mx.asarray(captured["hidden_states"], copy=False)
    mask = mx.asarray(captured["attention_mask"], copy=False)
    pos = mx.asarray(captured["position_embeddings"], copy=False)
    ref = mx.asarray(captured["reference_points"], copy=False)
    return hidden, mask, pos, ref


def mlx_to_torch(outputs):
    hidden, weights = outputs
    mx.eval(hidden, weights)
    sync_mlx()
    th = torch.as_tensor(hidden)
    tw = torch.as_tensor(weights)
    sync_torch()
    return th, tw


def benchmark_reference(layer, captured, warmup: int, iters: int):
    kwargs = dict(captured)

    with torch.inference_mode():
        for _ in range(warmup):
            sync_torch()
            out = layer(**kwargs)
            sync_torch()

        samples = []
        final = None
        for i in range(iters):
            sync_torch()
            t0 = time.perf_counter_ns()
            final = layer(**kwargs)
            sync_torch()
            dt = (time.perf_counter_ns() - t0) / 1e6
            samples.append(dt)
            print(f"  hybrid {i+1:02d}/{iters}: {dt:.3f} ms", flush=True)

    return final, stats(samples)


def benchmark_island(
    island: MlxDeformableIsland,
    captured,
    *,
    compiled: bool,
    warmup: int,
    iters: int,
):
    label = "mlx_compiled_island" if compiled else "mlx_eager_island"
    call = island.compiled_call if compiled else island.eager

    # First execution / compile is excluded.
    inputs = to_mlx_inputs(captured)
    first = call(*inputs)
    _ = mlx_to_torch(first)

    for _ in range(warmup):
        inputs = to_mlx_inputs(captured)
        out = call(*inputs)
        _ = mlx_to_torch(out)

    samples = []
    final = None
    for i in range(iters):
        sync_torch()
        t0 = time.perf_counter_ns()

        inputs = to_mlx_inputs(captured)
        out = call(*inputs)
        final = mlx_to_torch(out)

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
    ap.add_argument("--threadgroup", type=int, default=256)
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

    mx.set_default_device(mx.gpu)
    h = load_helpers(args.helper_script)

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

    # Reproduce MetalGround v1 before capturing the real layer-0 workload.
    fusion_items = h.collect_fusion_specializers(model)
    h.patch_msda(model, args.threadgroup)
    h.set_fusion_mode(fusion_items, "fully_folded")

    layer = model.model.encoder.layers[0].deformable_layer
    original_forward = layer.forward
    captured: dict[str, Any] = {}

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
            captured.update(
                {
                    "hidden_states": hidden_states.detach(),
                    "attention_mask": attention_mask.detach(),
                    "position_embeddings": position_embeddings.detach(),
                    "reference_points": reference_points.detach(),
                    "spatial_shapes": spatial_shapes.detach(),
                    "spatial_shapes_list": [
                        (int(hh), int(ww))
                        for hh, ww in spatial_shapes_list
                    ],
                    "level_start_index": level_start_index.detach(),
                    "output_attentions": output_attentions,
                }
            )

        return original_forward(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_embeddings=position_embeddings,
            reference_points=reference_points,
            spatial_shapes=spatial_shapes,
            spatial_shapes_list=spatial_shapes_list,
            level_start_index=level_start_index,
            output_attentions=output_attentions,
        )

    layer.forward = types.MethodType(capture_forward, layer)

    print("Capturing real MetalGround-v1 layer-0 deformable workload...", flush=True)
    with torch.inference_mode():
        _ = model(**inputs)
        sync_torch()

    layer.forward = original_forward

    if not captured:
        raise RuntimeError("Failed to capture deformable-layer inputs.")

    print(
        f"hidden={tuple(captured['hidden_states'].shape)} "
        f"reference_points={tuple(captured['reference_points'].shape)} "
        f"activation={model.config.activation_function}",
        flush=True,
    )
    print(f"spatial={captured['spatial_shapes_list']}", flush=True)

    # Reference output on the exact captured trajectory.
    print("\nReference hybrid layer output...", flush=True)
    with torch.inference_mode():
        ref = original_forward(**captured)
        sync_torch()

    # Sync before zero-copy parameter import.
    sync_torch()
    island = MlxDeformableIsland(
        layer,
        activation_name=model.config.activation_function,
        spatial_shapes_list=captured["spatial_shapes_list"],
        threadgroup=args.threadgroup,
    )

    # Preflight eager.
    print("\nMLX island eager correctness preflight...", flush=True)
    eager_out = mlx_to_torch(island.eager(*to_mlx_inputs(captured)))
    eager_corr = {
        "hidden_states": compare(eager_out[0], ref[0]),
        "attention_weights": compare(eager_out[1], ref[1]),
    }
    print(json.dumps(eager_corr, indent=2), flush=True)

    compiled_supported = island.compiled is not None
    compiled_corr = None
    if compiled_supported:
        try:
            print("\nMLX island compiled correctness preflight...", flush=True)
            compiled_out = mlx_to_torch(
                island.compiled_call(*to_mlx_inputs(captured))
            )
            compiled_corr = {
                "hidden_states": compare(compiled_out[0], ref[0]),
                "attention_weights": compare(compiled_out[1], ref[1]),
            }
            print(json.dumps(compiled_corr, indent=2), flush=True)
        except Exception as exc:
            compiled_supported = False
            island.compiled_error = f"{type(exc).__name__}: {exc}"
            print(
                f"mx.compile path unavailable; continuing eager only: "
                f"{island.compiled_error}",
                flush=True,
            )

    print("\nBenchmark current MetalGround-v1 hybrid deformable layer...", flush=True)
    hybrid_out, hybrid_lat = benchmark_reference(
        original_forward, captured, args.warmup, args.iters
    )

    print("\nBenchmark MLX eager deformable island...", flush=True)
    eager_final, eager_lat = benchmark_island(
        island,
        captured,
        compiled=False,
        warmup=args.warmup,
        iters=args.iters,
    )
    eager_final_corr = {
        "hidden_states": compare(eager_final[0], hybrid_out[0]),
        "attention_weights": compare(eager_final[1], hybrid_out[1]),
    }

    compiled_lat = None
    compiled_final_corr = None
    if compiled_supported:
        print("\nBenchmark MLX compiled deformable island...", flush=True)
        compiled_final, compiled_lat = benchmark_island(
            island,
            captured,
            compiled=True,
            warmup=args.warmup,
            iters=args.iters,
        )
        compiled_final_corr = {
            "hidden_states": compare(compiled_final[0], hybrid_out[0]),
            "attention_weights": compare(compiled_final[1], hybrid_out[1]),
        }

    result = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "experiment": "0017",
        "purpose": (
            "Test a larger MLX execution island for the encoder deformable layer: "
            "MSDA projections + fused Metal MSDA + residual/norm + FFN, using one "
            "PyTorch->MLX and one MLX->PyTorch synchronization boundary."
        ),
        "model": args.model,
        "device": "mps",
        "dtype": "float32",
        "software": {
            "torch": torch.__version__,
            "transformers": transformers.__version__,
        },
        "workload": {
            "layer": "model.encoder.layers.0.deformable_layer",
            "hidden_shape": list(captured["hidden_states"].shape),
            "reference_points_shape": list(
                captured["reference_points"].shape
            ),
            "spatial_shapes_list": captured["spatial_shapes_list"],
            "activation": model.config.activation_function,
            "encoder_ffn_dim": int(model.config.encoder_ffn_dim),
        },
        "latency": {
            "current_hybrid_v1": hybrid_lat,
            "mlx_eager_island": eager_lat,
            "speedup_eager_vs_hybrid": (
                hybrid_lat["median_ms"] / eager_lat["median_ms"]
            ),
            "mlx_compiled_island": compiled_lat,
            "speedup_compiled_vs_hybrid": (
                None
                if compiled_lat is None
                else hybrid_lat["median_ms"] / compiled_lat["median_ms"]
            ),
        },
        "correctness": {
            "eager_preflight_vs_hybrid": eager_corr,
            "eager_final_vs_hybrid": eager_final_corr,
            "compiled_supported": compiled_supported,
            "compiled_error": island.compiled_error,
            "compiled_preflight_vs_hybrid": compiled_corr,
            "compiled_final_vs_hybrid": compiled_final_corr,
        },
        "notes": [
            "The current-hybrid reference already contains MetalGround MSDA v0.",
            "The MLX island keeps the same Metal MSDA v0 sampling core but moves its surrounding projections, residuals, layer norms, and encoder FFN into MLX.",
            "DLPack conversion is zero-copy when supported, but producer/consumer synchronization remains explicit because conversion itself does not synchronize pending Metal work.",
            "Dropout is inactive because the model is in eval mode.",
            "No retraining, pruning, approximation, or reduced precision is used.",
            "This is an isolated real layer workload; full-model integration is a later experiment."
        ],
    }

    out = Path("results/metalground_deformable_island_microbench.json")
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")

    print("\n=== Experiment 0017 summary ===", flush=True)
    print(
        f"hybrid median:    {hybrid_lat['median_ms']:.3f} ms",
        flush=True,
    )
    print(
        f"MLX eager median: {eager_lat['median_ms']:.3f} ms "
        f"({result['latency']['speedup_eager_vs_hybrid']:.3f}x)",
        flush=True,
    )
    if compiled_lat is not None:
        print(
            f"MLX compiled:     {compiled_lat['median_ms']:.3f} ms "
            f"({result['latency']['speedup_compiled_vs_hybrid']:.3f}x)",
            flush=True,
        )
    print(f"Saved: {out}", flush=True)


if __name__ == "__main__":
    main()
