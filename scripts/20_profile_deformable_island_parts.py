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

import mlx.core as mx
import torch
from PIL import Image
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

from metalground.msda_metal_v0 import msda_metal_v0


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


def compare_mx(a: mx.array, b: mx.array) -> dict:
    mx.eval(a, b)
    mx.synchronize()
    ta = torch.as_tensor(a)
    tb = torch.as_tensor(b)
    torch.mps.synchronize()
    d = (ta.float() - tb.float()).abs()
    return {
        "shape": list(ta.shape),
        "max_abs": float(d.max().item()),
        "mean_abs": float(d.mean().item()),
        "rmse": float(torch.sqrt(torch.mean((ta.float() - tb.float()) ** 2)).item()),
        "allclose_1e-5": bool(torch.allclose(ta, tb, rtol=1e-5, atol=1e-5)),
        "allclose_1e-4": bool(torch.allclose(ta, tb, rtol=1e-4, atol=1e-4)),
    }


def bench_mlx(label, fn, args, warmup: int, iters: int):
    # First compile/JIT execution excluded.
    first = fn(*args)
    if isinstance(first, tuple):
        mx.eval(*first)
    else:
        mx.eval(first)
    mx.synchronize()

    for _ in range(warmup):
        out = fn(*args)
        if isinstance(out, tuple):
            mx.eval(*out)
        else:
            mx.eval(out)
        mx.synchronize()

    samples = []
    final = None
    for i in range(iters):
        mx.synchronize()
        t0 = time.perf_counter_ns()
        final = fn(*args)
        if isinstance(final, tuple):
            mx.eval(*final)
        else:
            mx.eval(final)
        mx.synchronize()
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
        "--exp14-helper",
        type=Path,
        default=Path("scripts/14_full_model_fusion_algebraic.py"),
    )
    ap.add_argument(
        "--exp17-helper",
        type=Path,
        default=Path("scripts/17_bench_deformable_mlx_island.py"),
    )
    args = ap.parse_args()

    for p in (args.exp14_helper, args.exp17_helper):
        if not p.exists():
            raise SystemExit(f"Missing helper script: {p}")
    if not torch.backends.mps.is_available():
        raise SystemExit("MPS unavailable.")

    h14 = load_module(args.exp14_helper, "metalground_exp14_helpers_exp20")
    h17 = load_module(args.exp17_helper, "metalground_exp17_helpers_exp20")
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

    # Build the same pre-deformable trajectory as MetalGround v2 layer 0.
    fusion_items = h14.collect_fusion_specializers(model)
    h14.patch_msda(model, args.threadgroup)
    h14.set_fusion_mode(fusion_items, "fully_folded")

    layer = model.model.encoder.layers[0].deformable_layer
    original_forward = layer.forward
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
            captured.update(
                {
                    "hidden_states": hidden_states.detach(),
                    "attention_mask": attention_mask.detach(),
                    "position_embeddings": position_embeddings.detach(),
                    "reference_points": reference_points.detach(),
                    "spatial_shapes_list": [
                        (int(h), int(w)) for h, w in spatial_shapes_list
                    ],
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
    print("Capturing real layer-0 input...", flush=True)
    with torch.inference_mode():
        _ = model(**inputs)
        h14.sync()
    layer.forward = original_forward

    if not captured:
        raise RuntimeError("Failed to capture layer-0 workload.")

    print(
        f"hidden={tuple(captured['hidden_states'].shape)} "
        f"spatial={captured['spatial_shapes_list']}",
        flush=True,
    )

    torch.mps.synchronize()
    island = h17.MlxDeformableIsland(
        layer,
        activation_name=model.config.activation_function,
        spatial_shapes_list=captured["spatial_shapes_list"],
        threadgroup=args.threadgroup,
    )
    if island.compiled is None:
        raise RuntimeError(f"Full island compile unavailable: {island.compiled_error}")

    # One explicit entry boundary; component timings below are MLX-internal only.
    hidden = mx.asarray(captured["hidden_states"], copy=False)
    mask = mx.asarray(captured["attention_mask"], copy=False)
    pos = mx.asarray(captured["position_embeddings"], copy=False)
    ref = mx.asarray(captured["reference_points"], copy=False)
    mx.eval(hidden, mask, pos, ref)
    mx.synchronize()

    p = island.p

    def attn_half_impl(hidden, mask, pos, ref):
        B, S, _ = hidden.shape
        query = hidden + pos

        value = h17.linear(hidden, p["value_w"], p["value_b"])
        value = mx.where(mask[..., None], value, 0.0)
        value = value.reshape(B, S, island.H, island.HD)

        offsets = h17.linear(query, p["sampling_w"], p["sampling_b"])
        offsets = offsets.reshape(
            B, S, island.H, island.NL, island.NP, 2
        )

        weights = h17.linear(query, p["attn_w_w"], p["attn_w_b"])
        weights = weights.reshape(B, S, island.H, island.NL * island.NP)
        weights = mx.softmax(weights, axis=-1)
        weights = weights.reshape(B, S, island.H, island.NL, island.NP)

        locations = (
            ref[:, :, None, :, None, :]
            + offsets
            / island.offset_normalizer[None, None, None, :, None, :]
        )

        attn_out = msda_metal_v0(
            value,
            island.spatial_shapes,
            island.level_start,
            locations,
            weights,
            threadgroup_size=island.threadgroup,
        )
        attn_out = h17.linear(attn_out, p["output_w"], p["output_b"])
        x = hidden + attn_out
        x = mx.fast.layer_norm(
            x, p["self_ln_w"], p["self_ln_b"], island.self_ln_eps
        )
        return x, weights

    def ffn_half_impl(x):
        residual = x
        y = h17.linear(x, p["fc1_w"], p["fc1_b"])
        y = island._activation(y)
        y = h17.linear(y, p["fc2_w"], p["fc2_b"])
        y = residual + y
        y = mx.fast.layer_norm(
            y, p["final_ln_w"], p["final_ln_b"], island.final_ln_eps
        )
        return y

    def core_input_impl(hidden, mask, pos, ref):
        B, S, _ = hidden.shape
        query = hidden + pos

        value = h17.linear(hidden, p["value_w"], p["value_b"])
        value = mx.where(mask[..., None], value, 0.0)
        value = value.reshape(B, S, island.H, island.HD)

        offsets = h17.linear(query, p["sampling_w"], p["sampling_b"])
        offsets = offsets.reshape(
            B, S, island.H, island.NL, island.NP, 2
        )

        weights = h17.linear(query, p["attn_w_w"], p["attn_w_b"])
        weights = weights.reshape(B, S, island.H, island.NL * island.NP)
        weights = mx.softmax(weights, axis=-1)
        weights = weights.reshape(B, S, island.H, island.NL, island.NP)

        locations = (
            ref[:, :, None, :, None, :]
            + offsets
            / island.offset_normalizer[None, None, None, :, None, :]
        )
        return value, locations, weights

    full_compiled = island.compiled_call
    attn_compiled = mx.compile(attn_half_impl)
    ffn_compiled = mx.compile(ffn_half_impl)
    core_input_compiled = mx.compile(core_input_impl)

    print("\nPreparing real FFN input and MSDA core inputs...", flush=True)
    attn_pre = attn_compiled(hidden, mask, pos, ref)
    mx.eval(*attn_pre)
    mx.synchronize()
    ffn_input = attn_pre[0]

    core_inputs = core_input_compiled(hidden, mask, pos, ref)
    mx.eval(*core_inputs)
    mx.synchronize()

    def core_only(value, locations, weights):
        return msda_metal_v0(
            value,
            island.spatial_shapes,
            island.level_start,
            locations,
            weights,
            threadgroup_size=island.threadgroup,
        )

    core_compiled = mx.compile(core_only)

    # Correctness: split composition should match the full compiled island.
    print("\nCorrectness: split composition vs full island...", flush=True)
    full_out = full_compiled(hidden, mask, pos, ref)
    split_attn = attn_compiled(hidden, mask, pos, ref)
    split_final = ffn_compiled(split_attn[0])
    mx.eval(*full_out, *split_attn, split_final)
    mx.synchronize()

    correctness = {
        "split_final_vs_full_hidden": compare_mx(split_final, full_out[0]),
        "split_attention_weights_vs_full": compare_mx(
            split_attn[1], full_out[1]
        ),
    }
    print(json.dumps(correctness, indent=2), flush=True)

    print("\nBenchmark full compiled island (MLX internal only)...", flush=True)
    _, full_lat = bench_mlx(
        "full_island",
        full_compiled,
        (hidden, mask, pos, ref),
        args.warmup,
        args.iters,
    )

    print("\nBenchmark compiled self-attention half...", flush=True)
    _, attn_lat = bench_mlx(
        "attn_half",
        attn_compiled,
        (hidden, mask, pos, ref),
        args.warmup,
        args.iters,
    )

    print("\nBenchmark compiled Metal MSDA core only...", flush=True)
    _, core_lat = bench_mlx(
        "msda_core",
        core_compiled,
        core_inputs,
        args.warmup,
        args.iters,
    )

    print("\nBenchmark compiled FFN half...", flush=True)
    _, ffn_lat = bench_mlx(
        "ffn_half",
        ffn_compiled,
        (ffn_input,),
        args.warmup,
        args.iters,
    )

    result = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "experiment": "0020",
        "purpose": (
            "Decompose the MetalGround-v2 compiled encoder deformable island "
            "into self-attention/MSDA and FFN halves using the exact real "
            "layer-0 workload."
        ),
        "model": args.model,
        "device": "mps",
        "dtype": "float32",
        "workload": {
            "layer": "model.encoder.layers.0.deformable_layer",
            "hidden_shape": list(captured["hidden_states"].shape),
            "spatial_shapes_list": captured["spatial_shapes_list"],
            "ffn_dim": int(model.config.encoder_ffn_dim),
        },
        "latency_mlx_internal": {
            "full_compiled_island": full_lat,
            "compiled_self_attention_half": attn_lat,
            "compiled_msda_core_only": core_lat,
            "compiled_ffn_half": ffn_lat,
            "diagnostic_sum_halves_median_ms": (
                attn_lat["median_ms"] + ffn_lat["median_ms"]
            ),
            "full_minus_halves_sum_median_ms": (
                full_lat["median_ms"]
                - attn_lat["median_ms"]
                - ffn_lat["median_ms"]
            ),
            "self_attention_minus_core_median_ms": (
                attn_lat["median_ms"] - core_lat["median_ms"]
            ),
        },
        "correctness": correctness,
        "notes": [
            "Component timings are MLX-internal and exclude PyTorch<->MLX entry/exit bridge time.",
            "Each component is separately mx.compile'd, so sums are diagnostic and not an additive GPU timeline.",
            "The full compiled island is measured in the same MLX-internal timing regime for comparison.",
            "The MSDA core benchmark uses precomputed real value/location/weight tensors.",
            "No retraining, approximation, pruning, or reduced precision is used."
        ],
    }

    out = Path("results/metalground_v2_deformable_island_breakdown.json")
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")

    print("\n=== Experiment 0020 summary ===", flush=True)
    print(f"full island: {full_lat['median_ms']:.3f} ms", flush=True)
    print(f"attn half:   {attn_lat['median_ms']:.3f} ms", flush=True)
    print(f"MSDA core:   {core_lat['median_ms']:.3f} ms", flush=True)
    print(f"FFN half:    {ffn_lat['median_ms']:.3f} ms", flush=True)
    print(f"Saved: {out}", flush=True)


if __name__ == "__main__":
    main()
