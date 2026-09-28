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

    af = ta.float()
    bf = tb.float()
    d = (af - bf).abs()
    return {
        "shape": list(ta.shape),
        "max_abs": float(d.max().item()),
        "mean_abs": float(d.mean().item()),
        "rmse": float(torch.sqrt(torch.mean((af - bf) ** 2)).item()),
        "allclose_1e-5": bool(torch.allclose(ta, tb, rtol=1e-5, atol=1e-5)),
        "allclose_1e-4": bool(torch.allclose(ta, tb, rtol=1e-4, atol=1e-4)),
        "finite": bool(torch.isfinite(ta).all().item()),
    }


def bench(label, fn, args, warmup: int, iters: int):
    first = fn(*args)
    mx.eval(first)
    mx.synchronize()

    for _ in range(warmup):
        y = fn(*args)
        mx.eval(y)
        mx.synchronize()

    samples = []
    final = None
    for i in range(iters):
        mx.synchronize()
        t0 = time.perf_counter_ns()
        final = fn(*args)
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
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--iters", type=int, default=30)
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

    h14 = load_module(args.exp14_helper, "metalground_exp14_helpers_exp21")
    h17 = load_module(args.exp17_helper, "metalground_exp17_helpers_exp21")
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
    print("Capturing real layer-0 workload...", flush=True)
    with torch.inference_mode():
        _ = model(**inputs)
        h14.sync()
    layer.forward = original_forward

    if not captured:
        raise RuntimeError("Failed to capture layer-0 workload.")

    torch.mps.synchronize()
    island = h17.MlxDeformableIsland(
        layer,
        activation_name=model.config.activation_function,
        spatial_shapes_list=captured["spatial_shapes_list"],
        threadgroup=args.threadgroup,
    )

    hidden = mx.asarray(captured["hidden_states"], copy=False)
    mask = mx.asarray(captured["attention_mask"], copy=False)
    pos = mx.asarray(captured["position_embeddings"], copy=False)
    ref = mx.asarray(captured["reference_points"], copy=False)
    mx.eval(hidden, mask, pos, ref)
    mx.synchronize()

    # Get the exact post-self-attention input to the FFN.
    attn_out = island.compiled_call(hidden, mask, pos, ref)
    mx.eval(*attn_out)
    mx.synchronize()
    ffn_input = attn_out[0]

    p = island.p
    B, S, D = ffn_input.shape
    FF = int(model.config.encoder_ffn_dim)

    def baseline_3d(x):
        residual = x
        y = h17.linear(x, p["fc1_w"], p["fc1_b"])
        y = island._activation(y)
        y = h17.linear(y, p["fc2_w"], p["fc2_b"])
        y = residual + y
        y = mx.fast.layer_norm(
            y, p["final_ln_w"], p["final_ln_b"], island.final_ln_eps
        )
        return y

    def flat_matmul(x):
        residual = x
        x2 = x.reshape(B * S, D)

        y = x2 @ p["fc1_w"].T
        y = y + p["fc1_b"]
        y = island._activation(y)

        y = y @ p["fc2_w"].T
        y = y + p["fc2_b"]
        y = y.reshape(B, S, D)

        y = residual + y
        y = mx.fast.layer_norm(
            y, p["final_ln_w"], p["final_ln_b"], island.final_ln_eps
        )
        return y

    def flat_addmm(x):
        residual = x
        x2 = x.reshape(B * S, D)

        # Explicitly express GEMM+bias as one operation.
        y = mx.addmm(p["fc1_b"], x2, p["fc1_w"].T)
        y = island._activation(y)
        y = mx.addmm(p["fc2_b"], y, p["fc2_w"].T)
        y = y.reshape(B, S, D)

        y = residual + y
        y = mx.fast.layer_norm(
            y, p["final_ln_w"], p["final_ln_b"], island.final_ln_eps
        )
        return y

    baseline_compiled = mx.compile(baseline_3d)
    flat_matmul_compiled = mx.compile(flat_matmul)
    flat_addmm_compiled = mx.compile(flat_addmm)

    print(
        f"FFN workload: input={tuple(ffn_input.shape)}, "
        f"hidden={FF}, flattened_rows={B*S}",
        flush=True,
    )

    print("\nCorrectness preflight...", flush=True)
    ref = baseline_compiled(ffn_input)
    mm = flat_matmul_compiled(ffn_input)
    amm = flat_addmm_compiled(ffn_input)
    mx.eval(ref, mm, amm)
    mx.synchronize()

    correctness = {
        "flat_matmul_vs_baseline": compare_mx(mm, ref),
        "flat_addmm_vs_baseline": compare_mx(amm, ref),
    }
    print(json.dumps(correctness, indent=2), flush=True)

    print("\nBenchmark baseline 3D compiled FFN...", flush=True)
    baseline_final, baseline_lat = bench(
        "baseline_3d",
        baseline_compiled,
        (ffn_input,),
        args.warmup,
        args.iters,
    )

    print("\nBenchmark flattened matmul FFN...", flush=True)
    mm_final, mm_lat = bench(
        "flat_matmul",
        flat_matmul_compiled,
        (ffn_input,),
        args.warmup,
        args.iters,
    )

    print("\nBenchmark flattened addmm FFN...", flush=True)
    amm_final, amm_lat = bench(
        "flat_addmm",
        flat_addmm_compiled,
        (ffn_input,),
        args.warmup,
        args.iters,
    )

    final_correctness = {
        "flat_matmul_vs_baseline": compare_mx(mm_final, baseline_final),
        "flat_addmm_vs_baseline": compare_mx(amm_final, baseline_final),
    }

    macs = int(B * S * D * FF * 2)

    result = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "experiment": "0021",
        "purpose": (
            "Test whether the dominant MetalGround-v2 encoder FFN can be "
            "improved with explicit 2D GEMM layout and MLX addmm fusion before "
            "investing in a custom Metal MLP kernel."
        ),
        "model": args.model,
        "device": "mps",
        "dtype": "float32",
        "workload": {
            "input_shape": list(ffn_input.shape),
            "external_dim": int(D),
            "ffn_dim": int(FF),
            "activation": model.config.activation_function,
            "analytical_two_gemm_macs": macs,
        },
        "latency_mlx_internal": {
            "baseline_3d_compiled": baseline_lat,
            "flat_matmul_compiled": mm_lat,
            "flat_addmm_compiled": amm_lat,
            "speedup_flat_matmul": (
                baseline_lat["median_ms"] / mm_lat["median_ms"]
            ),
            "speedup_flat_addmm": (
                baseline_lat["median_ms"] / amm_lat["median_ms"]
            ),
        },
        "correctness": {
            "preflight": correctness,
            "final": final_correctness,
        },
        "decision_rule": (
            "If the best exact API/layout variant improves median FFN latency "
            "by >=5%, integrate it into all six deformable islands. Otherwise "
            "treat the dense FFN as effectively GEMM-bound under current MLX "
            "primitives and move the next optimization campaign to Swin whole-block."
        ),
        "notes": [
            "All variants use the same FP32 weights and exact ReLU/LayerNorm semantics.",
            "Timings are MLX-internal and exclude PyTorch<->MLX bridge time.",
            "No approximation, quantization, pruning, retraining, or reduced precision is used.",
            "mx.addmm explicitly represents bias + matrix multiplication as one primitive."
        ],
    }

    out = Path("results/metalground_v2_ffn_impl_sweep.json")
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")

    print("\n=== Experiment 0021 summary ===", flush=True)
    print(
        f"baseline 3D: {baseline_lat['median_ms']:.3f} ms",
        flush=True,
    )
    print(
        f"flat matmul: {mm_lat['median_ms']:.3f} ms "
        f"({result['latency_mlx_internal']['speedup_flat_matmul']:.3f}x)",
        flush=True,
    )
    print(
        f"flat addmm:  {amm_lat['median_ms']:.3f} ms "
        f"({result['latency_mlx_internal']['speedup_flat_addmm']:.3f}x)",
        flush=True,
    )
    print(f"Saved: {out}", flush=True)


if __name__ == "__main__":
    main()
