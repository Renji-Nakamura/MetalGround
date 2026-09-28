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

import torch
import transformers
from PIL import Image
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor


def sync() -> None:
    torch.mps.synchronize()


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


def benchmark(block, hidden, dims, warmup: int, iters: int, label: str):
    with torch.inference_mode():
        for _ in range(warmup):
            sync()
            out = block(hidden, dims, always_partition=False)
            sync()

        samples = []
        final = None
        for i in range(iters):
            sync()
            t0 = time.perf_counter_ns()
            final = block(hidden, dims, always_partition=False)
            sync()
            dt = (time.perf_counter_ns() - t0) / 1e6
            samples.append(dt)
            print(f"    {label} {i+1:02d}/{iters}: {dt:.3f} ms", flush=True)

    return final, stats(samples)


def precompute_static_geometry(block, dims):
    block.set_shift_and_window_size(dims)
    h, w = int(dims[0]), int(dims[1])
    ws = int(block.window_size)

    pad_right = (ws - w % ws) % ws
    pad_bottom = (ws - h % ws) % ws
    hp = h + pad_bottom
    wp = w + pad_right

    with torch.inference_mode():
        rel_bias = block.attention.relative_position_bias().detach()
        shift_mask = block.get_attn_mask(
            hp,
            wp,
            dtype=rel_bias.dtype,
            device=rel_bias.device,
        )
        if shift_mask is not None:
            shift_mask = shift_mask.detach()
        sync()

    return {
        "relative_position_bias": rel_bias,
        "shift_mask": shift_mask,
        "padded_dimensions": [hp, wp],
    }


def run_variant(
    block,
    hidden,
    dims,
    static,
    *,
    cache_rel_bias: bool,
    cache_shift_mask: bool,
    warmup: int,
    iters: int,
    label: str,
):
    rel_module = block.attention.relative_position_bias
    orig_rel_forward = rel_module.forward
    orig_mask = block.get_attn_mask

    try:
        if cache_rel_bias:
            cached_rel = static["relative_position_bias"]

            def rel_forward(self):
                return cached_rel

            # `relative_position_bias` is an nn.Module child. Replacing the
            # parent attribute with a function violates PyTorch's module
            # registry; patch only the child module's forward method.
            rel_module.forward = types.MethodType(rel_forward, rel_module)

        if cache_shift_mask and static["shift_mask"] is not None:
            cached_mask = static["shift_mask"]

            def mask_forward(self, height, width, dtype, device):
                # This experiment specializes to the captured fixed resolution.
                return cached_mask

            block.get_attn_mask = types.MethodType(mask_forward, block)

        return benchmark(
            block, hidden, dims, warmup, iters, label
        )
    finally:
        rel_module.forward = orig_rel_forward
        block.get_attn_mask = orig_mask


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="IDEA-Research/grounding-dino-tiny")
    ap.add_argument("--image", type=Path, default=Path("assets/input.jpg"))
    ap.add_argument("--prompt", nargs="+", default=["a cat", "a dog"])
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--iters", type=int, default=30)
    args = ap.parse_args()

    if not torch.backends.mps.is_available():
        raise SystemExit("MPS unavailable.")

    print("Loading model...", flush=True)
    processor = AutoProcessor.from_pretrained(args.model)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(args.model)
    model.eval().to("mps")
    sync()

    image = Image.open(args.image).convert("RGB")
    text_labels = [args.prompt]
    cpu_inputs = processor(images=image, text=text_labels, return_tensors="pt")
    inputs = {k: v.to("mps") for k, v in cpu_inputs.items()}
    sync()

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

    print("Capturing real stage-0 inputs...", flush=True)
    with torch.inference_mode():
        _ = model(**inputs)
        sync()

    for name, block in targets:
        block.forward = originals[name]

    if set(captured) != {name for name, _ in targets}:
        raise RuntimeError(f"Capture failure: {sorted(captured)}")

    result_blocks = {}

    for name, block in targets:
        print(f"\n=== {name} ===", flush=True)
        cap = captured[name]
        hidden = cap["hidden_states"]
        dims = cap["input_dimensions"]

        static = precompute_static_geometry(block, dims)
        has_shift_mask = static["shift_mask"] is not None

        print(
            f"  hidden={tuple(hidden.shape)} dims={dims} "
            f"window={block.window_size} shift={block.shift_size} "
            f"has_shift_mask={has_shift_mask}",
            flush=True,
        )

        print("  baseline...", flush=True)
        baseline_out, baseline_lat = run_variant(
            block,
            hidden,
            dims,
            static,
            cache_rel_bias=False,
            cache_shift_mask=False,
            warmup=args.warmup,
            iters=args.iters,
            label="baseline",
        )

        print("  cache relative-position bias...", flush=True)
        rel_out, rel_lat = run_variant(
            block,
            hidden,
            dims,
            static,
            cache_rel_bias=True,
            cache_shift_mask=False,
            warmup=args.warmup,
            iters=args.iters,
            label="cache_rel",
        )

        mask_out = None
        mask_lat = None
        both_out = None
        both_lat = None

        if has_shift_mask:
            print("  cache shift mask...", flush=True)
            mask_out, mask_lat = run_variant(
                block,
                hidden,
                dims,
                static,
                cache_rel_bias=False,
                cache_shift_mask=True,
                warmup=args.warmup,
                iters=args.iters,
                label="cache_mask",
            )

            print("  cache both...", flush=True)
            both_out, both_lat = run_variant(
                block,
                hidden,
                dims,
                static,
                cache_rel_bias=True,
                cache_shift_mask=True,
                warmup=args.warmup,
                iters=args.iters,
                label="cache_both",
            )

        variants = {
            "baseline": baseline_lat,
            "cache_relative_position_bias": rel_lat,
        }
        correctness = {
            "cache_relative_position_bias": compare(rel_out[0], baseline_out[0])
        }

        if has_shift_mask:
            variants["cache_shift_mask"] = mask_lat
            variants["cache_both"] = both_lat
            correctness["cache_shift_mask"] = compare(
                mask_out[0], baseline_out[0]
            )
            correctness["cache_both"] = compare(
                both_out[0], baseline_out[0]
            )

        derived = {
            "speedup_cache_relative_position_bias": (
                baseline_lat["median_ms"] / rel_lat["median_ms"]
            ),
        }
        if has_shift_mask:
            derived.update(
                {
                    "speedup_cache_shift_mask": (
                        baseline_lat["median_ms"] / mask_lat["median_ms"]
                    ),
                    "speedup_cache_both": (
                        baseline_lat["median_ms"] / both_lat["median_ms"]
                    ),
                    "reduction_cache_both_ms": (
                        baseline_lat["median_ms"] - both_lat["median_ms"]
                    ),
                }
            )

        result_blocks[name] = {
            "input_shape": list(hidden.shape),
            "input_dimensions": list(dims),
            "window_size": int(block.window_size),
            "shift_size": int(block.shift_size),
            "padded_dimensions": static["padded_dimensions"],
            "has_shift_mask": has_shift_mask,
            "latency": variants,
            "derived": derived,
            "correctness": correctness,
        }

    result = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "experiment": "0023",
        "purpose": (
            "Causal ablation of static Swin geometry specialization in the "
            "original PyTorch/MPS SDPA path: cache relative-position bias lookup "
            "and shifted-window attention mask separately."
        ),
        "model": args.model,
        "device": "mps",
        "dtype": "float32",
        "software": {
            "torch": torch.__version__,
            "transformers": transformers.__version__,
        },
        "blocks": result_blocks,
        "notes": [
            "All timed variants remain in the original PyTorch/MPS Swin path and use the model's normal attention backend.",
            "Only tensors that are deterministic functions of fixed inference geometry and trained weights are precomputed.",
            "The shift-mask cache is specialized to the captured fixed input resolution.",
            "No approximation, retraining, pruning, quantization, or reduced precision is used.",
            "If shift-mask caching recovers most of Experiment 0022's shifted-block gain, backend migration is unnecessary for this optimization."
        ],
    }

    out = Path("results/metalground_swin_static_cache_ablation.json")
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")

    print("\n=== Experiment 0023 summary ===", flush=True)
    for name, row in result_blocks.items():
        base = row["latency"]["baseline"]["median_ms"]
        print(f"{name}: baseline {base:.3f} ms", flush=True)
        print(
            f"  cache rel: "
            f"{row['latency']['cache_relative_position_bias']['median_ms']:.3f} ms "
            f"({row['derived']['speedup_cache_relative_position_bias']:.3f}x)",
            flush=True,
        )
        if row["has_shift_mask"]:
            print(
                f"  cache mask: "
                f"{row['latency']['cache_shift_mask']['median_ms']:.3f} ms "
                f"({row['derived']['speedup_cache_shift_mask']:.3f}x)",
                flush=True,
            )
            print(
                f"  cache both: "
                f"{row['latency']['cache_both']['median_ms']:.3f} ms "
                f"({row['derived']['speedup_cache_both']:.3f}x)",
                flush=True,
            )

    print(f"Saved: {out}", flush=True)


if __name__ == "__main__":
    main()
