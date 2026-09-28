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


class MlxFusionDeformableWideIsland:
    """
    Layer-0 feasibility island:
      PyTorch producer
        -> one Torch->MLX boundary
        -> fusion LayerNorms
        -> algebraically fully-folded bidirectional fusion
        -> fusion layer-scale residuals
        -> deformable attention + FFN island
        -> one MLX->Torch boundary

    Text enhancer intentionally remains outside this island.

    GroundingDinoFusionLayer's drop_path is inactive in model.eval(), so the
    inference expression is:
        norm_v + vision_param * delta_v
        norm_t + text_param   * delta_t
    """

    def __init__(self, fusion_layer, fusion_core, deform_core):
        if fusion_layer.training:
            raise ValueError("Wide island requires model.eval().")
        if fusion_layer.drop_path.training:
            raise ValueError("Fusion drop_path must be in eval mode.")

        self.fusion_layer = fusion_layer
        self.fusion_core = fusion_core
        self.deform_core = deform_core

        self.v_ln_w = mx.asarray(
            fusion_layer.layer_norm_vision.weight, copy=False
        )
        self.v_ln_b = mx.asarray(
            fusion_layer.layer_norm_vision.bias, copy=False
        )
        self.t_ln_w = mx.asarray(
            fusion_layer.layer_norm_text.weight, copy=False
        )
        self.t_ln_b = mx.asarray(
            fusion_layer.layer_norm_text.bias, copy=False
        )
        self.v_ln_eps = float(fusion_layer.layer_norm_vision.eps)
        self.t_ln_eps = float(fusion_layer.layer_norm_text.eps)

        self.vision_param = mx.asarray(fusion_layer.vision_param, copy=False)
        self.text_param = mx.asarray(fusion_layer.text_param, copy=False)

        mx.eval(
            self.v_ln_w,
            self.v_ln_b,
            self.t_ln_w,
            self.t_ln_b,
            self.vision_param,
            self.text_param,
        )
        sync_mlx()

        self.compiled_diag = mx.compile(self._forward_diag)
        self.compiled_min = mx.compile(self._forward_min)

    def _fusion(
        self,
        vision,
        text,
        vision_mask,
        text_mask,
    ):
        v = mx.fast.layer_norm(
            vision, self.v_ln_w, self.v_ln_b, self.v_ln_eps
        )
        t = mx.fast.layer_norm(
            text, self.t_ln_w, self.t_ln_b, self.t_ln_eps
        )

        delta_v, vision_attn, delta_t, text_attn = (
            self.fusion_core._forward_impl(v, t, vision_mask, text_mask)
        )

        v = v + self.vision_param * delta_v
        t = t + self.text_param * delta_t
        return v, t, vision_attn, text_attn

    def _forward_diag(
        self,
        vision,
        text,
        vision_mask,
        text_mask,
        deform_attention_mask,
        position,
        reference_points,
    ):
        v, t, vision_attn, text_attn = self._fusion(
            vision, text, vision_mask, text_mask
        )

        v_out, deform_attn = self.deform_core._forward_impl(
            v,
            deform_attention_mask,
            position,
            reference_points,
        )

        return v_out, t, vision_attn, text_attn, deform_attn

    def _forward_min(
        self,
        vision,
        text,
        vision_mask,
        text_mask,
        deform_attention_mask,
        position,
        reference_points,
    ):
        v, t, _vision_attn, _text_attn = self._fusion(
            vision, text, vision_mask, text_mask
        )

        v_out, _deform_attn = self.deform_core._forward_impl(
            v,
            deform_attention_mask,
            position,
            reference_points,
        )

        # Production inference with output_attentions=False only needs the
        # updated vision/text features at the encoder-layer boundary.
        return v_out, t


class CurrentSeparateSegment:
    """
    Current candidate structure from Experiment 0032:
      Torch -> MLX fusion core -> Torch fusion wrapper
            -> MLX deformable island -> Torch

    Uses the exact same MLX fusion/deformable objects as the wide island.
    """

    def __init__(self, fusion_layer, fusion_core, deform_core):
        self.fusion_layer = fusion_layer
        self.fusion_core = fusion_core
        self.deform_core = deform_core
        self.attn = fusion_layer.attn
        self.original_attn_forward = self.attn.forward

        def mlx_attn_forward(
            _module_self,
            vision_features,
            text_features,
            vision_attention_mask=None,
            text_attention_mask=None,
        ):
            sync_torch()

            v = mx.asarray(vision_features, copy=False)
            t = mx.asarray(text_features, copy=False)
            vm = (
                None
                if vision_attention_mask is None
                else mx.asarray(vision_attention_mask, copy=False)
            )
            tm = (
                None
                if text_attention_mask is None
                else mx.asarray(text_attention_mask, copy=False)
            )

            out = self.fusion_core.compiled_call(v, t, vm, tm)
            mx.eval(*out)
            sync_mlx()

            dv = torch.as_tensor(out[0])
            va = torch.as_tensor(out[1])
            dt = torch.as_tensor(out[2])
            ta = torch.as_tensor(out[3])
            sync_torch()
            return (dv, va), (dt, ta)

        self.attn.forward = types.MethodType(mlx_attn_forward, self.attn)

    def restore(self):
        self.attn.forward = self.original_attn_forward

    def run(
        self,
        vision,
        text,
        vision_mask,
        text_mask,
        deform_attention_mask,
        position,
        reference_points,
    ):
        (v, vision_attn), (t, text_attn) = self.fusion_layer(
            vision,
            text,
            attention_mask_vision=vision_mask,
            attention_mask_text=text_mask,
        )

        # Current Experiment-0018 deformable bridge.
        sync_torch()
        v_mx = mx.asarray(v, copy=False)
        mask_mx = mx.asarray(deform_attention_mask, copy=False)
        pos_mx = mx.asarray(position, copy=False)
        ref_mx = mx.asarray(reference_points, copy=False)

        v_out_mx, deform_attn_mx = self.deform_core.compiled_call(
            v_mx, mask_mx, pos_mx, ref_mx
        )
        mx.eval(v_out_mx, deform_attn_mx)
        sync_mlx()

        v_out = torch.as_tensor(v_out_mx)
        deform_attn = torch.as_tensor(deform_attn_mx)
        sync_torch()

        return v_out, t, vision_attn, text_attn, deform_attn


def to_mlx_inputs(captured):
    sync_torch()
    return (
        mx.asarray(captured["vision"], copy=False),
        mx.asarray(captured["text"], copy=False),
        (
            None
            if captured["vision_mask"] is None
            else mx.asarray(captured["vision_mask"], copy=False)
        ),
        (
            None
            if captured["text_mask"] is None
            else mx.asarray(captured["text_mask"], copy=False)
        ),
        mx.asarray(captured["deform_attention_mask"], copy=False),
        mx.asarray(captured["position"], copy=False),
        mx.asarray(captured["reference_points"], copy=False),
    )


def diag_mlx_to_torch(out):
    mx.eval(*out)
    sync_mlx()
    result = tuple(torch.as_tensor(x) for x in out)
    sync_torch()
    return result


def min_mlx_to_torch(out):
    mx.eval(*out)
    sync_mlx()
    result = tuple(torch.as_tensor(x) for x in out)
    sync_torch()
    return result


def benchmark_current(segment, captured, warmup, iters):
    args = (
        captured["vision"],
        captured["text"],
        captured["vision_mask"],
        captured["text_mask"],
        captured["deform_attention_mask"],
        captured["position"],
        captured["reference_points"],
    )

    with torch.inference_mode():
        for _ in range(warmup):
            sync_torch()
            _ = segment.run(*args)
            sync_torch()

        samples = []
        final = None
        for i in range(iters):
            sync_torch()
            t0 = time.perf_counter_ns()
            final = segment.run(*args)
            sync_torch()
            dt = (time.perf_counter_ns() - t0) / 1e6
            samples.append(dt)
            print(
                f"  current separate {i+1:02d}/{iters}: {dt:.3f} ms",
                flush=True,
            )

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
        "--exp13-helper",
        type=Path,
        default=Path("scripts/13_bench_fusion_algebraic_specialization.py"),
    )
    ap.add_argument(
        "--exp17-helper",
        type=Path,
        default=Path("scripts/17_bench_deformable_mlx_island.py"),
    )
    ap.add_argument(
        "--exp30-helper",
        type=Path,
        default=Path("scripts/30_bench_fully_folded_fusion_mlx.py"),
    )
    args = ap.parse_args()

    for p in (args.exp13_helper, args.exp17_helper, args.exp30_helper):
        if not p.exists():
            raise SystemExit(f"Missing helper: {p}")

    if not torch.backends.mps.is_available():
        raise SystemExit("MPS unavailable.")

    h13 = load_module(args.exp13_helper, "mg33_h13")
    h17 = load_module(args.exp17_helper, "mg33_h17")
    h30 = load_module(args.exp30_helper, "mg33_h30")

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

    layer = model.model.encoder.layers[0]
    fusion_layer = layer.fusion_layer
    deform_layer = layer.deformable_layer

    fusion_capture = {}
    deform_capture = {}

    original_fusion_forward = fusion_layer.forward
    original_deform_forward = deform_layer.forward

    def capture_fusion(
        self,
        vision_features,
        text_features,
        attention_mask_vision=None,
        attention_mask_text=None,
    ):
        if not fusion_capture:
            fusion_capture.update(
                vision=vision_features.detach(),
                text=text_features.detach(),
                vision_mask=(
                    None
                    if attention_mask_vision is None
                    else attention_mask_vision.detach()
                ),
                text_mask=(
                    None
                    if attention_mask_text is None
                    else attention_mask_text.detach()
                ),
            )
        return original_fusion_forward(
            vision_features,
            text_features,
            attention_mask_vision=attention_mask_vision,
            attention_mask_text=attention_mask_text,
        )

    def capture_deform(
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
        if not deform_capture:
            deform_capture.update(
                hidden_states=hidden_states.detach(),
                deform_attention_mask=attention_mask.detach(),
                position=position_embeddings.detach(),
                reference_points=reference_points.detach(),
                spatial_shapes_list=[
                    (int(h), int(w)) for h, w in spatial_shapes_list
                ],
            )
        return original_deform_forward(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_embeddings=position_embeddings,
            reference_points=reference_points,
            spatial_shapes=spatial_shapes,
            spatial_shapes_list=spatial_shapes_list,
            level_start_index=level_start_index,
            output_attentions=output_attentions,
        )

    fusion_layer.forward = types.MethodType(capture_fusion, fusion_layer)
    deform_layer.forward = types.MethodType(capture_deform, deform_layer)

    print("Capturing real layer-0 fusion/deformable workload...", flush=True)
    with torch.inference_mode():
        _ = model(**inputs)
        sync_torch()

    fusion_layer.forward = original_fusion_forward
    deform_layer.forward = original_deform_forward

    if not fusion_capture or not deform_capture:
        raise RuntimeError("Failed to capture fusion/deformable inputs.")

    captured = {
        **fusion_capture,
        "deform_attention_mask": deform_capture["deform_attention_mask"],
        "position": deform_capture["position"],
        "reference_points": deform_capture["reference_points"],
    }

    print(
        f"vision={tuple(captured['vision'].shape)} "
        f"text={tuple(captured['text'].shape)} "
        f"spatial={deform_capture['spatial_shapes_list']}",
        flush=True,
    )

    sync_torch()
    torch_spec = h13.AlgebraicBiMHA(fusion_layer.attn)
    fusion_core = h30.MlxFullyFoldedFusion(torch_spec)
    deform_core = h17.MlxDeformableIsland(
        deform_layer,
        activation_name=model.config.activation_function,
        spatial_shapes_list=deform_capture["spatial_shapes_list"],
        threadgroup=args.threadgroup,
    )
    if deform_core.compiled is None:
        raise RuntimeError(deform_core.compiled_error or "deform compile failed")

    wide = MlxFusionDeformableWideIsland(
        fusion_layer, fusion_core, deform_core
    )
    current = CurrentSeparateSegment(
        fusion_layer, fusion_core, deform_core
    )

    torch_source_args = (
        captured["vision"],
        captured["text"],
        captured["vision_mask"],
        captured["text_mask"],
        captured["deform_attention_mask"],
        captured["position"],
        captured["reference_points"],
    )

    def wide_bridge_diag():
        sync_torch()
        mlx_args = tuple(
            None if x is None else mx.asarray(x, copy=False)
            for x in torch_source_args
        )
        out = wide.compiled_diag(*mlx_args)
        return diag_mlx_to_torch(out)

    def wide_bridge_min():
        sync_torch()
        mlx_args = tuple(
            None if x is None else mx.asarray(x, copy=False)
            for x in torch_source_args
        )
        out = wide.compiled_min(*mlx_args)
        return min_mlx_to_torch(out)

    try:
        print("Correctness preflight...", flush=True)
        with torch.inference_mode():
            current_ref = current.run(*torch_source_args)
            wide_ref = wide_bridge_diag()

        names = (
            "vision_output",
            "text_output",
            "vision_attention",
            "text_attention",
            "deformable_attention",
        )
        correctness = {
            n: compare(w, c)
            for n, w, c in zip(names, wide_ref, current_ref)
        }
        print(json.dumps(correctness, indent=2), flush=True)

        print("\nBenchmark current separate boundaries...", flush=True)
        current_final, current_lat = benchmark_current(
            current, captured, args.warmup, args.iters
        )

        print("\nBenchmark wide single-boundary island...", flush=True)
        # Compile outside timed region.
        _ = wide_bridge_min()

        for _ in range(args.warmup):
            _ = wide_bridge_min()

        wide_samples = []
        wide_final = None
        with torch.inference_mode():
            for i in range(args.iters):
                sync_torch()
                t0 = time.perf_counter_ns()
                wide_final = wide_bridge_min()
                sync_torch()
                dt = (time.perf_counter_ns() - t0) / 1e6
                wide_samples.append(dt)
                print(
                    f"  wide island {i+1:02d}/{args.iters}: {dt:.3f} ms",
                    flush=True,
                )

        wide_lat = stats(wide_samples)

        # Minimal output correctness against the current segment.
        final_min_correctness = {
            "vision_output": compare(wide_final[0], current_final[0]),
            "text_output": compare(wide_final[1], current_final[1]),
        }

        result = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "experiment": "0033",
            "purpose": (
                "Layer-0 feasibility test for widening the MLX encoder island "
                "across fusion LayerNorm/residual + fully-folded fusion + "
                "deformable attention/FFN, removing the intermediate "
                "MLX->PyTorch->MLX boundary."
            ),
            "model": args.model,
            "device": "mps",
            "dtype": "float32",
            "shape": {
                "vision": list(captured["vision"].shape),
                "text": list(captured["text"].shape),
                "spatial_shapes": [
                    list(x) for x in deform_capture["spatial_shapes_list"]
                ],
            },
            "configuration": {
                "current_path": (
                    "compiled MLX fusion attention -> PyTorch fusion "
                    "LayerNorm/residual wrapper -> compiled MLX deformable island"
                ),
                "wide_path": (
                    "single compiled MLX island containing fusion LayerNorms, "
                    "fully-folded fusion, fusion layer-scale residuals, "
                    "deformable attention, and deformable FFN"
                ),
                "text_enhancer_inside_island": False,
                "wide_timed_outputs": [
                    "vision_output",
                    "text_output",
                ],
                "output_attentions": False,
                "threadgroup": args.threadgroup,
                "approximation": False,
                "retraining": False,
                "reduced_precision": False,
            },
            "latency": {
                "current_separate_boundaries": current_lat,
                "wide_single_boundary": wide_lat,
            },
            "derived": {
                "speedup_wide_vs_current": (
                    current_lat["median_ms"] / wide_lat["median_ms"]
                ),
                "median_reduction_ms": (
                    current_lat["median_ms"] - wide_lat["median_ms"]
                ),
                "median_reduction_percent": (
                    100.0
                    * (current_lat["median_ms"] - wide_lat["median_ms"])
                    / current_lat["median_ms"]
                ),
            },
            "correctness": {
                "diagnostic_all_outputs_wide_vs_current": correctness,
                "timed_minimal_outputs_wide_vs_current": final_min_correctness,
            },
            "decision_rule": (
                "Proceed to full-model six-layer wide-island integration if "
                "the single-boundary path saves at least ~1.5 ms/layer or ~5% "
                "against the current separate MLX fusion + deformable path, "
                "while vision/text outputs remain within the established "
                "FP32 backend envelope (target allclose 1e-4)."
            ),
            "notes": [
                "Both paths use the same folded fusion weights and the same MetalGround MSDA kernel.",
                "The wide path keeps text enhancement in PyTorch and moves only the fusion-to-deformable segment into one MLX execution island.",
                "GroundingDinoFusionLayer drop_path is inactive because the model is in eval mode.",
                "The timed wide path does not bridge attention-weight tensors because output_attentions=False; diagnostic correctness evaluates them separately.",
                "No approximation, quantization, pruning, retraining, or reduced precision is used."
            ],
        }

        out = Path(
            "results/metalground_fusion_deformable_wide_island_layer0.json"
        )
        out.write_text(
            json.dumps(result, indent=2, ensure_ascii=False) + "\n"
        )

        print("\n=== Experiment 0033 summary ===", flush=True)
        print(
            f"current separate: {current_lat['median_ms']:.3f} ms",
            flush=True,
        )
        print(
            f"wide island:      {wide_lat['median_ms']:.3f} ms",
            flush=True,
        )
        print(
            f"reduction:        "
            f"{result['derived']['median_reduction_ms']:+.3f} ms "
            f"({result['derived']['speedup_wide_vs_current']:.3f}x)",
            flush=True,
        )
        print(f"Saved: {out}", flush=True)

    finally:
        current.restore()


if __name__ == "__main__":
    main()
