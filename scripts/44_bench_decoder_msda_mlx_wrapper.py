#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import inspect
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


def compare(a: torch.Tensor, b: torch.Tensor):
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
        "finite": bool(torch.isfinite(a).all().item() and torch.isfinite(b).all().item()),
    }


def linear(x, weight, bias):
    y = x @ weight.T
    if bias is not None:
        y = y + bias
    return y


class MlxDecoderMsdaWrapper:
    """
    Decoder deformable-attention module with PyTorch wrapper work moved into MLX.

    Kept exactly the same:
      - trained parameters
      - FP32
      - MetalGround fused MSDA core

    Moved from PyTorch/MPS into one compiled MLX region:
      - position add
      - value projection + mask
      - sampling-offset projection
      - attention-weight projection + softmax
      - sampling-location construction
      - output projection

    Timed production path bridges only the final attention output back to Torch.
    The attention-weight tensor is bridged only in the untimed correctness path.
    """

    def __init__(self, module, *, spatial_shapes_list, threadgroup):
        self.module = module
        self.threadgroup = int(threadgroup)

        self.H = int(module.n_heads)
        self.NL = int(module.n_levels)
        self.NP = int(module.n_points)
        self.DMODEL = int(module.d_model)
        self.HD = self.DMODEL // self.H

        self.p = {}

        def import_linear(prefix, lin):
            self.p[prefix + "_w"] = mx.asarray(lin.weight, copy=False)
            self.p[prefix + "_b"] = (
                None if lin.bias is None else mx.asarray(lin.bias, copy=False)
            )

        import_linear("sampling", module.sampling_offsets)
        import_linear("attn_w", module.attention_weights)
        import_linear("value", module.value_proj)
        import_linear("output", module.output_proj)

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

        arrays = [
            self.spatial_shapes,
            self.level_start,
            self.offset_normalizer,
        ]
        arrays.extend(v for v in self.p.values() if v is not None)
        mx.eval(*arrays)
        mx.synchronize()

        self.compiled = mx.compile(self._forward_impl)

    def _forward_impl(
        self,
        hidden_states,
        attention_mask,
        encoder_hidden_states,
        position_embeddings,
        reference_points,
    ):
        query = (
            hidden_states
            if position_embeddings is None
            else hidden_states + position_embeddings
        )

        B, Q, _ = query.shape
        _, S, _ = encoder_hidden_states.shape

        value = linear(
            encoder_hidden_states,
            self.p["value_w"],
            self.p["value_b"],
        )
        if attention_mask is not None:
            value = mx.where(attention_mask[..., None], value, 0.0)

        value = value.reshape(B, S, self.H, self.HD)

        offsets = linear(
            query,
            self.p["sampling_w"],
            self.p["sampling_b"],
        )
        offsets = offsets.reshape(
            B, Q, self.H, self.NL, self.NP, 2
        )

        weights = linear(
            query,
            self.p["attn_w_w"],
            self.p["attn_w_b"],
        )
        weights = weights.reshape(B, Q, self.H, self.NL * self.NP)
        weights = mx.softmax(weights, axis=-1)
        weights = weights.reshape(
            B, Q, self.H, self.NL, self.NP
        )

        if reference_points.shape[-1] == 2:
            locations = (
                reference_points[:, :, None, :, None, :]
                + offsets
                / self.offset_normalizer[
                    None, None, None, :, None, :
                ]
            )
        elif reference_points.shape[-1] == 4:
            locations = (
                reference_points[:, :, None, :, None, :2]
                + offsets
                / float(self.NP)
                * reference_points[
                    :, :, None, :, None, 2:
                ]
                * 0.5
            )
        else:
            raise ValueError(
                f"Unsupported reference point dim: "
                f"{reference_points.shape[-1]}"
            )

        out = msda_metal_v0(
            value,
            self.spatial_shapes,
            self.level_start,
            locations,
            weights,
            threadgroup_size=self.threadgroup,
        )
        out = linear(
            out,
            self.p["output_w"],
            self.p["output_b"],
        )
        return out, weights

    def call(self, *args):
        return self.compiled(*args)


def bind_capture(original_forward, captured):
    sig = inspect.signature(original_forward)

    def wrapper(*args, **kwargs):
        if not captured:
            bound = sig.bind_partial(*args, **kwargs)
            bound.apply_defaults()
            for name, value in bound.arguments.items():
                if isinstance(value, torch.Tensor):
                    captured[name] = value.detach()
                else:
                    captured[name] = value
        return original_forward(*args, **kwargs)

    return wrapper


def get_required(captured, name):
    if name not in captured:
        raise RuntimeError(
            f"Captured decoder MSDA call lacks '{name}'. "
            f"Available={sorted(captured.keys())}"
        )
    return captured[name]


def spatial_list_from_capture(captured):
    if captured.get("spatial_shapes_list") is not None:
        return [
            (int(h), int(w))
            for h, w in captured["spatial_shapes_list"]
        ]

    shapes = get_required(captured, "spatial_shapes")
    return [
        (int(h), int(w))
        for h, w in shapes.detach().cpu().tolist()
    ]


def to_mlx_inputs(captured, sync_fn):
    # Critical: the sync is after every PyTorch producer used below.
    sync_fn()

    return (
        mx.asarray(
            get_required(captured, "hidden_states"),
            copy=False,
        ),
        (
            None
            if captured.get("attention_mask") is None
            else mx.asarray(captured["attention_mask"], copy=False)
        ),
        mx.asarray(
            get_required(captured, "encoder_hidden_states"),
            copy=False,
        ),
        (
            None
            if captured.get("position_embeddings") is None
            else mx.asarray(
                captured["position_embeddings"], copy=False
            )
        ),
        mx.asarray(
            get_required(captured, "reference_points"),
            copy=False,
        ),
    )


def bridge_diagnostic(outputs, sync_fn):
    out, weights = outputs
    mx.eval(out, weights)
    mx.synchronize()
    out_t = torch.as_tensor(out)
    weights_t = torch.as_tensor(weights)
    sync_fn()
    return out_t, weights_t


def bridge_production(outputs, sync_fn):
    out, _weights = outputs
    mx.eval(out)
    mx.synchronize()
    out_t = torch.as_tensor(out)
    sync_fn()
    return out_t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="IDEA-Research/grounding-dino-tiny")
    ap.add_argument("--image", type=Path, default=Path("assets/input.jpg"))
    ap.add_argument("--prompt", nargs="+", default=["a cat", "a dog"])
    ap.add_argument("--threadgroup", type=int, default=256)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--pairs", type=int, default=30)

    ap.add_argument(
        "--exp13-helper", type=Path,
        default=Path("scripts/13_bench_fusion_algebraic_specialization.py"),
    )
    ap.add_argument(
        "--exp14-helper", type=Path,
        default=Path("scripts/14_full_model_fusion_algebraic.py"),
    )
    ap.add_argument(
        "--exp17-helper", type=Path,
        default=Path("scripts/17_bench_deformable_mlx_island.py"),
    )
    ap.add_argument(
        "--exp18-helper", type=Path,
        default=Path("scripts/18_full_model_deformable_islands.py"),
    )
    ap.add_argument(
        "--exp30-helper", type=Path,
        default=Path("scripts/30_bench_fully_folded_fusion_mlx.py"),
    )
    ap.add_argument(
        "--exp31-helper", type=Path,
        default=Path("scripts/31_full_model_mlx_fusion_paired.py"),
    )
    ap.add_argument(
        "--exp36-helper", type=Path,
        default=Path("scripts/36_wide_island_mask_sync_fix.py"),
    )
    ap.add_argument(
        "--exp37-helper", type=Path,
        default=Path("scripts/37_corrected_wide_island_multiprocess.py"),
    )
    ap.add_argument(
        "--exp38-helper", type=Path,
        default=Path("scripts/38_consolidated_runtime_ablation.py"),
    )
    args = ap.parse_args()

    for p in (
        args.exp13_helper, args.exp14_helper, args.exp17_helper,
        args.exp18_helper, args.exp30_helper, args.exp31_helper,
        args.exp36_helper, args.exp37_helper, args.exp38_helper,
    ):
        if not p.exists():
            raise SystemExit(f"Missing helper: {p}")

    if not torch.backends.mps.is_available():
        raise SystemExit("MPS unavailable.")

    h37 = load_module(args.exp37_helper, "mg44_h37")
    h38 = load_module(args.exp38_helper, "mg44_h38")

    print("Building current consolidated runtime...", flush=True)
    rt = h37.build_runtime(args)
    model = rt["model"]
    h14 = rt["h14"]
    inputs = rt["inputs"]

    cache = h38.ExactTextBackboneCache(
        model.model.text_backbone
    )

    try:
        h38.set_runtime_mode(rt, cache, "wide_cache")
        cache.prime(model, inputs, h14.sync)
        cache.set_enabled(True)

        decoder_layer0 = model.model.decoder.layers[0]
        target = decoder_layer0.encoder_attn
        original_forward = target.forward
        captured = {}

        target.forward = types.MethodType(
            lambda _self, *a, **kw: capture_wrapper(*a, **kw),
            target,
        )
        capture_wrapper = bind_capture(
            original_forward, captured
        )

        print("Capturing real decoder layer0 MSDA workload...", flush=True)
        with torch.inference_mode():
            _ = model(**inputs)
            h14.sync()

        target.forward = original_forward

        if not captured:
            raise RuntimeError("Failed to capture decoder encoder_attn call.")

        spatial_shapes_list = spatial_list_from_capture(captured)

        print(
            "captured keys=" + ", ".join(sorted(captured.keys())),
            flush=True,
        )
        print(
            f"hidden={tuple(get_required(captured, 'hidden_states').shape)} "
            f"memory={tuple(get_required(captured, 'encoder_hidden_states').shape)} "
            f"ref={tuple(get_required(captured, 'reference_points').shape)} "
            f"spatial={spatial_shapes_list}",
            flush=True,
        )

        # Exact current reference on the captured call.
        ref_kwargs = dict(captured)
        with torch.inference_mode():
            ref_out = original_forward(**ref_kwargs)
            h14.sync()

        if not isinstance(ref_out, tuple) or len(ref_out) < 2:
            raise RuntimeError(
                f"Unexpected decoder MSDA output type: {type(ref_out)}"
            )

        # Synchronize before zero-copy parameter import.
        h14.sync()
        candidate = MlxDecoderMsdaWrapper(
            target,
            spatial_shapes_list=spatial_shapes_list,
            threadgroup=args.threadgroup,
        )

        print("\nCandidate correctness preflight...", flush=True)
        cand_diag = bridge_diagnostic(
            candidate.call(*to_mlx_inputs(captured, h14.sync)),
            h14.sync,
        )

        correctness = {
            "output": compare(cand_diag[0], ref_out[0]),
            "attention_weights": compare(
                cand_diag[1], ref_out[1]
            ),
        }
        print(json.dumps(correctness, indent=2), flush=True)

        correctness_pass = (
            correctness["output"]["allclose_1e-4"]
            and correctness["attention_weights"]["allclose_1e-4"]
        )

        latency = None
        derived = None

        if correctness_pass:
            print(
                "\nCorrectness passed; paired bridge-included benchmark...",
                flush=True,
            )

            # Compile/first-execution excluded.
            _ = bridge_production(
                candidate.call(
                    *to_mlx_inputs(captured, h14.sync)
                ),
                h14.sync,
            )

            with torch.inference_mode():
                for i in range(args.warmup):
                    h14.sync()
                    _ = original_forward(**ref_kwargs)
                    h14.sync()

                    _ = bridge_production(
                        candidate.call(
                            *to_mlx_inputs(captured, h14.sync)
                        ),
                        h14.sync,
                    )
                    print(
                        f"  warmup {i+1}/{args.warmup}",
                        flush=True,
                    )

            current_samples = []
            candidate_samples = []
            deltas = []

            for i in range(args.pairs):
                order = (
                    ("current", "candidate")
                    if i % 2 == 0
                    else ("candidate", "current")
                )
                local = {}

                for mode in order:
                    if mode == "current":
                        h14.sync()
                        t0 = time.perf_counter_ns()
                        with torch.inference_mode():
                            _ = original_forward(**ref_kwargs)
                        h14.sync()
                        dt = (time.perf_counter_ns() - t0) / 1e6
                        current_samples.append(dt)
                    else:
                        h14.sync()
                        t0 = time.perf_counter_ns()
                        _ = bridge_production(
                            candidate.call(
                                *to_mlx_inputs(captured, h14.sync)
                            ),
                            h14.sync,
                        )
                        dt = (time.perf_counter_ns() - t0) / 1e6
                        candidate_samples.append(dt)

                    local[mode] = dt

                delta = local["current"] - local["candidate"]
                deltas.append(delta)

                print(
                    f"  pair {i+1:02d}/{args.pairs}: "
                    f"current={local['current']:.3f} ms "
                    f"mlx={local['candidate']:.3f} ms "
                    f"delta={delta:+.3f} ms",
                    flush=True,
                )

            c = stats(current_samples)
            m = stats(candidate_samples)
            d = stats(deltas)

            latency = {
                "current_pytorch_wrapper_plus_metal_core": c,
                "compiled_mlx_wrapper_plus_same_metal_core": m,
                "paired_delta_current_minus_mlx": d,
            }
            derived = {
                "speedup_from_medians":
                    c["median_ms"] / m["median_ms"],
                "median_difference_of_marginals_ms":
                    c["median_ms"] - m["median_ms"],
                "paired_delta_median_ms":
                    d["median_ms"],
                "paired_delta_mean_ms":
                    d["mean_ms"],
                "reduction_percent_from_medians":
                    100.0
                    * (c["median_ms"] - m["median_ms"])
                    / c["median_ms"],
            }
        else:
            print(
                "\nCorrectness failed; timing intentionally skipped.",
                flush=True,
            )

        result = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "experiment": "0044",
            "purpose": (
                "Isolated real-workload test of moving the Grounding DINO "
                "decoder deformable-attention wrapper from PyTorch/MPS into "
                "compiled MLX while retaining the exact same MetalGround MSDA core."
            ),
            "model": args.model,
            "device": "mps",
            "dtype": "float32",
            "software": {
                "torch": torch.__version__,
                "transformers": transformers.__version__,
            },
            "configuration": {
                "decoder_layer": 0,
                "query_shape": list(
                    get_required(captured, "hidden_states").shape
                ),
                "memory_shape": list(
                    get_required(
                        captured, "encoder_hidden_states"
                    ).shape
                ),
                "reference_points_shape": list(
                    get_required(
                        captured, "reference_points"
                    ).shape
                ),
                "spatial_shapes_list": spatial_shapes_list,
                "current": (
                    "PyTorch/MPS projections + softmax/location construction "
                    "+ custom Metal MSDA core + PyTorch/MPS output projection"
                ),
                "candidate": (
                    "compiled MLX projections + softmax/location construction "
                    "+ same custom Metal MSDA core + MLX output projection"
                ),
                "timed_candidate_bridge": (
                    "Torch->MLX input bridge and MLX->Torch final output bridge "
                    "included; attention weights are not bridged in timed "
                    "production path"
                ),
                "approximation": False,
                "retraining": False,
                "reduced_precision": False,
            },
            "captured_argument_names": sorted(captured.keys()),
            "correctness": correctness,
            "correctness_gate_passed": correctness_pass,
            "latency": latency,
            "derived": derived,
            "decision_rule": (
                "Proceed to all-six-decoder integration if output and attention "
                "weights remain within the established FP32 1e-4 envelope and "
                "the bridge-included paired median improves by at least 1.0 ms "
                "or 25% versus the current decoder deformable-attention module. "
                "Otherwise close this wrapper-port path."
            ),
            "notes": [
                "The Metal MSDA kernel is identical between current and candidate paths.",
                "Only the wrapper-side projections, softmax, sampling-location construction, and output projection move to MLX.",
                "The timed production candidate does not bridge attention weights because output_attentions=False in the target runtime.",
                "No approximation, pruning, quantization, retraining, or reduced precision is used.",
                "This is an isolated module benchmark, not a full-model authority."
            ],
        }

        out = Path(
            "results/metalground_decoder_msda_mlx_wrapper.json"
        )
        out.write_text(
            json.dumps(result, indent=2, ensure_ascii=False) + "\n"
        )

        print("\n=== Experiment 0044 summary ===", flush=True)
        print(f"correctness gate: {correctness_pass}", flush=True)
        if derived is not None:
            print(
                f"current median: "
                f"{latency['current_pytorch_wrapper_plus_metal_core']['median_ms']:.3f} ms",
                flush=True,
            )
            print(
                f"MLX median:     "
                f"{latency['compiled_mlx_wrapper_plus_same_metal_core']['median_ms']:.3f} ms",
                flush=True,
            )
            print(
                f"paired saving:  "
                f"{derived['paired_delta_median_ms']:+.3f} ms",
                flush=True,
            )
            print(
                f"speedup:        "
                f"{derived['speedup_from_medians']:.3f}x",
                flush=True,
            )
        print(f"Saved: {out}", flush=True)

    finally:
        # Restore target if capture failed mid-flight.
        try:
            target.forward = original_forward
        except Exception:
            pass
        cache.restore()
        for d in rt["layer_dispatchers"]:
            d.restore()
        for fd in rt["fusion_dispatchers"]:
            fd.restore()


if __name__ == "__main__":
    main()
