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
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import mlx.core as mx
import torch
import transformers


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


def mask_aware_error(a: torch.Tensor, b: torch.Tensor):
    a = a.detach()
    b = b.detach()
    fa = torch.isfinite(a)
    fb = torch.isfinite(b)
    common = fa & fb

    if bool(common.any().item()):
        af = a[common].float()
        bf = b[common].float()
        d = (af - bf).abs()
        max_abs = float(d.max().item())
        mean_abs = float(d.mean().item())
        rmse = float(torch.sqrt(torch.mean((af - bf) ** 2)).item())
    else:
        max_abs = mean_abs = rmse = 0.0

    return {
        "finite_mask_equal": bool(torch.equal(fa, fb)),
        "nan_mask_equal": bool(torch.equal(torch.isnan(a), torch.isnan(b))),
        "posinf_mask_equal": bool(torch.equal(torch.isposinf(a), torch.isposinf(b))),
        "neginf_mask_equal": bool(torch.equal(torch.isneginf(a), torch.isneginf(b))),
        "finite_max_abs": max_abs,
        "finite_mean_abs": mean_abs,
        "finite_rmse": rmse,
        "allclose_1e-5": bool(torch.allclose(a, b, rtol=1e-5, atol=1e-5)),
        "allclose_1e-4": bool(torch.allclose(a, b, rtol=1e-4, atol=1e-4)),
    }


def pair_audit(a, b, model, h14):
    return {
        "logits": mask_aware_error(a.logits, b.logits),
        "pred_boxes": mask_aware_error(a.pred_boxes, b.pred_boxes),
        "topk": h14.topk_audit(a, b, model.config.num_queries),
    }


def to_mlx_or_none(x):
    return None if x is None else mx.asarray(x, copy=False)


@dataclass
class SuperState:
    current_layer_calls: int = 0
    super_root_calls: int = 0
    super_identity_calls: int = 0


class SixLayerSuperDispatcher:
    """
    Wrap the already-corrected encoder layer dispatchers.

    current mode:
      use the adopted corrected-wide layer path independently per layer,
      returning to PyTorch for each text enhancer.

    super mode:
      layer 0 executes all six corrected wide islands plus six MLX text
      enhancers while vision/text remain MLX arrays.
      layers 1..5 become mathematical identities because layer 0 has already
      advanced both branches through all six encoder layers.

    This preserves the outer HF encoder loop and its return structure without
    reimplementing GroundingDinoEncoder.forward.
    """

    def __init__(
        self,
        layers,
        underlying_dispatchers,
        wide_islands,
        text_islands,
        sync_fn,
    ):
        self.layers = layers
        self.underlying_dispatchers = underlying_dispatchers
        self.wide_islands = wide_islands
        self.text_islands = text_islands
        self.sync_fn = sync_fn
        self.state = SuperState()
        self.mode = "current"
        self.original_forwards = [layer.forward for layer in layers]

        for i, layer in enumerate(layers):
            original = self.original_forwards[i]

            def make_forward(index, bound_original):
                def dispatched(
                    _layer_self,
                    vision_features,
                    vision_position_embedding,
                    spatial_shapes,
                    spatial_shapes_list,
                    level_start_index,
                    key_padding_mask,
                    reference_points,
                    text_features=None,
                    text_attention_mask=None,
                    text_position_embedding=None,
                    text_self_attention_masks=None,
                    text_position_ids=None,
                ):
                    if self.mode == "current":
                        self.state.current_layer_calls += 1
                        return bound_original(
                            vision_features=vision_features,
                            vision_position_embedding=vision_position_embedding,
                            spatial_shapes=spatial_shapes,
                            spatial_shapes_list=spatial_shapes_list,
                            level_start_index=level_start_index,
                            key_padding_mask=key_padding_mask,
                            reference_points=reference_points,
                            text_features=text_features,
                            text_attention_mask=text_attention_mask,
                            text_position_embedding=text_position_embedding,
                            text_self_attention_masks=text_self_attention_masks,
                            text_position_ids=text_position_ids,
                        )

                    if index > 0:
                        self.state.super_identity_calls += 1
                        return (
                            (vision_features, text_features),
                            (None, None, None, None),
                        )

                    self.state.super_root_calls += 1

                    # Resolve every layer's text positional embedding while
                    # still in PyTorch. These may themselves schedule MPS work,
                    # so the producer sync MUST happen only after all six are
                    # resolved.
                    resolved_text_positions = [
                        lyr.get_text_position_embeddings(
                            text_features,
                            text_position_embedding,
                            text_position_ids,
                        )
                        for lyr in self.layers
                    ]

                    # True final PyTorch producer boundary.
                    self.sync_fn()

                    v = mx.asarray(vision_features, copy=False)
                    t = mx.asarray(text_features, copy=False)
                    key_padding = mx.asarray(key_padding_mask, copy=False)
                    text_padding = to_mlx_or_none(text_attention_mask)
                    vision_pos = mx.asarray(
                        vision_position_embedding, copy=False
                    )
                    refs = mx.asarray(reference_points, copy=False)

                    if text_self_attention_masks is None:
                        allowed_text = None
                    else:
                        text_self_mask_mx = mx.asarray(
                            text_self_attention_masks, copy=False
                        )
                        allowed_text = mx.logical_not(text_self_mask_mx)

                    pos_mx = [
                        to_mlx_or_none(x) for x in resolved_text_positions
                    ]

                    # No mx.eval / no Torch bridge between layers.
                    for i in range(6):
                        v, t = self.wide_islands[i].compiled_min(
                            v,
                            t,
                            key_padding,
                            text_padding,
                            vision_pos,
                            refs,
                        )
                        t, _attn = self.text_islands[i].compiled_call(
                            t,
                            allowed_text,
                            pos_mx[i],
                        )

                    mx.eval(v, t)
                    mx.synchronize()

                    v_t = torch.as_tensor(v)
                    t_t = torch.as_tensor(t)
                    self.sync_fn()

                    return (
                        (v_t, t_t),
                        (None, None, None, None),
                    )

                return dispatched

            layer.forward = types.MethodType(
                make_forward(i, original),
                layer,
            )

    def set_mode(self, mode: str):
        if mode not in ("current", "super"):
            raise ValueError(mode)
        self.mode = mode

    def reset_counts(self):
        self.state = SuperState()

    def restore(self):
        for layer, original in zip(self.layers, self.original_forwards):
            layer.forward = original


def run_timed(model, inputs, sync_fn):
    sync_fn()
    t0 = time.perf_counter_ns()
    with torch.inference_mode():
        out = model(**inputs)
    sync_fn()
    return out, (time.perf_counter_ns() - t0) / 1e6


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="IDEA-Research/grounding-dino-tiny")
    ap.add_argument("--image", type=Path, default=Path("assets/input.jpg"))
    ap.add_argument("--prompt", nargs="+", default=["a cat", "a dog"])
    ap.add_argument("--threadgroup", type=int, default=256)
    ap.add_argument("--warmup-per-mode", type=int, default=3)
    ap.add_argument("--pairs", type=int, default=20)

    ap.add_argument(
        "--exp13-helper",
        type=Path,
        default=Path("scripts/13_bench_fusion_algebraic_specialization.py"),
    )
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
    ap.add_argument(
        "--exp18-helper",
        type=Path,
        default=Path("scripts/18_full_model_deformable_islands.py"),
    )
    ap.add_argument(
        "--exp30-helper",
        type=Path,
        default=Path("scripts/30_bench_fully_folded_fusion_mlx.py"),
    )
    ap.add_argument(
        "--exp31-helper",
        type=Path,
        default=Path("scripts/31_full_model_mlx_fusion_paired.py"),
    )
    ap.add_argument(
        "--exp36-helper",
        type=Path,
        default=Path("scripts/36_wide_island_mask_sync_fix.py"),
    )
    ap.add_argument(
        "--exp37-helper",
        type=Path,
        default=Path("scripts/37_corrected_wide_island_multiprocess.py"),
    )
    ap.add_argument(
        "--exp38-helper",
        type=Path,
        default=Path("scripts/38_consolidated_runtime_ablation.py"),
    )
    ap.add_argument(
        "--exp41-helper",
        type=Path,
        default=Path("scripts/41_two_layer_encoder_super_island.py"),
    )
    args = ap.parse_args()

    for p in (
        args.exp13_helper,
        args.exp14_helper,
        args.exp17_helper,
        args.exp18_helper,
        args.exp30_helper,
        args.exp31_helper,
        args.exp36_helper,
        args.exp37_helper,
        args.exp38_helper,
        args.exp41_helper,
    ):
        if not p.exists():
            raise SystemExit(f"Missing helper: {p}")

    if not torch.backends.mps.is_available():
        raise SystemExit("MPS unavailable.")

    h37 = load_module(args.exp37_helper, "mg42_h37")
    h38 = load_module(args.exp38_helper, "mg42_h38")
    h41 = load_module(args.exp41_helper, "mg42_h41")

    print("Building consolidated current runtime...", flush=True)
    rt = h37.build_runtime(args)
    model = rt["model"]
    h14 = rt["h14"]
    h36 = rt["h36"]
    inputs = rt["inputs"]

    # Both current and candidate use the exact prompt cache so this experiment
    # isolates encoder boundary removal only.
    cache = h38.ExactTextBackboneCache(model.model.text_backbone)

    # Current adopted encoder path must be corrected-wide.
    h36.set_mode(rt["layer_dispatchers"], "fixed")

    layers = list(model.model.encoder.layers)
    if len(layers) != 6:
        raise RuntimeError(f"Expected 6 encoder layers, got {len(layers)}")

    print("Building six MLX text-enhancer ports...", flush=True)
    text_islands = [
        h41.MlxTextEnhancer(
            layer.text_enhancer_layer,
            activation_name=model.config.activation_function,
        )
        for layer in layers
    ]

    wide_islands = [d.fixed_island for d in rt["layer_dispatchers"]]

    super_dispatcher = SixLayerSuperDispatcher(
        layers,
        rt["layer_dispatchers"],
        wide_islands,
        text_islands,
        h14.sync,
    )

    try:
        # Prime cache outside correctness/timing.
        super_dispatcher.set_mode("current")
        cache.prime(model, inputs, h14.sync)

        # Current consolidated runtime reference.
        cache.set_enabled(True)
        super_dispatcher.set_mode("current")
        with torch.inference_mode():
            current_ref = model(**inputs)
            h14.sync()

        # Six-layer super-island candidate.
        super_dispatcher.set_mode("super")
        with torch.inference_mode():
            super_ref = model(**inputs)
            h14.sync()

        correctness = {
            "super_vs_current": pair_audit(
                super_ref, current_ref, model, h14
            ),
            "current_vs_original": pair_audit(
                current_ref, rt["original_ref"], model, h14
            ),
            "super_vs_original": pair_audit(
                super_ref, rt["original_ref"], model, h14
            ),
        }

        print(
            "Top-k preflight:",
            json.dumps(
                {k: v["topk"] for k, v in correctness.items()},
                indent=2,
            ),
            flush=True,
        )

        topk_pass = (
            correctness["super_vs_current"]["topk"]["set_overlap"] == 900
            and correctness["super_vs_current"]["topk"][
                "changed_membership_each_side"
            ] == 0
        )

        latency = None
        derived = None
        call_validation = None

        if topk_pass:
            print(
                "Correctness membership gate passed; warming both modes...",
                flush=True,
            )

            for i in range(args.warmup_per_mode):
                super_dispatcher.set_mode("current")
                _ = run_timed(model, inputs, h14.sync)
                super_dispatcher.set_mode("super")
                _ = run_timed(model, inputs, h14.sync)
                print(
                    f"  warmup pair {i+1}/{args.warmup_per_mode}",
                    flush=True,
                )

            # Benchmark-only counters.
            super_dispatcher.reset_counts()
            cache.reset_counts()
            rt["msda_state"].call_count = 0

            current_samples = []
            super_samples = []
            deltas = []

            print(f"Paired full-model benchmark x{args.pairs}...", flush=True)
            for i in range(args.pairs):
                order = (
                    ("current", "super")
                    if i % 2 == 0
                    else ("super", "current")
                )
                local = {}

                for mode in order:
                    super_dispatcher.set_mode(mode)
                    _out, dt = run_timed(model, inputs, h14.sync)
                    local[mode] = dt
                    if mode == "current":
                        current_samples.append(dt)
                    else:
                        super_samples.append(dt)

                delta = local["current"] - local["super"]
                deltas.append(delta)
                print(
                    f"  pair {i+1:02d}/{args.pairs}: "
                    f"current={local['current']:.3f} ms, "
                    f"super={local['super']:.3f} ms, "
                    f"delta={delta:+.3f} ms",
                    flush=True,
                )

            c = stats(current_samples)
            s = stats(super_samples)
            d = stats(deltas)

            latency = {
                "current_consolidated": c,
                "six_layer_super_island": s,
                "paired_delta_current_minus_super": d,
            }
            derived = {
                "speedup_from_medians": c["median_ms"] / s["median_ms"],
                "median_difference_of_marginals_ms":
                    c["median_ms"] - s["median_ms"],
                "paired_delta_median_ms": d["median_ms"],
                "paired_delta_mean_ms": d["mean_ms"],
                "super_median_fps": 1000.0 / s["median_ms"],
            }

            st = super_dispatcher.state
            expected_current_calls = args.pairs * 6
            expected_super_roots = args.pairs
            expected_super_identities = args.pairs * 5
            expected_cache_hits = args.pairs * 2
            expected_decoder_msda = args.pairs * 2 * 6

            call_validation = {
                "current_encoder_layer_calls": st.current_layer_calls,
                "super_root_calls": st.super_root_calls,
                "super_identity_calls": st.super_identity_calls,
                "cache_hits": cache.hits,
                "cache_misses": cache.misses,
                "decoder_metal_msda_calls": rt["msda_state"].call_count,
                "expected_current_encoder_layer_calls": expected_current_calls,
                "expected_super_root_calls": expected_super_roots,
                "expected_super_identity_calls": expected_super_identities,
                "expected_cache_hits": expected_cache_hits,
                "expected_decoder_metal_msda_calls": expected_decoder_msda,
            }

            if st.current_layer_calls != expected_current_calls:
                raise RuntimeError(
                    f"current layer calls {st.current_layer_calls} "
                    f"!= {expected_current_calls}"
                )
            if st.super_root_calls != expected_super_roots:
                raise RuntimeError(
                    f"super roots {st.super_root_calls} "
                    f"!= {expected_super_roots}"
                )
            if st.super_identity_calls != expected_super_identities:
                raise RuntimeError(
                    f"super identities {st.super_identity_calls} "
                    f"!= {expected_super_identities}"
                )
            if cache.hits != expected_cache_hits or cache.misses != 0:
                raise RuntimeError(
                    f"cache hits/misses {cache.hits}/{cache.misses}, "
                    f"expected {expected_cache_hits}/0"
                )
            if rt["msda_state"].call_count != expected_decoder_msda:
                raise RuntimeError(
                    f"decoder MSDA calls {rt['msda_state'].call_count} "
                    f"!= {expected_decoder_msda}"
                )
        else:
            print(
                "Top-k membership gate FAILED; latency intentionally skipped.",
                flush=True,
            )

        result = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "experiment": "0042",
            "purpose": (
                "Full-model feasibility test of a six-layer MLX encoder "
                "super-island that keeps both vision and text branches in MLX "
                "across all encoder layers, eliminating five intermediate "
                "MLX->PyTorch->MLX boundaries."
            ),
            "model": args.model,
            "device": "mps",
            "dtype": "float32",
            "software": {
                "torch": torch.__version__,
                "transformers": transformers.__version__,
            },
            "configuration": {
                "baseline": (
                    "current consolidated runtime: corrected per-layer wide "
                    "fusion+deformable islands + PyTorch text enhancers + "
                    "exact prompt cache + decoder Metal MSDA"
                ),
                "candidate": (
                    "single six-layer encoder MLX super-island: six corrected "
                    "wide vision islands + six MLX text enhancers, one entry "
                    "and one exit for the whole encoder"
                ),
                "prompt_cache": "enabled identically in both modes",
                "output_attentions": False,
                "warmup_per_mode": args.warmup_per_mode,
                "paired_iterations": args.pairs,
                "approximation": False,
                "retraining": False,
                "reduced_precision": False,
            },
            "correctness": correctness,
            "topk_membership_gate_passed": topk_pass,
            "latency": latency,
            "derived": derived,
            "call_validation": call_validation,
            "decision_rule": (
                "Proceed to multi-process replication and adopt the six-layer "
                "encoder super-island if top-900 membership is preserved and "
                "the same-process paired saving is materially positive "
                "(target >=5 ms full-model). If membership changes, debug "
                "numerical/semantic divergence before timing. If the saving "
                "is <5 ms, keep the current per-layer wide runtime."
            ),
            "notes": [
                "Both modes use the exact fixed-prompt BERT cache, so the experiment isolates encoder execution scheduling.",
                "The candidate resolves all six text positional embeddings before the final PyTorch producer sync.",
                "No Torch bridge or mx.eval occurs between the six encoder layers in super mode.",
                "Layers 1-5 are identity wrappers only because layer 0 has already executed their exact encoder computations inside the super-island.",
                "No approximation, pruning, quantization, retraining, or reduced precision is used.",
                "Dataset-level accuracy equivalence remains unmeasured."
            ],
        }

        out = Path("results/metalground_six_layer_encoder_super_island.json")
        out.write_text(
            json.dumps(result, indent=2, ensure_ascii=False) + "\n"
        )

        print("\n=== Experiment 0042 summary ===", flush=True)
        print(f"top-k membership gate: {topk_pass}", flush=True)
        if derived is not None:
            print(
                f"current median: {latency['current_consolidated']['median_ms']:.3f} ms",
                flush=True,
            )
            print(
                f"super median:   {latency['six_layer_super_island']['median_ms']:.3f} ms",
                flush=True,
            )
            print(
                f"paired saving:  {derived['paired_delta_median_ms']:+.3f} ms",
                flush=True,
            )
            print(
                f"super FPS:      {derived['super_median_fps']:.3f}",
                flush=True,
            )
        print(f"Saved: {out}", flush=True)

    finally:
        super_dispatcher.restore()
        cache.restore()
        for d in rt["layer_dispatchers"]:
            d.restore()
        for fd in rt["fusion_dispatchers"]:
            fd.restore()


if __name__ == "__main__":
    main()
