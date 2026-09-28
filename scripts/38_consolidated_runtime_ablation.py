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


def mask_aware_error(a: torch.Tensor, b: torch.Tensor) -> dict[str, Any]:
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


class ExactTextBackboneCache:
    def __init__(self, module):
        self.module = module
        self.original_forward = module.forward
        self.cached_output = None
        self.enabled = False
        self.calls = 0
        self.hits = 0
        self.misses = 0
        self.bypass = 0

        def patched_forward(_module_self, *args, **kwargs):
            self.calls += 1

            if not self.enabled:
                self.bypass += 1
                return self.original_forward(*args, **kwargs)

            if self.cached_output is None:
                self.misses += 1
                out = self.original_forward(*args, **kwargs)
                self.cached_output = out
                return out

            self.hits += 1
            return self.cached_output

        module.forward = types.MethodType(patched_forward, module)

    def set_enabled(self, enabled: bool):
        self.enabled = bool(enabled)

    def reset_counts(self):
        self.calls = self.hits = self.misses = self.bypass = 0

    def prime(self, model, inputs, sync_fn):
        self.cached_output = None
        self.reset_counts()
        self.set_enabled(True)
        with torch.inference_mode():
            _ = model(**inputs)
            sync_fn()
        if self.misses != 1 or self.cached_output is None:
            raise RuntimeError(
                f"Cache prime failed: misses={self.misses}, "
                f"cached={self.cached_output is not None}"
            )

    def restore(self):
        self.module.forward = self.original_forward


def pair_audit(a, b, model, h14):
    return {
        "logits": mask_aware_error(a.logits, b.logits),
        "pred_boxes": mask_aware_error(a.pred_boxes, b.pred_boxes),
        "topk": h14.topk_audit(a, b, model.config.num_queries),
    }


def run_timed(model, inputs, sync_fn):
    sync_fn()
    t0 = time.perf_counter_ns()
    with torch.inference_mode():
        out = model(**inputs)
    sync_fn()
    return out, (time.perf_counter_ns() - t0) / 1e6


def set_runtime_mode(rt, cache, mode: str):
    """
    Four cumulative exact-runtime stages.

    v2:
      PyTorch fully-folded fusion + separate MLX deformable islands

    mlx_fusion:
      compiled MLX fully-folded fusion + separate MLX deformable islands

    wide:
      corrected single-boundary MLX fusion+deformable islands

    wide_cache:
      wide + exact fixed-prompt BERT cache
    """
    h36 = rt["h36"]
    layer_dispatchers = rt["layer_dispatchers"]
    fusion_dispatchers = rt["fusion_dispatchers"]

    if mode == "v2":
        h36.set_mode(layer_dispatchers, "baseline")
        for fd in fusion_dispatchers:
            fd.set_mode("torch")
        cache.set_enabled(False)
    elif mode == "mlx_fusion":
        h36.set_mode(layer_dispatchers, "baseline")
        for fd in fusion_dispatchers:
            fd.set_mode("mlx")
        cache.set_enabled(False)
    elif mode == "wide":
        h36.set_mode(layer_dispatchers, "fixed")
        for fd in fusion_dispatchers:
            fd.set_mode("mlx")
        cache.set_enabled(False)
    elif mode == "wide_cache":
        h36.set_mode(layer_dispatchers, "fixed")
        for fd in fusion_dispatchers:
            fd.set_mode("mlx")
        cache.set_enabled(True)
    else:
        raise ValueError(mode)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=20)
    ap.add_argument("--warmup-per-mode", type=int, default=2)
    ap.add_argument("--model", default="IDEA-Research/grounding-dino-tiny")
    ap.add_argument("--image", type=Path, default=Path("assets/input.jpg"))
    ap.add_argument("--prompt", nargs="+", default=["a cat", "a dog"])
    ap.add_argument("--threadgroup", type=int, default=256)

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
    args = ap.parse_args()

    if args.rounds % 4 != 0:
        raise SystemExit("--rounds must be divisible by 4 for balanced mode order.")

    for p in (
        args.exp13_helper,
        args.exp14_helper,
        args.exp17_helper,
        args.exp18_helper,
        args.exp30_helper,
        args.exp31_helper,
        args.exp36_helper,
        args.exp37_helper,
    ):
        if not p.exists():
            raise SystemExit(f"Missing helper: {p}")

    if not torch.backends.mps.is_available():
        raise SystemExit("MPS unavailable.")

    h37 = load_module(args.exp37_helper, "mg38_h37")

    print("Building consolidated runtime components...", flush=True)
    rt = h37.build_runtime(args)

    model = rt["model"]
    h14 = rt["h14"]
    h31 = rt["h31"]
    h36 = rt["h36"]
    inputs = rt["inputs"]

    cache = ExactTextBackboneCache(model.model.text_backbone)

    modes = ["v2", "mlx_fusion", "wide", "wide_cache"]

    try:
        # Prime exact fixed-prompt cache outside all correctness/timing regions.
        set_runtime_mode(rt, cache, "wide_cache")
        print("Priming exact prompt cache...", flush=True)
        cache.prime(model, inputs, h14.sync)

        # Correctness outputs for the full cumulative ladder.
        print("Collecting four-mode correctness ladder...", flush=True)
        correctness_outputs = {}
        for mode in modes:
            set_runtime_mode(rt, cache, mode)
            with torch.inference_mode():
                correctness_outputs[mode] = model(**inputs)
                h14.sync()
            print(f"  {mode}: done", flush=True)

        original_ref = rt["original_ref"]

        correctness = {
            "each_mode_vs_original": {
                mode: pair_audit(
                    correctness_outputs[mode],
                    original_ref,
                    model,
                    h14,
                )
                for mode in modes
            },
            "incremental": {
                "mlx_fusion_vs_v2": pair_audit(
                    correctness_outputs["mlx_fusion"],
                    correctness_outputs["v2"],
                    model,
                    h14,
                ),
                "wide_vs_mlx_fusion": pair_audit(
                    correctness_outputs["wide"],
                    correctness_outputs["mlx_fusion"],
                    model,
                    h14,
                ),
                "wide_cache_vs_wide": pair_audit(
                    correctness_outputs["wide_cache"],
                    correctness_outputs["wide"],
                    model,
                    h14,
                ),
            },
        }

        # Prompt caching must remain exact on top of the corrected wide runtime.
        cache_exact = (
            correctness["incremental"]["wide_cache_vs_wide"]["logits"][
                "finite_max_abs"
            ] == 0.0
            and correctness["incremental"]["wide_cache_vs_wide"]["pred_boxes"][
                "finite_max_abs"
            ] == 0.0
            and correctness["incremental"]["wide_cache_vs_wide"]["topk"][
                "rankwise_identical"
            ] == model.config.num_queries
        )
        if not cache_exact:
            raise RuntimeError(
                "Prompt cache is not exact on the consolidated wide runtime."
            )

        # Warm all modes equally.
        print("Warming all four modes...", flush=True)
        for wi in range(args.warmup_per_mode):
            for mode in modes:
                set_runtime_mode(rt, cache, mode)
                _ = run_timed(model, inputs, h14.sync)
            print(
                f"  warmup cycle {wi+1}/{args.warmup_per_mode}",
                flush=True,
            )

        # Reset benchmark-only counters.
        cache.reset_counts()
        rt["state"].baseline_calls = 0
        rt["state"].fixed_calls = 0
        for fd in rt["fusion_dispatchers"]:
            fd.reset_counts()
        rt["island_state"].calls = 0
        rt["msda_state"].call_count = 0

        samples = {mode: [] for mode in modes}
        deltas = {
            "v2_minus_mlx_fusion": [],
            "mlx_fusion_minus_wide": [],
            "wide_minus_wide_cache": [],
            "v2_minus_wide_cache": [],
        }
        round_records = []

        print(
            f"Balanced four-mode benchmark: {args.rounds} rounds...",
            flush=True,
        )

        for r in range(args.rounds):
            shift = r % 4
            order = modes[shift:] + modes[:shift]
            local = {}

            for mode in order:
                set_runtime_mode(rt, cache, mode)
                _out, dt = run_timed(model, inputs, h14.sync)
                samples[mode].append(dt)
                local[mode] = dt

            d1 = local["v2"] - local["mlx_fusion"]
            d2 = local["mlx_fusion"] - local["wide"]
            d3 = local["wide"] - local["wide_cache"]
            df = local["v2"] - local["wide_cache"]

            deltas["v2_minus_mlx_fusion"].append(d1)
            deltas["mlx_fusion_minus_wide"].append(d2)
            deltas["wide_minus_wide_cache"].append(d3)
            deltas["v2_minus_wide_cache"].append(df)

            round_records.append({
                "round": r,
                "order": order,
                "latency_ms": local,
                "deltas_ms": {
                    "v2_minus_mlx_fusion": d1,
                    "mlx_fusion_minus_wide": d2,
                    "wide_minus_wide_cache": d3,
                    "v2_minus_wide_cache": df,
                },
            })

            print(
                f"  round {r+1:02d}/{args.rounds} "
                f"v2={local['v2']:.3f} "
                f"mlx={local['mlx_fusion']:.3f} "
                f"wide={local['wide']:.3f} "
                f"final={local['wide_cache']:.3f} | "
                f"final gain={df:+.3f} ms",
                flush=True,
            )

        latency_stats = {
            mode: stats(xs) for mode, xs in samples.items()
        }
        delta_stats = {
            name: stats(xs) for name, xs in deltas.items()
        }

        med = {
            mode: latency_stats[mode]["median_ms"] for mode in modes
        }

        fusion_counts = h31.sum_counts(rt["fusion_dispatchers"])

        expected_v2_fusion_torch = args.rounds * 6
        expected_mlx_fusion_calls = args.rounds * 6
        expected_baseline_layer_calls = args.rounds * 2 * 6
        expected_fixed_layer_calls = args.rounds * 2 * 6
        expected_baseline_deform_calls = args.rounds * 2 * 6
        expected_decoder_msda = args.rounds * 4 * 6

        expected_cache_hits = args.rounds
        expected_cache_bypass = args.rounds * 3

        call_validation = {
            "text_cache": {
                "calls": cache.calls,
                "hits": cache.hits,
                "misses": cache.misses,
                "bypass": cache.bypass,
                "expected_hits": expected_cache_hits,
                "expected_bypass": expected_cache_bypass,
            },
            "encoder_layers": {
                "baseline_calls": rt["state"].baseline_calls,
                "fixed_calls": rt["state"].fixed_calls,
                "expected_baseline_calls": expected_baseline_layer_calls,
                "expected_fixed_calls": expected_fixed_layer_calls,
            },
            "fusion_dispatchers": {
                **fusion_counts,
                "expected_torch_calls": expected_v2_fusion_torch,
                "expected_mlx_calls": expected_mlx_fusion_calls,
            },
            "deformable_island_wrapper": {
                "calls": rt["island_state"].calls,
                "expected_calls": expected_baseline_deform_calls,
            },
            "decoder_metal_msda": {
                "calls": rt["msda_state"].call_count,
                "expected_calls": expected_decoder_msda,
            },
        }

        if cache.hits != expected_cache_hits or cache.misses != 0:
            raise RuntimeError(
                f"Prompt-cache benchmark mismatch: "
                f"hits={cache.hits}, misses={cache.misses}, "
                f"expected hits={expected_cache_hits}"
            )
        if cache.bypass != expected_cache_bypass:
            raise RuntimeError(
                f"Prompt-cache bypass mismatch: {cache.bypass} "
                f"!= {expected_cache_bypass}"
            )
        if rt["state"].baseline_calls != expected_baseline_layer_calls:
            raise RuntimeError("Baseline encoder-layer call count mismatch.")
        if rt["state"].fixed_calls != expected_fixed_layer_calls:
            raise RuntimeError("Wide encoder-layer call count mismatch.")
        if fusion_counts["torch_calls"] != expected_v2_fusion_torch:
            raise RuntimeError("PyTorch fusion call count mismatch.")
        if fusion_counts["mlx_calls"] != expected_mlx_fusion_calls:
            raise RuntimeError("MLX fusion call count mismatch.")
        if rt["island_state"].calls != expected_baseline_deform_calls:
            raise RuntimeError("Deformable island call count mismatch.")
        if rt["msda_state"].call_count != expected_decoder_msda:
            raise RuntimeError("Decoder Metal MSDA call count mismatch.")

        result = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "experiment": "0038",
            "purpose": (
                "Balanced same-process cumulative ablation of the consolidated "
                "exact MetalGround runtime: v2 -> compiled MLX fusion -> "
                "corrected wide fusion+deformable islands -> exact fixed-prompt "
                "BERT cache."
            ),
            "model": args.model,
            "device": "mps",
            "dtype": "float32",
            "software": {
                "torch": torch.__version__,
                "transformers": transformers.__version__,
            },
            "protocol": {
                "rounds": args.rounds,
                "warmup_per_mode": args.warmup_per_mode,
                "modes": modes,
                "order": (
                    "four-way cyclic rotation; with rounds divisible by 4, "
                    "each mode occupies each execution position equally often"
                ),
                "prompt": args.prompt,
                "prompt_cache_prime_excluded": True,
                "approximation": False,
                "retraining": False,
                "reduced_precision": False,
            },
            "mode_definitions": {
                "v2": (
                    "PyTorch fully-folded fusion + six compiled MLX deformable "
                    "islands + decoder Metal MSDA"
                ),
                "mlx_fusion": (
                    "compiled MLX fully-folded fusion + separate compiled MLX "
                    "deformable islands + decoder Metal MSDA"
                ),
                "wide": (
                    "corrected single-boundary compiled MLX fusion+deformable "
                    "islands + decoder Metal MSDA"
                ),
                "wide_cache": (
                    "wide + exact fixed-prompt BERT text-backbone cache"
                ),
            },
            "latency": latency_stats,
            "within_round_deltas": delta_stats,
            "derived": {
                "speedup_mlx_fusion_vs_v2":
                    med["v2"] / med["mlx_fusion"],
                "speedup_wide_vs_mlx_fusion":
                    med["mlx_fusion"] / med["wide"],
                "speedup_prompt_cache_vs_wide":
                    med["wide"] / med["wide_cache"],
                "speedup_final_vs_v2":
                    med["v2"] / med["wide_cache"],
                "median_final_fps": 1000.0 / med["wide_cache"],
                "median_final_reduction_vs_v2_ms":
                    med["v2"] - med["wide_cache"],
                "median_final_reduction_vs_v2_percent":
                    100.0 * (med["v2"] - med["wide_cache"]) / med["v2"],
            },
            "correctness": correctness,
            "prompt_cache_exact_on_wide": cache_exact,
            "call_validation": call_validation,
            "round_records": round_records,
            "decision_rule": (
                "Promote the consolidated runtime if all three incremental "
                "within-round deltas remain materially positive, prompt caching "
                "is exact on top of the corrected wide path, and final top-900 "
                "membership remains preserved. Use a subsequent multi-process "
                "paired benchmark for the final headline rather than comparing "
                "against historical single-process medians."
            ),
            "notes": [
                "This is a cumulative same-process ablation, not a historical cross-run comparison.",
                "The corrected wide path derives the deformable valid-token mask inside MLX after importing a synchronized key_padding_mask.",
                "The prompt cache reuses only the initial image-independent BERT output; image-conditioned text enhancement and fusion still run each frame.",
                "No approximation, pruning, quantization, retraining, or reduced precision is used.",
                "Dataset-level accuracy equivalence remains unmeasured."
            ],
        }

        out = Path("results/metalground_consolidated_ablation.json")
        out.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")

        print("\n=== Experiment 0038 summary ===", flush=True)
        for mode in modes:
            print(
                f"{mode:12s}: {latency_stats[mode]['median_ms']:.3f} ms",
                flush=True,
            )
        print(
            f"final vs v2 within-round paired median: "
            f"{delta_stats['v2_minus_wide_cache']['median_ms']:+.3f} ms",
            flush=True,
        )
        print(
            f"final median FPS: "
            f"{result['derived']['median_final_fps']:.3f}",
            flush=True,
        )
        print(f"Saved: {out}", flush=True)

    finally:
        cache.restore()
        for d in rt["layer_dispatchers"]:
            d.restore()
        for fd in rt["fusion_dispatchers"]:
            fd.restore()


if __name__ == "__main__":
    main()
