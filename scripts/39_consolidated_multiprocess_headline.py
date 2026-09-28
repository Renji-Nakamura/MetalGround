#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import statistics
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

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
        "p10_ms": percentile(xs, 0.10),
        "p90_ms": percentile(xs, 0.90),
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


def run_timed(model, inputs, sync_fn):
    sync_fn()
    t0 = time.perf_counter_ns()
    with torch.inference_mode():
        out = model(**inputs)
    sync_fn()
    return out, (time.perf_counter_ns() - t0) / 1e6


def worker(args):
    if not torch.backends.mps.is_available():
        raise SystemExit("MPS unavailable.")

    h37 = load_module(args.exp37_helper, f"mg39_h37_{os.getpid()}")
    h38 = load_module(args.exp38_helper, f"mg39_h38_{os.getpid()}")

    print(f"[worker {args.worker_index}] building runtime...", flush=True)
    rt = h37.build_runtime(args)
    model = rt["model"]
    h14 = rt["h14"]
    h31 = rt["h31"]
    h36 = rt["h36"]
    inputs = rt["inputs"]

    cache = h38.ExactTextBackboneCache(model.model.text_backbone)

    try:
        # Prime exact prompt cache on the final runtime, excluded from timing.
        h38.set_runtime_mode(rt, cache, "wide_cache")
        cache.prime(model, inputs, h14.sync)

        # Three-way correctness:
        # original PyTorch/MPS vs controlled v2 vs final consolidated.
        h38.set_runtime_mode(rt, cache, "v2")
        with torch.inference_mode():
            v2_ref = model(**inputs)
            h14.sync()

        h38.set_runtime_mode(rt, cache, "wide_cache")
        with torch.inference_mode():
            final_ref = model(**inputs)
            h14.sync()

        correctness = {
            "v2_vs_original": pair_audit(
                v2_ref, rt["original_ref"], model, h14
            ),
            "final_vs_v2": pair_audit(
                final_ref, v2_ref, model, h14
            ),
            "final_vs_original": pair_audit(
                final_ref, rt["original_ref"], model, h14
            ),
        }

        # Warm both benchmark modes equally.
        for i in range(args.warmup_per_mode):
            h38.set_runtime_mode(rt, cache, "v2")
            _ = run_timed(model, inputs, h14.sync)
            h38.set_runtime_mode(rt, cache, "wide_cache")
            _ = run_timed(model, inputs, h14.sync)
            print(
                f"[worker {args.worker_index}] warmup "
                f"{i+1}/{args.warmup_per_mode}",
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

        v2_samples = []
        final_samples = []
        deltas = []

        worker_flip = args.worker_index % 2

        for i in range(args.pairs_per_process):
            v2_first = ((i + worker_flip) % 2 == 0)
            order = (
                ("v2", "final")
                if v2_first
                else ("final", "v2")
            )
            local = {}

            for mode in order:
                if mode == "v2":
                    h38.set_runtime_mode(rt, cache, "v2")
                else:
                    h38.set_runtime_mode(rt, cache, "wide_cache")

                _out, dt = run_timed(model, inputs, h14.sync)
                local[mode] = dt
                if mode == "v2":
                    v2_samples.append(dt)
                else:
                    final_samples.append(dt)

            delta = local["v2"] - local["final"]
            deltas.append(delta)

            print(
                f"[worker {args.worker_index}] pair "
                f"{i+1:02d}/{args.pairs_per_process}: "
                f"v2={local['v2']:.3f} "
                f"final={local['final']:.3f} "
                f"delta={delta:+.3f} ms",
                flush=True,
            )

        fusion_counts = h31.sum_counts(rt["fusion_dispatchers"])

        expected_each_encoder = args.pairs_per_process * 6
        expected_decoder = args.pairs_per_process * 2 * 6
        expected_cache_hits = args.pairs_per_process
        expected_cache_bypass = args.pairs_per_process

        # v2 uses baseline encoder-layer path; final uses fixed wide path.
        if rt["state"].baseline_calls != expected_each_encoder:
            raise RuntimeError(
                f"baseline encoder calls {rt['state'].baseline_calls} "
                f"!= {expected_each_encoder}"
            )
        if rt["state"].fixed_calls != expected_each_encoder:
            raise RuntimeError(
                f"fixed encoder calls {rt['state'].fixed_calls} "
                f"!= {expected_each_encoder}"
            )
        if fusion_counts["torch_calls"] != expected_each_encoder:
            raise RuntimeError(
                f"torch fusion calls {fusion_counts['torch_calls']} "
                f"!= {expected_each_encoder}"
            )
        if fusion_counts["mlx_calls"] != 0:
            raise RuntimeError(
                f"unexpected standalone MLX fusion calls: "
                f"{fusion_counts['mlx_calls']}"
            )
        if rt["island_state"].calls != expected_each_encoder:
            raise RuntimeError(
                f"separate deform calls {rt['island_state'].calls} "
                f"!= {expected_each_encoder}"
            )
        if rt["msda_state"].call_count != expected_decoder:
            raise RuntimeError(
                f"decoder MSDA calls {rt['msda_state'].call_count} "
                f"!= {expected_decoder}"
            )
        if cache.hits != expected_cache_hits or cache.misses != 0:
            raise RuntimeError(
                f"cache hits/misses {cache.hits}/{cache.misses}, "
                f"expected {expected_cache_hits}/0"
            )
        if cache.bypass != expected_cache_bypass:
            raise RuntimeError(
                f"cache bypass {cache.bypass} "
                f"!= {expected_cache_bypass}"
            )

        result = {
            "worker_index": args.worker_index,
            "pid": os.getpid(),
            "latency": {
                "controlled_v2": stats(v2_samples),
                "consolidated_final": stats(final_samples),
                "paired_delta_v2_minus_final": stats(deltas),
            },
            "derived": {
                "speedup_from_medians": (
                    statistics.median(v2_samples)
                    / statistics.median(final_samples)
                ),
                "median_difference_of_marginals_ms": (
                    statistics.median(v2_samples)
                    - statistics.median(final_samples)
                ),
                "final_median_fps": (
                    1000.0 / statistics.median(final_samples)
                ),
            },
            "correctness": correctness,
            "call_validation": {
                "baseline_encoder_layer_calls":
                    rt["state"].baseline_calls,
                "fixed_encoder_layer_calls":
                    rt["state"].fixed_calls,
                "fusion_torch_calls": fusion_counts["torch_calls"],
                "fusion_mlx_calls": fusion_counts["mlx_calls"],
                "separate_deformable_calls":
                    rt["island_state"].calls,
                "decoder_metal_msda_calls":
                    rt["msda_state"].call_count,
                "cache_hits": cache.hits,
                "cache_misses": cache.misses,
                "cache_bypass": cache.bypass,
                "expected_each_encoder_mode": expected_each_encoder,
                "expected_decoder_msda_calls": expected_decoder,
                "expected_cache_hits": expected_cache_hits,
                "expected_cache_bypass": expected_cache_bypass,
            },
        }

        args.worker_out.write_text(
            json.dumps(result, indent=2, ensure_ascii=False) + "\n"
        )

        print(
            f"[worker {args.worker_index}] paired median "
            f"{result['latency']['paired_delta_v2_minus_final']['median_ms']:+.3f} ms, "
            f"final median {result['latency']['consolidated_final']['median_ms']:.3f} ms",
            flush=True,
        )

    finally:
        cache.restore()
        for d in rt["layer_dispatchers"]:
            d.restore()
        for fd in rt["fusion_dispatchers"]:
            fd.restore()


def driver(args):
    script_path = Path(__file__).resolve()
    workers = []

    with tempfile.TemporaryDirectory(prefix="metalground_exp39_") as td:
        td_path = Path(td)

        for i in range(args.processes):
            worker_out = td_path / f"worker_{i}.json"
            cmd = [
                sys.executable,
                str(script_path),
                "--worker",
                "--worker-index", str(i),
                "--worker-out", str(worker_out),
                "--model", args.model,
                "--image", str(args.image),
                "--warmup-per-mode", str(args.warmup_per_mode),
                "--pairs-per-process", str(args.pairs_per_process),
                "--threadgroup", str(args.threadgroup),
                "--exp13-helper", str(args.exp13_helper),
                "--exp14-helper", str(args.exp14_helper),
                "--exp17-helper", str(args.exp17_helper),
                "--exp18-helper", str(args.exp18_helper),
                "--exp30-helper", str(args.exp30_helper),
                "--exp31-helper", str(args.exp31_helper),
                "--exp36-helper", str(args.exp36_helper),
                "--exp37-helper", str(args.exp37_helper),
                "--exp38-helper", str(args.exp38_helper),
                "--prompt",
                *args.prompt,
            ]

            print(
                f"\n=== fresh process {i+1}/{args.processes} ===",
                flush=True,
            )
            proc = subprocess.run(cmd, check=False)
            if proc.returncode != 0:
                raise RuntimeError(
                    f"Worker {i} failed with exit code {proc.returncode}"
                )
            workers.append(json.loads(worker_out.read_text()))

    process_deltas = [
        w["latency"]["paired_delta_v2_minus_final"]["median_ms"]
        for w in workers
    ]
    v2_medians = [
        w["latency"]["controlled_v2"]["median_ms"] for w in workers
    ]
    final_medians = [
        w["latency"]["consolidated_final"]["median_ms"] for w in workers
    ]

    positive = sum(x > 0 for x in process_deltas)

    membership_preserved = sum(
        w["correctness"]["final_vs_original"]["topk"]["set_overlap"] == 900
        and w["correctness"]["final_vs_original"]["topk"][
            "changed_membership_each_side"
        ] == 0
        for w in workers
    )

    final_vs_v2_membership_preserved = sum(
        w["correctness"]["final_vs_v2"]["topk"]["set_overlap"] == 900
        and w["correctness"]["final_vs_v2"]["topk"][
            "changed_membership_each_side"
        ] == 0
        for w in workers
    )

    median_v2 = statistics.median(v2_medians)
    median_final = statistics.median(final_medians)
    median_delta = statistics.median(process_deltas)

    result = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "experiment": "0039",
        "purpose": (
            "Multi-process paired headline validation of the consolidated "
            "MetalGround runtime against the controlled MetalGround-v2 baseline."
        ),
        "model": args.model,
        "device": "mps",
        "dtype": "float32",
        "software": {
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "python": sys.version,
        },
        "protocol": {
            "fresh_processes": args.processes,
            "pairs_per_process": args.pairs_per_process,
            "warmup_per_mode": args.warmup_per_mode,
            "pair_order": (
                "alternating v2/final; initial order balanced by process parity"
            ),
            "prompt_cache": "enabled only in consolidated final mode",
            "primary_performance_unit": "per-process paired-delta median",
        },
        "runtime_definitions": {
            "controlled_v2": (
                "PyTorch fully-folded fusion + six separate compiled-MLX "
                "deformable islands + decoder Metal MSDA"
            ),
            "consolidated_final": (
                "corrected single-boundary compiled-MLX fusion+deformable "
                "islands + exact fixed-prompt BERT cache + decoder Metal MSDA"
            ),
        },
        "workers": workers,
        "aggregate": {
            "paired_delta_medians_across_processes": stats(process_deltas),
            "positive_process_count": positive,
            "total_process_count": args.processes,
            "median_of_process_paired_medians_ms": median_delta,
            "controlled_v2_process_medians": stats(v2_medians),
            "final_process_medians": stats(final_medians),
            "median_absolute_controlled_v2_ms": median_v2,
            "median_absolute_final_ms": median_final,
            "speedup_from_median_process_medians":
                median_v2 / median_final,
            "final_fps_from_median_process_median":
                1000.0 / median_final,
            "final_reduction_vs_controlled_v2_ms":
                median_v2 - median_final,
            "final_reduction_vs_controlled_v2_percent":
                100.0 * (median_v2 - median_final) / median_v2,
            "positive_processes": positive,
            "final_vs_original_topk_membership_preserved_processes":
                membership_preserved,
            "final_vs_v2_topk_membership_preserved_processes":
                final_vs_v2_membership_preserved,
            "cross_process_final_median_range_ms":
                max(final_medians) - min(final_medians),
        },
        "decision_rule": (
            "Promote the consolidated runtime if all or nearly all fresh "
            "processes show a large positive v2-minus-final paired median, "
            "and top-900 membership is preserved in all processes. Treat the "
            "aggregate final absolute median as a robust multi-process "
            "runtime estimate; keep historical original-PyTorch comparisons "
            "separate from the causal v2-to-final paired effect."
        ),
        "notes": [
            "Each worker independently rebuilds the runtime.",
            "Both modes run on the same model instance inside each worker.",
            "The final mode combines corrected wide encoder islands and exact fixed-prompt BERT caching.",
            "The prompt-cache prime is excluded from timing.",
            "No approximation, pruning, quantization, retraining, or reduced precision is used.",
            "Dataset-level accuracy equivalence remains unmeasured."
        ],
    }

    out = Path("results/metalground_consolidated_multiprocess_headline.json")
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")

    print("\n=== Experiment 0039 aggregate ===", flush=True)
    for w in workers:
        d = w["latency"]["paired_delta_v2_minus_final"]["median_ms"]
        b = w["latency"]["controlled_v2"]["median_ms"]
        f = w["latency"]["consolidated_final"]["median_ms"]
        tk = w["correctness"]["final_vs_original"]["topk"]
        print(
            f"process {w['worker_index']}: "
            f"v2={b:.3f} ms, final={f:.3f} ms, "
            f"paired={d:+.3f} ms, "
            f"final-vs-original overlap={tk['set_overlap']}/900, "
            f"rankwise={tk['rankwise_identical']}/900",
            flush=True,
        )

    print(
        f"positive processes: {positive}/{args.processes}",
        flush=True,
    )
    print(
        "median of process paired medians: "
        f"{median_delta:+.3f} ms",
        flush=True,
    )
    print(
        f"aggregate final median: {median_final:.3f} ms "
        f"({1000.0/median_final:.3f} FPS)",
        flush=True,
    )
    print(f"Saved: {out}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker", action="store_true")
    ap.add_argument("--worker-index", type=int, default=0)
    ap.add_argument("--worker-out", type=Path)
    ap.add_argument("--processes", type=int, default=5)
    ap.add_argument("--pairs-per-process", type=int, default=8)
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
    ap.add_argument(
        "--exp38-helper",
        type=Path,
        default=Path("scripts/38_consolidated_runtime_ablation.py"),
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
    ):
        if not p.exists():
            raise SystemExit(f"Missing helper: {p}")

    if args.worker:
        if args.worker_out is None:
            raise SystemExit("--worker requires --worker-out")
        worker(args)
    else:
        driver(args)


if __name__ == "__main__":
    main()
