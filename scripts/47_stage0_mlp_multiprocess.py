#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import torch


def load_module(path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load helper: {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod


def pct(xs, p):
    ys = sorted(xs)
    if len(ys) == 1:
        return ys[0]
    k = (len(ys) - 1) * p
    lo = int(k)
    hi = min(lo + 1, len(ys) - 1)
    f = k - lo
    return ys[lo] * (1 - f) + ys[hi] * f


def stats(xs):
    return {
        "n": len(xs),
        "mean_ms": statistics.fmean(xs),
        "median_ms": statistics.median(xs),
        "p90_ms": pct(xs, 0.90),
        "p95_ms": pct(xs, 0.95),
        "min_ms": min(xs),
        "max_ms": max(xs),
    }


def worker(args):
    if not torch.backends.mps.is_available():
        raise SystemExit("MPS unavailable.")

    h37 = load_module(args.exp37_helper, f"mg47_h37_{args.worker_index}")
    h38 = load_module(args.exp38_helper, f"mg47_h38_{args.worker_index}")
    h45 = load_module(args.exp45_helper, f"mg47_h45_{args.worker_index}")
    h46 = load_module(args.exp46_helper, f"mg47_h46_{args.worker_index}")

    print(
        f"[worker {args.worker_index}] building fresh runtime",
        flush=True,
    )
    rt = h37.build_runtime(args)
    model = rt["model"]
    h14 = rt["h14"]
    inputs = rt["inputs"]

    cache = h38.ExactTextBackboneCache(model.model.text_backbone)
    dispatchers = []

    try:
        h38.set_runtime_mode(rt, cache, "wide_cache")
        cache.prime(model, inputs, h14.sync)
        cache.set_enabled(True)

        stages = model.model.backbone.conv_encoder.model.swin.encoder.layers
        stage0 = stages[0]
        mlp_modules = [stage0.blocks[0].mlp, stage0.blocks[1].mlp]

        # Parameters are PyTorch/MPS producers; synchronize before shared import.
        h14.sync()
        candidates = [h45.MlxSwinMLP(m) for m in mlp_modules]
        dispatchers = [
            h46.SwinMlpDispatcher(m, c, h14.sync)
            for m, c in zip(mlp_modules, candidates)
        ]

        def set_mode(mode):
            for d in dispatchers:
                d.set_mode(mode)

        # First MLX compilation/execution excluded.
        set_mode("mlx")
        with torch.inference_mode():
            _ = model(**inputs)
        h14.sync()

        # Correctness preflight.
        set_mode("current")
        with torch.inference_mode():
            current_ref = model(**inputs)
        h14.sync()

        set_mode("mlx")
        with torch.inference_mode():
            candidate_ref = model(**inputs)
        h14.sync()

        correctness = {
            "candidate_vs_current": h46.audit(
                candidate_ref, current_ref, model, h14
            ),
            "current_vs_original": h46.audit(
                current_ref, rt["original_ref"], model, h14
            ),
            "candidate_vs_original": h46.audit(
                candidate_ref, rt["original_ref"], model, h14
            ),
        }

        topk_cc = correctness["candidate_vs_current"]["topk"]
        membership_pass = (
            topk_cc["set_overlap"] == model.config.num_queries
            and topk_cc["changed_membership_each_side"] == 0
        )

        if not membership_pass:
            result = {
                "worker_index": args.worker_index,
                "correctness": correctness,
                "topk_membership_gate_passed": False,
                "latency": None,
                "call_validation": None,
            }
            args.worker_output.parent.mkdir(parents=True, exist_ok=True)
            args.worker_output.write_text(
                json.dumps(result, indent=2) + "\n"
            )
            print(
                f"[worker {args.worker_index}] membership FAILED; timing skipped",
                flush=True,
            )
            return

        # Balanced per-mode warmup.
        for i in range(args.warmup_per_mode):
            set_mode("current")
            _ = h46.timed_forward(model, inputs, h14.sync)
            set_mode("mlx")
            _ = h46.timed_forward(model, inputs, h14.sync)
            print(
                f"[worker {args.worker_index}] warmup "
                f"{i+1}/{args.warmup_per_mode}",
                flush=True,
            )

        # Benchmark-only counters.
        for d in dispatchers:
            d.reset_counts()
        cache.reset_counts()
        rt["msda_state"].call_count = 0

        current_samples = []
        candidate_samples = []
        deltas = []

        for i in range(args.pairs_per_process):
            order = (
                ("current", "mlx")
                if i % 2 == 0
                else ("mlx", "current")
            )
            local = {}

            for mode in order:
                set_mode(mode)
                _out, dt = h46.timed_forward(model, inputs, h14.sync)
                local[mode] = dt
                if mode == "current":
                    current_samples.append(dt)
                else:
                    candidate_samples.append(dt)

            delta = local["current"] - local["mlx"]
            deltas.append(delta)

            print(
                f"[worker {args.worker_index}] pair "
                f"{i+1:02d}/{args.pairs_per_process}: "
                f"current={local['current']:.3f} ms "
                f"candidate={local['mlx']:.3f} ms "
                f"delta={delta:+.3f} ms",
                flush=True,
            )

        expected_each = args.pairs_per_process
        expected_cache_hits = args.pairs_per_process * 2
        expected_decoder_msda = args.pairs_per_process * 2 * 6

        call_validation = {
            "block0_current_calls": dispatchers[0].current_calls,
            "block0_mlx_calls": dispatchers[0].mlx_calls,
            "block1_current_calls": dispatchers[1].current_calls,
            "block1_mlx_calls": dispatchers[1].mlx_calls,
            "expected_each_mode_per_mlp": expected_each,
            "cache_hits": cache.hits,
            "cache_misses": cache.misses,
            "expected_cache_hits": expected_cache_hits,
            "decoder_metal_msda_calls": rt["msda_state"].call_count,
            "expected_decoder_metal_msda_calls": expected_decoder_msda,
        }

        calls_pass = (
            dispatchers[0].current_calls == expected_each
            and dispatchers[0].mlx_calls == expected_each
            and dispatchers[1].current_calls == expected_each
            and dispatchers[1].mlx_calls == expected_each
            and cache.hits == expected_cache_hits
            and cache.misses == 0
            and rt["msda_state"].call_count == expected_decoder_msda
        )

        result = {
            "worker_index": args.worker_index,
            "correctness": correctness,
            "topk_membership_gate_passed": membership_pass,
            "latency": {
                "current_consolidated": stats(current_samples),
                "stage0_two_mlp_mlx": stats(candidate_samples),
                "paired_delta_current_minus_candidate": stats(deltas),
            },
            "paired_deltas_ms": deltas,
            "call_validation": call_validation,
            "call_validation_passed": calls_pass,
        }

        args.worker_output.parent.mkdir(parents=True, exist_ok=True)
        args.worker_output.write_text(
            json.dumps(result, indent=2) + "\n"
        )

        print(
            f"[worker {args.worker_index}] "
            f"paired median={statistics.median(deltas):+.3f} ms "
            f"membership={membership_pass} calls={calls_pass}",
            flush=True,
        )

    finally:
        for d in dispatchers:
            try:
                d.restore()
            except Exception:
                pass
        cache.restore()
        for d in rt["layer_dispatchers"]:
            d.restore()
        for fd in rt["fusion_dispatchers"]:
            fd.restore()


def parent(args):
    worker_dir = Path("results/exp0047_workers")
    worker_dir.mkdir(parents=True, exist_ok=True)

    worker_results = []

    for i in range(args.processes):
        worker_out = worker_dir / f"worker_{i}.json"
        if worker_out.exists():
            worker_out.unlink()

        cmd = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--worker",
            "--worker-index", str(i),
            "--worker-output", str(worker_out),
            "--pairs-per-process", str(args.pairs_per_process),
            "--warmup-per-mode", str(args.warmup_per_mode),
            "--model", args.model,
            "--image", str(args.image),
            "--threadgroup", str(args.threadgroup),
        ]

        for prompt in args.prompt:
            # Prompt is passed below in one explicit --prompt group.
            pass
        cmd += ["--prompt", *args.prompt]

        helper_fields = [
            "exp13_helper", "exp14_helper", "exp17_helper",
            "exp18_helper", "exp30_helper", "exp31_helper",
            "exp36_helper", "exp37_helper", "exp38_helper",
            "exp45_helper", "exp46_helper",
        ]
        for field in helper_fields:
            flag = "--" + field.replace("_", "-")
            cmd += [flag, str(getattr(args, field))]

        print(
            f"\n=== fresh process {i+1}/{args.processes} ===",
            flush=True,
        )
        completed = subprocess.run(cmd)
        if completed.returncode != 0:
            raise SystemExit(
                f"worker {i} failed with exit code {completed.returncode}"
            )
        if not worker_out.exists():
            raise SystemExit(f"worker {i} produced no result file")

        worker_results.append(json.loads(worker_out.read_text()))

    membership_all = all(
        w["topk_membership_gate_passed"] for w in worker_results
    )
    calls_all = all(
        bool(w.get("call_validation_passed", False))
        for w in worker_results
    )

    per_process_paired_medians = [
        w["latency"]["paired_delta_current_minus_candidate"]["median_ms"]
        for w in worker_results
    ]
    per_process_current_medians = [
        w["latency"]["current_consolidated"]["median_ms"]
        for w in worker_results
    ]
    per_process_candidate_medians = [
        w["latency"]["stage0_two_mlp_mlx"]["median_ms"]
        for w in worker_results
    ]

    positive_processes = sum(
        x > 0 for x in per_process_paired_medians
    )
    median_of_paired_medians = statistics.median(
        per_process_paired_medians
    )
    current_median_of_process_medians = statistics.median(
        per_process_current_medians
    )
    candidate_median_of_process_medians = statistics.median(
        per_process_candidate_medians
    )

    # Adoption is intentionally stricter than a single-process pass.
    robust_go = (
        membership_all
        and calls_all
        and positive_processes == args.processes
        and median_of_paired_medians >= 5.0
    )

    result = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "experiment": "0047",
        "purpose": (
            "Fresh-process robustness replication of the two-stage0-Swin-MLP "
            "full-model optimization that passed the same-process gate in "
            "Experiment 0046."
        ),
        "model": args.model,
        "device": "mps",
        "dtype": "float32",
        "configuration": {
            "processes": args.processes,
            "pairs_per_process": args.pairs_per_process,
            "warmup_per_mode_per_process": args.warmup_per_mode,
            "baseline": (
                "current consolidated runtime: corrected encoder wide islands "
                "+ exact fixed-prompt BERT cache + decoder Metal MSDA"
            ),
            "candidate_delta": (
                "stage0.block0.mlp and stage0.block1.mlp replaced by compiled "
                "MLX exact fc1->erf-GELU->fc2"
            ),
            "fresh_runtime_rebuild_each_process": True,
            "fresh_torch_to_mlx_bridge_each_mlp_call": True,
            "final_mlx_to_torch_bridge_each_mlp_call": True,
            "approximation": False,
            "retraining": False,
            "reduced_precision": False,
        },
        "workers": worker_results,
        "aggregate": {
            "topk_membership_passed_all_processes": membership_all,
            "call_validation_passed_all_processes": calls_all,
            "positive_paired_median_processes": positive_processes,
            "total_processes": args.processes,
            "per_process_paired_medians_ms":
                per_process_paired_medians,
            "median_of_process_paired_medians_ms":
                median_of_paired_medians,
            "per_process_current_medians_ms":
                per_process_current_medians,
            "per_process_candidate_medians_ms":
                per_process_candidate_medians,
            "current_median_of_process_medians_ms":
                current_median_of_process_medians,
            "candidate_median_of_process_medians_ms":
                candidate_median_of_process_medians,
            "candidate_fps_from_median_process_median":
                1000.0 / candidate_median_of_process_medians,
            "speedup_from_median_process_medians":
                current_median_of_process_medians
                / candidate_median_of_process_medians,
        },
        "decision_rule": (
            "ROBUST GO / adopt only if all fresh processes preserve top-900 "
            "membership, all call validations pass, every process has a "
            "positive paired median, and the median of per-process paired "
            "medians is at least 5 ms."
        ),
        "robust_go": robust_go,
        "notes": [
            "Primary causal metric is the median of per-process paired medians, not a cross-process subtraction of unrelated historical authorities.",
            "Absolute medians are reported only from this controlled fresh-process protocol.",
            "No approximation, pruning, quantization, retraining, or reduced precision is used.",
            "Experiment 0039 remains the robust runtime authority unless this experiment passes the robust adoption rule."
        ],
    }

    out = Path(
        "results/metalground_stage0_mlp_multiprocess.json"
    )
    out.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n"
    )

    print("\n=== Experiment 0047 aggregate ===", flush=True)
    print(
        "paired medians per process: "
        + ", ".join(f"{x:+.3f}" for x in per_process_paired_medians)
        + " ms",
        flush=True,
    )
    print(
        f"positive processes: {positive_processes}/{args.processes}",
        flush=True,
    )
    print(
        f"median of paired medians: "
        f"{median_of_paired_medians:+.3f} ms",
        flush=True,
    )
    print(
        f"controlled current median-of-process-medians: "
        f"{current_median_of_process_medians:.3f} ms",
        flush=True,
    )
    print(
        f"candidate median-of-process-medians: "
        f"{candidate_median_of_process_medians:.3f} ms "
        f"({1000.0/candidate_median_of_process_medians:.3f} FPS)",
        flush=True,
    )
    print(f"membership all: {membership_all}", flush=True)
    print(f"call validation all: {calls_all}", flush=True)
    print(f"ROBUST GO: {robust_go}", flush=True)
    print(f"Saved: {out}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker", action="store_true")
    ap.add_argument("--worker-index", type=int, default=0)
    ap.add_argument("--worker-output", type=Path)

    ap.add_argument("--processes", type=int, default=5)
    ap.add_argument("--pairs-per-process", type=int, default=8)
    ap.add_argument("--warmup-per-mode", type=int, default=2)

    ap.add_argument("--model", default="IDEA-Research/grounding-dino-tiny")
    ap.add_argument("--image", type=Path, default=Path("assets/input.jpg"))
    ap.add_argument("--prompt", nargs="+", default=["a cat", "a dog"])
    ap.add_argument("--threadgroup", type=int, default=256)

    helpers = {
        13: "bench_fusion_algebraic_specialization.py",
        14: "full_model_fusion_algebraic.py",
        17: "bench_deformable_mlx_island.py",
        18: "full_model_deformable_islands.py",
        30: "bench_fully_folded_fusion_mlx.py",
        31: "full_model_mlx_fusion_paired.py",
        36: "wide_island_mask_sync_fix.py",
        37: "corrected_wide_island_multiprocess.py",
        38: "consolidated_runtime_ablation.py",
        45: "bench_swin_stage0_mlp_mlx.py",
        46: "full_model_stage0_mlp_mlx.py",
    }

    for n, fn in helpers.items():
        ap.add_argument(
            f"--exp{n}-helper",
            type=Path,
            default=Path("scripts") / f"{n:02d}_{fn}",
        )

    args = ap.parse_args()

    for n in helpers:
        p = getattr(args, f"exp{n}_helper")
        if not p.exists():
            raise SystemExit(f"Missing helper: {p}")

    if args.worker:
        if args.worker_output is None:
            raise SystemExit("--worker-output required in worker mode")
        worker(args)
    else:
        parent(args)


if __name__ == "__main__":
    main()
