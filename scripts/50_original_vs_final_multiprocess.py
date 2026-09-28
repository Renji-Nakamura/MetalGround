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
from transformers import AutoModelForZeroShotObjectDetection


def load_module(path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load helper: {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
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
        "mean": statistics.fmean(xs),
        "median": statistics.median(xs),
        "p10": percentile(xs, 0.10),
        "p90": percentile(xs, 0.90),
        "p95": percentile(xs, 0.95),
        "min": min(xs),
        "max": max(xs),
    }


def stats_ms(xs):
    s = stats(xs)
    return {f"{k}_ms" if k != "n" else k: v for k, v in s.items()}


def timed_forward(model, inputs, sync_fn):
    sync_fn()
    t0 = time.perf_counter_ns()
    with torch.inference_mode():
        out = model(**inputs)
    sync_fn()
    return out, (time.perf_counter_ns() - t0) / 1e6


def worker(args):
    if not torch.backends.mps.is_available():
        raise SystemExit("MPS unavailable.")

    tag = f"{os.getpid()}_{args.worker_index}"
    h37 = load_module(args.exp37_helper, f"mg50_h37_{tag}")
    h38 = load_module(args.exp38_helper, f"mg50_h38_{tag}")
    h45 = load_module(args.exp45_helper, f"mg50_h45_{tag}")
    h46 = load_module(args.exp46_helper, f"mg50_h46_{tag}")

    print(
        f"[worker {args.worker_index}] building adopted MetalGround runtime...",
        flush=True,
    )
    rt = h37.build_runtime(args)
    final_model = rt["model"]
    inputs = rt["inputs"]
    h14 = rt["h14"]

    cache = h38.ExactTextBackboneCache(
        final_model.model.text_backbone
    )
    stage0_dispatchers = []

    try:
        # Adopted pre-0050 runtime:
        # corrected wide encoder islands + exact prompt cache +
        # decoder Metal MSDA + robust stage0 two-MLP MLX replacement.
        h38.set_runtime_mode(rt, cache, "wide_cache")
        cache.prime(final_model, inputs, h14.sync)
        cache.set_enabled(True)

        stages = (
            final_model.model.backbone.conv_encoder.model.swin.encoder.layers
        )
        stage0_modules = [
            stages[0].blocks[0].mlp,
            stages[0].blocks[1].mlp,
        ]

        h14.sync()
        stage0_candidates = [
            h45.MlxSwinMLP(m) for m in stage0_modules
        ]
        stage0_dispatchers = [
            h46.SwinMlpDispatcher(m, c, h14.sync)
            for m, c in zip(stage0_modules, stage0_candidates)
        ]
        for d in stage0_dispatchers:
            d.set_mode("mlx")

        # Compile/prime adopted MLX MLPs and prompt cache outside timing.
        with torch.inference_mode():
            _ = final_model(**inputs)
        h14.sync()

        # Build a second, completely untouched HF/PyTorch/MPS model in the
        # same process. It receives the exact same already-preprocessed inputs.
        print(
            f"[worker {args.worker_index}] building untouched original model...",
            flush=True,
        )
        original_model = (
            AutoModelForZeroShotObjectDetection
            .from_pretrained(args.model)
            .eval()
            .to("mps")
        )
        h14.sync()

        if bool(original_model.config.output_attentions):
            raise RuntimeError(
                "Experiment 0050 requires output_attentions=False."
            )

        # Optional context-only memory snapshot after both models are resident.
        memory_context = {}
        try:
            memory_context["mps_current_allocated_bytes"] = int(
                torch.mps.current_allocated_memory()
            )
        except Exception:
            pass
        try:
            memory_context["mps_driver_allocated_bytes"] = int(
                torch.mps.driver_allocated_memory()
            )
        except Exception:
            pass

        # ---------------- Correctness ----------------
        print(
            f"[worker {args.worker_index}] correctness preflight...",
            flush=True,
        )

        with torch.inference_mode():
            original_ref_2 = original_model(**inputs)
        h14.sync()

        with torch.inference_mode():
            final_ref = final_model(**inputs)
        h14.sync()

        correctness = {
            # Verifies that the separately loaded untouched model is a valid
            # baseline for the same checkpoint/input.
            "second_original_vs_build_runtime_original": h46.audit(
                original_ref_2,
                rt["original_ref"],
                final_model,
                h14,
            ),
            "final_vs_original": h46.audit(
                final_ref,
                original_ref_2,
                final_model,
                h14,
            ),
        }

        topk = correctness["final_vs_original"]["topk"]
        membership_pass = (
            topk["set_overlap"] == final_model.config.num_queries
            and topk["changed_membership_each_side"] == 0
        )

        original_identity = correctness[
            "second_original_vs_build_runtime_original"
        ]
        original_identity_pass = (
            original_identity["topk"]["rankwise_identical"]
            == final_model.config.num_queries
            and original_identity["logits"]["allclose_1e-5"]
            and original_identity["pred_boxes"]["allclose_1e-5"]
        )

        if not membership_pass or not original_identity_pass:
            result = {
                "worker_index": args.worker_index,
                "pid": os.getpid(),
                "correctness": correctness,
                "topk_membership_gate_passed": membership_pass,
                "original_model_identity_gate_passed":
                    original_identity_pass,
                "latency": None,
                "call_validation": None,
                "memory_context": memory_context,
            }
            args.worker_out.write_text(
                json.dumps(result, indent=2, ensure_ascii=False) + "\n"
            )
            print(
                f"[worker {args.worker_index}] correctness gate failed; "
                "timing skipped",
                flush=True,
            )
            return

        # ---------------- Balanced warmup ----------------
        worker_flip = args.worker_index % 2
        for i in range(args.warmup_per_mode):
            original_first = ((i + worker_flip) % 2 == 0)
            order = (
                ("original", "final")
                if original_first
                else ("final", "original")
            )
            for mode in order:
                if mode == "original":
                    _ = timed_forward(
                        original_model, inputs, h14.sync
                    )
                else:
                    _ = timed_forward(
                        final_model, inputs, h14.sync
                    )
            print(
                f"[worker {args.worker_index}] warmup "
                f"{i+1}/{args.warmup_per_mode}",
                flush=True,
            )

        # Reset candidate-only benchmark counters after all preflight/warmup.
        for d in stage0_dispatchers:
            d.reset_counts()
        cache.reset_counts()
        rt["msda_state"].call_count = 0
        rt["state"].baseline_calls = 0
        rt["state"].fixed_calls = 0

        original_samples = []
        final_samples = []
        deltas = []
        pairwise_speedups = []

        for i in range(args.pairs_per_process):
            original_first = ((i + worker_flip) % 2 == 0)
            order = (
                ("original", "final")
                if original_first
                else ("final", "original")
            )
            local = {}

            for mode in order:
                if mode == "original":
                    _out, dt = timed_forward(
                        original_model, inputs, h14.sync
                    )
                    original_samples.append(dt)
                else:
                    _out, dt = timed_forward(
                        final_model, inputs, h14.sync
                    )
                    final_samples.append(dt)
                local[mode] = dt

            delta = local["original"] - local["final"]
            speedup = local["original"] / local["final"]
            deltas.append(delta)
            pairwise_speedups.append(speedup)

            print(
                f"[worker {args.worker_index}] pair "
                f"{i+1:02d}/{args.pairs_per_process}: "
                f"original={local['original']:.3f} ms "
                f"final={local['final']:.3f} ms "
                f"delta={delta:+.3f} ms "
                f"speedup={speedup:.3f}x",
                flush=True,
            )

        expected_final_calls = args.pairs_per_process
        expected_stage0_each = expected_final_calls
        expected_cache_hits = expected_final_calls
        expected_decoder_msda = expected_final_calls * 6
        expected_fixed_encoder = expected_final_calls * 6

        call_validation = {
            "stage0_block0_mlx_calls":
                stage0_dispatchers[0].mlx_calls,
            "stage0_block1_mlx_calls":
                stage0_dispatchers[1].mlx_calls,
            "expected_each_stage0_mlx_calls":
                expected_stage0_each,
            "prompt_cache_hits": cache.hits,
            "prompt_cache_misses": cache.misses,
            "prompt_cache_bypass": cache.bypass,
            "expected_prompt_cache_hits": expected_cache_hits,
            "decoder_metal_msda_calls":
                rt["msda_state"].call_count,
            "expected_decoder_metal_msda_calls":
                expected_decoder_msda,
            "fixed_encoder_layer_calls": rt["state"].fixed_calls,
            "baseline_encoder_layer_calls":
                rt["state"].baseline_calls,
            "expected_fixed_encoder_layer_calls":
                expected_fixed_encoder,
        }

        calls_pass = (
            stage0_dispatchers[0].mlx_calls
            == expected_stage0_each
            and stage0_dispatchers[1].mlx_calls
            == expected_stage0_each
            and cache.hits == expected_cache_hits
            and cache.misses == 0
            and cache.bypass == 0
            and rt["msda_state"].call_count
            == expected_decoder_msda
            and rt["state"].fixed_calls
            == expected_fixed_encoder
            and rt["state"].baseline_calls == 0
        )

        result = {
            "worker_index": args.worker_index,
            "pid": os.getpid(),
            "correctness": correctness,
            "topk_membership_gate_passed": membership_pass,
            "original_model_identity_gate_passed":
                original_identity_pass,
            "latency": {
                "untouched_original_pytorch_mps":
                    stats_ms(original_samples),
                "adopted_metalground_final":
                    stats_ms(final_samples),
                "paired_delta_original_minus_final":
                    stats_ms(deltas),
                "pairwise_speedup_original_over_final":
                    stats(pairwise_speedups),
            },
            "call_validation": call_validation,
            "call_validation_passed": calls_pass,
            "memory_context": memory_context,
        }

        args.worker_out.write_text(
            json.dumps(result, indent=2, ensure_ascii=False) + "\n"
        )

        print(
            f"[worker {args.worker_index}] "
            f"paired delta median="
            f"{statistics.median(deltas):+.3f} ms, "
            f"paired speedup median="
            f"{statistics.median(pairwise_speedups):.3f}x",
            flush=True,
        )

    finally:
        for d in stage0_dispatchers:
            try:
                d.restore()
            except Exception:
                pass
        cache.restore()
        for d in rt["layer_dispatchers"]:
            d.restore()
        for fd in rt["fusion_dispatchers"]:
            fd.restore()


def driver(args):
    script = Path(__file__).resolve()
    workers = []

    with tempfile.TemporaryDirectory(
        prefix="metalground_exp50_"
    ) as td:
        td = Path(td)

        for i in range(args.processes):
            out = td / f"worker_{i}.json"

            cmd = [
                sys.executable,
                str(script),
                "--worker",
                "--worker-index", str(i),
                "--worker-out", str(out),
                "--model", args.model,
                "--image", str(args.image),
                "--threadgroup", str(args.threadgroup),
                "--warmup-per-mode", str(args.warmup_per_mode),
                "--pairs-per-process", str(args.pairs_per_process),
                "--prompt", *args.prompt,
            ]

            helper_fields = [
                "exp13_helper", "exp14_helper", "exp17_helper",
                "exp18_helper", "exp30_helper", "exp31_helper",
                "exp36_helper", "exp37_helper", "exp38_helper",
                "exp45_helper", "exp46_helper",
            ]
            for field in helper_fields:
                cmd += [
                    "--" + field.replace("_", "-"),
                    str(getattr(args, field)),
                ]

            print(
                f"\n=== fresh process {i+1}/{args.processes} ===",
                flush=True,
            )
            proc = subprocess.run(cmd, check=False)
            if proc.returncode != 0:
                raise RuntimeError(
                    f"Worker {i} failed with exit code "
                    f"{proc.returncode}"
                )
            if not out.exists():
                raise RuntimeError(
                    f"Worker {i} produced no output."
                )
            workers.append(json.loads(out.read_text()))

    if not all(w["latency"] is not None for w in workers):
        raise RuntimeError(
            "At least one worker failed correctness before timing."
        )

    deltas = [
        w["latency"]["paired_delta_original_minus_final"][
            "median_ms"
        ]
        for w in workers
    ]
    speedups = [
        w["latency"]["pairwise_speedup_original_over_final"][
            "median"
        ]
        for w in workers
    ]
    original_medians = [
        w["latency"]["untouched_original_pytorch_mps"][
            "median_ms"
        ]
        for w in workers
    ]
    final_medians = [
        w["latency"]["adopted_metalground_final"][
            "median_ms"
        ]
        for w in workers
    ]

    membership_all = all(
        w["topk_membership_gate_passed"] for w in workers
    )
    identity_all = all(
        w["original_model_identity_gate_passed"]
        for w in workers
    )
    calls_all = all(
        w["call_validation_passed"] for w in workers
    )
    positive_all = all(x > 0 for x in deltas)
    speedup_all = all(x > 1.0 for x in speedups)

    median_delta = statistics.median(deltas)
    median_pairwise_speedup = statistics.median(speedups)
    median_original = statistics.median(original_medians)
    median_final = statistics.median(final_medians)

    robust_headline_pass = (
        membership_all
        and identity_all
        and calls_all
        and positive_all
        and speedup_all
    )

    result = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "experiment": "0050",
        "purpose": (
            "Direct same-protocol end-to-end comparison of untouched "
            "Hugging Face PyTorch/MPS Grounding DINO against the currently "
            "adopted MetalGround runtime, using alternating paired timing "
            "inside independently rebuilt fresh processes."
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
            "warmup_per_mode_per_process":
                args.warmup_per_mode,
            "same_process_pairing": True,
            "two_model_instances_per_worker": True,
            "pair_order": (
                "alternating original/final; initial order balanced "
                "by worker parity"
            ),
            "same_preprocessed_mps_inputs": True,
            "primary_causal_metrics": [
                "median of per-process paired latency-delta medians",
                "median of per-process pairwise-speedup medians",
            ],
        },
        "runtime_definitions": {
            "original": (
                "untouched transformers 5.17 Grounding DINO on "
                "PyTorch/MPS FP32"
            ),
            "metalground_final": (
                "corrected wide encoder MLX islands + exact fixed-prompt "
                "BERT cache + decoder custom Metal MSDA + robust exact "
                "compiled-MLX stage0 block0/block1 Swin MLPs"
            ),
        },
        "workers": workers,
        "aggregate": {
            "all_original_identity_checks_passed": identity_all,
            "all_final_topk_membership_checks_passed":
                membership_all,
            "all_call_validations_passed": calls_all,
            "all_process_paired_delta_medians_positive":
                positive_all,
            "all_process_pairwise_speedup_medians_gt_1":
                speedup_all,
            "per_process_paired_delta_medians_ms": deltas,
            "median_of_process_paired_delta_medians_ms":
                median_delta,
            "per_process_pairwise_speedup_medians": speedups,
            "median_of_process_pairwise_speedup_medians":
                median_pairwise_speedup,
            "per_process_original_medians_ms":
                original_medians,
            "per_process_final_medians_ms": final_medians,
            "original_median_of_process_medians_ms":
                median_original,
            "final_median_of_process_medians_ms":
                median_final,
            "ratio_of_median_process_medians":
                median_original / median_final,
            "original_fps_from_median_process_median":
                1000.0 / median_original,
            "final_fps_from_median_process_median":
                1000.0 / median_final,
            "robust_headline_pass": robust_headline_pass,
        },
        "decision_rule": (
            "Promote a direct end-to-end speedup headline only if the "
            "separately loaded untouched original model matches the "
            "build-runtime original reference in every process, final "
            "top-900 membership is preserved in every process, all call "
            "validations pass, and every process has both a positive "
            "original-minus-final paired median and pairwise speedup "
            "median >1."
        ),
        "notes": [
            "This experiment is specifically designed to replace the previously non-causal historical 999.117 ms / 476.360 ms ratio with a direct matched-protocol comparison.",
            "Both models are resident in the same worker process; this may shift absolute latency through memory pressure, so Experiment 0047 remains the preferred standalone absolute runtime authority.",
            "The pairwise speedup is the preferred end-to-end headline metric from this experiment.",
            "No approximation, pruning, quantization, retraining, or reduced precision is used.",
            "Top-k proposal membership is checked, but dataset-level detection accuracy equivalence remains unmeasured."
        ],
    }

    out = Path(
        "results/metalground_original_vs_final_multiprocess.json"
    )
    out.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n"
    )

    print("\n=== Experiment 0050 aggregate ===", flush=True)
    print(
        "per-process paired delta medians: "
        + ", ".join(f"{x:+.3f}" for x in deltas)
        + " ms",
        flush=True,
    )
    print(
        "per-process pairwise speedup medians: "
        + ", ".join(f"{x:.3f}x" for x in speedups),
        flush=True,
    )
    print(
        f"median paired saving: {median_delta:+.3f} ms",
        flush=True,
    )
    print(
        f"median pairwise speedup: "
        f"{median_pairwise_speedup:.3f}x",
        flush=True,
    )
    print(
        f"controlled median-of-process-medians: "
        f"original={median_original:.3f} ms, "
        f"final={median_final:.3f} ms",
        flush=True,
    )
    print(
        f"robust headline pass: {robust_headline_pass}",
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

    ap.add_argument(
        "--model",
        default="IDEA-Research/grounding-dino-tiny",
    )
    ap.add_argument(
        "--image",
        type=Path,
        default=Path("assets/input.jpg"),
    )
    ap.add_argument(
        "--prompt",
        nargs="+",
        default=["a cat", "a dog"],
    )
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
        if args.worker_out is None:
            raise SystemExit("--worker requires --worker-out")
        worker(args)
    else:
        driver(args)


if __name__ == "__main__":
    main()
