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
import types
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import mlx.core as mx
import torch
import transformers
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
        "p10_ms": percentile(xs, 0.10),
        "p90_ms": percentile(xs, 0.90),
        "min_ms": min(xs),
        "max_ms": max(xs),
    }


def mask_aware_compare(a: torch.Tensor, b: torch.Tensor) -> dict[str, Any]:
    a = a.detach()
    b = b.detach()
    finite_a = torch.isfinite(a)
    finite_b = torch.isfinite(b)
    common = finite_a & finite_b

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
        "finite_mask_equal": bool(torch.equal(finite_a, finite_b)),
        "nan_mask_equal": bool(torch.equal(torch.isnan(a), torch.isnan(b))),
        "posinf_mask_equal": bool(torch.equal(torch.isposinf(a), torch.isposinf(b))),
        "neginf_mask_equal": bool(torch.equal(torch.isneginf(a), torch.isneginf(b))),
        "finite_max_abs": max_abs,
        "finite_mean_abs": mean_abs,
        "finite_rmse": rmse,
        "allclose_1e-5": bool(torch.allclose(a, b, rtol=1e-5, atol=1e-5)),
    }


class TextBackboneCache:
    def __init__(self, module):
        self.module = module
        self.original_forward = module.forward
        self.cached_output = None
        self.mode = "baseline"
        self.calls = 0
        self.hits = 0
        self.misses = 0

        def patched_forward(_module_self, *args, **kwargs):
            self.calls += 1
            if self.mode == "baseline":
                return self.original_forward(*args, **kwargs)
            if self.cached_output is None:
                self.misses += 1
                out = self.original_forward(*args, **kwargs)
                self.cached_output = out
                return out
            self.hits += 1
            return self.cached_output

        module.forward = types.MethodType(patched_forward, module)

    def set_baseline(self):
        self.mode = "baseline"

    def set_cached(self):
        self.mode = "cached"

    def reset_counts(self):
        self.calls = self.hits = self.misses = 0

    def prime(self, model, inputs, sync_fn):
        self.cached_output = None
        self.reset_counts()
        self.set_cached()
        with torch.inference_mode():
            _ = model(**inputs)
            sync_fn()
        if self.misses != 1 or self.cached_output is None:
            raise RuntimeError(
                f"Prompt-cache prime failed: misses={self.misses}"
            )

    def restore(self):
        self.module.forward = self.original_forward


def run_timed(model, inputs, sync_fn):
    sync_fn()
    t0 = time.perf_counter_ns()
    with torch.inference_mode():
        out = model(**inputs)
    sync_fn()
    return out, (time.perf_counter_ns() - t0) / 1e6


def build_v2(args):
    h14 = load_module(args.exp14_helper, f"mg_h14_{os.getpid()}")
    h17 = load_module(args.exp17_helper, f"mg_h17_{os.getpid()}")
    h18 = load_module(args.exp18_helper, f"mg_h18_{os.getpid()}")

    mx.set_default_device(mx.gpu)

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

    layer0 = model.model.encoder.layers[0].deformable_layer
    original_layer0_forward = layer0.forward
    capture = {}

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
        if not capture:
            capture["spatial_shapes_list"] = [
                (int(h), int(w)) for h, w in spatial_shapes_list
            ]
        return original_layer0_forward(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_embeddings=position_embeddings,
            reference_points=reference_points,
            spatial_shapes=spatial_shapes,
            spatial_shapes_list=spatial_shapes_list,
            level_start_index=level_start_index,
            output_attentions=output_attentions,
        )

    layer0.forward = types.MethodType(capture_forward, layer0)
    with torch.inference_mode():
        _ = model(**inputs)
        h14.sync()
    layer0.forward = original_layer0_forward

    if not capture:
        raise RuntimeError("Failed to capture encoder spatial shapes.")

    h14.sync()
    h18.patch_encoder_deformable_islands(
        model,
        h17.MlxDeformableIsland,
        activation_name=model.config.activation_function,
        spatial_shapes_list=capture["spatial_shapes_list"],
        threadgroup=args.threadgroup,
    )

    return model, inputs, h14


def worker(args):
    if not torch.backends.mps.is_available():
        raise SystemExit("MPS unavailable.")

    print(f"[worker {args.worker_index}] building MetalGround v2...", flush=True)
    model, inputs, h14 = build_v2(args)
    cache = TextBackboneCache(model.model.text_backbone)

    try:
        cache.prime(model, inputs, h14.sync)

        cache.set_baseline()
        with torch.inference_mode():
            baseline_correct = model(**inputs)
            h14.sync()

        cache.set_cached()
        with torch.inference_mode():
            cached_correct = model(**inputs)
            h14.sync()

        correctness = {
            "logits": mask_aware_compare(
                cached_correct.logits, baseline_correct.logits
            ),
            "pred_boxes": mask_aware_compare(
                cached_correct.pred_boxes, baseline_correct.pred_boxes
            ),
            "topk": h14.topk_audit(
                cached_correct, baseline_correct, model.config.num_queries
            ),
        }

        for i in range(args.warmup_per_mode):
            cache.set_baseline()
            _ = run_timed(model, inputs, h14.sync)
            cache.set_cached()
            _ = run_timed(model, inputs, h14.sync)
            print(
                f"[worker {args.worker_index}] warmup "
                f"{i+1}/{args.warmup_per_mode}",
                flush=True,
            )

        cache.reset_counts()
        baseline_samples = []
        cached_samples = []
        deltas = []

        # Flip the first order by worker parity so process-level ordering is
        # balanced as well as pair-level ordering.
        worker_flip = args.worker_index % 2

        for i in range(args.pairs_per_process):
            baseline_first = ((i + worker_flip) % 2 == 0)
            order = (
                ("baseline", "cached")
                if baseline_first
                else ("cached", "baseline")
            )
            local = {}

            for mode in order:
                if mode == "baseline":
                    cache.set_baseline()
                else:
                    cache.set_cached()

                _out, dt = run_timed(model, inputs, h14.sync)
                local[mode] = dt
                if mode == "baseline":
                    baseline_samples.append(dt)
                else:
                    cached_samples.append(dt)

            delta = local["baseline"] - local["cached"]
            deltas.append(delta)
            print(
                f"[worker {args.worker_index}] pair "
                f"{i+1:02d}/{args.pairs_per_process}: "
                f"base={local['baseline']:.3f} "
                f"cache={local['cached']:.3f} "
                f"delta={delta:+.3f} ms",
                flush=True,
            )

        expected_hits = args.pairs_per_process
        if cache.hits != expected_hits or cache.misses != 0:
            raise RuntimeError(
                f"Unexpected timed cache state: hits={cache.hits}, "
                f"misses={cache.misses}, expected hits={expected_hits}"
            )

        result = {
            "worker_index": args.worker_index,
            "pid": os.getpid(),
            "baseline": stats(baseline_samples),
            "cached": stats(cached_samples),
            "paired_delta": stats(deltas),
            "speedup_from_medians": (
                statistics.median(baseline_samples)
                / statistics.median(cached_samples)
            ),
            "median_difference_of_marginals_ms": (
                statistics.median(baseline_samples)
                - statistics.median(cached_samples)
            ),
            "correctness": correctness,
            "cache_hits": cache.hits,
            "cache_misses": cache.misses,
        }

        args.worker_out.write_text(
            json.dumps(result, indent=2, ensure_ascii=False) + "\n"
        )
        print(
            f"[worker {args.worker_index}] paired median "
            f"{result['paired_delta']['median_ms']:+.3f} ms",
            flush=True,
        )
    finally:
        cache.restore()


def driver(args):
    script_path = Path(__file__).resolve()
    worker_results = []

    with tempfile.TemporaryDirectory(prefix="metalground_exp29_") as td:
        td_path = Path(td)

        for i in range(args.processes):
            worker_out = td_path / f"worker_{i}.json"
            cmd = [
                sys.executable,
                str(script_path),
                "--worker",
                "--worker-index",
                str(i),
                "--worker-out",
                str(worker_out),
                "--model",
                args.model,
                "--image",
                str(args.image),
                "--warmup-per-mode",
                str(args.warmup_per_mode),
                "--pairs-per-process",
                str(args.pairs_per_process),
                "--threadgroup",
                str(args.threadgroup),
                "--exp14-helper",
                str(args.exp14_helper),
                "--exp17-helper",
                str(args.exp17_helper),
                "--exp18-helper",
                str(args.exp18_helper),
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
            worker_results.append(json.loads(worker_out.read_text()))

    process_delta_medians = [
        r["paired_delta"]["median_ms"] for r in worker_results
    ]
    process_delta_means = [
        r["paired_delta"]["mean_ms"] for r in worker_results
    ]
    baseline_medians = [
        r["baseline"]["median_ms"] for r in worker_results
    ]
    cached_medians = [
        r["cached"]["median_ms"] for r in worker_results
    ]
    all_pair_deltas = []
    # Worker stats do not retain raw samples; process-level medians are the
    # deliberate primary unit to prevent one noisy process dominating.
    positive_processes = sum(x > 0 for x in process_delta_medians)

    aggregate = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "experiment": "0029",
        "purpose": (
            "Multi-process robustness study of exact fixed-prompt BERT caching: "
            "independent fresh MetalGround-v2 processes, each with paired "
            "baseline-vs-cache full-model measurements."
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
            "warmup_per_mode_per_process": args.warmup_per_mode,
            "pairs_per_process": args.pairs_per_process,
            "pair_order": "alternating; initial order balanced by worker parity",
            "primary_unit": "per-process paired-delta median",
        },
        "workers": worker_results,
        "aggregate": {
            "paired_delta_median_across_processes": stats(
                process_delta_medians
            ),
            "paired_delta_mean_across_processes": stats(
                process_delta_means
            ),
            "absolute_baseline_median_across_processes": stats(
                baseline_medians
            ),
            "absolute_cached_median_across_processes": stats(
                cached_medians
            ),
            "positive_process_count": positive_processes,
            "total_process_count": args.processes,
            "median_of_process_paired_medians_ms": statistics.median(
                process_delta_medians
            ),
            "median_absolute_baseline_ms": statistics.median(
                baseline_medians
            ),
            "median_absolute_cached_ms": statistics.median(
                cached_medians
            ),
            "cross_process_baseline_median_range_ms": (
                max(baseline_medians) - min(baseline_medians)
            ),
            "cross_process_cached_median_range_ms": (
                max(cached_medians) - min(cached_medians)
            ),
        },
        "decision_rule": (
            "Adopt prompt caching as a robust runtime optimization if at least "
            "4/5 fresh processes have positive paired-delta medians and the "
            "median of process paired medians is materially positive. Keep "
            "absolute historical v2/v3 single-run medians separate from this "
            "causal estimate."
        ),
        "notes": [
            "Each worker is a new Python process and rebuilds MetalGround v2 independently.",
            "Within each worker, baseline and cached executions alternate on the same model and inputs.",
            "The per-process paired-delta median is the primary causal statistic.",
            "Absolute medians across processes quantify run-to-run drift and are not treated as paired effects.",
            "No approximation, quantization, pruning, retraining, or reduced precision is used."
        ],
    }

    out = Path("results/metalground_prompt_cache_multiprocess_robustness.json")
    out.write_text(json.dumps(aggregate, indent=2, ensure_ascii=False) + "\n")

    print("\n=== Experiment 0029 aggregate ===", flush=True)
    for r in worker_results:
        print(
            f"process {r['worker_index']}: "
            f"baseline={r['baseline']['median_ms']:.3f} ms, "
            f"cached={r['cached']['median_ms']:.3f} ms, "
            f"paired_delta={r['paired_delta']['median_ms']:+.3f} ms",
            flush=True,
        )
    print(
        f"positive processes: {positive_processes}/{args.processes}",
        flush=True,
    )
    print(
        "median of process paired medians: "
        f"{aggregate['aggregate']['median_of_process_paired_medians_ms']:+.3f} ms",
        flush=True,
    )
    print(
        "baseline cross-process median range: "
        f"{aggregate['aggregate']['cross_process_baseline_median_range_ms']:.3f} ms",
        flush=True,
    )
    print(f"Saved: {out}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker", action="store_true")
    ap.add_argument("--worker-index", type=int, default=0)
    ap.add_argument("--worker-out", type=Path)
    ap.add_argument("--processes", type=int, default=5)
    ap.add_argument("--pairs-per-process", type=int, default=10)
    ap.add_argument("--warmup-per-mode", type=int, default=2)
    ap.add_argument("--model", default="IDEA-Research/grounding-dino-tiny")
    ap.add_argument("--image", type=Path, default=Path("assets/input.jpg"))
    ap.add_argument("--prompt", nargs="+", default=["a cat", "a dog"])
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
    ap.add_argument(
        "--exp18-helper",
        type=Path,
        default=Path("scripts/18_full_model_deformable_islands.py"),
    )
    args = ap.parse_args()

    for p in (args.exp14_helper, args.exp17_helper, args.exp18_helper):
        if not p.exists():
            raise SystemExit(f"Missing helper script: {p}")

    if args.worker:
        if args.worker_out is None:
            raise SystemExit("--worker requires --worker-out")
        worker(args)
    else:
        driver(args)


if __name__ == "__main__":
    main()
