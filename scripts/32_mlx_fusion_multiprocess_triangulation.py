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
        "shape": list(a.shape),
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


def run_timed(model, inputs, sync_fn):
    sync_fn()
    t0 = time.perf_counter_ns()
    with torch.inference_mode():
        out = model(**inputs)
    sync_fn()
    return out, (time.perf_counter_ns() - t0) / 1e6


def three_way_correctness(
    original,
    v2,
    candidate,
    model,
    h14,
    processor,
    inputs,
    text_labels,
    image_size,
    box_threshold,
    text_threshold,
):
    def pair(a, b):
        return {
            "logits": mask_aware_error(a.logits, b.logits),
            "pred_boxes": mask_aware_error(a.pred_boxes, b.pred_boxes),
            "topk": h14.topk_audit(a, b, model.config.num_queries),
        }

    return {
        "v2_vs_original": pair(v2, original),
        "candidate_vs_v2": pair(candidate, v2),
        "candidate_vs_original": pair(candidate, original),
        "detections": {
            "original": h14.detection_summary(
                processor,
                original,
                inputs["input_ids"],
                text_labels,
                image_size,
                box_threshold,
                text_threshold,
            ),
            "v2": h14.detection_summary(
                processor,
                v2,
                inputs["input_ids"],
                text_labels,
                image_size,
                box_threshold,
                text_threshold,
            ),
            "candidate": h14.detection_summary(
                processor,
                candidate,
                inputs["input_ids"],
                text_labels,
                image_size,
                box_threshold,
                text_threshold,
            ),
        },
    }


def worker(args):
    if not torch.backends.mps.is_available():
        raise SystemExit("MPS unavailable.")

    h13 = load_module(args.exp13_helper, f"mg32_h13_{os.getpid()}")
    h14 = load_module(args.exp14_helper, f"mg32_h14_{os.getpid()}")
    h17 = load_module(args.exp17_helper, f"mg32_h17_{os.getpid()}")
    h18 = load_module(args.exp18_helper, f"mg32_h18_{os.getpid()}")
    h30 = load_module(args.exp30_helper, f"mg32_h30_{os.getpid()}")
    h31 = load_module(args.exp31_helper, f"mg32_h31_{os.getpid()}")

    mx.set_default_device(mx.gpu)

    print(f"[worker {args.worker_index}] loading model...", flush=True)
    processor = AutoProcessor.from_pretrained(args.model)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(args.model)
    model.eval().to("mps")
    h14.sync()

    image = Image.open(args.image).convert("RGB")
    text_labels = [args.prompt]
    cpu_inputs = processor(images=image, text=text_labels, return_tensors="pt")
    inputs = {k: v.to("mps") for k, v in cpu_inputs.items()}
    h14.sync()

    # Original unmodified PyTorch/MPS output for trajectory triangulation.
    with torch.inference_mode():
        original_ref = model(**inputs)
        h14.sync()

    # Build the MetalGround-v2 base except that fusion is controlled by
    # dispatchers so baseline and candidate share one exact model instance.
    msda_state = h14.patch_msda(model, args.threadgroup)

    dispatchers = []
    h14.sync()
    for layer in model.model.encoder.layers:
        attn = layer.fusion_layer.attn
        torch_spec = h13.AlgebraicBiMHA(attn)
        mlx_spec = h30.MlxFullyFoldedFusion(torch_spec)
        dispatchers.append(
            h31.FusionDispatcher(attn, torch_spec, mlx_spec)
        )

    h31.set_all(dispatchers, "torch")

    # Capture spatial shapes needed by the existing six deformable islands.
    layer0 = model.model.encoder.layers[0].deformable_layer
    original_layer0_forward = layer0.forward
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
            captured["spatial_shapes_list"] = [
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

    if not captured:
        raise RuntimeError("Failed to capture encoder spatial shapes.")

    h14.sync()
    island_state, _ = h18.patch_encoder_deformable_islands(
        model,
        h17.MlxDeformableIsland,
        activation_name=model.config.activation_function,
        spatial_shapes_list=captured["spatial_shapes_list"],
        threadgroup=args.threadgroup,
    )

    try:
        # Three-way correctness in one process/model state.
        h31.set_all(dispatchers, "torch")
        with torch.inference_mode():
            v2_ref = model(**inputs)
            h14.sync()

        h31.set_all(dispatchers, "mlx")
        with torch.inference_mode():
            candidate_ref = model(**inputs)
            h14.sync()

        correctness = three_way_correctness(
            original_ref,
            v2_ref,
            candidate_ref,
            model,
            h14,
            processor,
            inputs,
            text_labels,
            image.size,
            args.box_threshold,
            args.text_threshold,
        )

        print(
            f"[worker {args.worker_index}] top-k triangulation:",
            json.dumps(
                {
                    k: v["topk"]
                    for k, v in correctness.items()
                    if k.endswith("_vs_original") or k == "candidate_vs_v2"
                },
                indent=2,
            ),
            flush=True,
        )

        # Warm both performance modes. Original reference is not part of timing.
        for i in range(args.warmup_per_mode):
            h31.set_all(dispatchers, "torch")
            _ = run_timed(model, inputs, h14.sync)
            h31.set_all(dispatchers, "mlx")
            _ = run_timed(model, inputs, h14.sync)
            print(
                f"[worker {args.worker_index}] warmup "
                f"{i+1}/{args.warmup_per_mode}",
                flush=True,
            )

        h31.reset_all_counts(dispatchers)
        baseline_samples = []
        candidate_samples = []
        deltas = []

        worker_flip = args.worker_index % 2
        for i in range(args.pairs_per_process):
            baseline_first = ((i + worker_flip) % 2 == 0)
            order = (
                ("torch", "mlx")
                if baseline_first
                else ("mlx", "torch")
            )
            local = {}

            for mode in order:
                h31.set_all(dispatchers, mode)
                _out, dt = run_timed(model, inputs, h14.sync)
                local[mode] = dt
                if mode == "torch":
                    baseline_samples.append(dt)
                else:
                    candidate_samples.append(dt)

            delta = local["torch"] - local["mlx"]
            deltas.append(delta)
            print(
                f"[worker {args.worker_index}] pair "
                f"{i+1:02d}/{args.pairs_per_process}: "
                f"torch={local['torch']:.3f} "
                f"mlx={local['mlx']:.3f} "
                f"delta={delta:+.3f} ms",
                flush=True,
            )

        counts = h31.sum_counts(dispatchers)
        expected_each = args.pairs_per_process * 6
        expected_total = expected_each * 2

        if counts["calls"] != expected_total:
            raise RuntimeError(
                f"Fusion call count mismatch: {counts}, "
                f"expected total {expected_total}"
            )
        if counts["torch_calls"] != expected_each:
            raise RuntimeError(
                f"Torch call count mismatch: {counts['torch_calls']} "
                f"!= {expected_each}"
            )
        if counts["mlx_calls"] != expected_each:
            raise RuntimeError(
                f"MLX call count mismatch: {counts['mlx_calls']} "
                f"!= {expected_each}"
            )

        result = {
            "worker_index": args.worker_index,
            "pid": os.getpid(),
            "latency": {
                "v2_torch_fusion": stats(baseline_samples),
                "mlx_fusion_candidate": stats(candidate_samples),
                "paired_delta_torch_minus_mlx": stats(deltas),
            },
            "derived": {
                "speedup_from_medians": (
                    statistics.median(baseline_samples)
                    / statistics.median(candidate_samples)
                ),
                "median_difference_of_marginals_ms": (
                    statistics.median(baseline_samples)
                    - statistics.median(candidate_samples)
                ),
            },
            "correctness": correctness,
            "call_validation": {
                **counts,
                "expected_each_mode": expected_each,
                "expected_total": expected_total,
            },
        }

        args.worker_out.write_text(
            json.dumps(result, indent=2, ensure_ascii=False) + "\n"
        )

    finally:
        for d in dispatchers:
            d.restore()


def driver(args):
    script_path = Path(__file__).resolve()
    workers = []

    with tempfile.TemporaryDirectory(prefix="metalground_exp32_") as td:
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
                "--box-threshold", str(args.box_threshold),
                "--text-threshold", str(args.text_threshold),
                "--exp13-helper", str(args.exp13_helper),
                "--exp14-helper", str(args.exp14_helper),
                "--exp17-helper", str(args.exp17_helper),
                "--exp18-helper", str(args.exp18_helper),
                "--exp30-helper", str(args.exp30_helper),
                "--exp31-helper", str(args.exp31_helper),
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
        w["latency"]["paired_delta_torch_minus_mlx"]["median_ms"]
        for w in workers
    ]
    baseline_medians = [
        w["latency"]["v2_torch_fusion"]["median_ms"] for w in workers
    ]
    candidate_medians = [
        w["latency"]["mlx_fusion_candidate"]["median_ms"] for w in workers
    ]

    # Summarize the discrete proposal-order relationship observed per process.
    topk_summary = []
    for w in workers:
        corr = w["correctness"]
        topk_summary.append(
            {
                "worker_index": w["worker_index"],
                "v2_vs_original": corr["v2_vs_original"]["topk"],
                "candidate_vs_v2": corr["candidate_vs_v2"]["topk"],
                "candidate_vs_original": corr["candidate_vs_original"]["topk"],
            }
        )

    positive = sum(x > 0 for x in process_deltas)

    result = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "experiment": "0032",
        "purpose": (
            "Replicate the full-model MLX-fusion speedup across independent "
            "fresh processes while triangulating numerical trajectory among "
            "original PyTorch/MPS, MetalGround v2, and the MLX-fusion candidate."
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
            "pair_order": "alternating; initial order balanced by worker parity",
            "prompt_cache": False,
            "primary_performance_unit": "per-process paired-delta median",
            "correctness_triangulation": [
                "MetalGround v2 vs original PyTorch/MPS",
                "MLX-fusion candidate vs MetalGround v2",
                "MLX-fusion candidate vs original PyTorch/MPS",
            ],
        },
        "workers": workers,
        "aggregate": {
            "paired_delta_median_across_processes": stats(process_deltas),
            "positive_process_count": positive,
            "total_process_count": args.processes,
            "median_of_process_paired_medians_ms": statistics.median(
                process_deltas
            ),
            "absolute_v2_medians_across_processes": stats(baseline_medians),
            "absolute_candidate_medians_across_processes": stats(
                candidate_medians
            ),
            "median_absolute_v2_ms": statistics.median(baseline_medians),
            "median_absolute_candidate_ms": statistics.median(
                candidate_medians
            ),
            "speedup_from_median_absolute_process_medians": (
                statistics.median(baseline_medians)
                / statistics.median(candidate_medians)
            ),
            "topk_triangulation": topk_summary,
        },
        "decision_rule": (
            "Adopt compiled MLX fusion if all or nearly all fresh processes "
            "show a large positive paired delta and the candidate preserves "
            "the top-900 proposal set with numerical behavior consistent with "
            "the previously characterized rank-sensitive FP32 trajectory. "
            "If adopted, the next experiment should merge fusion and the "
            "existing deformable MLX island to remove the intermediate "
            "PyTorch/MLX boundary."
        ),
        "notes": [
            "Original PyTorch/MPS is used only as a correctness reference, not a timed baseline.",
            "Performance pairing compares MetalGround v2's PyTorch fully-folded fusion with compiled MLX fully-folded fusion on the same model instance.",
            "Prompt caching is disabled to isolate the fusion change.",
            "No approximation, quantization, pruning, retraining, or reduced precision is used."
        ],
    }

    out = Path("results/metalground_mlx_fusion_multiprocess_triangulation.json")
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")

    print("\n=== Experiment 0032 aggregate ===", flush=True)
    for w in workers:
        d = w["latency"]["paired_delta_torch_minus_mlx"]["median_ms"]
        b = w["latency"]["v2_torch_fusion"]["median_ms"]
        c = w["latency"]["mlx_fusion_candidate"]["median_ms"]
        t = w["correctness"]["candidate_vs_original"]["topk"]
        print(
            f"process {w['worker_index']}: "
            f"v2={b:.3f} ms, candidate={c:.3f} ms, "
            f"paired={d:+.3f} ms, "
            f"candidate-vs-original rankwise={t['rankwise_identical']}/900 "
            f"mismatch={t['mismatched_ranks_zero_based']}",
            flush=True,
        )

    print(
        f"positive processes: {positive}/{args.processes}",
        flush=True,
    )
    print(
        "median paired gain across processes: "
        f"{result['aggregate']['median_of_process_paired_medians_ms']:+.3f} ms",
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
    ap.add_argument("--box-threshold", type=float, default=0.3)
    ap.add_argument("--text-threshold", type=float, default=0.25)
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
    args = ap.parse_args()

    for p in (
        args.exp13_helper,
        args.exp14_helper,
        args.exp17_helper,
        args.exp18_helper,
        args.exp30_helper,
        args.exp31_helper,
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
