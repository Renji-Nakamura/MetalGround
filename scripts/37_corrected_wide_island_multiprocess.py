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


def build_runtime(args):
    h13 = load_module(args.exp13_helper, f"mg37_h13_{os.getpid()}")
    h14 = load_module(args.exp14_helper, f"mg37_h14_{os.getpid()}")
    h17 = load_module(args.exp17_helper, f"mg37_h17_{os.getpid()}")
    h18 = load_module(args.exp18_helper, f"mg37_h18_{os.getpid()}")
    h30 = load_module(args.exp30_helper, f"mg37_h30_{os.getpid()}")
    h31 = load_module(args.exp31_helper, f"mg37_h31_{os.getpid()}")
    h36 = load_module(args.exp36_helper, f"mg37_h36_{os.getpid()}")

    mx.set_default_device(mx.gpu)

    processor = AutoProcessor.from_pretrained(args.model)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(args.model)
    model.eval().to("mps")
    h14.sync()

    if bool(model.config.output_attentions):
        raise RuntimeError("Experiment 0037 requires output_attentions=False.")

    image = Image.open(args.image).convert("RGB")
    text_labels = [args.prompt]
    cpu_inputs = processor(images=image, text=text_labels, return_tensors="pt")
    inputs = {k: v.to("mps") for k, v in cpu_inputs.items()}
    h14.sync()

    # Original PyTorch/MPS reference for trajectory audit.
    with torch.inference_mode():
        original_ref = model(**inputs)
        h14.sync()

    # Existing decoder Metal MSDA.
    msda_state = h14.patch_msda(model, args.threadgroup)

    # Robust compiled-MLX fully-folded fusion path.
    fusion_dispatchers = []
    for layer in model.model.encoder.layers:
        torch_spec = h13.AlgebraicBiMHA(layer.fusion_layer.attn)
        mlx_spec = h30.MlxFullyFoldedFusion(torch_spec)
        fd = h31.FusionDispatcher(
            layer.fusion_layer.attn,
            torch_spec,
            mlx_spec,
        )
        fd.set_mode("mlx")
        fusion_dispatchers.append(fd)

    # Capture encoder multiscale geometry before deformable island patching.
    layer0_deform = model.model.encoder.layers[0].deformable_layer
    original_deform_forward = layer0_deform.forward
    captured = {}

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
        if not captured:
            captured["spatial_shapes_list"] = [
                (int(h), int(w)) for h, w in spatial_shapes_list
            ]
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

    layer0_deform.forward = types.MethodType(capture_deform, layer0_deform)
    with torch.inference_mode():
        _ = model(**inputs)
        h14.sync()
    layer0_deform.forward = original_deform_forward

    if not captured:
        raise RuntimeError("Failed to capture encoder spatial shapes.")

    island_state, deform_islands = h18.patch_encoder_deformable_islands(
        model,
        h17.MlxDeformableIsland,
        activation_name=model.config.activation_function,
        spatial_shapes_list=captured["spatial_shapes_list"],
        threadgroup=args.threadgroup,
    )

    # Corrected wide islands from Experiment 0036.
    fixed_islands = []
    for i, layer in enumerate(model.model.encoder.layers):
        fixed_islands.append(
            h36.MlxFusionDeformableWideIslandFixed(
                layer.fusion_layer,
                fusion_dispatchers[i].mlx_specializer,
                deform_islands[i],
            )
        )

    state = h36.LayerCallState()
    layer_dispatchers = [
        h36.FixedWideLayerDispatcher(layer, fixed, state)
        for layer, fixed in zip(model.model.encoder.layers, fixed_islands)
    ]

    return {
        "model": model,
        "processor": processor,
        "image": image,
        "text_labels": text_labels,
        "inputs": inputs,
        "original_ref": original_ref,
        "h14": h14,
        "h31": h31,
        "h36": h36,
        "fusion_dispatchers": fusion_dispatchers,
        "layer_dispatchers": layer_dispatchers,
        "island_state": island_state,
        "msda_state": msda_state,
        "state": state,
    }


def worker(args):
    if not torch.backends.mps.is_available():
        raise SystemExit("MPS unavailable.")

    rt = build_runtime(args)
    model = rt["model"]
    h14 = rt["h14"]
    h31 = rt["h31"]
    h36 = rt["h36"]
    layer_dispatchers = rt["layer_dispatchers"]
    fusion_dispatchers = rt["fusion_dispatchers"]
    state = rt["state"]
    island_state = rt["island_state"]
    msda_state = rt["msda_state"]

    try:
        # Three-way correctness.
        h36.set_mode(layer_dispatchers, "baseline")
        with torch.inference_mode():
            baseline_ref = model(**rt["inputs"])
            h14.sync()

        h36.set_mode(layer_dispatchers, "fixed")
        with torch.inference_mode():
            fixed_ref = model(**rt["inputs"])
            h14.sync()

        correctness = {
            "fixed_vs_separate": pair_audit(
                fixed_ref, baseline_ref, model, h14
            ),
            "separate_vs_original": pair_audit(
                baseline_ref, rt["original_ref"], model, h14
            ),
            "fixed_vs_original": pair_audit(
                fixed_ref, rt["original_ref"], model, h14
            ),
        }

        # Warm both performance modes.
        for i in range(args.warmup_per_mode):
            h36.set_mode(layer_dispatchers, "baseline")
            _ = run_timed(model, rt["inputs"], h14.sync)
            h36.set_mode(layer_dispatchers, "fixed")
            _ = run_timed(model, rt["inputs"], h14.sync)
            print(
                f"[worker {args.worker_index}] warmup "
                f"{i+1}/{args.warmup_per_mode}",
                flush=True,
            )

        # Benchmark-only counters.
        state.baseline_calls = 0
        state.fixed_calls = 0
        for fd in fusion_dispatchers:
            fd.reset_counts()
        island_state.calls = 0
        msda_state.call_count = 0

        baseline_samples = []
        fixed_samples = []
        deltas = []

        worker_flip = args.worker_index % 2

        for i in range(args.pairs_per_process):
            baseline_first = ((i + worker_flip) % 2 == 0)
            order = (
                ("baseline", "fixed")
                if baseline_first
                else ("fixed", "baseline")
            )
            local = {}

            for mode in order:
                h36.set_mode(layer_dispatchers, mode)
                _out, dt = run_timed(model, rt["inputs"], h14.sync)
                local[mode] = dt
                if mode == "baseline":
                    baseline_samples.append(dt)
                else:
                    fixed_samples.append(dt)

            delta = local["baseline"] - local["fixed"]
            deltas.append(delta)
            print(
                f"[worker {args.worker_index}] pair "
                f"{i+1:02d}/{args.pairs_per_process}: "
                f"separate={local['baseline']:.3f} "
                f"fixed={local['fixed']:.3f} "
                f"delta={delta:+.3f} ms",
                flush=True,
            )

        fusion_counts = h31.sum_counts(fusion_dispatchers)
        expected_each = args.pairs_per_process * 6
        expected_decoder = args.pairs_per_process * 2 * 6

        if state.baseline_calls != expected_each:
            raise RuntimeError(
                f"baseline calls {state.baseline_calls} != {expected_each}"
            )
        if state.fixed_calls != expected_each:
            raise RuntimeError(
                f"fixed calls {state.fixed_calls} != {expected_each}"
            )
        if fusion_counts["mlx_calls"] != expected_each:
            raise RuntimeError(
                f"baseline fusion calls {fusion_counts['mlx_calls']} "
                f"!= {expected_each}"
            )
        if island_state.calls != expected_each:
            raise RuntimeError(
                f"baseline deform calls {island_state.calls} "
                f"!= {expected_each}"
            )
        if msda_state.call_count != expected_decoder:
            raise RuntimeError(
                f"decoder MSDA calls {msda_state.call_count} "
                f"!= {expected_decoder}"
            )

        result = {
            "worker_index": args.worker_index,
            "pid": os.getpid(),
            "latency": {
                "separate_baseline": stats(baseline_samples),
                "corrected_wide": stats(fixed_samples),
                "paired_delta_separate_minus_wide": stats(deltas),
            },
            "derived": {
                "speedup_from_medians": (
                    statistics.median(baseline_samples)
                    / statistics.median(fixed_samples)
                ),
                "median_difference_of_marginals_ms": (
                    statistics.median(baseline_samples)
                    - statistics.median(fixed_samples)
                ),
            },
            "correctness": correctness,
            "call_validation": {
                "baseline_layer_calls": state.baseline_calls,
                "fixed_layer_calls": state.fixed_calls,
                "baseline_fusion_mlx_calls": fusion_counts["mlx_calls"],
                "baseline_deformable_island_calls": island_state.calls,
                "decoder_metal_msda_calls": msda_state.call_count,
                "expected_each_encoder_mode": expected_each,
                "expected_decoder_msda_calls": expected_decoder,
            },
        }

        args.worker_out.write_text(
            json.dumps(result, indent=2, ensure_ascii=False) + "\n"
        )

        print(
            f"[worker {args.worker_index}] paired median "
            f"{result['latency']['paired_delta_separate_minus_wide']['median_ms']:+.3f} ms, "
            f"topk overlap "
            f"{correctness['fixed_vs_separate']['topk']['set_overlap']}/900",
            flush=True,
        )

    finally:
        for d in layer_dispatchers:
            d.restore()
        for fd in fusion_dispatchers:
            fd.restore()


def driver(args):
    script_path = Path(__file__).resolve()
    workers = []

    with tempfile.TemporaryDirectory(prefix="metalground_exp37_") as td:
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
        w["latency"]["paired_delta_separate_minus_wide"]["median_ms"]
        for w in workers
    ]
    baseline_medians = [
        w["latency"]["separate_baseline"]["median_ms"] for w in workers
    ]
    fixed_medians = [
        w["latency"]["corrected_wide"]["median_ms"] for w in workers
    ]

    positive = sum(x > 0 for x in process_deltas)
    membership_preserved = sum(
        w["correctness"]["fixed_vs_separate"]["topk"]["set_overlap"] == 900
        and w["correctness"]["fixed_vs_separate"]["topk"][
            "changed_membership_each_side"
        ] == 0
        for w in workers
    )

    topk_rows = []
    for w in workers:
        topk_rows.append({
            "worker_index": w["worker_index"],
            "fixed_vs_separate": w["correctness"]["fixed_vs_separate"]["topk"],
            "separate_vs_original": w["correctness"]["separate_vs_original"]["topk"],
            "fixed_vs_original": w["correctness"]["fixed_vs_original"]["topk"],
        })

    result = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "experiment": "0037",
        "purpose": (
            "Multi-process robustness replication of the corrected wide "
            "fusion+deformable encoder island after the Experiment-0036 "
            "cross-runtime synchronization fix."
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
        },
        "workers": workers,
        "aggregate": {
            "paired_delta_median_across_processes": stats(process_deltas),
            "positive_process_count": positive,
            "total_process_count": args.processes,
            "topk_membership_preserved_process_count": membership_preserved,
            "median_of_process_paired_medians_ms":
                statistics.median(process_deltas),
            "absolute_separate_medians_across_processes":
                stats(baseline_medians),
            "absolute_fixed_wide_medians_across_processes":
                stats(fixed_medians),
            "median_absolute_separate_ms":
                statistics.median(baseline_medians),
            "median_absolute_fixed_wide_ms":
                statistics.median(fixed_medians),
            "speedup_from_median_absolute_process_medians": (
                statistics.median(baseline_medians)
                / statistics.median(fixed_medians)
            ),
            "topk_triangulation": topk_rows,
        },
        "decision_rule": (
            "Adopt the corrected wide island if at least 4/5 fresh processes "
            "show positive paired-delta medians and top-900 membership is "
            "preserved in all processes. Rank-order-only differences may be "
            "reported under the previously characterized near-tie trajectory "
            "phenomenon, but no membership changes are acceptable."
        ),
        "notes": [
            "Each process rebuilds the runtime independently.",
            "Performance pairing compares the robust separate MLX fusion+deformable path with the corrected single-boundary wide path on the same model instance.",
            "The deformable valid-token mask is derived inside MLX from a synchronized key_padding_mask.",
            "Prompt caching is disabled to isolate wide-island effects.",
            "No approximation, quantization, pruning, retraining, or reduced precision is used.",
            "Dataset-level accuracy equivalence remains unmeasured."
        ],
    }

    out = Path(
        "results/metalground_corrected_wide_island_multiprocess.json"
    )
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")

    print("\n=== Experiment 0037 aggregate ===", flush=True)
    for w in workers:
        d = w["latency"]["paired_delta_separate_minus_wide"]["median_ms"]
        b = w["latency"]["separate_baseline"]["median_ms"]
        f = w["latency"]["corrected_wide"]["median_ms"]
        tk = w["correctness"]["fixed_vs_separate"]["topk"]
        print(
            f"process {w['worker_index']}: "
            f"separate={b:.3f} ms, fixed={f:.3f} ms, "
            f"paired={d:+.3f} ms, "
            f"overlap={tk['set_overlap']}/900, "
            f"rankwise={tk['rankwise_identical']}/900",
            flush=True,
        )

    print(
        f"positive processes: {positive}/{args.processes}",
        flush=True,
    )
    print(
        f"membership preserved: "
        f"{membership_preserved}/{args.processes}",
        flush=True,
    )
    print(
        "median of process paired medians: "
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
    args = ap.parse_args()

    for p in (
        args.exp13_helper,
        args.exp14_helper,
        args.exp17_helper,
        args.exp18_helper,
        args.exp30_helper,
        args.exp31_helper,
        args.exp36_helper,
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
