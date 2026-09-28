#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from PIL import Image

MODE_PRISTINE = "pristine_no_opencv"
MODE_IMPORT = "opencv_import_only"
MODE_INIT = "camera_init_then_close_before_build"
MODE_ACTIVE_BUILD = "camera_active_during_build_then_close"

MODES = [
    MODE_PRISTINE,
    MODE_IMPORT,
    MODE_INIT,
    MODE_ACTIVE_BUILD,
]

# Five fresh-process replication rounds. The first four orders form a
# balanced Latin-square style rotation; round 5 repeats the first order.
ROUND_ORDERS = [
    [MODE_PRISTINE, MODE_IMPORT, MODE_INIT, MODE_ACTIVE_BUILD],
    [MODE_IMPORT, MODE_ACTIVE_BUILD, MODE_PRISTINE, MODE_INIT],
    [MODE_INIT, MODE_PRISTINE, MODE_ACTIVE_BUILD, MODE_IMPORT],
    [MODE_ACTIVE_BUILD, MODE_INIT, MODE_IMPORT, MODE_PRISTINE],
    [MODE_PRISTINE, MODE_IMPORT, MODE_INIT, MODE_ACTIVE_BUILD],
]

PROCESSES_PER_CONDITION = 5
INITIAL_WARMUPS = 3
POST_CONTEXT_WARMUPS = 1
MEASUREMENTS = 8
MATERIAL_EFFECT_MS = 10.0
INTER_WORKER_SETTLE_S = 2.0

HISTORICAL_STANDALONE_AUTHORITY_MS = 476.360333
HISTORICAL_0053B_CLOSED_MS = 525.020313
HISTORICAL_0053B_LIVE_MS = 538.6341045


def load_module(path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load helper: {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod


def stat(xs):
    arr = np.asarray(xs, dtype=np.float64)
    return {
        "n": int(arr.size),
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "p90": float(np.percentile(arr, 90)),
        "p95": float(np.percentile(arr, 95)),
        "min": float(arr.min()),
        "max": float(arr.max()),
    }


class ActiveCamera:
    def __init__(self, cv2, index, width, height, fps):
        self.cv2 = cv2
        self.index = index
        self.width = width
        self.height = height
        self.fps = fps
        self.cap = None
        self.backend = None
        self.actual_size = None
        self.actual_fps = None
        self.stop_event = threading.Event()
        self.thread = None
        self.frames = 0
        self.read_failures = 0

    def open(self):
        cv2 = self.cv2
        backend = (
            cv2.CAP_AVFOUNDATION
            if hasattr(cv2, "CAP_AVFOUNDATION")
            else cv2.CAP_ANY
        )
        cap = cv2.VideoCapture(self.index, backend)
        if not cap.isOpened():
            cap.release()
            cap = cv2.VideoCapture(self.index, cv2.CAP_ANY)
        if not cap.isOpened():
            raise RuntimeError("Could not open camera.")
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, float(self.width))
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, float(self.height))
        cap.set(cv2.CAP_PROP_FPS, float(self.fps))
        self.cap = cap
        try:
            self.backend = cap.getBackendName()
        except Exception:
            self.backend = None
        self.actual_size = [
            int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        ]
        self.actual_fps = float(cap.get(cv2.CAP_PROP_FPS))

    def start(self):
        self.open()
        self.stop_event.clear()
        self.thread = threading.Thread(
            target=self._run,
            daemon=True,
            name="metalground-0053c-camera",
        )
        self.thread.start()

    def _run(self):
        while not self.stop_event.is_set():
            ok, frame = self.cap.read()
            if ok and frame is not None:
                self.frames += 1
            else:
                self.read_failures += 1
                time.sleep(0.003)

    def stop_and_close(self):
        self.stop_event.set()
        if self.thread is not None:
            self.thread.join(timeout=2.0)
        self.thread = None
        if self.cap is not None:
            self.cap.release()
        self.cap = None


def open_read_close_camera(cv2, index, width, height, fps, reads=3):
    backend = (
        cv2.CAP_AVFOUNDATION
        if hasattr(cv2, "CAP_AVFOUNDATION")
        else cv2.CAP_ANY
    )
    cap = cv2.VideoCapture(index, backend)
    if not cap.isOpened():
        cap.release()
        cap = cv2.VideoCapture(index, cv2.CAP_ANY)
    if not cap.isOpened():
        raise RuntimeError("Could not open camera.")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, float(width))
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, float(height))
    cap.set(cv2.CAP_PROP_FPS, float(fps))

    try:
        actual_backend = cap.getBackendName()
    except Exception:
        actual_backend = None
    actual_size = [
        int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    ]
    actual_fps = float(cap.get(cv2.CAP_PROP_FPS))

    successful = 0
    for _ in range(reads):
        ok, frame = cap.read()
        if ok and frame is not None:
            successful += 1
    cap.release()

    return {
        "backend": actual_backend,
        "actual_size": actual_size,
        "actual_fps": actual_fps,
        "successful_reads": successful,
    }


def build_runtime(args):
    h37 = load_module(args.exp37_helper, "mg53c_h37")
    h38 = load_module(args.exp38_helper, "mg53c_h38")
    h45 = load_module(args.exp45_helper, "mg53c_h45")
    h46 = load_module(args.exp46_helper, "mg53c_h46")

    rt = h37.build_runtime(args)
    model = rt["model"]
    processor = rt["processor"]
    h14 = rt["h14"]

    cache = h38.ExactTextBackboneCache(model.model.text_backbone)
    h38.set_runtime_mode(rt, cache, "wide_cache")

    stages = model.model.backbone.conv_encoder.model.swin.encoder.layers
    stage0_modules = [
        stages[0].blocks[0].mlp,
        stages[0].blocks[1].mlp,
    ]
    h14.sync()
    candidates = [h45.MlxSwinMLP(m) for m in stage0_modules]
    stage0_dispatchers = [
        h46.SwinMlpDispatcher(m, c, h14.sync)
        for m, c in zip(stage0_modules, candidates)
    ]
    for d in stage0_dispatchers:
        d.set_mode("mlx")

    image = Image.open(args.image).convert("RGB")
    cpu_inputs = processor(
        images=image,
        text=[args.prompt],
        return_tensors="pt",
    )
    inputs = {k: v.to("mps") for k, v in cpu_inputs.items()}
    h14.sync()

    cache.cached_output = None
    cache.prime(model, inputs, h14.sync)
    cache.set_enabled(True)

    return {
        "rt": rt,
        "model": model,
        "inputs": inputs,
        "cache": cache,
        "stage0_dispatchers": stage0_dispatchers,
        "h14": h14,
    }


def measure_one(model, inputs, sync):
    sync()
    t0 = time.perf_counter_ns()
    with torch.inference_mode():
        out = model(**inputs)
    sync()
    t1 = time.perf_counter_ns()
    del out
    return (t1 - t0) / 1_000_000.0


def cleanup_runtime(runtime):
    if runtime is None:
        return
    for d in runtime["stage0_dispatchers"]:
        try:
            d.restore()
        except Exception:
            pass
    try:
        runtime["cache"].restore()
    except Exception:
        pass
    try:
        for d in runtime["rt"]["layer_dispatchers"]:
            d.restore()
        for fd in runtime["rt"]["fusion_dispatchers"]:
            fd.restore()
    except Exception:
        pass


def run_worker(args):
    mode = args.worker_mode
    runtime = None
    active_camera = None
    context_info = {
        "mode": mode,
        "opencv_imported": False,
        "camera_initialized": False,
        "camera_active_during_build": False,
    }

    try:
        # Critical methodological property:
        # pristine mode never imports cv2 in this process.
        cv2 = None

        if mode == MODE_PRISTINE:
            pass

        elif mode == MODE_IMPORT:
            import cv2 as cv2_mod
            cv2 = cv2_mod
            context_info["opencv_imported"] = True

        elif mode == MODE_INIT:
            import cv2 as cv2_mod
            cv2 = cv2_mod
            context_info["opencv_imported"] = True
            context_info["camera_initialized"] = True
            context_info["camera_init"] = open_read_close_camera(
                cv2,
                args.camera_index,
                args.camera_width,
                args.camera_height,
                args.camera_fps,
                reads=3,
            )

        elif mode == MODE_ACTIVE_BUILD:
            import cv2 as cv2_mod
            cv2 = cv2_mod
            context_info["opencv_imported"] = True
            context_info["camera_initialized"] = True
            context_info["camera_active_during_build"] = True
            active_camera = ActiveCamera(
                cv2,
                args.camera_index,
                args.camera_width,
                args.camera_height,
                args.camera_fps,
            )
            active_camera.start()
            time.sleep(0.20)
            context_info["camera_active"] = {
                "backend": active_camera.backend,
                "actual_size": active_camera.actual_size,
                "actual_fps": active_camera.actual_fps,
            }

        else:
            raise RuntimeError(f"Unknown worker mode: {mode}")

        build_t0 = time.perf_counter_ns()
        runtime = build_runtime(args)
        build_t1 = time.perf_counter_ns()

        model = runtime["model"]
        inputs = runtime["inputs"]
        sync = runtime["h14"].sync

        # Initial warmups intentionally happen with the context still active.
        initial_warmups_ms = []
        for _ in range(INITIAL_WARMUPS):
            initial_warmups_ms.append(measure_one(model, inputs, sync))

        if active_camera is not None:
            active_camera.stop_and_close()
            context_info["active_camera_frames_during_build_and_warmup"] = (
                active_camera.frames
            )
            context_info["active_camera_read_failures"] = (
                active_camera.read_failures
            )
            active_camera = None

        # One identical post-context warmup before measured samples.
        post_context_warmups_ms = []
        for _ in range(POST_CONTEXT_WARMUPS):
            post_context_warmups_ms.append(measure_one(model, inputs, sync))

        measured = []
        for i in range(MEASUREMENTS):
            ms = measure_one(model, inputs, sync)
            measured.append(ms)
            print(
                f"[worker {mode}] {i + 1}/{MEASUREMENTS}: "
                f"{ms:.3f} ms",
                flush=True,
            )

        result = {
            "worker_mode": mode,
            "pid": os.getpid(),
            "environment": {
                "torch": torch.__version__,
                "device": "mps",
                "dtype": "float32",
            },
            "context": context_info,
            "build_wall_ms": (build_t1 - build_t0) / 1_000_000.0,
            "initial_warmups_ms": initial_warmups_ms,
            "post_context_warmups_ms": post_context_warmups_ms,
            "measurements_ms": measured,
            "summary_ms": stat(measured),
        }
        args.worker_output.parent.mkdir(parents=True, exist_ok=True)
        args.worker_output.write_text(
            json.dumps(result, indent=2) + "\n"
        )
        return 0

    finally:
        if active_camera is not None:
            active_camera.stop_and_close()
        cleanup_runtime(runtime)
        gc.collect()


def contrast_result(round_deltas):
    med = float(np.median(np.asarray(round_deltas, dtype=np.float64)))
    all_positive = all(x > 0 for x in round_deltas)
    all_negative = all(x < 0 for x in round_deltas)
    return {
        "round_deltas_ms": round_deltas,
        "median_round_delta_ms": med,
        "all_5_positive": all_positive,
        "all_5_negative": all_negative,
        "material_threshold_ms": MATERIAL_EFFECT_MS,
        "material_slowdown": (
            all_positive and med >= MATERIAL_EFFECT_MS
        ),
        "material_speedup": (
            all_negative and med <= -MATERIAL_EFFECT_MS
        ),
    }


def worker_command(args, mode, output_path):
    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker-mode",
        mode,
        "--worker-output",
        str(output_path),
        "--model",
        args.model,
        "--image",
        str(args.image),
        "--camera-index",
        str(args.camera_index),
        "--camera-width",
        str(args.camera_width),
        "--camera-height",
        str(args.camera_height),
        "--camera-fps",
        str(args.camera_fps),
        "--threadgroup",
        str(args.threadgroup),
        "--prompt",
        *args.prompt,
    ]
    for n in (13, 14, 17, 18, 30, 31, 36, 37, 38, 45, 46):
        cmd += [
            f"--exp{n}-helper",
            str(getattr(args, f"exp{n}_helper")),
        ]
    return cmd


def run_parent(args):
    if not args.image.exists():
        raise SystemExit(f"Missing fixed input: {args.image}")
    if not torch.backends.mps.is_available():
        raise SystemExit("MPS unavailable.")
    for n in (13, 14, 17, 18, 30, 31, 36, 37, 38, 45, 46):
        p = getattr(args, f"exp{n}_helper")
        if not p.exists():
            raise SystemExit(f"Missing helper: {p}")

    print(
        "Experiment 0053c fresh-process runtime-context audit\n"
        f"  rounds: {PROCESSES_PER_CONDITION}\n"
        f"  conditions: {len(MODES)}\n"
        f"  measurements/worker: {MEASUREMENTS}\n"
        f"  total workers: {PROCESSES_PER_CONDITION * len(MODES)}",
        flush=True,
    )

    records = []
    with tempfile.TemporaryDirectory(prefix="metalground_0053c_") as td:
        td_path = Path(td)

        for round_idx, order in enumerate(ROUND_ORDERS):
            print(
                f"\n=== Fresh-process round {round_idx + 1}/5 ===\n"
                f"order: {order}",
                flush=True,
            )
            round_modes = {}

            for mode_idx, mode in enumerate(order):
                worker_out = (
                    td_path
                    / f"round{round_idx}_{mode_idx}_{mode}.json"
                )
                cmd = worker_command(args, mode, worker_out)

                print(f"\nLaunching fresh worker: {mode}", flush=True)
                proc = subprocess.run(cmd)
                if proc.returncode != 0:
                    raise RuntimeError(
                        f"Worker failed: round={round_idx}, mode={mode}, "
                        f"returncode={proc.returncode}"
                    )
                if not worker_out.exists():
                    raise RuntimeError(
                        f"Worker did not produce output: {worker_out}"
                    )

                rec = json.loads(worker_out.read_text())
                rec["round_index"] = round_idx
                rec["order_index"] = mode_idx
                records.append(rec)
                round_modes[mode] = rec

                print(
                    f"Worker median {mode}: "
                    f"{rec['summary_ms']['median']:.3f} ms",
                    flush=True,
                )
                time.sleep(INTER_WORKER_SETTLE_S)

            if set(round_modes) != set(MODES):
                raise RuntimeError(
                    f"Round {round_idx} missing modes: "
                    f"{set(MODES) - set(round_modes)}"
                )

    # Per-mode process-median summary.
    per_mode_process_medians = {}
    per_mode = {}
    for mode in MODES:
        vals = [
            rec["summary_ms"]["median"]
            for rec in records
            if rec["worker_mode"] == mode
        ]
        if len(vals) != PROCESSES_PER_CONDITION:
            raise RuntimeError(
                f"Expected {PROCESSES_PER_CONDITION} workers for {mode}; "
                f"got {len(vals)}"
            )
        per_mode_process_medians[mode] = vals
        per_mode[mode] = stat(vals)

    # Fresh-process round-paired contrasts.
    contrasts = {
        "opencv_import_only_minus_pristine": [],
        "camera_init_increment_minus_import_only": [],
        "active_during_build_increment_minus_camera_init": [],
        "camera_init_minus_pristine": [],
        "active_during_build_minus_pristine": [],
    }

    round_summaries = []
    for round_idx in range(PROCESSES_PER_CONDITION):
        rm = {
            rec["worker_mode"]: rec["summary_ms"]["median"]
            for rec in records
            if rec["round_index"] == round_idx
        }
        row = {
            "round_index": round_idx,
            "mode_process_medians_ms": rm,
            "contrasts_ms": {
                "opencv_import_only_minus_pristine": (
                    rm[MODE_IMPORT] - rm[MODE_PRISTINE]
                ),
                "camera_init_increment_minus_import_only": (
                    rm[MODE_INIT] - rm[MODE_IMPORT]
                ),
                "active_during_build_increment_minus_camera_init": (
                    rm[MODE_ACTIVE_BUILD] - rm[MODE_INIT]
                ),
                "camera_init_minus_pristine": (
                    rm[MODE_INIT] - rm[MODE_PRISTINE]
                ),
                "active_during_build_minus_pristine": (
                    rm[MODE_ACTIVE_BUILD] - rm[MODE_PRISTINE]
                ),
            },
        }
        round_summaries.append(row)
        for key, value in row["contrasts_ms"].items():
            contrasts[key].append(value)

    contrast_decisions = {
        key: contrast_result(vals)
        for key, vals in contrasts.items()
    }

    material = [
        key
        for key, value in contrast_decisions.items()
        if value["material_slowdown"]
    ]

    pristine_median_of_process_medians = per_mode[MODE_PRISTINE]["median"]
    active_median_of_process_medians = per_mode[MODE_ACTIVE_BUILD]["median"]

    result = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "experiment": "0053c",
        "purpose": (
            "Fresh-process audit of the persistent runtime-context gap, "
            "separating OpenCV import, camera initialization, and camera "
            "activity during MetalGround runtime construction."
        ),
        "protocol": {
            "conditions": MODES,
            "fresh_processes_per_condition": PROCESSES_PER_CONDITION,
            "initial_warmups_per_worker": INITIAL_WARMUPS,
            "post_context_warmups_per_worker": POST_CONTEXT_WARMUPS,
            "measurements_per_worker": MEASUREMENTS,
            "round_orders": ROUND_ORDERS,
            "inter_worker_settle_s": INTER_WORKER_SETTLE_S,
            "fixed_image": str(args.image),
            "prompt": args.prompt,
            "performance_timing": True,
            "approximation": False,
            "retraining": False,
            "quantization": False,
            "reduced_precision": False,
        },
        "historical_context_ms": {
            "exp0047_standalone_authority": (
                HISTORICAL_STANDALONE_AUTHORITY_MS
            ),
            "exp0053b_closed_same_process": (
                HISTORICAL_0053B_CLOSED_MS
            ),
            "exp0053b_live_same_process": (
                HISTORICAL_0053B_LIVE_MS
            ),
        },
        "per_mode_process_medians_ms": per_mode_process_medians,
        "per_mode_summary_of_process_medians_ms": per_mode,
        "round_summaries": round_summaries,
        "pre_registered_contrast_rule": {
            "definition": (
                "A fresh-process context effect is material iff all 5 "
                "round-paired deltas have the same sign and the median "
                "round delta magnitude is >=10 ms."
            ),
            "material_threshold_ms": MATERIAL_EFFECT_MS,
            "contrasts": contrast_decisions,
            "material_slowdown_contrasts": material,
        },
        "headline_context": {
            "pristine_median_of_process_medians_ms": (
                pristine_median_of_process_medians
            ),
            "active_build_median_of_process_medians_ms": (
                active_median_of_process_medians
            ),
            "pristine_minus_historical_0047_ms": (
                pristine_median_of_process_medians
                - HISTORICAL_STANDALONE_AUTHORITY_MS
            ),
            "active_build_minus_pristine_ms": (
                active_median_of_process_medians
                - pristine_median_of_process_medians
            ),
        },
        "workers": records,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n"
    )

    print("\n=== Experiment 0053c summary ===", flush=True)
    for mode in MODES:
        print(
            f"{mode}: process medians="
            f"{per_mode_process_medians[mode]} "
            f"median-of-medians={per_mode[mode]['median']:.3f} ms",
            flush=True,
        )
    print("\nFresh-process causal contrasts:", flush=True)
    for key, value in contrast_decisions.items():
        print(
            f"{key}: {value['round_deltas_ms']} "
            f"median={value['median_round_delta_ms']:+.3f} ms "
            f"material={value['material_slowdown']}",
            flush=True,
        )
    print(
        f"Material slowdown contrasts: {material or 'none'}",
        flush=True,
    )
    print(f"Saved: {args.output}", flush=True)


def make_parser():
    ap = argparse.ArgumentParser()
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
    ap.add_argument("--camera-index", type=int, default=0)
    ap.add_argument("--camera-width", type=int, default=1280)
    ap.add_argument("--camera-height", type=int, default=720)
    ap.add_argument("--camera-fps", type=float, default=30.0)
    ap.add_argument(
        "--output",
        type=Path,
        default=Path(
            "results/metalground_runtime_context_audit_0053c.json"
        ),
    )
    ap.add_argument("--threadgroup", type=int, default=256)

    # Internal worker args.
    ap.add_argument("--worker-mode", choices=MODES)
    ap.add_argument("--worker-output", type=Path)

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
    return ap


def main():
    args = make_parser().parse_args()
    if args.worker_mode is not None:
        if args.worker_output is None:
            raise SystemExit("--worker-output is required in worker mode.")
        raise SystemExit(run_worker(args))
    run_parent(args)


if __name__ == "__main__":
    main()
