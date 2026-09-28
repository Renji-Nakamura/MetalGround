#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import statistics
import sys
import tempfile
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from PIL import Image

MODE_CLOSED = "camera_closed_no_display"
MODE_OPEN_IDLE = "camera_open_idle_no_display"
MODE_CAPTURE = "camera_capture_active_no_display"
MODE_CAPTURE_DISPLAY = "camera_capture_active_display_active"

MODES = [
    MODE_CLOSED,
    MODE_OPEN_IDLE,
    MODE_CAPTURE,
    MODE_CAPTURE_DISPLAY,
]

# Balanced 4x4 Latin-square style order to reduce monotonic thermal/order bias.
ROUND_ORDERS = [
    [MODE_CLOSED, MODE_OPEN_IDLE, MODE_CAPTURE, MODE_CAPTURE_DISPLAY],
    [MODE_OPEN_IDLE, MODE_CAPTURE_DISPLAY, MODE_CLOSED, MODE_CAPTURE],
    [MODE_CAPTURE, MODE_CLOSED, MODE_CAPTURE_DISPLAY, MODE_OPEN_IDLE],
    [MODE_CAPTURE_DISPLAY, MODE_CAPTURE, MODE_OPEN_IDLE, MODE_CLOSED],
]

MATERIAL_COMPONENT_EFFECT_MS = 10.0
HISTORICAL_STANDALONE_AUTHORITY_MS = 476.360333
HISTORICAL_0053A_LIVE_INFERENCE_MS = 539.272854


def load_module(path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load helper: {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod


def summary(xs):
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


class CameraReader:
    def __init__(self, cv2, index, width, height, fps):
        self.cv2 = cv2
        self.index = index
        self.width = width
        self.height = height
        self.fps = fps
        self.cap = None
        self.backend_actual = None
        self.actual_size = None
        self.actual_fps = None

        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = None
        self._latest = None
        self._seq = 0
        self._timestamps = []

    def open(self):
        if self.cap is not None:
            return
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
            raise RuntimeError(
                "Could not open camera. Check macOS camera permission "
                "and --camera-index."
            )
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, float(self.width))
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, float(self.height))
        cap.set(cv2.CAP_PROP_FPS, float(self.fps))
        self.cap = cap
        try:
            self.backend_actual = cap.getBackendName()
        except Exception:
            self.backend_actual = None
        self.actual_size = [
            int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        ]
        self.actual_fps = float(cap.get(cv2.CAP_PROP_FPS))

    def read_one(self):
        self.open()
        ok, frame = self.cap.read()
        t_ns = time.perf_counter_ns()
        if not ok or frame is None:
            raise RuntimeError("Camera read failed.")
        with self._lock:
            self._seq += 1
            self._latest = (self._seq, t_ns, frame.copy())
            self._timestamps.append(t_ns)
        return self._latest

    def start_capture(self):
        self.open()
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        # Prime one frame so display has data immediately.
        self.read_one()
        self._thread = threading.Thread(
            target=self._run,
            name="metalground-0053b-camera",
            daemon=True,
        )
        self._thread.start()

    def _run(self):
        while not self._stop.is_set():
            ok, frame = self.cap.read()
            t_ns = time.perf_counter_ns()
            if not ok or frame is None:
                time.sleep(0.003)
                continue
            with self._lock:
                self._seq += 1
                self._latest = (self._seq, t_ns, frame.copy())
                self._timestamps.append(t_ns)

    def latest(self):
        with self._lock:
            if self._latest is None:
                return None
            seq, t_ns, frame = self._latest
            return seq, t_ns, frame.copy()

    def stop_capture_keep_open(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        self._thread = None

    def close(self):
        self.stop_capture_keep_open()
        if self.cap is not None:
            self.cap.release()
        self.cap = None

    def capture_interval_ms(self):
        with self._lock:
            ts = list(self._timestamps)
        return [
            (b - a) / 1_000_000.0
            for a, b in zip(ts, ts[1:])
            if b > a
        ]


def build_adopted_runtime(args, fixed_image_path):
    h37 = load_module(args.exp37_helper, "mg53b_h37")
    h38 = load_module(args.exp38_helper, "mg53b_h38")
    h45 = load_module(args.exp45_helper, "mg53b_h45")
    h46 = load_module(args.exp46_helper, "mg53b_h46")

    args.image = fixed_image_path
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

    image = Image.open(fixed_image_path).convert("RGB")
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
        "processor": processor,
        "inputs": inputs,
        "cache": cache,
        "stage0_dispatchers": stage0_dispatchers,
        "h14": h14,
    }


def prepare_mode(camera, cv2, mode):
    # All setup/teardown is outside measured inference timing.
    if mode == MODE_CLOSED:
        camera.close()
        try:
            cv2.destroyWindow("MetalGround 0053b")
        except Exception:
            pass
        return

    if mode == MODE_OPEN_IDLE:
        camera.close()
        camera.open()
        camera.read_one()
        camera.stop_capture_keep_open()
        try:
            cv2.destroyWindow("MetalGround 0053b")
        except Exception:
            pass
        return

    if mode == MODE_CAPTURE:
        camera.close()
        camera.open()
        camera.start_capture()
        try:
            cv2.destroyWindow("MetalGround 0053b")
        except Exception:
            pass
        # Let capture settle.
        time.sleep(0.10)
        return

    if mode == MODE_CAPTURE_DISPLAY:
        camera.close()
        camera.open()
        camera.start_capture()
        cv2.namedWindow("MetalGround 0053b", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("MetalGround 0053b", 640, 360)
        time.sleep(0.10)
        latest = camera.latest()
        if latest is not None:
            cv2.imshow("MetalGround 0053b", latest[2])
            cv2.waitKey(1)
        return

    raise ValueError(mode)


def pump_display(camera, cv2):
    latest = camera.latest()
    if latest is None:
        return
    cv2.imshow("MetalGround 0053b", latest[2])
    cv2.waitKey(1)


def measure_one(model, inputs, h14):
    h14.sync()
    t0 = time.perf_counter_ns()
    with torch.inference_mode():
        outputs = model(**inputs)
    h14.sync()
    t1 = time.perf_counter_ns()
    # Retain no output state across iterations.
    del outputs
    return (t1 - t0) / 1_000_000.0


def component_result(round_deltas):
    med = float(np.median(np.asarray(round_deltas)))
    all_positive = all(x > 0 for x in round_deltas)
    all_negative = all(x < 0 for x in round_deltas)
    material_slowdown = (
        all_positive and med >= MATERIAL_COMPONENT_EFFECT_MS
    )
    material_speedup = (
        all_negative and med <= -MATERIAL_COMPONENT_EFFECT_MS
    )
    return {
        "round_deltas_ms": round_deltas,
        "median_round_delta_ms": med,
        "all_rounds_positive": all_positive,
        "all_rounds_negative": all_negative,
        "material_threshold_ms": MATERIAL_COMPONENT_EFFECT_MS,
        "material_slowdown": material_slowdown,
        "material_speedup": material_speedup,
    }


def main():
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
    ap.add_argument("--rounds", type=int, default=4)
    ap.add_argument("--samples-per-mode", type=int, default=5)
    ap.add_argument("--warmup-per-mode", type=int, default=1)
    ap.add_argument(
        "--output",
        type=Path,
        default=Path(
            "results/metalground_camera_contention_audit_0053b.json"
        ),
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

    if args.rounds != 4:
        raise SystemExit(
            "0053b pre-registered protocol uses exactly --rounds 4."
        )
    if args.samples_per_mode != 5:
        raise SystemExit(
            "0053b pre-registered protocol uses exactly "
            "--samples-per-mode 5."
        )
    if args.warmup_per_mode != 1:
        raise SystemExit(
            "0053b pre-registered protocol uses exactly "
            "--warmup-per-mode 1."
        )
    if not args.image.exists():
        raise SystemExit(f"Missing fixed input: {args.image}")
    if not torch.backends.mps.is_available():
        raise SystemExit("MPS unavailable.")
    for n in helpers:
        p = getattr(args, f"exp{n}_helper")
        if not p.exists():
            raise SystemExit(f"Missing helper: {p}")

    try:
        import cv2
    except Exception as exc:
        raise SystemExit(
            "OpenCV is required; 0053a already used this dependency."
        ) from exc

    camera = CameraReader(
        cv2,
        args.camera_index,
        args.camera_width,
        args.camera_height,
        args.camera_fps,
    )

    runtime = None
    try:
        # Match 0053a initialization context: camera capture is active while
        # the adopted runtime is constructed. Measurement begins only after
        # this initialization and after the camera is explicitly closed.
        print(
            "Opening/starting AVFoundation camera to match 0053a "
            "runtime-construction context...",
            flush=True,
        )
        camera.open()
        camera.start_capture()
        time.sleep(0.20)
        print(
            f"Camera backend: {camera.backend_actual}\n"
            f"Camera property: {camera.actual_size[0]}x"
            f"{camera.actual_size[1]} @ {camera.actual_fps:.3f} FPS",
            flush=True,
        )

        print("Building adopted MetalGround runtime...", flush=True)
        runtime = build_adopted_runtime(args, args.image)
        model = runtime["model"]
        inputs = runtime["inputs"]
        h14 = runtime["h14"]

        # Initial model warmup while camera is still active, as in 0053a.
        print("Initial runtime warmup...", flush=True)
        for _ in range(3):
            _ = measure_one(model, inputs, h14)

        camera.close()
        gc.collect()

        round_records = []
        all_samples = {mode: [] for mode in MODES}

        for round_idx, order in enumerate(ROUND_ORDERS):
            print(
                f"\n=== Round {round_idx + 1}/4: {order} ===",
                flush=True,
            )
            mode_medians = {}

            for mode in order:
                print(f"Preparing mode: {mode}", flush=True)
                prepare_mode(camera, cv2, mode)

                # Mode warmup is outside measured samples.
                for _ in range(args.warmup_per_mode):
                    if mode == MODE_CAPTURE_DISPLAY:
                        pump_display(camera, cv2)
                    _ = measure_one(model, inputs, h14)
                    if mode == MODE_CAPTURE_DISPLAY:
                        pump_display(camera, cv2)

                vals = []
                for j in range(args.samples_per_mode):
                    if mode == MODE_CAPTURE_DISPLAY:
                        pump_display(camera, cv2)

                    ms = measure_one(model, inputs, h14)

                    if mode == MODE_CAPTURE_DISPLAY:
                        pump_display(camera, cv2)

                    vals.append(ms)
                    all_samples[mode].append(ms)
                    print(
                        f"  {mode} [{j + 1}/{args.samples_per_mode}] "
                        f"{ms:.3f} ms",
                        flush=True,
                    )

                med = float(np.median(np.asarray(vals)))
                mode_medians[mode] = med
                print(f"  median: {med:.3f} ms", flush=True)

                # Avoid mode transition state leaking into next condition.
                if mode != MODE_CLOSED:
                    camera.close()
                try:
                    cv2.destroyWindow("MetalGround 0053b")
                except Exception:
                    pass
                time.sleep(0.05)

            record = {
                "round_index": round_idx,
                "order": order,
                "mode_medians_ms": mode_medians,
                "deltas_ms": {
                    "camera_open_idle_minus_closed": (
                        mode_medians[MODE_OPEN_IDLE]
                        - mode_medians[MODE_CLOSED]
                    ),
                    "active_capture_minus_open_idle": (
                        mode_medians[MODE_CAPTURE]
                        - mode_medians[MODE_OPEN_IDLE]
                    ),
                    "display_increment_minus_capture": (
                        mode_medians[MODE_CAPTURE_DISPLAY]
                        - mode_medians[MODE_CAPTURE]
                    ),
                    "full_live_minus_closed": (
                        mode_medians[MODE_CAPTURE_DISPLAY]
                        - mode_medians[MODE_CLOSED]
                    ),
                },
            }
            round_records.append(record)

        camera.close()
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass

        per_mode = {
            mode: summary(vals)
            for mode, vals in all_samples.items()
        }

        component_keys = [
            "camera_open_idle_minus_closed",
            "active_capture_minus_open_idle",
            "display_increment_minus_capture",
            "full_live_minus_closed",
        ]
        components = {}
        for key in component_keys:
            deltas = [
                r["deltas_ms"][key]
                for r in round_records
            ]
            components[key] = component_result(deltas)

        material_components = [
            key
            for key, value in components.items()
            if value["material_slowdown"]
        ]

        closed_med = per_mode[MODE_CLOSED]["median"]
        live_med = per_mode[MODE_CAPTURE_DISPLAY]["median"]

        result = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "experiment": "0053b",
            "purpose": (
                "Same-process causal audit of the ~63 ms inference slowdown "
                "observed in 0053a, separating camera-open, active-capture, "
                "and display activity while holding model input fixed."
            ),
            "environment": {
                "torch": torch.__version__,
                "opencv": cv2.__version__,
                "device": "mps",
                "dtype": "float32",
            },
            "configuration": {
                "model": args.model,
                "fixed_image": str(args.image),
                "prompt": args.prompt,
                "rounds": args.rounds,
                "samples_per_mode": args.samples_per_mode,
                "warmup_per_mode": args.warmup_per_mode,
                "round_orders": ROUND_ORDERS,
                "camera_requested": {
                    "index": args.camera_index,
                    "size": [args.camera_width, args.camera_height],
                    "fps": args.camera_fps,
                },
                "camera_actual_backend": camera.backend_actual,
                "historical_context_ms": {
                    "exp0047_standalone_authority": (
                        HISTORICAL_STANDALONE_AUTHORITY_MS
                    ),
                    "exp0053a_live_camera_inference_median": (
                        HISTORICAL_0053A_LIVE_INFERENCE_MS
                    ),
                    "cross_protocol_gap": (
                        HISTORICAL_0053A_LIVE_INFERENCE_MS
                        - HISTORICAL_STANDALONE_AUTHORITY_MS
                    ),
                },
                "performance_timing": True,
                "approximation": False,
                "retraining": False,
                "quantization": False,
                "reduced_precision": False,
            },
            "timing": {
                "per_mode_inference_ms": per_mode,
                "round_records": round_records,
            },
            "pre_registered_component_rule": {
                "material_threshold_ms": MATERIAL_COMPONENT_EFFECT_MS,
                "definition": (
                    "A component is classified as a material causal slowdown "
                    "iff its round-level median delta is positive in all 4 "
                    "rounds and the median of the 4 round deltas is >=10 ms."
                ),
                "components": components,
                "material_slowdown_components": material_components,
            },
            "same_process_summary": {
                "closed_no_display_median_ms": closed_med,
                "capture_display_median_ms": live_med,
                "full_live_minus_closed_raw_median_difference_ms": (
                    live_med - closed_med
                ),
                "historical_0047_comparison_is_context_only": True,
                "historical_0053a_comparison_is_context_only": True,
            },
            "camera_capture_interval_ms": summary(
                camera.capture_interval_ms()
            ),
            "decision_hint": (
                "Use round-level component deltas for causal attribution. "
                "Do not attribute the historical 0047->0053a cross-protocol "
                "gap directly to camera/display without same-process evidence."
            ),
        }

        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(result, indent=2, ensure_ascii=False) + "\n"
        )

        print("\n=== Experiment 0053b summary ===", flush=True)
        for mode in MODES:
            s = per_mode[mode]
            print(
                f"{mode}: median={s['median']:.3f} ms "
                f"p95={s['p95']:.3f} ms",
                flush=True,
            )
        print("\nRound-level causal components:", flush=True)
        for key, value in components.items():
            print(
                f"{key}: deltas={value['round_deltas_ms']} "
                f"median={value['median_round_delta_ms']:+.3f} ms "
                f"material={value['material_slowdown']}",
                flush=True,
            )
        print(
            "Material slowdown components: "
            f"{material_components or 'none'}",
            flush=True,
        )
        print(f"Saved: {args.output}", flush=True)

    finally:
        camera.close()
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass

        if runtime is not None:
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


if __name__ == "__main__":
    main()
