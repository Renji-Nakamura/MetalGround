#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import math
import statistics
import sys
import tempfile
import threading
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from PIL import Image

CANVAS_WH = (1200, 901)
CANVAS_FILL = (114, 114, 114)

PRE_REGISTERED_GATES = {
    "required_samples": 30,
    "capture_freshness_p95_ms_max": 75.0,
    "non_inference_overhead_median_ms_max": 100.0,
    "processed_fps_min": 1.8,
    "app_ingress_to_ui_submit_median_ms_max": 650.0,
}


def load_module(path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load helper: {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod


def percentile(xs, q):
    if not xs:
        return None
    return float(np.percentile(np.asarray(xs, dtype=np.float64), q))


def stats(xs):
    if not xs:
        return None
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


def ns_to_ms(delta_ns):
    return delta_ns / 1_000_000.0


def letterbox(image: Image.Image, size_wh=CANVAS_WH):
    W, H = size_wh
    image = image.convert("RGB")
    w, h = image.size
    scale = min(W / w, H / h)
    nw = max(1, int(round(w * scale)))
    nh = max(1, int(round(h * scale)))
    resized = image.resize((nw, nh), Image.Resampling.LANCZOS)
    left = (W - nw) // 2
    top = (H - nh) // 2
    canvas = Image.new("RGB", (W, H), CANVAS_FILL)
    canvas.paste(resized, (left, top))
    return canvas


class LatestFrameCamera:
    def __init__(self, cv2, index, width, height, fps):
        self.cv2 = cv2
        self.index = index
        self.width = width
        self.height = height
        self.fps = fps

        self.cap = None
        self.backend_requested = None
        self.backend_actual = None

        self._cond = threading.Condition()
        self._latest = None
        self._seq = 0
        self._stopped = False
        self._thread = None
        self._error = None
        self.capture_timestamps_ns = deque(maxlen=10000)

    def open(self):
        cv2 = self.cv2

        attempts = []
        if hasattr(cv2, "CAP_AVFOUNDATION"):
            attempts.append(("CAP_AVFOUNDATION", cv2.CAP_AVFOUNDATION))
        attempts.append(("CAP_ANY", cv2.CAP_ANY))

        for name, backend in attempts:
            cap = cv2.VideoCapture(self.index, backend)
            if cap.isOpened():
                self.cap = cap
                self.backend_requested = name
                break
            cap.release()

        if self.cap is None:
            raise RuntimeError(
                "Could not open camera. Check macOS camera permission for "
                "Terminal/iTerm and verify --camera-index."
            )

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, float(self.width))
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, float(self.height))
        self.cap.set(cv2.CAP_PROP_FPS, float(self.fps))

        # Not every backend supports this; record the result but do not depend on it.
        buffer_set_result = None
        if hasattr(cv2, "CAP_PROP_BUFFERSIZE"):
            try:
                buffer_set_result = bool(
                    self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1.0)
                )
            except Exception:
                buffer_set_result = False
        self.buffer_size_set_result = buffer_set_result

        try:
            self.backend_actual = self.cap.getBackendName()
        except Exception:
            self.backend_actual = None

        self.actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.actual_fps_property = float(self.cap.get(cv2.CAP_PROP_FPS))

    def start(self):
        if self.cap is None:
            self.open()
        self._thread = threading.Thread(
            target=self._run,
            name="metalground-camera",
            daemon=True,
        )
        self._thread.start()

    def _run(self):
        try:
            while True:
                with self._cond:
                    if self._stopped:
                        return

                ok, frame = self.cap.read()
                t_ns = time.perf_counter_ns()
                if not ok or frame is None:
                    time.sleep(0.005)
                    continue

                with self._cond:
                    self._seq += 1
                    self._latest = (self._seq, t_ns, frame)
                    self.capture_timestamps_ns.append(t_ns)
                    self._cond.notify_all()
        except BaseException as exc:
            with self._cond:
                self._error = repr(exc)
                self._cond.notify_all()

    def get_latest_after(self, last_seq, timeout=5.0):
        deadline = time.monotonic() + timeout
        with self._cond:
            while True:
                if self._error is not None:
                    raise RuntimeError(
                        f"Camera thread failed: {self._error}"
                    )
                if self._latest is not None and self._latest[0] > last_seq:
                    seq, t_ns, frame = self._latest
                    # Copy because the camera backend may reuse underlying buffers.
                    return seq, t_ns, frame.copy()

                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError(
                        "Timed out waiting for a fresh camera frame."
                    )
                self._cond.wait(timeout=min(remaining, 0.25))

    def capture_timestamps_between(self, start_ns, end_ns):
        with self._cond:
            return [
                x
                for x in list(self.capture_timestamps_ns)
                if start_ns <= x <= end_ns
            ]

    def stop(self):
        with self._cond:
            self._stopped = True
            self._cond.notify_all()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        if self.cap is not None:
            self.cap.release()


def make_inputs(processor, canvas, prompt_labels):
    cpu = processor(
        images=canvas,
        text=[prompt_labels],
        return_tensors="pt",
    )
    return cpu


def overlay_detections(cv2, canvas_rgb, result, max_detections=25):
    frame = cv2.cvtColor(canvas_rgb, cv2.COLOR_RGB2BGR)

    boxes = result["boxes"]
    scores = result["scores"]
    if hasattr(boxes, "detach"):
        boxes = boxes.detach().cpu().numpy()
    else:
        boxes = np.asarray(boxes)
    if hasattr(scores, "detach"):
        scores = scores.detach().cpu().numpy()
    else:
        scores = np.asarray(scores)

    labels = result.get("text_labels")
    if labels is None:
        labels = result.get("labels", [])
    labels = list(labels)

    n = min(len(boxes), max_detections)
    for i in range(n):
        x1, y1, x2, y2 = [int(round(float(v))) for v in boxes[i]]
        x1 = max(0, min(frame.shape[1] - 1, x1))
        x2 = max(0, min(frame.shape[1] - 1, x2))
        y1 = max(0, min(frame.shape[0] - 1, y1))
        y2 = max(0, min(frame.shape[0] - 1, y2))
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 2)

        label = str(labels[i]) if i < len(labels) else "object"
        score = float(scores[i]) if i < len(scores) else float("nan")
        text = f"{label} {score:.2f}"
        cv2.putText(
            frame,
            text,
            (x1, max(18, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

    return frame, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera-index", type=int, default=0)
    ap.add_argument("--camera-width", type=int, default=1280)
    ap.add_argument("--camera-height", type=int, default=720)
    ap.add_argument("--camera-fps", type=float, default=30.0)
    ap.add_argument(
        "--model",
        default="IDEA-Research/grounding-dino-tiny",
    )
    ap.add_argument(
        "--prompt",
        nargs="+",
        default=["a cat", "a dog"],
        help="Grounding DINO labels; historical default preserves cat/dog workload.",
    )
    ap.add_argument("--box-threshold", type=float, default=0.30)
    ap.add_argument("--text-threshold", type=float, default=0.25)
    ap.add_argument("--warmup-frames", type=int, default=3)
    ap.add_argument("--samples", type=int, default=30)
    ap.add_argument("--no-display", action="store_true")
    ap.add_argument(
        "--output",
        type=Path,
        default=Path("results/metalground_webcam_probe_0053a.json"),
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

    if args.samples < 1:
        raise SystemExit("--samples must be >= 1")
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
            "OpenCV is required. Add the pinned dependency with:\n"
            "  uv add 'opencv-python==4.14.0.94'"
        ) from exc

    camera = LatestFrameCamera(
        cv2=cv2,
        index=args.camera_index,
        width=args.camera_width,
        height=args.camera_height,
        fps=args.camera_fps,
    )

    stage0_dispatchers = []
    cache = None
    rt = None

    try:
        print("Opening camera before model construction...", flush=True)
        camera.start()

        seq0, t0, frame0 = camera.get_latest_after(0, timeout=10.0)
        print(
            f"Camera backend: {camera.backend_actual or camera.backend_requested}\n"
            f"Camera property: {camera.actual_width}x{camera.actual_height} "
            f"@ {camera.actual_fps_property:.3f} FPS",
            flush=True,
        )

        rgb0 = cv2.cvtColor(frame0, cv2.COLOR_BGR2RGB)
        canvas0 = letterbox(Image.fromarray(rgb0))

        with tempfile.TemporaryDirectory(prefix="metalground_0053a_") as td:
            temp_image = Path(td) / "first_canvas.png"
            canvas0.save(temp_image, format="PNG", optimize=False)

            args.image = temp_image

            h37 = load_module(args.exp37_helper, "mg53_h37")
            h38 = load_module(args.exp38_helper, "mg53_h38")
            h45 = load_module(args.exp45_helper, "mg53_h45")
            h46 = load_module(args.exp46_helper, "mg53_h46")

            print("Building adopted MetalGround runtime...", flush=True)
            rt = h37.build_runtime(args)
            final_model = rt["model"]
            processor = rt["processor"]
            h14 = rt["h14"]

            cache = h38.ExactTextBackboneCache(
                final_model.model.text_backbone
            )
            h38.set_runtime_mode(rt, cache, "wide_cache")

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

            first_cpu_inputs = make_inputs(
                processor,
                canvas0,
                args.prompt,
            )
            base_pixel_shape = tuple(
                first_cpu_inputs["pixel_values"].shape
            )
            if base_pixel_shape != (1, 3, 800, 1065):
                raise RuntimeError(
                    "Unexpected processor geometry: "
                    f"{base_pixel_shape}; expected (1,3,800,1065)."
                )

            first_inputs = {
                k: v.to("mps")
                for k, v in first_cpu_inputs.items()
            }
            h14.sync()

            cache.cached_output = None
            cache.prime(final_model, first_inputs, h14.sync)
            cache.set_enabled(True)

            print(
                f"Prompt: {args.prompt}\n"
                f"Processor shape: {base_pixel_shape}\n"
                f"Warmup frames: {args.warmup_frames}",
                flush=True,
            )

            if not args.no_display:
                cv2.namedWindow(
                    "MetalGround 0053a",
                    cv2.WINDOW_NORMAL,
                )
                cv2.resizeWindow("MetalGround 0053a", 900, 676)

            last_seq = seq0

            # Warmup with genuinely fresh frames.
            for w in range(args.warmup_frames):
                seq, t_capture, frame = camera.get_latest_after(
                    last_seq, timeout=5.0
                )
                last_seq = seq
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                canvas = letterbox(Image.fromarray(rgb))
                cpu_inputs = make_inputs(
                    processor, canvas, args.prompt
                )
                inputs = {
                    k: v.to("mps")
                    for k, v in cpu_inputs.items()
                }
                h14.sync()
                with torch.inference_mode():
                    outputs = final_model(**inputs)
                h14.sync()
                _ = processor.post_process_grounded_object_detection(
                    outputs,
                    inputs["input_ids"],
                    threshold=args.box_threshold,
                    text_threshold=args.text_threshold,
                    target_sizes=[(CANVAS_WH[1], CANVAS_WH[0])],
                )
                h14.sync()
                print(
                    f"  warmup {w + 1}/{args.warmup_frames}",
                    flush=True,
                )

            print(
                f"Collecting {args.samples} measured webcam samples...",
                flush=True,
            )
            samples = []
            measure_start_ns = time.perf_counter_ns()
            previous_selected_seq = last_seq
            ui_end_times = []
            interrupted = False

            for i in range(args.samples):
                seq, t_capture_ns, frame = camera.get_latest_after(
                    last_seq, timeout=5.0
                )
                last_seq = seq

                t_worker_start_ns = time.perf_counter_ns()
                dropped_since_prev = max(
                    0,
                    seq - previous_selected_seq - 1,
                )
                previous_selected_seq = seq

                # Stage 1: camera BGR -> RGB PIL -> fixed 1200x901 canvas.
                t0_ns = time.perf_counter_ns()
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                canvas = letterbox(Image.fromarray(rgb))
                canvas_rgb = np.asarray(canvas)
                t1_ns = time.perf_counter_ns()

                # Stage 2: HF image/text processor on CPU.
                cpu_inputs = make_inputs(
                    processor,
                    canvas,
                    args.prompt,
                )
                t2_ns = time.perf_counter_ns()

                if tuple(cpu_inputs["pixel_values"].shape) != base_pixel_shape:
                    raise RuntimeError(
                        f"Processor geometry changed at sample {i}: "
                        f"{tuple(cpu_inputs['pixel_values'].shape)}"
                    )

                # Stage 3: CPU -> MPS.
                inputs = {
                    k: v.to("mps")
                    for k, v in cpu_inputs.items()
                }
                h14.sync()
                t3_ns = time.perf_counter_ns()

                # Stage 4: adopted MetalGround inference.
                with torch.inference_mode():
                    outputs = final_model(**inputs)
                h14.sync()
                t4_ns = time.perf_counter_ns()

                # Stage 5: HF postprocess + materialize detections to CPU.
                result = processor.post_process_grounded_object_detection(
                    outputs,
                    inputs["input_ids"],
                    threshold=args.box_threshold,
                    text_threshold=args.text_threshold,
                    target_sizes=[(CANVAS_WH[1], CANVAS_WH[0])],
                )[0]
                h14.sync()

                # Force result materialization to CPU inside postprocess timing.
                result_cpu = {
                    "boxes": result["boxes"].detach().cpu(),
                    "scores": result["scores"].detach().cpu(),
                    "text_labels": list(
                        result.get(
                            "text_labels",
                            result.get("labels", []),
                        )
                    ),
                }
                t5_ns = time.perf_counter_ns()

                # Stage 6: overlay.
                display_frame, detections_drawn = overlay_detections(
                    cv2,
                    canvas_rgb,
                    result_cpu,
                )
                t6_ns = time.perf_counter_ns()

                # Stage 7: UI submit.
                if not args.no_display:
                    cv2.imshow("MetalGround 0053a", display_frame)
                    key = cv2.waitKey(1) & 0xFF
                else:
                    key = 255
                t7_ns = time.perf_counter_ns()
                ui_end_times.append(t7_ns)

                queue_age_ms = ns_to_ms(
                    t_worker_start_ns - t_capture_ns
                )
                inference_ms = ns_to_ms(t4_ns - t3_ns)
                worker_total_ms = ns_to_ms(t7_ns - t_worker_start_ns)
                non_inference_ms = worker_total_ms - inference_ms
                ingress_to_ui_ms = ns_to_ms(t7_ns - t_capture_ns)

                sample = {
                    "sample_index": i,
                    "capture_sequence": seq,
                    "dropped_by_latest_frame_policy_since_previous": (
                        dropped_since_prev
                    ),
                    "capture_to_worker_start_ms": queue_age_ms,
                    "frame_convert_letterbox_ms": ns_to_ms(t1_ns - t0_ns),
                    "processor_cpu_ms": ns_to_ms(t2_ns - t1_ns),
                    "cpu_to_mps_ms": ns_to_ms(t3_ns - t2_ns),
                    "inference_ms": inference_ms,
                    "postprocess_to_cpu_ms": ns_to_ms(t5_ns - t4_ns),
                    "overlay_ms": ns_to_ms(t6_ns - t5_ns),
                    "ui_submit_ms": ns_to_ms(t7_ns - t6_ns),
                    "worker_total_ms": worker_total_ms,
                    "non_inference_overhead_ms": non_inference_ms,
                    "app_ingress_to_ui_submit_ms": ingress_to_ui_ms,
                    "detections_drawn": detections_drawn,
                }
                samples.append(sample)

                print(
                    f"[{i + 1:02d}/{args.samples}] "
                    f"age={queue_age_ms:6.1f} ms  "
                    f"infer={inference_ms:7.1f} ms  "
                    f"e2e={ingress_to_ui_ms:7.1f} ms  "
                    f"drop={dropped_since_prev}",
                    flush=True,
                )

                del outputs, result, result_cpu, inputs, cpu_inputs
                if (i + 1) % 10 == 0:
                    gc.collect()

                if key in (ord("q"), 27):
                    interrupted = True
                    print("User requested stop.", flush=True)
                    break

            measure_end_ns = time.perf_counter_ns()

            capture_ts = camera.capture_timestamps_between(
                measure_start_ns,
                measure_end_ns,
            )
            capture_intervals_ms = [
                ns_to_ms(b - a)
                for a, b in zip(capture_ts, capture_ts[1:])
                if b > a
            ]

            inter_ui_ms = [
                ns_to_ms(b - a)
                for a, b in zip(ui_end_times, ui_end_times[1:])
                if b > a
            ]

            if len(ui_end_times) >= 2:
                processed_fps = (
                    (len(ui_end_times) - 1)
                    / (
                        (ui_end_times[-1] - ui_end_times[0])
                        / 1_000_000_000.0
                    )
                )
            else:
                processed_fps = 0.0

            metric_fields = [
                "capture_to_worker_start_ms",
                "frame_convert_letterbox_ms",
                "processor_cpu_ms",
                "cpu_to_mps_ms",
                "inference_ms",
                "postprocess_to_cpu_ms",
                "overlay_ms",
                "ui_submit_ms",
                "worker_total_ms",
                "non_inference_overhead_ms",
                "app_ingress_to_ui_submit_ms",
            ]
            summary_metrics = {
                field: stats([s[field] for s in samples])
                for field in metric_fields
            }

            dropped_total = sum(
                s["dropped_by_latest_frame_policy_since_previous"]
                for s in samples
            )

            gate_results = {
                "required_samples": {
                    "threshold": PRE_REGISTERED_GATES["required_samples"],
                    "observed": len(samples),
                    "pass": len(samples)
                    >= PRE_REGISTERED_GATES["required_samples"],
                },
                "capture_freshness_p95_ms": {
                    "threshold_max": PRE_REGISTERED_GATES[
                        "capture_freshness_p95_ms_max"
                    ],
                    "observed": (
                        summary_metrics[
                            "capture_to_worker_start_ms"
                        ]["p95"]
                        if samples
                        else None
                    ),
                    "pass": (
                        bool(samples)
                        and summary_metrics[
                            "capture_to_worker_start_ms"
                        ]["p95"]
                        <= PRE_REGISTERED_GATES[
                            "capture_freshness_p95_ms_max"
                        ]
                    ),
                },
                "non_inference_overhead_median_ms": {
                    "threshold_max": PRE_REGISTERED_GATES[
                        "non_inference_overhead_median_ms_max"
                    ],
                    "observed": (
                        summary_metrics[
                            "non_inference_overhead_ms"
                        ]["median"]
                        if samples
                        else None
                    ),
                    "pass": (
                        bool(samples)
                        and summary_metrics[
                            "non_inference_overhead_ms"
                        ]["median"]
                        <= PRE_REGISTERED_GATES[
                            "non_inference_overhead_median_ms_max"
                        ]
                    ),
                },
                "processed_fps": {
                    "threshold_min": PRE_REGISTERED_GATES[
                        "processed_fps_min"
                    ],
                    "observed": processed_fps,
                    "pass": processed_fps
                    >= PRE_REGISTERED_GATES["processed_fps_min"],
                },
                "app_ingress_to_ui_submit_median_ms": {
                    "threshold_max": PRE_REGISTERED_GATES[
                        "app_ingress_to_ui_submit_median_ms_max"
                    ],
                    "observed": (
                        summary_metrics[
                            "app_ingress_to_ui_submit_ms"
                        ]["median"]
                        if samples
                        else None
                    ),
                    "pass": (
                        bool(samples)
                        and summary_metrics[
                            "app_ingress_to_ui_submit_ms"
                        ]["median"]
                        <= PRE_REGISTERED_GATES[
                            "app_ingress_to_ui_submit_median_ms_max"
                        ]
                    ),
                },
            }
            all_gates_pass = all(
                x["pass"] for x in gate_results.values()
            )

            result_json = {
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "experiment": "0053a",
                "purpose": (
                    "Application-level webcam pipeline characterization of "
                    "the adopted MetalGround runtime using an OpenCV camera "
                    "frontend with a one-slot latest-frame policy."
                ),
                "scope": "prototype_webcam_characterization",
                "environment": {
                    "torch": torch.__version__,
                    "opencv": cv2.__version__,
                    "device": "mps",
                    "dtype": "float32",
                },
                "camera": {
                    "requested_index": args.camera_index,
                    "requested_size": [
                        args.camera_width,
                        args.camera_height,
                    ],
                    "requested_fps": args.camera_fps,
                    "backend_requested": camera.backend_requested,
                    "backend_actual": camera.backend_actual,
                    "actual_size_property": [
                        camera.actual_width,
                        camera.actual_height,
                    ],
                    "actual_fps_property": camera.actual_fps_property,
                    "cap_prop_buffersize_set_to_1_result": (
                        camera.buffer_size_set_result
                    ),
                    "successful_capture_frames_during_measurement": len(
                        capture_ts
                    ),
                    "capture_interval_ms": stats(capture_intervals_ms),
                },
                "configuration": {
                    "prompt": args.prompt,
                    "box_threshold": args.box_threshold,
                    "text_threshold": args.text_threshold,
                    "canvas_wh": list(CANVAS_WH),
                    "processor_pixel_values_shape": list(base_pixel_shape),
                    "warmup_frames": args.warmup_frames,
                    "requested_samples": args.samples,
                    "completed_samples": len(samples),
                    "display_enabled": not args.no_display,
                    "latest_frame_mailbox_depth": 1,
                    "performance_timing": True,
                    "approximation": False,
                    "retraining": False,
                    "quantization": False,
                    "reduced_precision": False,
                },
                "timing_semantics": {
                    "capture_timestamp": (
                        "time.perf_counter_ns() immediately after "
                        "cv2.VideoCapture.read() returns a frame"
                    ),
                    "app_ingress_to_ui_submit": (
                        "capture-read return timestamp to completion of "
                        "cv2.imshow()/waitKey(1). This is NOT sensor "
                        "exposure-to-photon latency and may exclude backend "
                        "camera buffering before read() returns."
                    ),
                    "ui_submit": (
                        "window-system submission timing only; not physical "
                        "display scanout."
                    ),
                },
                "summary": {
                    **summary_metrics,
                    "processed_fps": processed_fps,
                    "inter_processed_frame_ms": stats(inter_ui_ms),
                    "dropped_by_latest_frame_policy_total": dropped_total,
                    "mean_superseded_capture_frames_per_processed_frame": (
                        dropped_total / len(samples)
                        if samples
                        else None
                    ),
                    "interrupted": interrupted,
                },
                "pre_registered_gates": {
                    "definition": PRE_REGISTERED_GATES,
                    "results": gate_results,
                    "all_pass": all_gates_pass,
                    "interpretation": (
                        "These gates qualify the Python/OpenCV prototype "
                        "pipeline only. They are not claims of real-time "
                        "performance or native zero-copy capture."
                    ),
                },
                "samples": samples,
                "notes": [
                    (
                        "The one-slot latest-frame mailbox intentionally "
                        "drops superseded camera frames instead of queuing "
                        "them behind ~0.5 s inference."
                    ),
                    (
                        "This prototype uses CPU/PIL/Hugging Face "
                        "preprocessing and OpenCV display. It is a "
                        "characterization step before native AVFoundation / "
                        "CVPixelBuffer / Metal preprocessing and rendering."
                    ),
                ],
            }

            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(
                json.dumps(result_json, indent=2, ensure_ascii=False) + "\n"
            )

            print("\n=== Experiment 0053a summary ===", flush=True)
            print(f"samples: {len(samples)}", flush=True)
            if samples:
                print(
                    "capture freshness p95: "
                    f"{summary_metrics['capture_to_worker_start_ms']['p95']:.3f} ms",
                    flush=True,
                )
                print(
                    "inference median: "
                    f"{summary_metrics['inference_ms']['median']:.3f} ms",
                    flush=True,
                )
                print(
                    "non-inference median: "
                    f"{summary_metrics['non_inference_overhead_ms']['median']:.3f} ms",
                    flush=True,
                )
                print(
                    "app ingress -> UI submit median: "
                    f"{summary_metrics['app_ingress_to_ui_submit_ms']['median']:.3f} ms",
                    flush=True,
                )
            print(f"processed FPS: {processed_fps:.4f}", flush=True)
            print(
                "latest-frame superseded captures: "
                f"{dropped_total}",
                flush=True,
            )
            print(
                f"prototype gates: {'PASS' if all_gates_pass else 'FAIL'}",
                flush=True,
            )
            print(f"Saved: {args.output}", flush=True)

    finally:
        if not args.no_display:
            try:
                cv2.destroyAllWindows()
            except Exception:
                pass
        for d in stage0_dispatchers:
            try:
                d.restore()
            except Exception:
                pass
        if cache is not None:
            try:
                cache.restore()
            except Exception:
                pass
        if rt is not None:
            try:
                for d in rt["layer_dispatchers"]:
                    d.restore()
                for fd in rt["fusion_dispatchers"]:
                    fd.restore()
            except Exception:
                pass
        camera.stop()


if __name__ == "__main__":
    main()
