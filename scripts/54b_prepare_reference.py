#!/usr/bin/env python3
import argparse
import json
import time
from pathlib import Path

import numpy as np
from PIL import Image
from transformers import AutoProcessor


def json_safe(obj):
    """Recursively convert metadata-only objects to JSON-native types."""
    if obj is None or isinstance(obj, (str, bool, int, float)):
        return obj
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, dict):
        return {str(k): json_safe(v) for k, v in obj.items()}
    if hasattr(obj, "items"):
        return {str(k): json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_safe(v) for v in obj]
    if hasattr(obj, "to_dict"):
        try:
            return json_safe(obj.to_dict())
        except Exception:
            pass
    return str(obj)

CAM_W, CAM_H = 1280, 720
LETTER_W, LETTER_H = 1200, 901
MODEL_H, MODEL_W = 800, 1065
FILL = 114

def percentile(xs, p):
    return float(np.percentile(np.asarray(xs, dtype=np.float64), p))

def stats(xs):
    a = np.asarray(xs, dtype=np.float64)
    return {
        "n": int(a.size),
        "mean": float(a.mean()),
        "median": float(np.median(a)),
        "p90": percentile(a, 90),
        "p95": percentile(a, 95),
        "min": float(a.min()),
        "max": float(a.max()),
    }

def synthetic_rgb():
    y, x = np.mgrid[0:CAM_H, 0:CAM_W]
    r = ((x * 255) // (CAM_W - 1)).astype(np.uint8)
    g = ((y * 255) // (CAM_H - 1)).astype(np.uint8)
    b = (((x // 32 + y // 24) & 1) * 180 + ((x * 13 + y * 7) % 76)).astype(np.uint8)
    img = np.stack([r, g, b], axis=-1)
    # Hard edges / small structures for interpolation diagnostics.
    img[80:240, 90:330] = np.array([255, 16, 32], np.uint8)
    img[400:640, 850:1180] = np.array([12, 240, 80], np.uint8)
    for k in range(0, min(CAM_W, CAM_H), 23):
        yy = np.arange(CAM_H)
        xx = (yy + k) % CAM_W
        img[yy, xx] = np.array([250, 250, 250], np.uint8)
    return img

def letterbox_pil(rgb):
    src = Image.fromarray(rgb, mode="RGB")
    scale = min(LETTER_W / CAM_W, LETTER_H / CAM_H)
    rw = int(round(CAM_W * scale))
    rh = int(round(CAM_H * scale))
    resized = src.resize((rw, rh), resample=Image.Resampling.BILINEAR)
    canvas = Image.new("RGB", (LETTER_W, LETTER_H), (FILL, FILL, FILL))
    left = (LETTER_W - rw) // 2
    top = (LETTER_H - rh) // 2
    canvas.paste(resized, (left, top))
    return canvas, {"resized_width": rw, "resized_height": rh, "left": left, "top": top}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="IDEA-Research/grounding-dino-tiny")
    ap.add_argument("--out-dir", default="results/0054b_reference")
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--samples", type=int, default=30)
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    rgb = synthetic_rgb()
    # Raw BGRA exactly as the native harness expects.
    bgra = np.empty((CAM_H, CAM_W, 4), dtype=np.uint8)
    bgra[..., 0] = rgb[..., 2]
    bgra[..., 1] = rgb[..., 1]
    bgra[..., 2] = rgb[..., 0]
    bgra[..., 3] = 255
    bgra.tofile(out / "input_1280x720_bgra8.raw")

    letter, geom = letterbox_pil(rgb)
    letter.save(out / "letterbox_reference.png")

    processor = AutoProcessor.from_pretrained(args.model)
    ip = processor.image_processor

    def run_once():
        return ip(images=letter, return_tensors="np")["pixel_values"][0].astype(np.float32, copy=False)

    for _ in range(args.warmup):
        ref = run_once()

    timings = []
    for _ in range(args.samples):
        t0 = time.perf_counter_ns()
        ref = run_once()
        t1 = time.perf_counter_ns()
        timings.append((t1 - t0) / 1e6)

    if tuple(ref.shape) != (3, MODEL_H, MODEL_W):
        raise RuntimeError(f"Unexpected processor output shape {ref.shape}; expected (3,{MODEL_H},{MODEL_W})")

    ref.tofile(out / "reference_pixel_values_f32_chw.bin")

    mean = [float(x) for x in getattr(ip, "image_mean")]
    std = [float(x) for x in getattr(ip, "image_std")]
    metadata = {
        "experiment": "0054b",
        "model": args.model,
        "input": {
            "width": CAM_W, "height": CAM_H, "format": "BGRA8",
            "raw_file": "input_1280x720_bgra8.raw",
        },
        "letterbox": {
            "width": LETTER_W, "height": LETTER_H, "fill_u8": FILL, **geom,
            "reference_png": "letterbox_reference.png",
            "resize_resample": "PIL.Image.Resampling.BILINEAR",
        },
        "processor": {
            "class": type(ip).__name__,
            "output_shape_chw": [int(x) for x in ref.shape],
            "image_mean": mean,
            "image_std": std,
            "do_resize": bool(getattr(ip, "do_resize", True)),
            "size": (
                dict(getattr(ip, "size"))
                if hasattr(getattr(ip, "size", None), "items")
                else getattr(ip, "size", None)
            ),
            "resample": str(getattr(ip, "resample", None)),
            "do_rescale": bool(getattr(ip, "do_rescale", True)),
            "rescale_factor": float(getattr(ip, "rescale_factor", 1.0 / 255.0)),
            "do_normalize": bool(getattr(ip, "do_normalize", True)),
        },
        "cpu_reference_timing_ms": stats(timings),
        "reference_file": "reference_pixel_values_f32_chw.bin",
        "reference_stats": {
            "min": float(ref.min()), "max": float(ref.max()),
            "mean": float(ref.mean()), "std": float(ref.std()),
        },
    }
    (out / "metadata.json").write_text(json.dumps(json_safe(metadata), indent=2, ensure_ascii=False))
    print("=== Experiment 0054b CPU reference ===")
    print("shape:", ref.shape)
    print("letterbox geometry:", geom)
    print("mean/std:", mean, std)
    print("CPU preprocess median ms:", metadata["cpu_reference_timing_ms"]["median"])
    print("Saved:", out / "metadata.json")

if __name__ == "__main__":
    main()
