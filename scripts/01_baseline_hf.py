#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import platform
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path

import psutil
import torch
from PIL import Image
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor


def sync(device: torch.device) -> None:
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize(device)


def percentile(xs: list[float], p: float) -> float:
    if not xs:
        return math.nan
    ys = sorted(xs)
    if len(ys) == 1:
        return ys[0]
    k = (len(ys) - 1) * p
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return ys[f]
    return ys[f] * (c - k) + ys[c] * (k - f)


def summary_ms(xs: list[float]) -> dict[str, float]:
    return {
        "n": len(xs),
        "mean_ms": statistics.fmean(xs),
        "median_ms": statistics.median(xs),
        "p90_ms": percentile(xs, 0.90),
        "p95_ms": percentile(xs, 0.95),
        "p99_ms": percentile(xs, 0.99),
        "min_ms": min(xs),
        "max_ms": max(xs),
        "fps_from_median": 1000.0 / statistics.median(xs),
    }


def mps_memory() -> dict[str, int]:
    out: dict[str, int] = {}
    for name in ("current_allocated_memory", "driver_allocated_memory", "recommended_max_memory"):
        fn = getattr(torch.mps, name, None)
        if fn is not None:
            try:
                out[name] = int(fn())
            except Exception:
                pass
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", choices=["mps", "cpu"], default="mps")
    ap.add_argument("--image", type=Path, default=Path("assets/input.jpg"))
    ap.add_argument("--model", default="IDEA-Research/grounding-dino-tiny")
    ap.add_argument("--prompt", nargs="+", default=["person", "laptop", "cup"])
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--threshold", type=float, default=0.35)
    ap.add_argument("--text-threshold", type=float, default=0.25)
    ap.add_argument("--tag", default="baseline")
    args = ap.parse_args()

    if not args.image.exists():
        raise SystemExit(f"Missing image: {args.image}. Put a fixed JPEG at assets/input.jpg")
    if args.device == "mps" and not torch.backends.mps.is_available():
        raise SystemExit("MPS is not available in this Python/PyTorch environment.")

    device = torch.device(args.device)
    image = Image.open(args.image).convert("RGB")
    text_labels = [args.prompt]

    print(f"Loading {args.model} ...")
    t0 = time.perf_counter()
    processor = AutoProcessor.from_pretrained(args.model)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(args.model)
    model.eval().to(device)
    sync(device)
    load_s = time.perf_counter() - t0

    # Prepare the exact same tensor for pure-forward timing.
    cpu_inputs = processor(images=image, text=text_labels, return_tensors="pt")
    inputs = {k: v.to(device) if hasattr(v, "to") else v for k, v in cpu_inputs.items()}

    print(f"Device: {device}; image={image.size}; prompt={args.prompt}")
    print(f"Warmup: {args.warmup}")
    with torch.inference_mode():
        for _ in range(args.warmup):
            _ = model(**inputs)
            sync(device)

    forward_ms: list[float] = []
    with torch.inference_mode():
        for i in range(args.iters):
            sync(device)
            t0 = time.perf_counter_ns()
            outputs = model(**inputs)
            sync(device)
            dt_ms = (time.perf_counter_ns() - t0) / 1e6
            forward_ms.append(dt_ms)
            print(f"forward {i + 1:02d}/{args.iters}: {dt_ms:.2f} ms")

    # Measure the user-visible inference path separately: preprocess + transfer + forward + postprocess.
    e2e_ms: list[float] = []
    last_result = None
    with torch.inference_mode():
        for i in range(args.iters):
            sync(device)
            t0 = time.perf_counter_ns()
            iter_inputs = processor(images=image, text=text_labels, return_tensors="pt")
            iter_inputs = {k: v.to(device) if hasattr(v, "to") else v for k, v in iter_inputs.items()}
            outputs = model(**iter_inputs)
            result = processor.post_process_grounded_object_detection(
                outputs,
                input_ids=iter_inputs.get("input_ids"),
                threshold=args.threshold,
                text_threshold=args.text_threshold,
                target_sizes=[(image.height, image.width)],
            )[0]
            # Force materialization/synchronization of result tensors.
            _ = result["scores"].detach().cpu().tolist()
            _ = result["boxes"].detach().cpu().tolist()
            sync(device)
            dt_ms = (time.perf_counter_ns() - t0) / 1e6
            e2e_ms.append(dt_ms)
            last_result = result
            print(f"e2e     {i + 1:02d}/{args.iters}: {dt_ms:.2f} ms")

    detections = []
    if last_result is not None:
        labels = last_result.get("text_labels", last_result.get("labels", []))
        for box, score, label in zip(last_result["boxes"], last_result["scores"], labels):
            detections.append({
                "label": str(label),
                "score": float(score.detach().cpu()),
                "box_xyxy": [float(x) for x in box.detach().cpu().tolist()],
            })

    process = psutil.Process()
    record = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "tag": args.tag,
        "model": args.model,
        "device": args.device,
        "dtype": "float32",
        "image": str(args.image),
        "image_size_wh": list(image.size),
        "prompt": args.prompt,
        "warmup": args.warmup,
        "iterations": args.iters,
        "load_seconds": load_s,
        "forward": summary_ms(forward_ms),
        "end_to_end": summary_ms(e2e_ms),
        "rss_bytes": int(process.memory_info().rss),
        "mps_memory": mps_memory() if args.device == "mps" else {},
        "software": {
            "python": platform.python_version(),
            "torch": torch.__version__,
        },
        "detections": detections,
    }

    out_dir = Path("results")
    out_dir.mkdir(exist_ok=True)
    out = out_dir / f"hf_{args.device}_{args.tag}.json"
    out.write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n")

    print("\n=== Forward ===")
    print(json.dumps(record["forward"], indent=2))
    print("\n=== End-to-end ===")
    print(json.dumps(record["end_to_end"], indent=2))
    print(f"\nDetections: {len(detections)}")
    for d in detections[:10]:
        print(f"  {d['label']}: {d['score']:.3f} {d['box_xyxy']}")
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
