#!/usr/bin/env python3
import argparse, json
from pathlib import Path
import numpy as np

def stats(a):
    return {
        "mean_abs": float(np.mean(np.abs(a))),
        "median_abs": float(np.median(np.abs(a))),
        "p95_abs": float(np.percentile(np.abs(a), 95)),
        "p99_abs": float(np.percentile(np.abs(a), 99)),
        "max_abs": float(np.max(np.abs(a))),
        "rmse": float(np.sqrt(np.mean(a.astype(np.float64) ** 2))),
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reference-dir", default="results/0054b_reference")
    ap.add_argument("--metal-json", default="results/metalground_metal_preprocess_0054b.json")
    ap.add_argument("--output", default="results/metalground_preprocess_compare_0054b.json")
    args = ap.parse_args()

    ref_dir = Path(args.reference_dir)
    meta = json.loads((ref_dir / "metadata.json").read_text())
    shape = tuple(meta["processor"]["output_shape_chw"])
    ref = np.fromfile(ref_dir / meta["reference_file"], dtype=np.float32).reshape(shape)

    metal_meta = json.loads(Path(args.metal_json).read_text())
    cand = np.fromfile(metal_meta["output_file"], dtype=np.float32).reshape(shape)

    diff = cand - ref
    ds = stats(diff)

    gates = {
        "shape_exact": list(cand.shape) == list(ref.shape),
        "mean_abs_le_0_010": ds["mean_abs"] <= 0.010,
        "p99_abs_le_0_050": ds["p99_abs"] <= 0.050,
        "max_abs_le_0_250": ds["max_abs"] <= 0.250,
        "metal_p95_ms_le_3": float(metal_meta["timing_ms"]["p95"]) <= 3.0,
    }
    result = {
        "experiment": "0054b",
        "reference_shape": list(ref.shape),
        "candidate_shape": list(cand.shape),
        "difference": ds,
        "cpu_reference_timing_ms": meta["cpu_reference_timing_ms"],
        "metal_timing_ms": metal_meta["timing_ms"],
        "pre_registered_gates": gates,
        "all_gates_pass": all(gates.values()),
        "notes": [
            "This compares a faithful two-pass Metal implementation against the current PIL + HF image processor reference on a deterministic 1280x720 frame.",
            "No Grounding DINO inference is run in 0054b.",
            "A failure of numerical gates triggers interpolation-semantics diagnosis before any one-pass fusion."
        ]
    }
    Path(args.output).write_text(json.dumps(result, indent=2))
    print("=== Experiment 0054b comparison ===")
    print(json.dumps(result, indent=2))
    print("Saved:", args.output)

if __name__ == "__main__":
    main()
