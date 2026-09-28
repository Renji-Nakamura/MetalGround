#!/usr/bin/env python3
import json
from pathlib import Path
import numpy as np
from PIL import Image

REF_DIR = Path("results/0054b_reference")
METAL_JSON = Path("results/metalground_metal_preprocess_0054b.json")
OUT = Path("results/metalground_preprocess_diagnostic_0054b1.json")

def err_stats(diff):
    a = np.abs(np.asarray(diff, dtype=np.float64))
    return {
        "n": int(a.size),
        "mean_abs": float(a.mean()),
        "median_abs": float(np.median(a)),
        "p95_abs": float(np.percentile(a, 95)),
        "p99_abs": float(np.percentile(a, 99)),
        "max_abs": float(a.max()),
        "rmse": float(np.sqrt(np.mean(np.asarray(diff, dtype=np.float64) ** 2))),
    }

meta = json.loads((REF_DIR / "metadata.json").read_text())
metal = json.loads(METAL_JSON.read_text())

# ---- Pass 1: compare RGBA8 intermediate with PIL letterbox reference. ----
pil = np.asarray(Image.open(REF_DIR / meta["letterbox"]["reference_png"]).convert("RGB"), dtype=np.int16)
mid_path = Path(metal["diagnostic_intermediate_file"])
mid_rgba = np.fromfile(mid_path, dtype=np.uint8).reshape(
    meta["letterbox"]["height"], meta["letterbox"]["width"], 4
)
mid_rgb = mid_rgba[..., :3].astype(np.int16)
pass1_diff_u8 = mid_rgb - pil
pass1 = {
    "overall_u8": err_stats(pass1_diff_u8),
    "per_channel_u8": {
        "R": err_stats(pass1_diff_u8[..., 0]),
        "G": err_stats(pass1_diff_u8[..., 1]),
        "B": err_stats(pass1_diff_u8[..., 2]),
    },
    "exact_pixel_fraction_rgb_triplet": float(np.mean(np.all(pass1_diff_u8 == 0, axis=-1))),
    "pixels_with_any_channel_abs_gt_1_fraction": float(
        np.mean(np.any(np.abs(pass1_diff_u8) > 1, axis=-1))
    ),
}

# ---- Validate what HF does after the saved letterbox image. ----
shape = tuple(meta["processor"]["output_shape_chw"])
hf_ref = np.fromfile(REF_DIR / meta["reference_file"], dtype=np.float32).reshape(shape)

out_h, out_w = shape[1], shape[2]
manual_img = Image.fromarray(pil.astype(np.uint8), "RGB").resize(
    (out_w, out_h), resample=Image.Resampling.BILINEAR
)
manual = np.asarray(manual_img, dtype=np.float32) / 255.0
mean = np.asarray(meta["processor"]["image_mean"], dtype=np.float32)
std = np.asarray(meta["processor"]["image_std"], dtype=np.float32)
manual = ((manual - mean) / std).transpose(2, 0, 1)
manual_vs_hf = err_stats(manual - hf_ref)

# ---- Final Metal vs HF, with spatial localization. ----
cand = np.fromfile(metal["output_file"], dtype=np.float32).reshape(shape)
diff = cand - hf_ref
absdiff = np.abs(diff)

per_channel = {
    "R": err_stats(diff[0]),
    "G": err_stats(diff[1]),
    "B": err_stats(diff[2]),
}

# Map output row centers back into the 1200x901 letterbox coordinate system.
letter_h = meta["letterbox"]["height"]
y_src = (np.arange(out_h, dtype=np.float64) + 0.5) * letter_h / out_h - 0.5
top = float(meta["letterbox"]["top"])
bottom = float(meta["letterbox"]["top"] + meta["letterbox"]["resized_height"] - 1)
boundary_mask = (
    (np.abs(y_src - top) <= 3.0) |
    (np.abs(y_src - bottom) <= 3.0)
)
padding_mask = (y_src < top - 3.0) | (y_src > bottom + 3.0)
interior_mask = (~boundary_mask) & (~padding_mask)

def region(mask_rows):
    vals = diff[:, mask_rows, :]
    return err_stats(vals)

regions = {
    "letterbox_boundary_rows_pm3_src_px": region(boundary_mask),
    "padding_away_from_boundary": region(padding_mask),
    "image_interior_away_from_boundary": region(interior_mask),
}

# Error vs reference gradient magnitude: interpolation mismatch should
# concentrate on edges/high-frequency regions if channel/layout/normalization
# are correct.
ref_hwc = hf_ref.transpose(1, 2, 0).astype(np.float64)
gx = np.zeros((out_h, out_w), dtype=np.float64)
gy = np.zeros((out_h, out_w), dtype=np.float64)
gx[:, 1:] = np.linalg.norm(ref_hwc[:, 1:] - ref_hwc[:, :-1], axis=-1)
gy[1:, :] = np.linalg.norm(ref_hwc[1:, :] - ref_hwc[:-1, :], axis=-1)
grad = np.maximum(gx, gy)
q50, q90, q99 = np.percentile(grad, [50, 90, 99])
pix_err = np.max(absdiff, axis=0)

def pix_stats(mask):
    return err_stats(pix_err[mask])

gradient_buckets = {
    "grad_le_q50": pix_stats(grad <= q50),
    "q50_to_q90": pix_stats((grad > q50) & (grad <= q90)),
    "q90_to_q99": pix_stats((grad > q90) & (grad <= q99)),
    "grad_gt_q99": pix_stats(grad > q99),
    "gradient_quantiles": {"q50": float(q50), "q90": float(q90), "q99": float(q99)},
}

diagnosis = []
if pass1["overall_u8"]["max_abs"] <= 1 and pass1["pixels_with_any_channel_abs_gt_1_fraction"] == 0:
    diagnosis.append("pass1_matches_PIL_within_1_u8_everywhere")
else:
    diagnosis.append("pass1_has_material_resize_or_rounding_difference")

if manual_vs_hf["max_abs"] < 1e-5:
    diagnosis.append("HF_post_letterbox_path_matches_explicit_PIL_bilinear_plus_normalize")
else:
    diagnosis.append("HF_post_letterbox_path_contains_additional_semantics")

if gradient_buckets["grad_gt_q99"]["mean_abs"] > 5 * max(
    gradient_buckets["grad_le_q50"]["mean_abs"], 1e-12
):
    diagnosis.append("final_error_is_strongly_edge_correlated")

result = {
    "experiment": "0054b-1",
    "purpose": "Localize 0054b fidelity failure without changing any preregistered gate.",
    "pass1_metal_vs_pil_letterbox": pass1,
    "manual_pil_second_resize_vs_hf_reference": manual_vs_hf,
    "final_metal_vs_hf": {
        "overall": err_stats(diff),
        "per_channel": per_channel,
        "regions": regions,
        "gradient_buckets_using_max_channel_abs_error": gradient_buckets,
    },
    "diagnosis_flags": diagnosis,
    "notes": [
        "Diagnostic intermediate readback occurs after all measured Metal timing and does not alter the 0054b timing region.",
        "0054b preregistered gates are unchanged.",
        "This diagnostic is causal localization, not a new accuracy gate."
    ],
}
OUT.write_text(json.dumps(result, indent=2))
print(json.dumps(result, indent=2))
print("Saved:", OUT)
