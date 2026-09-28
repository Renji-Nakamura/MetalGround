# Experiment 0054b-2 — Pillow-compatible Metal resize

0054b-1 localized the numerical failure to resize semantics. This candidate replaces Metal hardware linear sampling with a separable, scale-aware bilinear resampler modeled on Pillow `Resample.c`, including 22-bit fixed-point coefficient rounding and uint8 rounding after each resize.

The original 0054b gates are reused unchanged.

Run:
```bash
chmod +x scripts/54b2_pillow_compatible_probe.sh
caffeinate -i bash scripts/54b2_pillow_compatible_probe.sh
```

Upload `results/metalground_preprocess_compare_0054b2.json`.
