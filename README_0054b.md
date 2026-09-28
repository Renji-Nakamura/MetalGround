# MetalGround Experiment 0054b — Metal GPU preprocessing

This experiment isolates preprocessing before live-camera integration.

Pipeline under test:

```text
deterministic 1280x720 BGRA frame
→ Metal pass 1: bilinear resize + 1200x901 letterbox (u8 intermediate)
→ Metal pass 2: bilinear resize to 1065x800 + ImageNet-style normalization
→ float32 CHW [3,800,1065]
```

The CPU reference uses the current PIL letterbox plus the actual Hugging Face
`GroundingDinoImageProcessor` loaded from `IDEA-Research/grounding-dino-tiny`.

## Pre-registered gates

Before seeing 0054b results:

- shape exact: `[3,800,1065]`
- normalized tensor mean absolute error <= `0.010`
- normalized tensor p99 absolute error <= `0.050`
- normalized tensor max absolute error <= `0.250`
- Metal preprocessing p95 <= `3.0 ms`

The numerical gates are intentionally much stricter than downstream detector
equivalence, but allow small interpolation/backend differences.

If fidelity fails, diagnose resize-coordinate / filtering semantics before any
one-pass fusion. Do not move the gates after observing results.

## Run

Copy/overwrite these paths into the repository and run:

```bash
chmod +x scripts/54b_metal_preprocess_probe.sh

caffeinate -i bash scripts/54b_metal_preprocess_probe.sh
```

Expected outputs:

- `results/0054b_reference/metadata.json`
- `results/metalground_metal_preprocess_0054b.json`
- `results/metalground_preprocess_compare_0054b.json`

Upload `results/metalground_preprocess_compare_0054b.json` after completion.
If the script errors, paste the terminal error instead.


## Fix 1 — Transformers SizeDict JSON serialization

Transformers 5.17 exposes `image_processor.size` as a `SizeDict`, which is not
directly JSON serializable. The reference script now converts mapping-like
`size` objects to a plain Python `dict` before writing `metadata.json`.

No experiment protocol, timing, numerical gate, or workload was changed.
The failed run stopped while writing metadata before Metal compilation or any
0054b candidate measurement.


## Fix 2 — recursive JSON-safe metadata serialization

The first SizeDict-specific fix was insufficient because a `SizeDict`-like
object remained nested inside processor metadata. This revision serializes the
entire metadata tree through a recursive JSON-safe converter.

Only metadata representation changes. The generated test image, PIL/HF
reference tensor, timing workload, Metal candidate, and all pre-registered
gates remain unchanged.


## Fix 3 — no full Xcode requirement

The standalone Command Line Tools installation does not necessarily provide the
external `metal` / `metallib` executables. The experiment now compiles the same
MSL source at runtime with `MTLDevice.makeLibrary(source:options:)`.

This changes only shader-library creation. The shader source, preprocessing
workload, timing region, warmups, samples, numerical gates, and performance gate
are unchanged.

The CPU/HF reference stage had completed in the failed run (median 8.6067705 ms),
but the Metal candidate never ran, so no 0054b scientific decision was made.
