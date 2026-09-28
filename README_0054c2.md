# Experiment 0054c-2 — actual Metal-preprocess tensor through the full model

0054c-0 proved the MLX Metal → PyTorch MPS handoff is zero-copy.
0054c-1 proved that an exact-value bridged tensor is consumed by the adopted
MetalGround full model with exact outputs and no material latency regression.

0054c-2 now composes the **actual numerical output of 0054b-2** with that bridge
and the full model.

Reference:

```text
saved PIL 1200x901 letterbox
→ GroundingDinoImageProcessor
→ adopted MetalGround
```

Candidate:

```text
actual 0054b-2 Pillow-compatible Metal tensor
→ MLX Metal producer
→ explicit producer sync
→ DLPack copy=False
→ PyTorch MPS
→ adopted MetalGround
```

The saved candidate file is used only to isolate numerical propagation. This is
not yet the final live CVPixelBuffer ownership/lifetime bridge.

## Pre-registered gates

The local input gates are reused unchanged from 0054b:

- mean abs <= 0.010
- p99 abs <= 0.050
- max abs <= 0.250

Downstream behavioral gates:

- top-900 membership = 900/900
- default postprocess count equal
- default postprocess label multiset equal
- same-label greedy minimum IoU >= 0.99

Raw logits/boxes and rankwise top-k equality are recorded descriptively rather
than used as gates, because prior MetalGround experiments established that tiny
floating-point perturbations can reorder near-tied proposals without changing
proposal membership or downstream detections.

## Run

```bash
caffeinate -i uv run python scripts/54c2_preprocess_full_model_fidelity.py \
  --output results/metalground_preprocess_full_model_0054c2.json
```

Upload:

`results/metalground_preprocess_full_model_0054c2.json`


## Fix 1 — historical runtime-builder compatibility argument

`37_corrected_wide_island_multiprocess.py::build_runtime()` dereferences
`args.image` while constructing the adopted runtime. The first 0054c-2 harness
omitted that argparse field.

This revision restores the same compatibility image argument used by 0054c-1:

`--image assets/input.jpg`

The 0054c-2 scientific reference remains the saved 1200x901 PIL letterbox
(`results/0054b_reference/letterbox_reference.png`). The compatibility image is
only supplied because the historical runtime builder expects it.

No candidate tensor, correctness gate, preprocessing artifact, bridge rule, or
experiment semantics changed.


## Fix 2 — historical runtime-builder threadgroup argument

The historical `37_corrected_wide_island_multiprocess.py::build_runtime()`
also dereferences `args.threadgroup`. The 0054c-2 harness now restores the same
default used by 0054c-1:

`--threadgroup 256`

This is a runtime-construction compatibility argument only. No scientific
workload, candidate tensor, synchronization rule, or pre-registered gate changed.
