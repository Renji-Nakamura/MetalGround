# Experiment 0054c-1 — full-model consumption of the zero-copy bridge

0054c-0 established the MLX Metal ↔ PyTorch MPS DLPack boundary:

- `copy=False`
- device Metal/MPS
- exact values
- bidirectional storage aliasing
- bridge-only p95 below 1 microsecond

0054c-1 now isolates the next question:

> Can the adopted MetalGround full model consume that bridged storage with
> unchanged behavior and without a material integration penalty?

To keep causality clean, this experiment deliberately feeds the candidate
bridge **exactly the same FP32 pixel values** as the HF reference path. Thus a
failure cannot be blamed on resize/interpolation differences already studied
in 0054b.

The adopted runtime includes the robust-GO components already used in the
webcam system phase: corrected wide encoder islands, prompt cache, decoder
Metal MSDA, compiled MLX folded fusion/deformable path, and stage-0 MLX MLP.

## Pre-registered gates

Before seeing the result:

- bridged tensor is MPS FP32 `[1,3,800,1065]`
- bridged pixel values max abs = 0
- raw logits allclose `1e-5`
- raw boxes allclose `1e-5`
- top-900 membership = 900/900
- top-900 rankwise equality = 900/900
- default postprocess count equal
- default postprocess label multiset equal
- same-label greedy minimum IoU >= 0.9999
- paired candidate-minus-reference median regression <= 10 ms

Performance protocol:

- same process
- 3 warmups per mode
- 20 alternating A/B pairs
- reference timing = model only
- candidate timing = `torch.from_dlpack(..., copy=False)` + model
- MLX producer work is already evaluated/synchronized and excluded, because
  0054c-1 measures boundary consumption rather than preprocessing execution

## Run

This assumes the historical helper scripts and `assets/input.jpg` remain in the
MetalGround repository.

```bash
caffeinate -i uv run python scripts/54c1_full_model_bridge.py \
  --output results/metalground_full_model_bridge_0054c1.json
```

Upload:

`results/metalground_full_model_bridge_0054c1.json`

If 0054c-1 passes, the next experiment will compose the qualified preprocessing
producer and this bridge rather than testing the boundary in isolation.


## Fix 1 — topk_audit schema

The first run completed all 20 paired measurements but failed while constructing
the final gate dictionary because the harness expected non-existent keys
`membership_overlap` and `rankwise_equal`.

The historical `14_full_model_fusion_algebraic.py` helper actually returns:
- `set_overlap`
- `rankwise_identical`

This revision uses those authoritative key names and adds a schema guard.

No model execution, input values, warmups, pair count, timing region,
correctness threshold, or pre-registered gate changed.
