# MetalGround Experiment 0054c-0 — MLX Metal → PyTorch MPS DLPack bridge

This is the first sub-experiment of 0054c.

It does **not** yet run the full Grounding DINO model. Its job is to qualify
the exact output bridge before full-model integration.

The probe:

```text
0054b-2 [3,800,1065] FP32 values
→ MLX custom Metal producer
→ mx.eval + mx.synchronize
→ torch.from_dlpack(..., copy=False)
→ PyTorch MPS tensor
```

It additionally proves storage aliasing by writing a sentinel through the
PyTorch view and reading it through MLX, and checks the reverse
PyTorch-MPS → MLX `copy=False` path.

## Why MLX is used at this boundary

MLX 0.32.2 documents Metal DLPack interoperability with PyTorch MPS.
MLX arrays exported to PyTorch can be imported without a copy on Metal, and
PyTorch 2.12+ ordinary MPS tensors on Apple silicon use shared storage that MLX
can import with `copy=False`.

DLPack itself does not synchronize pending Metal work. The experiment therefore
uses the same producer-ordering rule established in MetalGround Experiment 0036:
producer work is evaluated and synchronized before the cross-runtime handoff.

## Pre-registered gates

Before seeing the result:

- MLX DLPack device type is Metal (`8`)
- PyTorch import is on `mps`
- dtype is FP32
- shape is exactly `[3,800,1065]`
- imported values are exactly equal to the MLX producer values (`max abs = 0`)
- MLX → PyTorch storage alias is observed
- PyTorch → MLX `copy=False` storage alias is observed
- bridge-only p95 <= `1.0 ms`

The bridge-only timing starts **after** producer synchronization, so it measures
the handoff itself and does not hide pending MLX GPU work inside the import.

## Run

This assumes the successful 0054b-2 files still exist in `results/`.

```bash
caffeinate -i uv run python scripts/54c0_dlpack_bridge_probe.py \
  --output results/metalground_dlpack_bridge_0054c0.json
```

Upload:

`results/metalground_dlpack_bridge_0054c0.json`

If this passes, 0054c-1 will feed the bridged tensor into a full-model
fixed-input comparison before the live camera path is connected.


## Fix 1 — MLX custom Metal kernel source

The initial probe referenced `inp_size` inside `mx.fast.metal_kernel`, but that
identifier is not provided to the kernel source by this API. The launch grid is
already exactly the tensor element count, so the redundant bounds check was
removed.

No bridge protocol, synchronization rule, timing region, sample count, alias
test, or pre-registered gate changed.
