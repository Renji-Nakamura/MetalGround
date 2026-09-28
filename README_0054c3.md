# Experiment 0054c-3 — foreign shared MTLBuffer → MLX → PyTorch MPS

0054c-0 proved MLX Metal → PyTorch MPS zero-copy DLPack.
0054c-1 proved the adopted full model consumes that storage correctly.
0054c-2 proved the actual 0054b-2 preprocessing values preserve proposal-set
and default detection behavior.

The remaining ownership gap is on the **native producer side**:

> Can a Metal buffer allocated outside MLX be imported by MLX without a copy,
> and then continue through the already-qualified PyTorch MPS boundary?

This probe builds a tiny Objective-C++ Python extension that allocates an
`MTLStorageModeShared` buffer and exposes it as a `kDLMetal` DLPack producer.

Pipeline:

```text
foreign MTLBuffer (Shared)
→ DLPack kDLMetal, copy=False
→ MLX
→ DLPack copy=False
→ PyTorch MPS
```

## Pre-registered gates

- DLPack device `(8,0)` / Metal
- MTLBuffer storage mode is Shared
- MLX and Torch shapes exact `[1,3,800,1065]`
- Torch device is MPS, dtype FP32
- selected initial values exact
- native `MTLBuffer.contents()` write is observed through Torch
- Torch write is observed through native `MTLBuffer.contents()`
- Torch write is also observed through MLX
- foreign → MLX import p95 <= 1 ms
- foreign → MLX → Torch boundary p95 <= 1 ms

A `copy=False` API call by itself is not considered sufficient evidence.
The cross-runtime mutation/alias checks are mandatory.

## Run

```bash
chmod +x scripts/54c3_foreign_mtlbuffer_bridge.sh
bash scripts/54c3_foreign_mtlbuffer_bridge.sh
```

Upload:

`results/metalground_foreign_mtlbuffer_bridge_0054c3.json`

This experiment is still synthetic: the foreign shared buffer is not yet the
live 0054a camera / 0054b preprocessing output. If it passes, the next step is
to make the native preprocessing producer write its FP32 CHW result directly
into this qualified foreign shared-buffer ownership path.
