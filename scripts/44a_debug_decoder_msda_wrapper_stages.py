#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import sys
import time
import types
from pathlib import Path

import mlx.core as mx
import torch

from metalground.msda_metal_v0 import msda_metal_v0


def load_module(path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load helper: {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod


def stamp(msg: str):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def eval_one(name: str, x):
    stamp(f"{name}: mx.eval start")
    t0 = time.perf_counter()
    mx.eval(x)
    mx.synchronize()
    stamp(
        f"{name}: PASS in {(time.perf_counter()-t0)*1000:.3f} ms "
        f"shape={tuple(x.shape)} dtype={x.dtype}"
    )
    return x


def linear(x, w, b):
    y = x @ w.T
    return y if b is None else y + b


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="IDEA-Research/grounding-dino-tiny")
    ap.add_argument("--image", type=Path, default=Path("assets/input.jpg"))
    ap.add_argument("--prompt", nargs="+", default=["a cat", "a dog"])
    ap.add_argument("--threadgroup", type=int, default=256)

    for n in (13, 14, 17, 18, 30, 31, 36, 37, 38, 44):
        ap.add_argument(
            f"--exp{n}-helper",
            type=Path,
            default=Path(f"scripts/{n:02d}_" + {
                13: "bench_fusion_algebraic_specialization.py",
                14: "full_model_fusion_algebraic.py",
                17: "bench_deformable_mlx_island.py",
                18: "full_model_deformable_islands.py",
                30: "bench_fully_folded_fusion_mlx.py",
                31: "full_model_mlx_fusion_paired.py",
                36: "wide_island_mask_sync_fix.py",
                37: "corrected_wide_island_multiprocess.py",
                38: "consolidated_runtime_ablation.py",
                44: "bench_decoder_msda_mlx_wrapper.py",
            }[n])
        )
    args = ap.parse_args()

    if not torch.backends.mps.is_available():
        raise SystemExit("MPS unavailable.")

    h37 = load_module(args.exp37_helper, "mg44a_h37")
    h38 = load_module(args.exp38_helper, "mg44a_h38")
    h44 = load_module(args.exp44_helper, "mg44a_h44")

    stamp("building consolidated runtime")
    rt = h37.build_runtime(args)
    model = rt["model"]
    h14 = rt["h14"]
    inputs = rt["inputs"]

    cache = h38.ExactTextBackboneCache(model.model.text_backbone)

    try:
        h38.set_runtime_mode(rt, cache, "wide_cache")
        cache.prime(model, inputs, h14.sync)
        cache.set_enabled(True)

        target = model.model.decoder.layers[0].encoder_attn
        original_forward = target.forward
        captured = {}

        target.forward = types.MethodType(
            lambda _self, *a, **kw: capture_wrapper(*a, **kw), target
        )
        capture_wrapper = h44.bind_capture(original_forward, captured)

        stamp("capturing decoder layer0 encoder_attn")
        with torch.inference_mode():
            _ = model(**inputs)
            h14.sync()
        target.forward = original_forward

        stamp("capture complete")
        for k in sorted(captured):
            v = captured[k]
            if isinstance(v, torch.Tensor):
                extra = ""
                if v.dtype == torch.bool:
                    h14.sync()
                    extra = f" true={int(v.sum().item())}/{v.numel()}"
                print(
                    f"  {k}: shape={tuple(v.shape)} dtype={v.dtype} "
                    f"device={v.device}{extra}",
                    flush=True,
                )
            else:
                print(f"  {k}: {type(v).__name__}={v}", flush=True)

        shapes = h44.spatial_list_from_capture(captured)
        h14.sync()

        stamp("constructing candidate (mx.compile object creation only)")
        cand = h44.MlxDecoderMsdaWrapper(
            target,
            spatial_shapes_list=shapes,
            threadgroup=args.threadgroup,
        )
        stamp("candidate constructed")

        stamp("synchronizing PyTorch producer before zero-copy import")
        h14.sync()

        hidden = mx.asarray(captured["hidden_states"], copy=False)
        enc = mx.asarray(captured["encoder_hidden_states"], copy=False)
        ref = mx.asarray(captured["reference_points"], copy=False)
        pos = (
            None if captured.get("position_embeddings") is None
            else mx.asarray(captured["position_embeddings"], copy=False)
        )
        mask = (
            None if captured.get("attention_mask") is None
            else mx.asarray(captured["attention_mask"], copy=False)
        )
        stamp(
            "zero-copy imports created: "
            f"hidden={tuple(hidden.shape)} enc={tuple(enc.shape)} "
            f"ref={tuple(ref.shape)} mask="
            f"{None if mask is None else tuple(mask.shape)}"
        )

        # Stage 1: query
        query = hidden if pos is None else hidden + pos
        eval_one("1 query add", query)

        B, Q, _ = query.shape
        _, S, _ = enc.shape

        # Stage 2: value projection
        value = linear(enc, cand.p["value_w"], cand.p["value_b"])
        eval_one("2 value projection", value)

        # Stage 3: value mask
        if mask is not None:
            value = mx.where(mask[..., None], value, 0.0)
            eval_one("3 value mask", value)
        else:
            stamp("3 value mask: SKIP (mask=None)")

        value = value.reshape(B, S, cand.H, cand.HD)
        eval_one("4 value reshape", value)

        # Stage 5: offsets
        offsets = linear(
            query, cand.p["sampling_w"], cand.p["sampling_b"]
        ).reshape(B, Q, cand.H, cand.NL, cand.NP, 2)
        eval_one("5 sampling offsets", offsets)

        # Stage 6: weights logits + softmax
        weights = linear(
            query, cand.p["attn_w_w"], cand.p["attn_w_b"]
        ).reshape(B, Q, cand.H, cand.NL * cand.NP)
        eval_one("6a attention logits", weights)
        weights = mx.softmax(weights, axis=-1).reshape(
            B, Q, cand.H, cand.NL, cand.NP
        )
        eval_one("6b attention softmax", weights)

        # Stage 7: locations
        if ref.shape[-1] == 2:
            locations = (
                ref[:, :, None, :, None, :]
                + offsets / cand.offset_normalizer[
                    None, None, None, :, None, :
                ]
            )
        elif ref.shape[-1] == 4:
            locations = (
                ref[:, :, None, :, None, :2]
                + offsets / float(cand.NP)
                * ref[:, :, None, :, None, 2:] * 0.5
            )
        else:
            raise RuntimeError(f"unexpected ref dim {ref.shape[-1]}")
        eval_one("7 sampling locations", locations)

        # Stage 8: exact custom Metal kernel, uncompiled outer graph.
        stamp("8 custom Metal MSDA: CALL start")
        t0 = time.perf_counter()
        core = msda_metal_v0(
            value,
            cand.spatial_shapes,
            cand.level_start,
            locations,
            weights,
            threadgroup_size=args.threadgroup,
        )
        stamp(
            f"8 custom Metal MSDA: call returned lazily in "
            f"{(time.perf_counter()-t0)*1000:.3f} ms; forcing eval"
        )
        eval_one("8 custom Metal MSDA", core)

        # Stage 9: output projection
        out = linear(core, cand.p["output_w"], cand.p["output_b"])
        eval_one("9 output projection", out)

        # Stage 10: bridge
        stamp("10 MLX->Torch bridge start")
        out_t = torch.as_tensor(out)
        h14.sync()
        stamp(
            f"10 MLX->Torch bridge PASS shape={tuple(out_t.shape)} "
            f"dtype={out_t.dtype}"
        )

        # Stage 11: uncompiled full function.
        stamp("11 full UNCOMPILED _forward_impl start")
        t0 = time.perf_counter()
        u_out, u_w = cand._forward_impl(hidden, mask, enc, pos, ref)
        stamp(
            f"11 uncompiled call returned lazily in "
            f"{(time.perf_counter()-t0)*1000:.3f} ms; forcing eval"
        )
        mx.eval(u_out, u_w)
        mx.synchronize()
        stamp("11 full UNCOMPILED _forward_impl PASS")

        print(
            "\nAll uncompiled stages passed.\n"
            "If the original Experiment 0044 hangs but this diagnostic reaches "
            "here, the hang is specifically in the first mx.compile execution "
            "of the composite wrapper rather than in the math or Metal kernel.",
            flush=True,
        )

    finally:
        try:
            target.forward = original_forward
        except Exception:
            pass
        cache.restore()
        for d in rt["layer_dispatchers"]:
            d.restore()
        for fd in rt["fusion_dispatchers"]:
            fd.restore()


if __name__ == "__main__":
    main()
