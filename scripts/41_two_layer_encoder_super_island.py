#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import inspect
import json
import math
import statistics
import sys
import time
import types
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import mlx.core as mx
import torch
import transformers


def load_module(path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load helper script: {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    try:
        spec.loader.exec_module(mod)
    except Exception:
        sys.modules.pop(module_name, None)
        raise
    return mod


def percentile(xs, p):
    ys = sorted(xs)
    if len(ys) == 1:
        return ys[0]
    k = (len(ys) - 1) * p
    f, c = math.floor(k), math.ceil(k)
    if f == c:
        return ys[f]
    return ys[f] * (c - k) + ys[c] * (k - f)


def stats(xs):
    return {
        "n": len(xs),
        "mean_ms": statistics.fmean(xs),
        "median_ms": statistics.median(xs),
        "p90_ms": percentile(xs, 0.90),
        "p95_ms": percentile(xs, 0.95),
        "min_ms": min(xs),
        "max_ms": max(xs),
    }


def compare(a: torch.Tensor, b: torch.Tensor):
    af = a.detach().float()
    bf = b.detach().float()
    d = (af - bf).abs()
    return {
        "shape": list(a.shape),
        "max_abs": float(d.max().item()),
        "mean_abs": float(d.mean().item()),
        "rmse": float(torch.sqrt(torch.mean((af - bf) ** 2)).item()),
        "allclose_1e-5": bool(torch.allclose(a, b, rtol=1e-5, atol=1e-5)),
        "allclose_1e-4": bool(torch.allclose(a, b, rtol=1e-4, atol=1e-4)),
        "finite": bool(torch.isfinite(a).all().item() and torch.isfinite(b).all().item()),
    }


def linear(x, weight, bias):
    y = x @ weight.T
    if bias is not None:
        y = y + bias
    return y


def activation(x, name: str):
    name = name.lower()
    if name == "relu":
        return mx.maximum(x, 0.0)
    if name in ("silu", "swish"):
        return x * mx.sigmoid(x)
    if name == "gelu":
        return 0.5 * x * (1.0 + mx.erf(x / math.sqrt(2.0)))
    if name in ("gelu_new", "gelu_fast"):
        c = math.sqrt(2.0 / math.pi)
        return 0.5 * x * (1.0 + mx.tanh(c * (x + 0.044715 * x * x * x)))
    raise ValueError(f"Unsupported activation: {name}")


class MlxTextEnhancer:
    """
    Inference-exact MLX port of GroundingDinoTextEnhancerLayer.

    Dropout is omitted only because model.eval() makes it an identity.
    The attention mask supplied by the encoder is an allowed-attention mask:
    True entries are allowed, False entries receive -inf.
    """

    def __init__(self, layer, activation_name: str):
        if layer.training:
            raise ValueError("Text enhancer must be in eval mode.")

        self.layer = layer
        self.activation_name = activation_name
        self.num_heads = int(layer.num_heads)

        attn = layer.self_attn

        # Transformers 5.17 GroundingDinoMultiheadAttention uses
        # query/key/value/out_proj. Older/alternate implementations may use
        # q_proj/k_proj/v_proj, so support both layouts explicitly.
        if all(hasattr(attn, name) for name in ("query", "key", "value", "out_proj")):
            q_module = attn.query
            k_module = attn.key
            v_module = attn.value
            attn_layout = "query_key_value"
        elif all(hasattr(attn, name) for name in ("q_proj", "k_proj", "v_proj", "out_proj")):
            q_module = attn.q_proj
            k_module = attn.k_proj
            v_module = attn.v_proj
            attn_layout = "q_proj_k_proj_v_proj"
        else:
            children = [name for name, _ in attn.named_children()]
            raise RuntimeError(
                "Unexpected GroundingDinoMultiheadAttention layout; "
                f"children={children}"
            )

        # Prefer the attention module's own metadata. In HF Transformers 5.17
        # these are num_attention_heads and attention_head_size.
        self.num_heads = int(
            getattr(attn, "num_attention_heads", getattr(layer, "num_heads"))
        )
        self.head_dim = int(
            getattr(
                attn,
                "attention_head_size",
                q_module.out_features // self.num_heads,
            )
        )
        embed_dim = int(q_module.out_features)

        if embed_dim != self.num_heads * self.head_dim:
            raise RuntimeError(
                f"Unexpected attention geometry: embed_dim={embed_dim}, "
                f"heads={self.num_heads}, head_dim={self.head_dim}"
            )

        # HF GroundingDinoMultiheadAttention divides the QK score by
        # sqrt(attention_head_size). Scaling q before matmul is algebraically
        # equivalent and avoids an extra full score-tensor operation.
        self.scale = self.head_dim ** -0.5
        self.attn_layout = attn_layout

        self.p = {}

        def import_linear(prefix, module):
            self.p[prefix + "_w"] = mx.asarray(module.weight, copy=False)
            self.p[prefix + "_b"] = (
                None if module.bias is None else mx.asarray(module.bias, copy=False)
            )

        import_linear("q", q_module)
        import_linear("k", k_module)
        import_linear("v", v_module)
        import_linear("out", attn.out_proj)
        import_linear("fc1", layer.fc1)
        import_linear("fc2", layer.fc2)

        self.p["ln1_w"] = mx.asarray(layer.layer_norm_before.weight, copy=False)
        self.p["ln1_b"] = mx.asarray(layer.layer_norm_before.bias, copy=False)
        self.p["ln2_w"] = mx.asarray(layer.layer_norm_after.weight, copy=False)
        self.p["ln2_b"] = mx.asarray(layer.layer_norm_after.bias, copy=False)
        self.ln1_eps = float(layer.layer_norm_before.eps)
        self.ln2_eps = float(layer.layer_norm_after.eps)

        arrays = [v for v in self.p.values() if v is not None]
        mx.eval(*arrays)
        mx.synchronize()

        self.compiled = mx.compile(self._forward_impl)

    def _forward_impl(self, hidden, allowed_mask, position):
        qk_input = hidden if position is None else hidden + position

        q = linear(qk_input, self.p["q_w"], self.p["q_b"]) * self.scale
        k = linear(qk_input, self.p["k_w"], self.p["k_b"])
        v = linear(hidden, self.p["v_w"], self.p["v_b"])

        B, T, D = q.shape

        q = q.reshape(B, T, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = k.reshape(B, T, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = v.reshape(B, T, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)

        scores = q @ k.transpose(0, 1, 3, 2)

        if allowed_mask is not None:
            mask = allowed_mask
            if mask.ndim == 3:
                mask = mask[:, None, :, :]
            # Broadcast [B,1,T,T] over heads.
            scores = mx.where(mask, scores, mx.array(float("-inf"), dtype=scores.dtype))

        probs = mx.softmax(scores, axis=-1)
        out = probs @ v
        out = out.transpose(0, 2, 1, 3).reshape(B, T, D)
        out = linear(out, self.p["out_w"], self.p["out_b"])

        x = hidden + out
        x = mx.fast.layer_norm(
            x, self.p["ln1_w"], self.p["ln1_b"], self.ln1_eps
        )

        residual = x
        x = linear(x, self.p["fc1_w"], self.p["fc1_b"])
        x = activation(x, self.activation_name)
        x = linear(x, self.p["fc2_w"], self.p["fc2_b"])
        x = residual + x
        x = mx.fast.layer_norm(
            x, self.p["ln2_w"], self.p["ln2_b"], self.ln2_eps
        )
        return x, probs

    def compiled_call(self, hidden, allowed_mask, position):
        return self.compiled(hidden, allowed_mask, position)


def detach_value(x):
    if isinstance(x, torch.Tensor):
        return x.detach()
    if isinstance(x, list):
        return list(x)
    if isinstance(x, tuple):
        return tuple(x)
    return x


def capture_bound_call(bound_forward, record: dict[str, Any]):
    sig = inspect.signature(bound_forward)

    def wrapper(*args, **kwargs):
        if not record:
            bound = sig.bind_partial(*args, **kwargs)
            bound.apply_defaults()
            for k, v in bound.arguments.items():
                record[k] = detach_value(v)
        return bound_forward(*args, **kwargs)

    return wrapper


def resolved_text_position(layer, record):
    return layer.get_text_position_embeddings(
        record["text_features"],
        record.get("text_position_embedding"),
        record.get("text_position_ids"),
    )


def current_two_layer_segment(layer0, layer1, rec0, rec1, sync_fn):
    kw0 = dict(rec0)
    kw1 = dict(rec1)

    with torch.inference_mode():
        out0 = layer0(**kw0)
        v0, t0 = out0[0]

        kw1["vision_features"] = v0
        kw1["text_features"] = t0
        out1 = layer1(**kw1)
        sync_fn()

    return out1[0]


def to_mlx_or_none(x):
    return None if x is None else mx.asarray(x, copy=False)


def super_two_layer_segment(
    wide0,
    wide1,
    txt0,
    txt1,
    rec0,
    pos0,
    pos1,
    sync_fn,
):
    # True producer boundary: no new MPS tensor creation after this sync.
    sync_fn()

    v = mx.asarray(rec0["vision_features"], copy=False)
    t = mx.asarray(rec0["text_features"], copy=False)
    key_padding = mx.asarray(rec0["key_padding_mask"], copy=False)
    text_padding = to_mlx_or_none(rec0.get("text_attention_mask"))
    vision_pos = mx.asarray(rec0["vision_position_embedding"], copy=False)
    refs = mx.asarray(rec0["reference_points"], copy=False)

    text_self_mask = rec0.get("text_self_attention_masks")
    if text_self_mask is None:
        allowed_text = None
    else:
        text_self_mask_mx = mx.asarray(text_self_mask, copy=False)
        # HF encoder calls text_enhancer(attention_masks=~text_self_attention_masks)
        allowed_text = mx.logical_not(text_self_mask_mx)

    pos0_mx = to_mlx_or_none(pos0)
    pos1_mx = to_mlx_or_none(pos1)

    # Layer 0: fusion+deformable, then text enhancer.
    v, t = wide0.compiled_min(
        v,
        t,
        key_padding,
        text_padding,
        vision_pos,
        refs,
    )
    t, _attn0 = txt0.compiled_call(t, allowed_text, pos0_mx)

    # Layer 1 consumes both layer-0 outputs, entirely inside MLX.
    v, t = wide1.compiled_min(
        v,
        t,
        key_padding,
        text_padding,
        vision_pos,
        refs,
    )
    t, _attn1 = txt1.compiled_call(t, allowed_text, pos1_mx)

    mx.eval(v, t)
    mx.synchronize()

    v_t = torch.as_tensor(v)
    t_t = torch.as_tensor(t)
    sync_fn()
    return v_t, t_t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="IDEA-Research/grounding-dino-tiny")
    ap.add_argument("--image", type=Path, default=Path("assets/input.jpg"))
    ap.add_argument("--prompt", nargs="+", default=["a cat", "a dog"])
    ap.add_argument("--threadgroup", type=int, default=256)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--iters", type=int, default=20)

    ap.add_argument(
        "--exp13-helper",
        type=Path,
        default=Path("scripts/13_bench_fusion_algebraic_specialization.py"),
    )
    ap.add_argument(
        "--exp14-helper",
        type=Path,
        default=Path("scripts/14_full_model_fusion_algebraic.py"),
    )
    ap.add_argument(
        "--exp17-helper",
        type=Path,
        default=Path("scripts/17_bench_deformable_mlx_island.py"),
    )
    ap.add_argument(
        "--exp18-helper",
        type=Path,
        default=Path("scripts/18_full_model_deformable_islands.py"),
    )
    ap.add_argument(
        "--exp30-helper",
        type=Path,
        default=Path("scripts/30_bench_fully_folded_fusion_mlx.py"),
    )
    ap.add_argument(
        "--exp31-helper",
        type=Path,
        default=Path("scripts/31_full_model_mlx_fusion_paired.py"),
    )
    ap.add_argument(
        "--exp36-helper",
        type=Path,
        default=Path("scripts/36_wide_island_mask_sync_fix.py"),
    )
    ap.add_argument(
        "--exp37-helper",
        type=Path,
        default=Path("scripts/37_corrected_wide_island_multiprocess.py"),
    )
    args = ap.parse_args()

    for p in (
        args.exp13_helper,
        args.exp14_helper,
        args.exp17_helper,
        args.exp18_helper,
        args.exp30_helper,
        args.exp31_helper,
        args.exp36_helper,
        args.exp37_helper,
    ):
        if not p.exists():
            raise SystemExit(f"Missing helper: {p}")

    if not torch.backends.mps.is_available():
        raise SystemExit("MPS unavailable.")

    h37 = load_module(args.exp37_helper, "mg41_h37")

    print("Building corrected-wide runtime...", flush=True)
    rt = h37.build_runtime(args)
    model = rt["model"]
    h14 = rt["h14"]
    h36 = rt["h36"]

    # Current adopted wide mode, but prompt cache is irrelevant for this
    # isolated encoder-prefix benchmark.
    h36.set_mode(rt["layer_dispatchers"], "fixed")

    layers = model.model.encoder.layers
    layer0 = layers[0]
    layer1 = layers[1]

    rec0 = {}
    rec1 = {}

    orig0 = layer0.forward
    orig1 = layer1.forward

    layer0.forward = types.MethodType(
        lambda _self, *a, **kw: capture_wrapped0(*a, **kw),
        layer0,
    )
    layer1.forward = types.MethodType(
        lambda _self, *a, **kw: capture_wrapped1(*a, **kw),
        layer1,
    )

    # Build capture wrappers after saving bound forwards.
    capture_wrapped0 = capture_bound_call(orig0, rec0)
    capture_wrapped1 = capture_bound_call(orig1, rec1)

    try:
        print("Capturing real final-runtime layer0/layer1 inputs...", flush=True)
        with torch.inference_mode():
            _ = model(**rt["inputs"])
            h14.sync()
    finally:
        layer0.forward = orig0
        layer1.forward = orig1

    if not rec0 or not rec1:
        raise RuntimeError("Failed to capture encoder layer inputs.")

    print(
        f"vision={tuple(rec0['vision_features'].shape)} "
        f"text={tuple(rec0['text_features'].shape)}",
        flush=True,
    )

    # Resolve text position embeddings exactly using the installed HF layer.
    pos0 = resolved_text_position(layer0, rec0)
    pos1 = resolved_text_position(layer1, rec1)
    h14.sync()

    # Sync before zero-copy parameter import.
    h14.sync()

    txt0 = MlxTextEnhancer(
        layer0.text_enhancer_layer,
        activation_name=model.config.activation_function,
    )
    txt1 = MlxTextEnhancer(
        layer1.text_enhancer_layer,
        activation_name=model.config.activation_function,
    )

    wide0 = rt["layer_dispatchers"][0].fixed_island
    wide1 = rt["layer_dispatchers"][1].fixed_island

    # ------------------------------------------------------------------
    # Text-enhancer correctness on the exact captured input to layer0's
    # text enhancer: obtain fused text from the wide core and compare.
    # ------------------------------------------------------------------
    print("\nText-enhancer MLX correctness preflight...", flush=True)

    h14.sync()
    v_mx = mx.asarray(rec0["vision_features"], copy=False)
    t_mx = mx.asarray(rec0["text_features"], copy=False)
    kp_mx = mx.asarray(rec0["key_padding_mask"], copy=False)
    tp_mx = to_mlx_or_none(rec0.get("text_attention_mask"))
    vp_mx = mx.asarray(rec0["vision_position_embedding"], copy=False)
    ref_mx = mx.asarray(rec0["reference_points"], copy=False)

    v0_fused_mx, t0_fused_mx = wide0.compiled_min(
        v_mx, t_mx, kp_mx, tp_mx, vp_mx, ref_mx
    )
    mx.eval(t0_fused_mx)
    mx.synchronize()
    t0_fused_torch = torch.as_tensor(t0_fused_mx)
    h14.sync()

    allowed_torch = (
        None
        if rec0.get("text_self_attention_masks") is None
        else ~rec0["text_self_attention_masks"]
    )
    with torch.inference_mode():
        ref_text0, ref_attn0 = layer0.text_enhancer_layer(
            hidden_states=t0_fused_torch,
            attention_masks=allowed_torch,
            position_embeddings=pos0,
        )
        h14.sync()

    allowed_mx = (
        None
        if rec0.get("text_self_attention_masks") is None
        else mx.logical_not(mx.asarray(rec0["text_self_attention_masks"], copy=False))
    )
    pos0_mx = to_mlx_or_none(pos0)
    cand_text0_mx, cand_attn0_mx = txt0.compiled_call(
        t0_fused_mx, allowed_mx, pos0_mx
    )
    mx.eval(cand_text0_mx, cand_attn0_mx)
    mx.synchronize()
    cand_text0 = torch.as_tensor(cand_text0_mx)
    cand_attn0 = torch.as_tensor(cand_attn0_mx)
    h14.sync()

    text_corr = {
        "hidden_states": compare(cand_text0, ref_text0),
        "attention_weights": compare(cand_attn0, ref_attn0),
    }
    print(json.dumps(text_corr, indent=2), flush=True)

    # ------------------------------------------------------------------
    # Two-layer segment correctness.
    # ------------------------------------------------------------------
    print("\nTwo-layer super-island correctness preflight...", flush=True)

    h36.set_mode(rt["layer_dispatchers"], "fixed")

    ref_v, ref_t = current_two_layer_segment(
        layer0, layer1, rec0, rec1, h14.sync
    )

    cand_v, cand_t = super_two_layer_segment(
        wide0, wide1, txt0, txt1, rec0, pos0, pos1, h14.sync
    )

    segment_corr = {
        "vision_output": compare(cand_v, ref_v),
        "text_output": compare(cand_t, ref_t),
    }
    print(json.dumps(segment_corr, indent=2), flush=True)

    correctness_pass = (
        segment_corr["vision_output"]["allclose_1e-4"]
        and segment_corr["text_output"]["allclose_1e-4"]
        and text_corr["hidden_states"]["allclose_1e-4"]
    )

    latency = None
    derived = None

    if correctness_pass:
        print("\nCorrectness passed; benchmarking two-layer paths...", flush=True)

        # Exclude first compile/execution.
        _ = super_two_layer_segment(
            wide0, wide1, txt0, txt1, rec0, pos0, pos1, h14.sync
        )

        for i in range(args.warmup):
            _ = current_two_layer_segment(
                layer0, layer1, rec0, rec1, h14.sync
            )
            _ = super_two_layer_segment(
                wide0, wide1, txt0, txt1, rec0, pos0, pos1, h14.sync
            )
            print(f"  warmup {i+1}/{args.warmup}", flush=True)

        current_samples = []
        super_samples = []
        deltas = []

        for i in range(args.iters):
            order = ("current", "super") if i % 2 == 0 else ("super", "current")
            local = {}

            for mode in order:
                if mode == "current":
                    h14.sync()
                    t0_ns = time.perf_counter_ns()
                    _ = current_two_layer_segment(
                        layer0, layer1, rec0, rec1, h14.sync
                    )
                    h14.sync()
                    dt = (time.perf_counter_ns() - t0_ns) / 1e6
                    current_samples.append(dt)
                    local["current"] = dt
                else:
                    h14.sync()
                    t0_ns = time.perf_counter_ns()
                    _ = super_two_layer_segment(
                        wide0, wide1, txt0, txt1, rec0, pos0, pos1, h14.sync
                    )
                    h14.sync()
                    dt = (time.perf_counter_ns() - t0_ns) / 1e6
                    super_samples.append(dt)
                    local["super"] = dt

            delta = local["current"] - local["super"]
            deltas.append(delta)
            print(
                f"  pair {i+1:02d}/{args.iters}: "
                f"current={local['current']:.3f} ms "
                f"super={local['super']:.3f} ms "
                f"delta={delta:+.3f} ms",
                flush=True,
            )

        c = stats(current_samples)
        s = stats(super_samples)
        d = stats(deltas)

        latency = {
            "current_two_layers": c,
            "mlx_super_island_two_layers": s,
            "paired_delta_current_minus_super": d,
        }
        derived = {
            "speedup_from_medians": c["median_ms"] / s["median_ms"],
            "median_difference_of_marginals_ms": c["median_ms"] - s["median_ms"],
            "paired_delta_median_ms": d["median_ms"],
            "paired_delta_mean_ms": d["mean_ms"],
        }
    else:
        print(
            "\nCorrectness failed; timing intentionally skipped.",
            flush=True,
        )

    result = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "experiment": "0041",
        "purpose": (
            "Feasibility test for a multi-layer encoder MLX super-island: "
            "port GroundingDino text enhancement to MLX and execute encoder "
            "layers 0-1 continuously without the intermediate "
            "MLX->PyTorch->MLX boundary."
        ),
        "model": args.model,
        "device": "mps",
        "dtype": "float32",
        "software": {
            "torch": torch.__version__,
            "transformers": transformers.__version__,
        },
        "configuration": {
            "layers": [0, 1],
            "current_path": (
                "wide layer0 -> PyTorch text enhancer -> return -> "
                "wide layer1 -> PyTorch text enhancer"
            ),
            "candidate_path": (
                "one Torch->MLX import -> wide0 -> MLX text enhancer0 -> "
                "wide1 -> MLX text enhancer1 -> one MLX->Torch return"
            ),
            "text_length": int(rec0["text_features"].shape[1]),
            "text_attention_layout": txt0.attn_layout,
            "text_attention_heads": txt0.num_heads,
            "text_attention_head_dim": txt0.head_dim,
            "prompt_cache": "irrelevant to isolated encoder-prefix segment",
            "approximation": False,
            "retraining": False,
            "reduced_precision": False,
        },
        "text_enhancer_correctness": text_corr,
        "two_layer_correctness": segment_corr,
        "correctness_gate_passed": correctness_pass,
        "latency": latency,
        "derived": derived,
        "decision_rule": (
            "Proceed to a six-layer encoder super-island if the two-layer "
            "candidate preserves vision/text outputs within the established "
            "FP32 1e-4 envelope and shows a materially positive paired saving "
            "(target >=2 ms over two layers). If correctness fails, fix the "
            "text-enhancer port before any full-model integration. If "
            "correctness passes but latency is flat/regressive, keep the "
            "current per-layer wide islands."
        ),
        "notes": [
            "The candidate keeps both vision and text branches in MLX between layers 0 and 1.",
            "Dropout is mathematically omitted because model.eval() makes it an identity.",
            "The text self-attention mask is inverted inside MLX from the captured encoder mask semantics.",
            "No approximation, pruning, quantization, retraining, or reduced precision is used.",
            "This is an isolated encoder-prefix feasibility benchmark, not a full-model authority."
        ],
    }

    out = Path("results/metalground_two_layer_encoder_super_island.json")
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")

    print("\n=== Experiment 0041 summary ===", flush=True)
    print(f"correctness gate: {correctness_pass}", flush=True)
    if derived is not None:
        print(
            f"current two-layer median: "
            f"{latency['current_two_layers']['median_ms']:.3f} ms",
            flush=True,
        )
        print(
            f"super-island median:      "
            f"{latency['mlx_super_island_two_layers']['median_ms']:.3f} ms",
            flush=True,
        )
        print(
            f"paired saving median:     "
            f"{derived['paired_delta_median_ms']:+.3f} ms",
            flush=True,
        )
    print(f"Saved: {out}", flush=True)

    for d in rt["layer_dispatchers"]:
        d.restore()
    for fd in rt["fusion_dispatchers"]:
        fd.restore()


if __name__ == "__main__":
    main()
