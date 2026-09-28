#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import statistics
import time
import types
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import transformers
from PIL import Image
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor


def sync() -> None:
    torch.mps.synchronize()


def percentile(xs: list[float], p: float) -> float:
    ys = sorted(xs)
    if len(ys) == 1:
        return ys[0]
    k = (len(ys) - 1) * p
    f, c = math.floor(k), math.ceil(k)
    if f == c:
        return ys[f]
    return ys[f] * (c - k) + ys[c] * (k - f)


def stats(xs: list[float]) -> dict[str, float]:
    return {
        "n": len(xs),
        "mean_ms": statistics.fmean(xs),
        "median_ms": statistics.median(xs),
        "p90_ms": percentile(xs, 0.90),
        "p95_ms": percentile(xs, 0.95),
        "min_ms": min(xs),
        "max_ms": max(xs),
    }


def compare(a: torch.Tensor, b: torch.Tensor) -> dict[str, Any]:
    a = a.detach().float()
    b = b.detach().float()
    d = (a - b).abs()
    return {
        "shape": list(a.shape),
        "max_abs": float(d.max().item()),
        "mean_abs": float(d.mean().item()),
        "rmse": float(torch.sqrt(torch.mean((a - b) ** 2)).item()),
        "allclose_1e-5": bool(torch.allclose(a, b, rtol=1e-5, atol=1e-5)),
        "allclose_1e-4": bool(torch.allclose(a, b, rtol=1e-4, atol=1e-4)),
        "finite": bool(torch.isfinite(a).all().item()),
    }


def flatten_output(out):
    (vision_out, vision_attn), (text_out, text_attn) = out
    return vision_out, vision_attn, text_out, text_attn


def compare_output(actual, expected):
    names = (
        "vision_output",
        "vision_attention_weights",
        "text_output",
        "text_attention_weights",
    )
    aa = flatten_output(actual)
    bb = flatten_output(expected)
    return {n: compare(a, b) for n, a, b in zip(names, aa, bb)}


class AlgebraicBiMHA:
    """
    Batch-1 inference specialization of GroundingDinoBiMultiHeadAttention.

    Two variants:
      value_folded:
        Preserve the exact reference q/k projection and score path, while
        algebraically folding value projections through the output projection.

      fully_folded:
        Additionally replace the large vision query projection with an
        equivalent head-wise bilinear score form.

    No learned parameter is changed and no approximation/reduced precision is used.
    Floating-point association changes, so this is not expected to be bitwise exact.
    """

    def __init__(self, module):
        self.module = module
        self.H = int(module.num_heads)
        self.D = int(module.head_dim)
        self.E = int(module.embed_dim)
        self.d = int(module.vision_dim)
        self.scale = float(module.scale)

        if self.d != int(module.text_dim):
            raise ValueError("This specialization assumes equal vision/text input dims.")
        if module.dropout != 0.0:
            raise ValueError("This v0 specialization requires fusion dropout == 0.")
        if self.E != self.H * self.D:
            raise ValueError("Invalid head partition.")

        self._build_folded_weights()

    def _blocks_out(self, linear):
        # Linear weight is [out_dim, H*D]. Split input columns by head.
        return torch.stack(
            [
                linear.weight[:, h * self.D : (h + 1) * self.D]
                for h in range(self.H)
            ],
            dim=0,
        )  # [H, d, D]

    def _blocks_in(self, linear):
        # Linear weight is [H*D, in_dim]. Split output rows by head.
        return torch.stack(
            [
                linear.weight[h * self.D : (h + 1) * self.D, :]
                for h in range(self.H)
            ],
            dim=0,
        )  # [H, D, d]

    def _bias_blocks(self, linear):
        return torch.stack(
            [
                linear.bias[h * self.D : (h + 1) * self.D]
                for h in range(self.H)
            ],
            dim=0,
        )  # [H,D]

    def _build_folded_weights(self):
        m = self.module

        Wq = self._blocks_in(m.vision_proj)       # [H,D,d]
        bq = self._bias_blocks(m.vision_proj)     # [H,D]
        Wk = self._blocks_in(m.text_proj)         # [H,D,d]
        bk = self._bias_blocks(m.text_proj)       # [H,D]

        Wvv = self._blocks_in(m.values_vision_proj)  # [H,D,d]
        bvv = self._bias_blocks(m.values_vision_proj)
        Wvt = self._blocks_in(m.values_text_proj)    # [H,D,d]
        bvt = self._bias_blocks(m.values_text_proj)

        Wov = self._blocks_out(m.out_vision_proj)  # [H,d,D]
        Wot = self._blocks_out(m.out_text_proj)    # [H,d,D]

        # Score:
        # (v Wq^T + bq)(t Wk^T + bk)^T
        #
        # M[h]  = Wq[h]^T Wk[h]                         [d,d]
        # cv[h] = Wq[h]^T bk[h]                         [d]
        # ct[h] = bq[h] Wk[h]                           [d]
        # c0[h] = bq[h] dot bk[h]                       []
        self.score_M = torch.bmm(Wq.transpose(1, 2), Wk).contiguous()
        self.score_cv = torch.bmm(
            Wq.transpose(1, 2), bk.unsqueeze(-1)
        ).squeeze(-1).contiguous()
        self.score_ct = torch.bmm(
            bq.unsqueeze(1), Wk
        ).squeeze(1).contiguous()
        self.score_c0 = (bq * bk).sum(-1).contiguous()

        # Vision output path:
        # [A_v @ (text Wvt^T + bvt)] @ Wov^T
        # = A_v @ (text C_v + c_v)
        self.C_v = torch.bmm(Wvt.transpose(1, 2), Wov.transpose(1, 2)).contiguous()
        self.c_v = torch.bmm(
            bvt.unsqueeze(1), Wov.transpose(1, 2)
        ).squeeze(1).contiguous()

        # Text output path:
        # [A_t @ (vision Wvv^T + bvv)] @ Wot^T
        # = (A_t @ vision) C_t + rowsum(A_t) c_t
        self.C_t = torch.bmm(Wvv.transpose(1, 2), Wot.transpose(1, 2)).contiguous()
        self.c_t = torch.bmm(
            bvv.unsqueeze(1), Wot.transpose(1, 2)
        ).squeeze(1).contiguous()

        self.out_vision_bias = m.out_vision_proj.bias
        self.out_text_bias = m.out_text_proj.bias

        sync()

    def _reshape_ref(self, x, seq_len, batch_size):
        # Same logical layout as HF _reshape + flatten batch/head.
        return (
            x.view(batch_size, seq_len, self.H, self.D)
            .transpose(1, 2)
            .contiguous()
            .view(batch_size * self.H, seq_len, self.D)
        )

    def _normalize_scores(
        self,
        score,
        vision_attention_mask,
        text_attention_mask,
        batch_size,
        V,
        T,
    ):
        # score: [H,V,T] for B=1
        score = score - score.max()
        score = torch.clamp(score, min=-50000, max=50000)

        text_score = score.transpose(1, 2)
        text_score = text_score - torch.max(text_score, dim=-1, keepdim=True)[0]
        text_score = torch.clamp(text_score, min=-50000, max=50000)

        if vision_attention_mask is not None:
            vm = (
                vision_attention_mask[:, None, None, :]
                .repeat(1, self.H, 1, 1)
                .flatten(0, 1)
            )
            text_score = text_score.masked_fill(vm, float("-inf"))

        text_attn = text_score.softmax(dim=-1)

        if text_attention_mask is not None:
            tm = (
                text_attention_mask[:, None, None, :]
                .repeat(1, self.H, 1, 1)
                .flatten(0, 1)
            )
            score = score.masked_fill(tm, float("-inf"))

        vision_attn = score.softmax(dim=-1)
        return vision_attn, text_attn

    def _folded_outputs(self, vision, text, vision_attn, text_attn):
        # Batch-1 specialization.
        v = vision[0]  # [V,d]
        t = text[0]    # [T,d]

        # Vision delta:
        # z_text[h,t,d] = text @ C_v[h] + c_v[h]
        z_text = torch.matmul(t.unsqueeze(0), self.C_v) + self.c_v[:, None, :]
        # [H,V,T] @ [H,T,d] -> [H,V,d], then sum head contributions.
        delta_v = torch.bmm(vision_attn, z_text).sum(dim=0)
        delta_v = delta_v + self.out_vision_bias
        delta_v = delta_v.unsqueeze(0)

        # Text delta:
        # First reduce raw 256-d vision tokens before any 256->1024 expansion.
        reduced_v = torch.matmul(text_attn, v)  # [H,T,d]
        row_sum = text_attn.sum(dim=-1, keepdim=True)
        delta_t_heads = torch.bmm(reduced_v, self.C_t)
        delta_t_heads = delta_t_heads + row_sum * self.c_t[:, None, :]
        delta_t = delta_t_heads.sum(dim=0) + self.out_text_bias
        delta_t = delta_t.unsqueeze(0)

        return delta_v, delta_t

    def value_folded(
        self,
        vision_features,
        text_features,
        vision_attention_mask=None,
        text_attention_mask=None,
    ):
        B, V, _ = vision_features.shape
        _, T, _ = text_features.shape
        if B != 1:
            raise ValueError("v0 specialization is batch-1 only.")

        # Preserve reference q/k projection and score generation.
        q = self.module.vision_proj(vision_features) * self.scale
        k = self.module.text_proj(text_features)

        q = self._reshape_ref(q, V, B)
        k = self._reshape_ref(k, T, B)

        score = torch.bmm(q, k.transpose(1, 2))
        vision_attn, text_attn = self._normalize_scores(
            score,
            vision_attention_mask,
            text_attention_mask,
            B,
            V,
            T,
        )

        delta_v, delta_t = self._folded_outputs(
            vision_features, text_features, vision_attn, text_attn
        )
        return (delta_v, vision_attn), (delta_t, text_attn)

    def fully_folded(
        self,
        vision_features,
        text_features,
        vision_attention_mask=None,
        text_attention_mask=None,
    ):
        B, V, d = vision_features.shape
        _, T, _ = text_features.shape
        if B != 1:
            raise ValueError("v0 specialization is batch-1 only.")

        v = vision_features[0]  # [V,d]
        t = text_features[0]    # [T,d]

        # Head-wise dynamic bilinear projection:
        # R[h,d,T] = M[h] @ t^T + cv[h,:,None]
        R = torch.matmul(self.score_M, t.transpose(0, 1))
        R = R + self.score_cv[:, :, None]

        # [1,V,d] @ [H,d,T] broadcasts to [H,V,T].
        score = torch.matmul(v.unsqueeze(0), R)

        # Remaining q-bias-dependent text term.
        text_bias = torch.matmul(t, self.score_ct.transpose(0, 1)).transpose(0, 1)
        text_bias = text_bias + self.score_c0[:, None]
        score = (score + text_bias[:, None, :]) * self.scale

        vision_attn, text_attn = self._normalize_scores(
            score,
            vision_attention_mask,
            text_attention_mask,
            B,
            V,
            T,
        )

        delta_v, delta_t = self._folded_outputs(
            vision_features, text_features, vision_attn, text_attn
        )
        return (delta_v, vision_attn), (delta_t, text_attn)


def benchmark(label, fn, args, kwargs, warmup, iters):
    with torch.inference_mode():
        for _ in range(warmup):
            sync()
            out = fn(*args, **kwargs)
            sync()

        samples = []
        out = None
        for i in range(iters):
            sync()
            t0 = time.perf_counter_ns()
            out = fn(*args, **kwargs)
            sync()
            dt = (time.perf_counter_ns() - t0) / 1e6
            samples.append(dt)
            print(f"  {label} {i+1:02d}/{iters}: {dt:.3f} ms", flush=True)
    return out, stats(samples)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="IDEA-Research/grounding-dino-tiny")
    ap.add_argument("--image", type=Path, default=Path("assets/input.jpg"))
    ap.add_argument("--prompt", nargs="+", default=["a cat", "a dog"])
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--iters", type=int, default=20)
    args = ap.parse_args()

    if not torch.backends.mps.is_available():
        raise SystemExit("MPS unavailable.")

    print("Loading model...", flush=True)
    processor = AutoProcessor.from_pretrained(args.model)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(args.model)
    model.eval().to("mps")
    sync()

    image = Image.open(args.image).convert("RGB")
    text_labels = [args.prompt]
    cpu_inputs = processor(images=image, text=text_labels, return_tensors="pt")
    inputs = {k: v.to("mps") for k, v in cpu_inputs.items()}
    sync()

    attn = model.model.encoder.layers[0].fusion_layer.attn
    original_forward = attn.forward
    captured: dict[str, Any] = {}

    def capture_forward(
        self,
        vision_features,
        text_features,
        vision_attention_mask=None,
        text_attention_mask=None,
    ):
        if not captured:
            captured["vision"] = vision_features.detach()
            captured["text"] = text_features.detach()
            captured["vision_mask"] = (
                None if vision_attention_mask is None else vision_attention_mask.detach()
            )
            captured["text_mask"] = (
                None if text_attention_mask is None else text_attention_mask.detach()
            )
        return original_forward(
            vision_features,
            text_features,
            vision_attention_mask=vision_attention_mask,
            text_attention_mask=text_attention_mask,
        )

    attn.forward = types.MethodType(capture_forward, attn)
    print("Capturing real layer-0 fusion workload...", flush=True)
    with torch.inference_mode():
        _ = model(**inputs)
        sync()
    attn.forward = original_forward

    vision = captured["vision"]
    text = captured["text"]
    vm = captured["vision_mask"]
    tm = captured["text_mask"]

    print(
        f"vision={tuple(vision.shape)} text={tuple(text.shape)} "
        f"embed={attn.embed_dim} heads={attn.num_heads} head_dim={attn.head_dim}",
        flush=True,
    )

    args_call = (vision, text)
    kwargs_call = {
        "vision_attention_mask": vm,
        "text_attention_mask": tm,
    }

    print("\nReference output...", flush=True)
    with torch.inference_mode():
        ref = original_forward(*args_call, **kwargs_call)
        sync()

    specialized = AlgebraicBiMHA(attn)

    print("\nCorrectness preflight: value-folded...", flush=True)
    with torch.inference_mode():
        value_pre = specialized.value_folded(*args_call, **kwargs_call)
        sync()
    value_corr = compare_output(value_pre, ref)
    print(json.dumps(value_corr, indent=2), flush=True)

    print("\nCorrectness preflight: fully-folded...", flush=True)
    with torch.inference_mode():
        full_pre = specialized.fully_folded(*args_call, **kwargs_call)
        sync()
    full_corr = compare_output(full_pre, ref)
    print(json.dumps(full_corr, indent=2), flush=True)

    if not all(x["finite"] for x in value_corr.values()):
        raise RuntimeError("Non-finite value-folded output.")
    if not all(x["finite"] for x in full_corr.values()):
        raise RuntimeError("Non-finite fully-folded output.")

    print("\nBenchmark reference PyTorch/MPS...", flush=True)
    ref_out, ref_lat = benchmark(
        "reference",
        original_forward,
        args_call,
        kwargs_call,
        args.warmup,
        args.iters,
    )

    print("\nBenchmark value-folded PyTorch/MPS...", flush=True)
    value_out, value_lat = benchmark(
        "value_folded",
        specialized.value_folded,
        args_call,
        kwargs_call,
        args.warmup,
        args.iters,
    )

    print("\nBenchmark fully-folded PyTorch/MPS...", flush=True)
    full_out, full_lat = benchmark(
        "fully_folded",
        specialized.fully_folded,
        args_call,
        kwargs_call,
        args.warmup,
        args.iters,
    )

    # Re-check the timed outputs.
    value_corr_final = compare_output(value_out, ref_out)
    full_corr_final = compare_output(full_out, ref_out)

    V = vision.shape[1]
    T = text.shape[1]
    d = vision.shape[2]
    H = attn.num_heads
    D = attn.head_dim
    E = attn.embed_dim

    # MAC accounting is an analytical operation-count estimate, not measured FLOPs.
    original_large_vision_projection_macs = int(
        V * d * E + V * d * E + V * E * d
    )
    cross_attention_macs = int(3 * H * V * T * D)

    result = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "experiment": "0013",
        "purpose": (
            "Test algebraic specialization of GroundingDinoBiMultiHeadAttention "
            "that avoids expanding all 17,821 vision tokens to the 1,024-d internal space."
        ),
        "model": args.model,
        "device": "mps",
        "dtype": "float32",
        "software": {
            "torch": torch.__version__,
            "transformers": transformers.__version__,
        },
        "shape": {
            "batch": int(vision.shape[0]),
            "vision_tokens": int(V),
            "text_tokens": int(T),
            "input_dim": int(d),
            "embed_dim": int(E),
            "heads": int(H),
            "head_dim": int(D),
        },
        "variants": {
            "reference": {
                "latency": ref_lat,
            },
            "value_folded": {
                "description": (
                    "Reference q/k and score path; fold value projections and output "
                    "projections around the attention reductions."
                ),
                "latency": value_lat,
                "speedup_vs_reference": ref_lat["median_ms"] / value_lat["median_ms"],
                "correctness": value_corr_final,
            },
            "fully_folded": {
                "description": (
                    "Value folding plus bilinear score specialization that removes "
                    "the full-length vision 256->1024 query projection."
                ),
                "latency": full_lat,
                "speedup_vs_reference": ref_lat["median_ms"] / full_lat["median_ms"],
                "correctness": full_corr_final,
            },
        },
        "analytical_operation_count": {
            "note": "MAC estimates only; not measured hardware FLOPs.",
            "original_three_large_vision_projection_macs": original_large_vision_projection_macs,
            "cross_attention_score_and_two_value_reductions_macs": cross_attention_macs,
        },
        "notes": [
            "Batch-1 specialization only.",
            "Weights are folded once outside the timed loop; model weights are unchanged.",
            "No approximation, pruning, retraining, or reduced precision is used.",
            "The specialization changes floating-point association, so bitwise equality is not expected.",
            "value_folded preserves the original q/k score and softmax path; therefore attention weights should be much more numerically stable than fully_folded.",
            "This experiment uses PyTorch/MPS only; there is no MLX bridge in the specialized operator timing."
        ],
    }

    out = Path("results/metalground_fusion_algebraic_specialization.json")
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")

    print("\n=== Experiment 0013 summary ===", flush=True)
    print(f"reference median:    {ref_lat['median_ms']:.3f} ms", flush=True)
    print(
        f"value-folded median: {value_lat['median_ms']:.3f} ms "
        f"({result['variants']['value_folded']['speedup_vs_reference']:.2f}x)",
        flush=True,
    )
    print(
        f"fully-folded median: {full_lat['median_ms']:.3f} ms "
        f"({result['variants']['fully_folded']['speedup_vs_reference']:.2f}x)",
        flush=True,
    )
    print(f"Saved: {out}", flush=True)


if __name__ == "__main__":
    main()
