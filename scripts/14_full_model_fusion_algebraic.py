#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import statistics
import time
import types
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import mlx.core as mx
import numpy as np
import torch
import transformers
from PIL import Image
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

from metalground.msda_metal_v0 import msda_metal_v0


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
        "p99_ms": percentile(xs, 0.99),
        "min_ms": min(xs),
        "max_ms": max(xs),
    }


def mask_aware_error(a: torch.Tensor, b: torch.Tensor) -> dict[str, Any]:
    a = a.detach().float()
    b = b.detach().float()
    af = torch.isfinite(a)
    bf = torch.isfinite(b)
    both = af & bf

    out = {
        "shape": list(a.shape),
        "finite_mask_equal": bool(torch.equal(af, bf)),
        "nan_mask_equal": bool(torch.equal(torch.isnan(a), torch.isnan(b))),
        "posinf_mask_equal": bool(torch.equal(torch.isposinf(a), torch.isposinf(b))),
        "neginf_mask_equal": bool(torch.equal(torch.isneginf(a), torch.isneginf(b))),
    }
    if bool(both.any()):
        aa, bb = a[both], b[both]
        d = (aa - bb).abs()
        out.update(
            {
                "max_abs": float(d.max().item()),
                "mean_abs": float(d.mean().item()),
                "rmse": float(torch.sqrt(torch.mean((aa - bb) ** 2)).item()),
                "allclose_1e-5": bool(torch.allclose(aa, bb, rtol=1e-5, atol=1e-5)),
                "allclose_1e-4": bool(torch.allclose(aa, bb, rtol=1e-4, atol=1e-4)),
            }
        )
    sync()
    return out


def detection_summary(
    processor,
    outputs,
    input_ids,
    text_labels,
    image_size_wh,
    box_threshold: float,
    text_threshold: float,
):
    det = processor.post_process_grounded_object_detection(
        outputs,
        input_ids,
        text_labels=text_labels,
        target_sizes=[image_size_wh[::-1]],
        threshold=box_threshold,
        text_threshold=text_threshold,
    )[0]

    return [
        {
            "label": label,
            "score": float(score),
            "box": [float(x) for x in box],
        }
        for label, score, box in zip(
            det["text_labels"],
            det["scores"].detach().float().cpu().tolist(),
            det["boxes"].detach().float().cpu().tolist(),
        )
    ]


def topk_audit(out, reference, k: int) -> dict[str, Any]:
    rs = reference.enc_outputs_class.detach().float().max(-1).values
    cs = out.enc_outputs_class.detach().float().max(-1).values
    _, ri = torch.topk(rs, k, dim=1)
    _, ci = torch.topk(cs, k, dim=1)

    r = ri[0].cpu().tolist()
    c = ci[0].cpu().tolist()
    rset = set(r)
    cset = set(c)

    score_diff = (cs - rs).abs()

    result = {
        "k": k,
        "set_overlap": len(rset & cset),
        "changed_membership_each_side": k - len(rset & cset),
        "rankwise_identical": int((ri == ci).sum().item()),
        "score_vector_max_abs": float(score_diff.max().item()),
        "score_vector_mean_abs": float(score_diff.mean().item()),
        "mismatched_ranks_zero_based": [
            i for i, (a, b) in enumerate(zip(r, c)) if a != b
        ],
    }
    sync()
    return result


@dataclass
class MsdaPatchState:
    threadgroup: int
    metadata_cache: dict = field(default_factory=dict)
    call_count: int = 0

    def metadata(self, shapes):
        key = tuple((int(h), int(w)) for h, w in shapes)
        if key not in self.metadata_cache:
            starts, acc = [], 0
            for h, w in key:
                starts.append(acc)
                acc += h * w
            sm = mx.array(np.asarray(key, dtype=np.int32), dtype=mx.int32)
            st = mx.array(np.asarray(starts, dtype=np.int32), dtype=mx.int32)
            mx.eval(sm, st)
            mx.synchronize()
            self.metadata_cache[key] = (sm, st)
        return self.metadata_cache[key]


def make_msda_forward(state: MsdaPatchState):
    def forward(
        self,
        value,
        value_spatial_shapes,
        value_spatial_shapes_list,
        level_start_index,
        sampling_locations,
        attention_weights,
        im2col_step,
    ):
        if not value.is_contiguous():
            value = value.contiguous()
        if not sampling_locations.is_contiguous():
            sampling_locations = sampling_locations.contiguous()
        if not attention_weights.is_contiguous():
            attention_weights = attention_weights.contiguous()

        sync()

        vm = mx.asarray(value, copy=False)
        lm = mx.asarray(sampling_locations, copy=False)
        wm = mx.asarray(attention_weights, copy=False)
        sm, st = state.metadata(value_spatial_shapes_list)

        out_mx = msda_metal_v0(
            vm, sm, st, lm, wm, threadgroup_size=state.threadgroup
        )
        mx.eval(out_mx)
        mx.synchronize()

        out = torch.as_tensor(out_mx)
        if out.device.type != "mps":
            raise RuntimeError(f"Unexpected MSDA output device: {out.device}")
        state.call_count += 1
        return out

    return forward


def patch_msda(model, threadgroup: int) -> MsdaPatchState:
    state = MsdaPatchState(threadgroup=threadgroup)
    fwd = make_msda_forward(state)
    n = 0
    for _, module in model.named_modules():
        if module.__class__.__name__ == "MultiScaleDeformableAttention":
            module.forward = types.MethodType(fwd, module)
            n += 1
    if n != 12:
        raise RuntimeError(f"Expected 12 MSDA cores, got {n}")
    return state


class AlgebraicBiMHA:
    def __init__(self, module):
        self.module = module
        self.H = int(module.num_heads)
        self.D = int(module.head_dim)
        self.E = int(module.embed_dim)
        self.d = int(module.vision_dim)
        self.scale = float(module.scale)

        if self.d != int(module.text_dim):
            raise ValueError("Expected equal vision/text external dims.")
        if float(module.dropout) != 0.0:
            raise ValueError("v0 specialization requires fusion dropout == 0.")
        if self.E != self.H * self.D:
            raise ValueError("Invalid head partition.")

        self._build_folded_weights()

    def _blocks_out(self, linear):
        return torch.stack(
            [
                linear.weight[:, h * self.D : (h + 1) * self.D]
                for h in range(self.H)
            ],
            dim=0,
        )

    def _blocks_in(self, linear):
        return torch.stack(
            [
                linear.weight[h * self.D : (h + 1) * self.D, :]
                for h in range(self.H)
            ],
            dim=0,
        )

    def _bias_blocks(self, linear):
        return torch.stack(
            [
                linear.bias[h * self.D : (h + 1) * self.D]
                for h in range(self.H)
            ],
            dim=0,
        )

    def _build_folded_weights(self):
        m = self.module

        Wq = self._blocks_in(m.vision_proj)
        bq = self._bias_blocks(m.vision_proj)
        Wk = self._blocks_in(m.text_proj)
        bk = self._bias_blocks(m.text_proj)

        Wvv = self._blocks_in(m.values_vision_proj)
        bvv = self._bias_blocks(m.values_vision_proj)
        Wvt = self._blocks_in(m.values_text_proj)
        bvt = self._bias_blocks(m.values_text_proj)

        Wov = self._blocks_out(m.out_vision_proj)
        Wot = self._blocks_out(m.out_text_proj)

        self.score_M = torch.bmm(Wq.transpose(1, 2), Wk).contiguous()
        self.score_cv = torch.bmm(
            Wq.transpose(1, 2), bk.unsqueeze(-1)
        ).squeeze(-1).contiguous()
        self.score_ct = torch.bmm(
            bq.unsqueeze(1), Wk
        ).squeeze(1).contiguous()
        self.score_c0 = (bq * bk).sum(-1).contiguous()

        self.C_v = torch.bmm(
            Wvt.transpose(1, 2), Wov.transpose(1, 2)
        ).contiguous()
        self.c_v = torch.bmm(
            bvt.unsqueeze(1), Wov.transpose(1, 2)
        ).squeeze(1).contiguous()

        self.C_t = torch.bmm(
            Wvv.transpose(1, 2), Wot.transpose(1, 2)
        ).contiguous()
        self.c_t = torch.bmm(
            bvv.unsqueeze(1), Wot.transpose(1, 2)
        ).squeeze(1).contiguous()

        self.out_vision_bias = m.out_vision_proj.bias
        self.out_text_bias = m.out_text_proj.bias
        sync()

    def _reshape_ref(self, x, seq_len, batch_size):
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
        score = score - score.max()
        score = torch.clamp(score, min=-50000, max=50000)

        text_score = score.transpose(1, 2)
        text_score = text_score - torch.max(
            text_score, dim=-1, keepdim=True
        )[0]
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
        v = vision[0]
        t = text[0]

        z_text = torch.matmul(t.unsqueeze(0), self.C_v) + self.c_v[:, None, :]
        delta_v = torch.bmm(vision_attn, z_text).sum(dim=0)
        delta_v = (delta_v + self.out_vision_bias).unsqueeze(0)

        reduced_v = torch.matmul(text_attn, v)
        row_sum = text_attn.sum(dim=-1, keepdim=True)
        delta_t_heads = torch.bmm(reduced_v, self.C_t)
        delta_t_heads = delta_t_heads + row_sum * self.c_t[:, None, :]
        delta_t = (
            delta_t_heads.sum(dim=0) + self.out_text_bias
        ).unsqueeze(0)

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
            raise ValueError("Batch-1 specialization only.")

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
        B, V, _ = vision_features.shape
        _, T, _ = text_features.shape
        if B != 1:
            raise ValueError("Batch-1 specialization only.")

        v = vision_features[0]
        t = text_features[0]

        R = torch.matmul(self.score_M, t.transpose(0, 1))
        R = R + self.score_cv[:, :, None]

        score = torch.matmul(v.unsqueeze(0), R)

        text_bias = torch.matmul(
            t, self.score_ct.transpose(0, 1)
        ).transpose(0, 1)
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


def collect_fusion_specializers(model):
    items = []
    for i in range(6):
        name = f"model.encoder.layers.{i}.fusion_layer.attn"
        module = model.model.encoder.layers[i].fusion_layer.attn
        items.append(
            {
                "name": name,
                "module": module,
                "original_forward": module.forward,
                "specialized": AlgebraicBiMHA(module),
            }
        )
    return items


def set_fusion_mode(items, mode: str):
    for item in items:
        module = item["module"]
        if mode == "reference":
            module.forward = item["original_forward"]
            continue

        spec = item["specialized"]
        fn = spec.value_folded if mode == "value_folded" else spec.fully_folded

        def make_forward(bound_fn):
            def forward(
                self,
                vision_features,
                text_features,
                vision_attention_mask=None,
                text_attention_mask=None,
            ):
                return bound_fn(
                    vision_features,
                    text_features,
                    vision_attention_mask=vision_attention_mask,
                    text_attention_mask=text_attention_mask,
                )
            return forward

        module.forward = types.MethodType(make_forward(fn), module)


def benchmark(model, inputs, warmup: int, iters: int):
    with torch.inference_mode():
        for i in range(warmup):
            _ = model(**inputs)
            sync()
            print(f"  warmup {i+1}/{warmup}", flush=True)

        samples = []
        out = None
        for i in range(iters):
            sync()
            t0 = time.perf_counter_ns()
            out = model(**inputs)
            sync()
            dt = (time.perf_counter_ns() - t0) / 1e6
            samples.append(dt)
            print(f"  {i+1:02d}/{iters}: {dt:.3f} ms", flush=True)

    return out, stats(samples)


def variant_correctness(
    out,
    reference,
    processor,
    input_ids,
    text_labels,
    image_size,
    box_threshold,
    text_threshold,
    k,
):
    return {
        "logits": mask_aware_error(out.logits, reference.logits),
        "pred_boxes": mask_aware_error(out.pred_boxes, reference.pred_boxes),
        "topk": topk_audit(out, reference, k),
        "detections": detection_summary(
            processor,
            out,
            input_ids,
            text_labels,
            image_size,
            box_threshold,
            text_threshold,
        ),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="IDEA-Research/grounding-dino-tiny")
    ap.add_argument("--image", type=Path, default=Path("assets/input.jpg"))
    ap.add_argument("--prompt", nargs="+", default=["a cat", "a dog"])
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--threadgroup", type=int, default=256)
    ap.add_argument("--original-baseline-ms", type=float, default=999.116584)
    ap.add_argument("--msda-v0-baseline-ms", type=float, default=632.6745625)
    ap.add_argument("--box-threshold", type=float, default=0.3)
    ap.add_argument("--text-threshold", type=float, default=0.25)
    args = ap.parse_args()

    if not torch.backends.mps.is_available():
        raise SystemExit("MPS unavailable.")

    mx.set_default_device(mx.gpu)

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

    print("Unpatched PyTorch/MPS reference forward...", flush=True)
    with torch.inference_mode():
        reference = model(**inputs)
        sync()

    reference_detections = detection_summary(
        processor,
        reference,
        inputs["input_ids"],
        text_labels,
        image.size,
        args.box_threshold,
        args.text_threshold,
    )

    print("Building 6 algebraic fusion specializers...", flush=True)
    fusion_items = collect_fusion_specializers(model)

    print("Patching 12 MSDA cores with MetalGround v0...", flush=True)
    msda_state = patch_msda(model, args.threadgroup)

    variants = {}
    modes = ("reference", "value_folded", "fully_folded")

    for mode in modes:
        label = "msda_only" if mode == "reference" else f"msda_plus_{mode}"
        print(f"\n=== {label} ===", flush=True)
        set_fusion_mode(fusion_items, mode)

        # First execution after changing graph, excluded from timing.
        msda_state.call_count = 0
        with torch.inference_mode():
            first = model(**inputs)
            sync()

        corr = variant_correctness(
            first,
            reference,
            processor,
            inputs["input_ids"],
            text_labels,
            image.size,
            args.box_threshold,
            args.text_threshold,
            model.config.num_queries,
        )

        print("Top-k:", json.dumps(corr["topk"], indent=2), flush=True)
        print("Detections:", json.dumps(corr["detections"], indent=2), flush=True)

        msda_state.call_count = 0
        print(
            f"Benchmark warmup={args.warmup}, iters={args.iters}...",
            flush=True,
        )
        final, lat = benchmark(model, inputs, args.warmup, args.iters)

        expected_calls = (args.warmup + args.iters) * 12
        if msda_state.call_count != expected_calls:
            raise RuntimeError(
                f"MSDA call-count mismatch: got {msda_state.call_count}, "
                f"expected {expected_calls}"
            )

        final_corr = variant_correctness(
            final,
            reference,
            processor,
            inputs["input_ids"],
            text_labels,
            image.size,
            args.box_threshold,
            args.text_threshold,
            model.config.num_queries,
        )

        variants[label] = {
            "fusion_mode": mode,
            "latency": lat,
            "speedup_vs_original_pytorch_mps": (
                args.original_baseline_ms / lat["median_ms"]
            ),
            "speedup_vs_authoritative_msda_v0": (
                args.msda_v0_baseline_ms / lat["median_ms"]
            ),
            "first_forward_correctness": corr,
            "final_forward_correctness": final_corr,
            "msda_calls_timed_region_including_warmup": msda_state.call_count,
        }

        print(
            f"{label}: median={lat['median_ms']:.3f} ms, "
            f"p95={lat['p95_ms']:.3f} ms, "
            f"vs original={variants[label]['speedup_vs_original_pytorch_mps']:.3f}x",
            flush=True,
        )

    result = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "experiment": "0014",
        "purpose": (
            "Integrate algebraically specialized GroundingDinoBiMultiHeadAttention "
            "across all six encoder fusion layers on top of MetalGround MSDA v0."
        ),
        "model": args.model,
        "device": "mps",
        "dtype": "float32",
        "image": str(args.image),
        "prompt": args.prompt,
        "software": {
            "torch": torch.__version__,
            "transformers": transformers.__version__,
        },
        "baselines": {
            "experiment_0001_original_pytorch_mps_median_ms": args.original_baseline_ms,
            "experiment_0009_msda_v0_median_ms": args.msda_v0_baseline_ms,
        },
        "reference_detections": reference_detections,
        "fusion_specialization": {
            "num_layers": 6,
            "batch_specialization": 1,
            "weight_folding": "precomputed once before timing",
            "approximation": False,
            "reduced_precision": False,
        },
        "variants": variants,
        "notes": [
            "All variants use the same 12 MetalGround MSDA v0 cores.",
            "The msda_only variant is a same-process sanity benchmark; Experiment 0009 remains the authoritative MSDA-v0 baseline.",
            "Fusion folded weights are computed once before timed inference.",
            "No retraining, pruning, approximation, or reduced precision is used.",
            "Raw query-wise tensors can be rank-sensitive; top-k set overlap/order and postprocessed detections are recorded separately.",
        ],
    }

    out = Path("results/metalground_v1_fusion_full_model.json")
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")

    print("\n=== Experiment 0014 summary ===", flush=True)
    for label, data in variants.items():
        print(
            f"{label:28s} "
            f"{data['latency']['median_ms']:8.3f} ms  "
            f"{data['speedup_vs_original_pytorch_mps']:.3f}x original  "
            f"{data['speedup_vs_authoritative_msda_v0']:.3f}x vs MSDA-v0",
            flush=True,
        )
    print(f"Saved: {out}", flush=True)


if __name__ == "__main__":
    main()
