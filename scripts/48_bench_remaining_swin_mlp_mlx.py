#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import statistics
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import torch
import transformers


def load_module(path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load helper: {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="IDEA-Research/grounding-dino-tiny")
    ap.add_argument("--image", type=Path, default=Path("assets/input.jpg"))
    ap.add_argument("--prompt", nargs="+", default=["a cat", "a dog"])
    ap.add_argument("--threadgroup", type=int, default=256)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--pairs", type=int, default=20)

    helpers = {
        13: "bench_fusion_algebraic_specialization.py",
        14: "full_model_fusion_algebraic.py",
        17: "bench_deformable_mlx_island.py",
        18: "full_model_deformable_islands.py",
        30: "bench_fully_folded_fusion_mlx.py",
        31: "full_model_mlx_fusion_paired.py",
        36: "wide_island_mask_sync_fix.py",
        37: "corrected_wide_island_multiprocess.py",
        38: "consolidated_runtime_ablation.py",
        45: "bench_swin_stage0_mlp_mlx.py",
        46: "full_model_stage0_mlp_mlx.py",
    }
    for n, fn in helpers.items():
        ap.add_argument(
            f"--exp{n}-helper",
            type=Path,
            default=Path("scripts") / f"{n:02d}_{fn}",
        )
    args = ap.parse_args()

    for n in helpers:
        p = getattr(args, f"exp{n}_helper")
        if not p.exists():
            raise SystemExit(f"Missing helper: {p}")

    if not torch.backends.mps.is_available():
        raise SystemExit("MPS unavailable.")

    h37 = load_module(args.exp37_helper, "mg48_h37")
    h38 = load_module(args.exp38_helper, "mg48_h38")
    h45 = load_module(args.exp45_helper, "mg48_h45")
    h46 = load_module(args.exp46_helper, "mg48_h46")

    print("Building adopted consolidated runtime...", flush=True)
    rt = h37.build_runtime(args)
    model = rt["model"]
    h14 = rt["h14"]
    inputs = rt["inputs"]

    cache = h38.ExactTextBackboneCache(model.model.text_backbone)
    stage0_dispatchers = []
    capture_originals = []

    try:
        # Base runtime = Experiment 0039 consolidated runtime.
        h38.set_runtime_mode(rt, cache, "wide_cache")
        cache.prime(model, inputs, h14.sync)
        cache.set_enabled(True)

        stages = model.model.backbone.conv_encoder.model.swin.encoder.layers

        # Install the newly adopted stage0 MLP optimization so that captured
        # downstream tensors are exactly from the current post-0047 runtime.
        stage0_modules = [stages[0].blocks[0].mlp, stages[0].blocks[1].mlp]
        h14.sync()
        stage0_candidates = [h45.MlxSwinMLP(m) for m in stage0_modules]
        stage0_dispatchers = [
            h46.SwinMlpDispatcher(m, c, h14.sync)
            for m, c in zip(stage0_modules, stage0_candidates)
        ]
        for d in stage0_dispatchers:
            d.set_mode("mlx")

        # Prime stage0 compiled kernels before capture.
        print("Priming adopted stage0 MLX MLPs...", flush=True)
        with torch.inference_mode():
            _ = model(**inputs)
        h14.sync()

        # Capture every remaining Swin MLP input under the adopted runtime.
        targets = []
        for stage_idx in range(1, len(stages)):
            stage = stages[stage_idx]
            for block_idx, block in enumerate(stage.blocks):
                name = f"stage{stage_idx}.block{block_idx}.mlp"
                targets.append((stage_idx, block_idx, name, block.mlp, {}))

        for _si, _bi, _name, module, store in targets:
            original = h45.capture_once(module, store)
            capture_originals.append((module, original))

        print(
            f"Capturing {len(targets)} remaining Swin MLP workloads "
            f"with adopted stage0 MLX path active...",
            flush=True,
        )
        with torch.inference_mode():
            _ = model(**inputs)
        h14.sync()

        for module, original in capture_originals:
            module.forward = original
        capture_originals.clear()

        for si, bi, name, _module, store in targets:
            print(
                f"  {name}: input={tuple(store['input'].shape)}",
                flush=True,
            )

        # Isolated exact microbenchmarks.
        results = []
        for si, bi, name, module, store in targets:
            print(
                f"\n=== benchmarking {name} ===",
                flush=True,
            )
            r = h45.bench_one(
                name=name,
                module=module,
                captured=store,
                sync_fn=h14.sync,
                warmup=args.warmup,
                pairs=args.pairs,
            )
            r["stage_index"] = si
            r["block_index"] = bi
            results.append(r)

        # Stage-level selection heuristic. This is only a pre-screen for a
        # full-model experiment; isolated savings are never claimed additive.
        by_stage = defaultdict(list)
        for r in results:
            by_stage[r["stage_index"]].append(r)

        stage_summaries = {}
        selected_stages = []

        for stage_idx in sorted(by_stage):
            rs = by_stage[stage_idx]
            correctness_all = all(
                r["correctness_gate_passed"] for r in rs
            )

            paired = [
                r["derived"]["paired_delta_median_ms"]
                for r in rs
                if r["derived"] is not None
            ]
            reductions = [
                r["derived"]["reduction_percent_from_medians"]
                for r in rs
                if r["derived"] is not None
            ]

            all_positive = (
                len(paired) == len(rs)
                and all(x > 0.0 for x in paired)
            )
            median_reduction = (
                statistics.median(reductions) if reductions else None
            )
            median_paired = (
                statistics.median(paired) if paired else None
            )

            # Pre-screen only:
            # - all blocks correct
            # - every block positive
            # - median block reduction >= 15%
            candidate_stage = bool(
                correctness_all
                and all_positive
                and median_reduction is not None
                and median_reduction >= 15.0
            )

            if candidate_stage:
                selected_stages.append(stage_idx)

            stage_summaries[str(stage_idx)] = {
                "blocks": len(rs),
                "correctness_all_passed": correctness_all,
                "all_block_paired_medians_positive": all_positive,
                "per_block_paired_medians_ms": paired,
                "median_block_paired_saving_ms": median_paired,
                "per_block_reduction_percent": reductions,
                "median_block_reduction_percent": median_reduction,
                "selected_for_full_model_test": candidate_stage,
            }

        result = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "experiment": "0048",
            "purpose": (
                "Exact real-workload microbenchmark of every remaining Swin "
                "MLP in stages 1-3 after adopting the robust stage0 MLX MLP "
                "optimization from Experiment 0047."
            ),
            "model": args.model,
            "device": "mps",
            "dtype": "float32",
            "software": {
                "torch": torch.__version__,
                "transformers": transformers.__version__,
            },
            "configuration": {
                "runtime_during_capture": (
                    "current adopted runtime including stage0 block0/block1 "
                    "compiled MLX MLPs from Experiment 0047"
                ),
                "targets": [
                    r["name"] for r in results
                ],
                "candidate_per_target": (
                    "compiled MLX fc1 -> exact erf GELU -> fc2"
                ),
                "fresh_torch_to_mlx_bridge_each_call": True,
                "final_mlx_to_torch_bridge_each_call": True,
                "warmup_per_target": args.warmup,
                "pairs_per_target": args.pairs,
                "approximation": False,
                "retraining": False,
                "reduced_precision": False,
            },
            "results": results,
            "stage_summaries": stage_summaries,
            "selected_stages_for_full_model_test": selected_stages,
            "stage_selection_rule": (
                "A stage proceeds to full-model testing only if every MLP "
                "passes 1e-4 correctness, every block has positive paired "
                "median saving, and the stage median of block-level reduction "
                "percentages is at least 15%. Isolated block savings are a "
                "selection heuristic only and must not be summed as a claimed "
                "full-model speedup."
            ),
            "notes": [
                "Stage0 is already adopted from Experiment 0047 and is not re-benchmarked here.",
                "Captured stage1-3 inputs are produced with the adopted stage0 MLX MLP path active.",
                "This does not reopen generic whole-Swin-block substitution; only MLP submodules are tested.",
                "Experiment 0047 is the current robust runtime authority: 476.360 ms / 2.099 FPS under its controlled multi-process protocol."
            ],
        }

        out = Path(
            "results/metalground_remaining_swin_mlp_mlx.json"
        )
        out.write_text(
            json.dumps(result, indent=2, ensure_ascii=False) + "\n"
        )

        print("\n=== Experiment 0048 summary ===", flush=True)
        for stage_idx in sorted(stage_summaries, key=int):
            s = stage_summaries[stage_idx]
            print(
                f"stage{stage_idx}: correct={s['correctness_all_passed']} "
                f"all-positive={s['all_block_paired_medians_positive']} "
                f"median reduction="
                f"{s['median_block_reduction_percent']:.2f}% "
                f"selected={s['selected_for_full_model_test']}",
                flush=True,
            )

        print(
            "selected stages: "
            + (", ".join(f"stage{x}" for x in selected_stages)
               if selected_stages else "none"),
            flush=True,
        )
        print(f"Saved: {out}", flush=True)

    finally:
        for module, original in capture_originals:
            try:
                module.forward = original
            except Exception:
                pass
        for d in stage0_dispatchers:
            try:
                d.restore()
            except Exception:
                pass
        cache.restore()
        for d in rt["layer_dispatchers"]:
            d.restore()
        for fd in rt["fusion_dispatchers"]:
            fd.restore()


if __name__ == "__main__":
    main()
