#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import platform
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import torch
import transformers
from transformers import AutoModelForZeroShotObjectDetection


KEYWORDS = (
    "backbone",
    "swin",
    "text",
    "bert",
    "encoder",
    "decoder",
    "deform",
    "attention",
    "fusion",
    "feature",
    "query",
)


def count_params(module: torch.nn.Module, recurse: bool) -> int:
    return sum(p.numel() for p in module.parameters(recurse=recurse))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="IDEA-Research/grounding-dino-tiny")
    ap.add_argument("--out", type=Path, default=Path("results/hf_model_inventory.json"))
    args = ap.parse_args()

    print(f"Loading {args.model} on CPU for structural inspection...")
    model = AutoModelForZeroShotObjectDetection.from_pretrained(args.model)
    model.eval()

    modules = []
    class_counts = Counter()

    for name, module in model.named_modules():
        cls = type(module).__name__
        class_counts[cls] += 1

        direct_params = count_params(module, recurse=False)
        total_params = count_params(module, recurse=True)

        rec = {
            "name": name or "<root>",
            "class": cls,
            "direct_params": direct_params,
            "total_params": total_params,
        }
        modules.append(rec)

    candidates = [
        m for m in modules
        if any(k in m["name"].lower() or k in m["class"].lower() for k in KEYWORDS)
    ]

    record = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "model": args.model,
        "model_class": type(model).__name__,
        "base_model_prefix": getattr(model, "base_model_prefix", None),
        "total_parameters": count_params(model, recurse=True),
        "software": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
        },
        "config": model.config.to_dict(),
        "class_counts": dict(class_counts.most_common()),
        "candidate_modules": candidates,
        "all_modules": modules,
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n")

    print(f"\nModel class: {record['model_class']}")
    print(f"Parameters:  {record['total_parameters']:,}")
    print(f"Modules:     {len(modules):,}")

    print("\n=== Candidate modules ===")
    for m in candidates:
        print(
            f"{m['name']:<72} "
            f"{m['class']:<48} "
            f"direct={m['direct_params']:,} total={m['total_params']:,}"
        )

    print(f"\nSaved: {args.out}")


if __name__ == "__main__":
    main()
