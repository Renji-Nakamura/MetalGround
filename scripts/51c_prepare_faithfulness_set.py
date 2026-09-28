#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import io
import json
import random
import shutil
import sys
import urllib.request
import zipfile
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image


COCO_VAL_URL = "http://images.cocodataset.org/zips/val2017.zip"
COCO_HF_REPO = "pcuenq/coco-2017-mirror"
COCO_HF_FILENAME = "val2017.zip"
COCO_VAL_SHA256 = "4f7e2ccb2866ec5041993c9cf2a952bbed69647b115d0f74da7ce8f4bef82f05"

PROMPT_SETS = [
    {
        "id": "animals",
        "labels": ["a cat", "a dog", "a bird", "a horse"],
    },
    {
        "id": "road",
        "labels": [
            "a person",
            "a bicycle",
            "a car",
            "a motorcycle",
            "a bus",
            "a truck",
            "a traffic light",
            "a stop sign",
        ],
    },
    {
        "id": "indoor",
        "labels": [
            "a chair",
            "a couch",
            "a dining table",
            "a television",
            "a laptop",
            "a book",
        ],
    },
    {
        "id": "food",
        "labels": [
            "a banana",
            "an apple",
            "a sandwich",
            "a pizza",
            "a cake",
        ],
    },
    {
        "id": "mixed_open_vocab",
        "labels": [
            "a backpack",
            "an umbrella",
            "a suitcase",
            "a skateboard",
            "a tennis racket",
            "a teddy bear",
        ],
    },
]


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def download(url: str, dest: Path):
    """
    Download COCO val2017 from the Hugging Face mirror first.

    Reason:
    The COCO HTTPS endpoint had a certificate-hostname problem in this
    environment, and the HTTP endpoint connected but transferred 0 bytes.
    We therefore avoid both for the reproducible experiment path.

    TLS verification remains enabled. The completed archive is verified with
    the expected SHA256.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)

    def verify_existing(path: Path):
        if not path.exists() or path.stat().st_size == 0:
            return False
        actual = sha256_file(path)
        if actual == COCO_VAL_SHA256:
            print(
                f"Using verified existing archive: {path} "
                f"(sha256={actual})",
                flush=True,
            )
            return True
        print(
            f"Existing archive checksum mismatch; removing {path}\n"
            f"  expected: {COCO_VAL_SHA256}\n"
            f"  actual:   {actual}",
            flush=True,
        )
        path.unlink()
        return False

    if verify_existing(dest):
        return

    stale_part = dest.with_suffix(dest.suffix + ".part")
    if stale_part.exists():
        print(f"Removing stale partial file: {stale_part}", flush=True)
        stale_part.unlink()

    print(
        "Downloading COCO val2017 from Hugging Face mirror "
        f"{COCO_HF_REPO}/{COCO_HF_FILENAME}",
        flush=True,
    )

    try:
        from huggingface_hub import hf_hub_download

        cached = Path(
            hf_hub_download(
                repo_id=COCO_HF_REPO,
                filename=COCO_HF_FILENAME,
                repo_type="dataset",
            )
        )
        print(f"HF download/cache complete: {cached}", flush=True)
        shutil.copyfile(cached, dest)
    except Exception as exc:
        raise RuntimeError(
            "Hugging Face mirror download failed.\n"
            f"Error: {exc!r}\n"
            "Manual fallback: download val2017.zip and place it at "
            f"{dest}."
        ) from exc

    actual = sha256_file(dest)
    if actual != COCO_VAL_SHA256:
        dest.unlink(missing_ok=True)
        raise RuntimeError(
            "Downloaded COCO val2017 archive failed SHA256 verification.\n"
            f"Expected: {COCO_VAL_SHA256}\n"
            f"Actual:   {actual}"
        )

    print(
        f"Verified COCO val2017 archive: sha256={actual}",
        flush=True,
    )


def letterbox(image: Image.Image, size_wh: tuple[int, int]):
    W, H = size_wh
    image = image.convert("RGB")
    w, h = image.size
    scale = min(W / w, H / h)
    nw = max(1, int(round(w * scale)))
    nh = max(1, int(round(h * scale)))
    resized = image.resize((nw, nh), Image.Resampling.LANCZOS)

    left = (W - nw) // 2
    top = (H - nh) // 2
    canvas = Image.new("RGB", (W, H), (114, 114, 114))
    canvas.paste(resized, (left, top))
    return canvas, {
        "source_size_wh": [w, h],
        "canvas_size_wh": [W, H],
        "scale": scale,
        "resized_size_wh": [nw, nh],
        "padding_ltrb": [left, top, W - nw - left, H - nh - top],
    }


def save_normalized(
    image: Image.Image,
    out_path: Path,
    canvas_wh: tuple[int, int],
):
    normalized, geom = letterbox(image, canvas_wh)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    normalized.save(out_path, format="PNG", optimize=False)
    geom["normalized_sha256"] = sha256_file(out_path)
    return geom


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--archive",
        type=Path,
        default=Path("datasets/coco/val2017.zip"),
    )
    ap.add_argument(
        "--output-dir",
        type=Path,
        default=Path("assets/faithfulness_0051"),
    )
    ap.add_argument("--count", type=int, default=32)
    ap.add_argument("--seed", type=int, default=51051)
    ap.add_argument("--width", type=int, default=1200)
    ap.add_argument("--height", type=int, default=901)
    ap.add_argument(
        "--sentinel",
        type=Path,
        default=Path("assets/input.jpg"),
    )
    ap.add_argument(
        "--skip-download",
        action="store_true",
        help="Fail instead of downloading if the COCO archive is absent.",
    )
    args = ap.parse_args()

    if args.count < 1:
        raise SystemExit("--count must be >=1")

    if not args.archive.exists():
        if args.skip_download:
            raise SystemExit(
                f"Missing {args.archive}; rerun without --skip-download."
            )
        download(COCO_VAL_URL, args.archive)

    out_dir = args.output_dir
    images_dir = out_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(args.archive) as zf:
        names = sorted(
            n
            for n in zf.namelist()
            if n.startswith("val2017/")
            and n.lower().endswith((".jpg", ".jpeg"))
        )
        if len(names) < args.count:
            raise RuntimeError(
                f"Archive contains only {len(names)} images; "
                f"requested {args.count}."
            )

        rng = random.Random(args.seed)
        selected = sorted(rng.sample(names, args.count))

        entries = []
        for i, name in enumerate(selected):
            raw = zf.read(name)
            image = Image.open(io.BytesIO(raw)).convert("RGB")
            out_name = f"coco_{i:03d}_{Path(name).stem}.png"
            out_path = images_dir / out_name

            geom = save_normalized(
                image,
                out_path,
                (args.width, args.height),
            )
            entries.append(
                {
                    "id": f"coco_{i:03d}_{Path(name).stem}",
                    "path": str(out_path),
                    "source": "COCO val2017",
                    "source_archive_member": name,
                    "source_sha256": sha256_bytes(raw),
                    "sentinel": False,
                    "geometry": geom,
                }
            )
            print(
                f"[{i+1:02d}/{args.count}] {name} -> {out_path}",
                flush=True,
            )

    # Preserve the long-running project's canonical cat/dog sample as a
    # sentinel case. It is normalized through the same fixed-canvas path.
    if args.sentinel.exists():
        sentinel_image = Image.open(args.sentinel).convert("RGB")
        out_path = images_dir / "sentinel_baseline_input.png"
        geom = save_normalized(
            sentinel_image,
            out_path,
            (args.width, args.height),
        )
        entries.insert(
            0,
            {
                "id": "sentinel_baseline_input",
                "path": str(out_path),
                "source": "MetalGround Experiment 0001 baseline asset",
                "source_path": str(args.sentinel),
                "source_sha256": sha256_file(args.sentinel),
                "sentinel": True,
                "geometry": geom,
            },
        )

    manifest = {
        "experiment": "0051",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": {
            "name": "COCO val2017 deterministic subset",
            "archive": str(args.archive),
            "archive_sha256": sha256_file(args.archive),
            "expected_archive_sha256": COCO_VAL_SHA256,
            "download_sources": [
                f"https://huggingface.co/datasets/{COCO_HF_REPO}/blob/main/{COCO_HF_FILENAME}",
            ],
            "sampling": {
                "method": "Python random.Random(seed).sample over sorted val2017 filenames",
                "seed": args.seed,
                "count_coco": args.count,
                "sentinel_included": bool(args.sentinel.exists()),
            },
        },
        "normalization": {
            "method": "aspect-ratio-preserving letterbox",
            "canvas_size_wh": [args.width, args.height],
            "fill_rgb": [114, 114, 114],
            "resample": "PIL.Image.Resampling.LANCZOS",
            "format": "PNG",
            "reason": (
                "Keep the webcam-style input geometry fixed so the current "
                "MetalGround shape-specialized runtime is tested across image "
                "content rather than across unrelated input shapes."
            ),
        },
        "images": entries,
    }

    prompts = {
        "experiment": "0051",
        "prompt_sets": PROMPT_SETS,
        "notes": [
            "Each prompt set is evaluated on every prepared image.",
            "The animals set includes cat/dog so the historical sentinel remains comparable.",
            "Prompts vary token count and semantic domain without changing the detector."
        ],
    }

    manifest_path = out_dir / "manifest.json"
    prompts_path = out_dir / "prompts.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n"
    )
    prompts_path.write_text(
        json.dumps(prompts, indent=2, ensure_ascii=False) + "\n"
    )

    print("\nPrepared Experiment 0051 dataset.", flush=True)
    print(f"Images: {len(entries)}", flush=True)
    print(f"Prompt sets: {len(PROMPT_SETS)}", flush=True)
    print(
        f"Total full sweep cases: {len(entries) * len(PROMPT_SETS)}",
        flush=True,
    )
    print(f"Manifest: {manifest_path}", flush=True)
    print(f"Prompts:  {prompts_path}", flush=True)


if __name__ == "__main__":
    main()
