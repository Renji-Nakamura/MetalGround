#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import zipfile
from pathlib import Path

HF_REPO = "pcuenq/coco-2017-mirror"
HF_FILENAME = "annotations_trainval2017.zip"
EXPECTED_SHA256 = "113a836d90195ee1f884e704da6304dfaaecff1f023f49b6ca93c4aaae470268"
MEMBER = "annotations/instances_val2017.json"


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk_size)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--archive",
        type=Path,
        default=Path("datasets/coco/annotations_trainval2017.zip"),
    )
    ap.add_argument(
        "--output",
        type=Path,
        default=Path("datasets/coco/annotations/instances_val2017.json"),
    )
    args = ap.parse_args()

    args.archive.parent.mkdir(parents=True, exist_ok=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    if args.archive.exists():
        actual = sha256_file(args.archive)
        if actual != EXPECTED_SHA256:
            print(
                "Existing annotation archive checksum mismatch; removing.\n"
                f"  expected: {EXPECTED_SHA256}\n"
                f"  actual:   {actual}",
                flush=True,
            )
            args.archive.unlink()

    if not args.archive.exists():
        print(
            f"Downloading {HF_REPO}/{HF_FILENAME} from Hugging Face mirror...",
            flush=True,
        )
        try:
            from huggingface_hub import hf_hub_download
        except Exception as exc:
            raise SystemExit(
                "huggingface_hub is required; it is normally installed by "
                "transformers in this project."
            ) from exc

        cached = Path(
            hf_hub_download(
                repo_id=HF_REPO,
                filename=HF_FILENAME,
                repo_type="dataset",
            )
        )
        print(f"HF download/cache complete: {cached}", flush=True)
        shutil.copyfile(cached, args.archive)

    actual = sha256_file(args.archive)
    if actual != EXPECTED_SHA256:
        raise SystemExit(
            "COCO annotation archive failed SHA256 verification.\n"
            f"Expected: {EXPECTED_SHA256}\n"
            f"Actual:   {actual}"
        )

    print(f"Verified annotation archive SHA256: {actual}", flush=True)

    with zipfile.ZipFile(args.archive) as zf:
        if MEMBER not in zf.namelist():
            raise SystemExit(f"Archive does not contain {MEMBER}")
        data = zf.read(MEMBER)
        args.output.write_bytes(data)

    parsed = json.loads(args.output.read_text())
    n_images = len(parsed.get("images", []))
    n_categories = len(parsed.get("categories", []))
    n_annotations = len(parsed.get("annotations", []))

    if n_images != 5000:
        raise SystemExit(f"Expected 5000 val2017 images; found {n_images}.")
    if n_categories != 80:
        raise SystemExit(f"Expected 80 COCO categories; found {n_categories}.")

    print(
        f"Extracted: {args.output}\n"
        f"  images: {n_images}\n"
        f"  categories: {n_categories}\n"
        f"  annotations: {n_annotations}\n"
        f"  json_sha256: {sha256_file(args.output)}",
        flush=True,
    )


if __name__ == "__main__":
    main()
