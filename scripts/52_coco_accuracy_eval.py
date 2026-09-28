#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.util
import io
import json
import sqlite3
import sys
import tempfile
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoModelForZeroShotObjectDetection

EXPECTED_VAL2017_SHA256 = (
    "4f7e2ccb2866ec5041993c9cf2a952bbed69647b115d0f74da7ce8f4bef82f05"
)
CANVAS_WH = (1200, 901)
CANVAS_FILL = (114, 114, 114)
NUM_SELECT = 300
PRIMARY_EQUIVALENCE_MARGIN_AP_POINTS = 0.10


def load_module(path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load helper: {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod


def sha256_file(path: Path, chunk_size: int = 1024 * 1024):
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk_size)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def letterbox(image: Image.Image, size_wh=CANVAS_WH):
    W, H = size_wh
    image = image.convert("RGB")
    w, h = image.size
    scale = min(W / w, H / h)
    nw = max(1, int(round(w * scale)))
    nh = max(1, int(round(h * scale)))
    resized = image.resize((nw, nh), Image.Resampling.LANCZOS)
    left = (W - nw) // 2
    top = (H - nh) // 2
    canvas = Image.new("RGB", (W, H), CANVAS_FILL)
    canvas.paste(resized, (left, top))
    return canvas, {
        "source_size_wh": [w, h],
        "canvas_size_wh": [W, H],
        "resized_size_wh": [nw, nh],
        "padding_ltrb": [left, top, W - nw - left, H - nh - top],
        # Use exact realized pixel scales when mapping predictions back.
        "scale_x": nw / w,
        "scale_y": nh / h,
    }


def build_caption_and_spans(categories):
    """
    Reproduce GroundingDINO's official build_captions_and_token_span behavior
    for COCO category names. COCO names do not contain '/' alternatives.
    """
    caption = ""
    spans_by_catid = {}

    for cat in categories:
        class_name = cat["name"].lower()
        token_spans = []
        for subname in [x.strip() for x in class_name.strip().split(" ")]:
            if not subname:
                continue
            if caption:
                caption += " "
            beg = len(caption)
            end = beg + len(subname)
            token_spans.append([beg, end])
            caption += subname
        if token_spans:
            caption += " ."
            spans_by_catid[int(cat["id"])] = token_spans

    return caption, spans_by_catid


def create_positive_map(tokenizer, caption, spans_by_catid, max_text_len=256):
    tokenized = tokenizer(caption, return_tensors="pt")
    max_cat_id = max(spans_by_catid)
    positive_map = torch.zeros(
        (max_cat_id + 1, max_text_len),
        dtype=torch.float32,
    )

    for cat_id, spans in spans_by_catid.items():
        for beg, end in spans:
            beg_pos = tokenized.char_to_token(beg)
            end_pos = tokenized.char_to_token(end - 1)

            if beg_pos is None:
                for delta in (1, 2):
                    beg_pos = tokenized.char_to_token(beg + delta)
                    if beg_pos is not None:
                        break
            if end_pos is None:
                for delta in (2, 3):
                    end_pos = tokenized.char_to_token(end - delta)
                    if end_pos is not None:
                        break
            if beg_pos is None or end_pos is None:
                raise RuntimeError(
                    f"Could not map category {cat_id} character span "
                    f"[{beg}, {end}) to tokenizer positions."
                )

            positive_map[cat_id, beg_pos : end_pos + 1] = 1.0

    denom = positive_map.sum(-1, keepdim=True) + 1e-6
    positive_map = positive_map / denom
    return positive_map, tokenized


def cxcywh_to_xyxy(boxes):
    cx, cy, w, h = boxes.unbind(-1)
    return torch.stack(
        [cx - 0.5 * w, cy - 0.5 * h, cx + 0.5 * w, cy + 0.5 * h],
        dim=-1,
    )


def outputs_to_coco_predictions(
    outputs,
    image_id,
    geom,
    positive_map_mps,
    valid_category_ids,
    num_select=NUM_SELECT,
):
    logits = outputs.logits.detach().float()
    boxes_norm = outputs.pred_boxes.detach().float()

    if logits.shape[0] != 1 or boxes_norm.shape[0] != 1:
        raise RuntimeError("0052 currently expects batch size 1.")
    if logits.shape[-1] != positive_map_mps.shape[-1]:
        raise RuntimeError(
            f"Text logit width {logits.shape[-1]} != "
            f"positive-map width {positive_map_mps.shape[-1]}"
        )

    prob_to_token = logits.sigmoid()[0]  # [900, 256]
    prob_to_label = prob_to_token @ positive_map_mps.T  # [900, 91]

    flat = prob_to_label.flatten()
    values, indexes = torch.topk(flat, num_select)
    n_labels = prob_to_label.shape[1]
    query_idx = indexes // n_labels
    category_ids = indexes % n_labels

    cat_cpu = category_ids.detach().cpu().tolist()
    bad = [int(x) for x in cat_cpu if int(x) not in valid_category_ids]
    if bad:
        raise RuntimeError(
            "Official-style top-k selected a non-COCO category-id row. "
            f"Unexpected ids: {sorted(set(bad))}"
        )

    boxes = boxes_norm[0][query_idx]
    boxes = cxcywh_to_xyxy(boxes)

    W, H = geom["canvas_size_wh"]
    scale = torch.tensor(
        [W, H, W, H],
        dtype=boxes.dtype,
        device=boxes.device,
    )
    boxes = boxes * scale
    boxes_cpu = boxes.detach().cpu()
    scores_cpu = values.detach().cpu().tolist()

    left, top, _, _ = geom["padding_ltrb"]
    sx = geom["scale_x"]
    sy = geom["scale_y"]
    src_w, src_h = geom["source_size_wh"]

    records = []
    for score, cat_id, box in zip(scores_cpu, cat_cpu, boxes_cpu.tolist()):
        x1, y1, x2, y2 = box
        x1 = (x1 - left) / sx
        x2 = (x2 - left) / sx
        y1 = (y1 - top) / sy
        y2 = (y2 - top) / sy

        x1 = min(max(x1, 0.0), float(src_w))
        x2 = min(max(x2, 0.0), float(src_w))
        y1 = min(max(y1, 0.0), float(src_h))
        y2 = min(max(y2, 0.0), float(src_h))

        if x2 < x1:
            x1, x2 = x2, x1
        if y2 < y1:
            y1, y2 = y2, y1

        records.append(
            {
                "image_id": int(image_id),
                "category_id": int(cat_id),
                "bbox": [
                    float(x1),
                    float(y1),
                    float(max(0.0, x2 - x1)),
                    float(max(0.0, y2 - y1)),
                ],
                "score": float(score),
            }
        )

    if len(records) != num_select:
        raise RuntimeError(
            f"Expected {num_select} predictions; got {len(records)}."
        )
    return records


def make_inputs(processor, image, caption):
    cpu = processor(
        images=image,
        text=caption,
        return_tensors="pt",
    )
    return {k: v.to("mps") for k, v in cpu.items()}


def config_hash(config):
    raw = json.dumps(config, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(raw).hexdigest()


def open_checkpoint(path: Path, cfg):
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS meta (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS image_results (
            image_id INTEGER PRIMARY KEY,
            original_json TEXT NOT NULL,
            final_json TEXT NOT NULL
        )
        """
    )
    h = config_hash(cfg)
    row = conn.execute(
        "SELECT value FROM meta WHERE key='config_hash'"
    ).fetchone()
    if row is None:
        conn.execute(
            "INSERT INTO meta(key, value) VALUES('config_hash', ?)",
            (h,),
        )
        conn.execute(
            "INSERT INTO meta(key, value) VALUES('config_json', ?)",
            (json.dumps(cfg, sort_keys=True),),
        )
        conn.commit()
    elif row[0] != h:
        raise RuntimeError(
            f"Checkpoint {path} belongs to a different 0052 configuration. "
            "Use a different --output filename or remove the checkpoint."
        )
    return conn


def save_checkpoint_image(conn, image_id, original_preds, final_preds):
    conn.execute(
        """
        INSERT OR REPLACE INTO image_results(
            image_id, original_json, final_json
        ) VALUES (?, ?, ?)
        """,
        (
            int(image_id),
            json.dumps(original_preds, separators=(",", ":")),
            json.dumps(final_preds, separators=(",", ":")),
        ),
    )
    conn.commit()


def completed_image_ids(conn):
    return {
        int(row[0])
        for row in conn.execute("SELECT image_id FROM image_results")
    }


def load_predictions_from_checkpoint(conn, image_ids, side):
    if side not in ("original_json", "final_json"):
        raise ValueError(side)
    all_preds = []
    for image_id in image_ids:
        row = conn.execute(
            f"SELECT {side} FROM image_results WHERE image_id=?",
            (int(image_id),),
        ).fetchone()
        if row is None:
            raise RuntimeError(f"Missing checkpoint row for image {image_id}")
        all_preds.extend(json.loads(row[0]))
    return all_preds


def evaluate_coco(coco_gt, predictions, image_ids, categories):
    from pycocotools.cocoeval import COCOeval

    coco_dt = coco_gt.loadRes(predictions)
    ev = COCOeval(coco_gt, coco_dt, iouType="bbox")
    ev.params.imgIds = [int(x) for x in image_ids]
    ev.params.catIds = [int(c["id"]) for c in categories]
    ev.evaluate()
    ev.accumulate()
    ev.summarize()

    stat_names = [
        "AP",
        "AP50",
        "AP75",
        "AP_small",
        "AP_medium",
        "AP_large",
        "AR_maxDets1",
        "AR_maxDets10",
        "AR_maxDets100",
        "AR_small",
        "AR_medium",
        "AR_large",
    ]
    stats = {
        name: float(value)
        for name, value in zip(stat_names, ev.stats.tolist())
    }

    # precision: [IoU, recall, category, area, maxDets]
    precision = ev.eval["precision"]
    cat_ids = list(ev.params.catIds)
    per_category = {}
    for k, cat_id in enumerate(cat_ids):
        p = precision[:, :, k, 0, -1]
        p = p[p > -1]
        ap = float(p.mean()) if p.size else float("nan")
        cat = next(c for c in categories if int(c["id"]) == int(cat_id))
        per_category[str(cat_id)] = {
            "name": cat["name"],
            "AP": ap,
        }

    return stats, per_category


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--image-archive",
        type=Path,
        default=Path("datasets/coco/val2017.zip"),
    )
    ap.add_argument(
        "--annotations",
        type=Path,
        default=Path("datasets/coco/annotations/instances_val2017.json"),
    )
    ap.add_argument(
        "--output",
        type=Path,
        default=Path("results/metalground_coco_accuracy_0052.json"),
    )
    ap.add_argument("--limit-images", type=int)
    ap.add_argument(
        "--model",
        default="IDEA-Research/grounding-dino-tiny",
    )
    ap.add_argument("--threadgroup", type=int, default=256)

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

    if not args.image_archive.exists():
        raise SystemExit(
            f"Missing {args.image_archive}. Reuse the verified val2017.zip "
            "from Experiment 0051 preparation."
        )
    if not args.annotations.exists():
        raise SystemExit(
            f"Missing {args.annotations}. Run "
            "scripts/52_prepare_coco_annotations.py first."
        )
    for n in helpers:
        p = getattr(args, f"exp{n}_helper")
        if not p.exists():
            raise SystemExit(f"Missing helper: {p}")
    if not torch.backends.mps.is_available():
        raise SystemExit("MPS unavailable.")

    try:
        from pycocotools.coco import COCO
    except Exception as exc:
        raise SystemExit(
            "pycocotools is required. Add the reproducible dependency with:\n"
            "  uv add 'pycocotools==2.0.11'"
        ) from exc

    print("Verifying COCO val2017 archive...", flush=True)
    image_archive_sha = sha256_file(args.image_archive)
    if image_archive_sha != EXPECTED_VAL2017_SHA256:
        raise SystemExit(
            "val2017.zip SHA256 mismatch.\n"
            f"Expected: {EXPECTED_VAL2017_SHA256}\n"
            f"Actual:   {image_archive_sha}"
        )

    annotation_sha = sha256_file(args.annotations)
    coco_gt = COCO(str(args.annotations))
    categories = list(coco_gt.dataset["categories"])
    if len(categories) != 80:
        raise SystemExit(f"Expected 80 categories; got {len(categories)}")

    image_ids = sorted(int(x) for x in coco_gt.getImgIds())
    if args.limit_images is not None:
        if args.limit_images < 1:
            raise SystemExit("--limit-images must be >=1")
        image_ids = image_ids[: args.limit_images]
    if not image_ids:
        raise SystemExit("No COCO images selected.")

    caption, spans_by_catid = build_caption_and_spans(categories)
    valid_category_ids = {int(c["id"]) for c in categories}

    print(
        f"COCO images selected: {len(image_ids)}\n"
        f"COCO categories: {len(categories)}\n"
        f"Official-style caption chars: {len(caption)}",
        flush=True,
    )

    with zipfile.ZipFile(args.image_archive) as zf:
        first_info = coco_gt.loadImgs([image_ids[0]])[0]
        member = f"val2017/{first_info['file_name']}"
        first_raw = zf.read(member)
        first_image = Image.open(io.BytesIO(first_raw)).convert("RGB")
        first_canvas, first_geom = letterbox(first_image)

        with tempfile.TemporaryDirectory(prefix="metalground_0052_") as td:
            temp_image = Path(td) / "first_canvas.png"
            first_canvas.save(temp_image, format="PNG", optimize=False)

            # build_runtime expects these historical fields.
            args.image = temp_image
            # Initial build prompt only establishes runtime state. The exact
            # official-style caption is reprocessed and the exact text cache
            # is re-primed below before any evaluated inference.
            args.prompt = [c["name"].lower() for c in categories]

            h37 = load_module(args.exp37_helper, "mg52_h37")
            h38 = load_module(args.exp38_helper, "mg52_h38")
            h45 = load_module(args.exp45_helper, "mg52_h45")
            h46 = load_module(args.exp46_helper, "mg52_h46")

            print("Building adopted MetalGround runtime...", flush=True)
            rt = h37.build_runtime(args)
            final_model = rt["model"]
            processor = rt["processor"]
            h14 = rt["h14"]

            positive_map, tokenized = create_positive_map(
                processor.tokenizer,
                caption,
                spans_by_catid,
                max_text_len=256,
            )
            prompt_token_count = int(tokenized["attention_mask"].sum().item())
            print(
                f"Official-style prompt token count: {prompt_token_count}/256",
                flush=True,
            )
            if prompt_token_count > 256:
                raise RuntimeError("COCO prompt exceeds 256-token model limit.")

            cache = h38.ExactTextBackboneCache(
                final_model.model.text_backbone
            )
            stage0_dispatchers = []

            try:
                h38.set_runtime_mode(rt, cache, "wide_cache")

                stages = (
                    final_model.model.backbone.conv_encoder.model.swin.encoder.layers
                )
                stage0_modules = [
                    stages[0].blocks[0].mlp,
                    stages[0].blocks[1].mlp,
                ]
                h14.sync()
                stage0_candidates = [
                    h45.MlxSwinMLP(m) for m in stage0_modules
                ]
                stage0_dispatchers = [
                    h46.SwinMlpDispatcher(m, c, h14.sync)
                    for m, c in zip(stage0_modules, stage0_candidates)
                ]
                for d in stage0_dispatchers:
                    d.set_mode("mlx")

                print("Loading untouched original model...", flush=True)
                original_model = (
                    AutoModelForZeroShotObjectDetection
                    .from_pretrained(args.model)
                    .eval()
                    .to("mps")
                )
                h14.sync()

                first_inputs = make_inputs(
                    processor,
                    first_canvas,
                    caption,
                )

                # Exact prompt/tokenization identity check.
                if not torch.equal(
                    first_inputs["input_ids"].detach().cpu(),
                    tokenized["input_ids"],
                ):
                    raise RuntimeError(
                        "Processor input_ids differ from tokenizer input_ids "
                        "used to construct the official positive map."
                    )

                base_pixel_shape = tuple(first_inputs["pixel_values"].shape)
                if base_pixel_shape != (1, 3, 800, 1065):
                    raise RuntimeError(
                        "Unexpected fixed processor geometry: "
                        f"{base_pixel_shape}; expected (1,3,800,1065)."
                    )

                cache.cached_output = None
                cache.prime(final_model, first_inputs, h14.sync)
                cache.set_enabled(True)

                positive_map_mps = positive_map.to("mps")
                h14.sync()

                config = {
                    "experiment": "0052",
                    "model": args.model,
                    "image_archive_sha256": image_archive_sha,
                    "annotations_sha256": annotation_sha,
                    "image_ids": image_ids,
                    "canvas_wh": list(CANVAS_WH),
                    "canvas_fill": list(CANVAS_FILL),
                    "processor_pixel_shape": list(base_pixel_shape),
                    "caption": caption,
                    "category_ids": [int(c["id"]) for c in categories],
                    "num_select": NUM_SELECT,
                    "dtype": "float32",
                    "device": "mps",
                }
                checkpoint = args.output.with_suffix(
                    args.output.suffix + ".checkpoint.sqlite3"
                )
                conn = open_checkpoint(checkpoint, config)
                done = completed_image_ids(conn)
                if done:
                    print(
                        f"Resuming checkpoint: {len(done)} images already complete.",
                        flush=True,
                    )

                for i, image_id in enumerate(image_ids, start=1):
                    if image_id in done:
                        continue

                    info = coco_gt.loadImgs([image_id])[0]
                    member = f"val2017/{info['file_name']}"
                    raw = zf.read(member)
                    image = Image.open(io.BytesIO(raw)).convert("RGB")
                    canvas, geom = letterbox(image)
                    inputs = make_inputs(processor, canvas, caption)

                    if tuple(inputs["pixel_values"].shape) != base_pixel_shape:
                        raise RuntimeError(
                            f"Fixed-geometry invariant failed for {image_id}: "
                            f"{tuple(inputs['pixel_values'].shape)}"
                        )
                    if not torch.equal(
                        inputs["input_ids"],
                        first_inputs["input_ids"],
                    ):
                        raise RuntimeError(
                            f"Prompt tokenization changed at image {image_id}."
                        )

                    print(
                        f"[{i}/{len(image_ids)}] image_id={image_id}",
                        flush=True,
                    )

                    with torch.inference_mode():
                        original_out = original_model(**inputs)
                    h14.sync()

                    with torch.inference_mode():
                        final_out = final_model(**inputs)
                    h14.sync()

                    original_preds = outputs_to_coco_predictions(
                        original_out,
                        image_id,
                        geom,
                        positive_map_mps,
                        valid_category_ids,
                    )
                    final_preds = outputs_to_coco_predictions(
                        final_out,
                        image_id,
                        geom,
                        positive_map_mps,
                        valid_category_ids,
                    )

                    save_checkpoint_image(
                        conn,
                        image_id,
                        original_preds,
                        final_preds,
                    )

                    del original_out, final_out, inputs
                    if i % 100 == 0:
                        gc.collect()

                missing = set(image_ids) - completed_image_ids(conn)
                if missing:
                    raise RuntimeError(
                        f"Checkpoint incomplete; missing {len(missing)} images."
                    )

                # Inference complete. Release model references before loading
                # all COCO predictions for evaluation.
                for d in stage0_dispatchers:
                    d.restore()
                stage0_dispatchers = []
                cache.restore()
                for d in rt["layer_dispatchers"]:
                    d.restore()
                for fd in rt["fusion_dispatchers"]:
                    fd.restore()

                del original_model, final_model, positive_map_mps, rt
                h14.sync()
                try:
                    torch.mps.empty_cache()
                except Exception:
                    pass
                gc.collect()

                args.output.parent.mkdir(parents=True, exist_ok=True)
                stem = args.output.stem
                original_pred_path = args.output.with_name(
                    stem + "_original_predictions.json"
                )
                final_pred_path = args.output.with_name(
                    stem + "_final_predictions.json"
                )

                print("Evaluating untouched original...", flush=True)
                original_preds = load_predictions_from_checkpoint(
                    conn, image_ids, "original_json"
                )
                original_pred_path.write_text(
                    json.dumps(original_preds, separators=(",", ":"))
                )
                original_stats, original_per_category = evaluate_coco(
                    coco_gt, original_preds, image_ids, categories
                )
                del original_preds
                gc.collect()

                print("Evaluating MetalGround final...", flush=True)
                final_preds = load_predictions_from_checkpoint(
                    conn, image_ids, "final_json"
                )
                final_pred_path.write_text(
                    json.dumps(final_preds, separators=(",", ":"))
                )
                final_stats, final_per_category = evaluate_coco(
                    coco_gt, final_preds, image_ids, categories
                )
                del final_preds
                gc.collect()

                metric_deltas_ap_points = {
                    key: (final_stats[key] - original_stats[key]) * 100.0
                    for key in original_stats
                }

                category_deltas = {}
                max_abs_cat = None
                for cat_id in original_per_category:
                    o = original_per_category[cat_id]
                    f = final_per_category[cat_id]
                    delta = (f["AP"] - o["AP"]) * 100.0
                    category_deltas[cat_id] = {
                        "name": o["name"],
                        "original_AP": o["AP"],
                        "final_AP": f["AP"],
                        "delta_AP_points": delta,
                    }
                    if max_abs_cat is None or abs(delta) > abs(
                        max_abs_cat["delta_AP_points"]
                    ):
                        max_abs_cat = {
                            "category_id": int(cat_id),
                            "name": o["name"],
                            "delta_AP_points": delta,
                        }

                full_dataset = len(image_ids) == 5000
                primary_delta = metric_deltas_ap_points["AP"]
                gate_pass = (
                    abs(primary_delta)
                    <= PRIMARY_EQUIVALENCE_MARGIN_AP_POINTS
                )

                result = {
                    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                    "experiment": "0052",
                    "purpose": (
                        "Ground-truth COCO bbox accuracy comparison between "
                        "untouched PyTorch/MPS Grounding DINO and adopted "
                        "MetalGround under the fixed webcam-style geometry."
                    ),
                    "scope": (
                        "full_coco_val2017"
                        if full_dataset
                        else "smoke_subset_not_for_accuracy_claim"
                    ),
                    "model": args.model,
                    "device": "mps",
                    "dtype": "float32",
                    "images_evaluated": len(image_ids),
                    "configuration": {
                        "image_archive": str(args.image_archive),
                        "image_archive_sha256": image_archive_sha,
                        "annotations": str(args.annotations),
                        "annotations_sha256": annotation_sha,
                        "canvas_wh": list(CANVAS_WH),
                        "canvas_fill_rgb": list(CANVAS_FILL),
                        "processor_pixel_values_shape": list(base_pixel_shape),
                        "prompt_protocol": (
                            "GroundingDINO official COCO zero-shot style: "
                            "80 lower-case COCO category names in one caption, "
                            "token-span normalized positive map, sigmoid token "
                            "logits projected to category scores, top-300 over "
                            "query x category."
                        ),
                        "prompt_caption": caption,
                        "prompt_token_count": prompt_token_count,
                        "num_select": NUM_SELECT,
                        "performance_timing": False,
                        "approximation": False,
                        "retraining": False,
                        "reduced_precision": False,
                        "absolute_AP_protocol_note": (
                            "Input geometry is MetalGround's fixed 1200x901 "
                            "letterbox, not the official repository's native-"
                            "aspect RandomResize evaluation. Therefore absolute "
                            "AP is protocol-specific; the primary scientific "
                            "quantity is the matched original-vs-final delta."
                        ),
                    },
                    "pre_registered_primary_gate": {
                        "metric": "COCO bbox AP@[0.50:0.95]",
                        "equivalence_margin_AP_points": (
                            PRIMARY_EQUIVALENCE_MARGIN_AP_POINTS
                        ),
                        "definition": (
                            "PASS iff absolute(final-original) AP delta is "
                            "<= 0.10 AP points on the full 5000-image val2017."
                        ),
                        "applicable": full_dataset,
                        "pass": gate_pass if full_dataset else None,
                    },
                    "original": {
                        "metrics": original_stats,
                        "per_category": original_per_category,
                        "predictions_file": str(original_pred_path),
                    },
                    "final": {
                        "metrics": final_stats,
                        "per_category": final_per_category,
                        "predictions_file": str(final_pred_path),
                    },
                    "delta_AP_points_final_minus_original": (
                        metric_deltas_ap_points
                    ),
                    "per_category_delta": category_deltas,
                    "max_absolute_category_AP_delta": max_abs_cat,
                    "checkpoint": str(checkpoint),
                    "notes": [
                        (
                            "Subset/smoke AP is harness validation only and "
                            "must not be used for a dataset-level accuracy claim."
                        ),
                        (
                            "COCO boxes are mapped from the fixed letterbox "
                            "canvas back to original-image coordinates using "
                            "the exact realized x/y resize scales."
                        ),
                    ],
                }

                args.output.write_text(
                    json.dumps(result, indent=2, ensure_ascii=False) + "\n"
                )

                print("\n=== Experiment 0052 summary ===", flush=True)
                print(
                    f"images: {len(image_ids)} "
                    f"({'FULL' if full_dataset else 'SMOKE'})",
                    flush=True,
                )
                print(
                    "original AP: "
                    f"{original_stats['AP'] * 100:.6f}",
                    flush=True,
                )
                print(
                    "final AP:    "
                    f"{final_stats['AP'] * 100:.6f}",
                    flush=True,
                )
                print(
                    "delta AP points (final-original): "
                    f"{primary_delta:+.6f}",
                    flush=True,
                )
                if full_dataset:
                    print(
                        "primary equivalence gate: "
                        f"{'PASS' if gate_pass else 'FAIL'}",
                        flush=True,
                    )
                else:
                    print(
                        "primary equivalence gate: N/A on smoke subset",
                        flush=True,
                    )
                print(f"Saved: {args.output}", flush=True)

                conn.close()

            finally:
                for d in stage0_dispatchers:
                    try:
                        d.restore()
                    except Exception:
                        pass
                try:
                    cache.restore()
                except Exception:
                    pass
                try:
                    for d in rt["layer_dispatchers"]:
                        d.restore()
                    for fd in rt["fusion_dispatchers"]:
                        fd.restore()
                except Exception:
                    pass


if __name__ == "__main__":
    main()
