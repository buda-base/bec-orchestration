"""A/B a few pages: raw PaddleOCR vs header/footer-masked (paddleocr_v2).

Intended to run on the GPU worker (``paddlocr_job_v1``) after the vLLM engine
is available. Fetches selected pages from archive.tbrc.org, applies the same
preprocess + optional layout mask the job uses, OCRs both variants with one
loaded engine, and writes::

    <out_dir>/<stem>_before.jpg
    <out_dir>/<stem>_after.jpg
    <out_dir>/<stem>_crops.jpg     (top+bottom strips, before | after)
    <out_dir>/results.json

Usage (on the box, env sourced)::

    cd /home/ubuntu/bec-orchestration
    /opt/pytorch/bin/python -m bec_orch.jobs.paddleocr.abtest_layout_mask \\
        --out-dir /tmp/paddleocr_v2_ab
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import boto3
from PIL import Image, ImageDraw

from bec_orch.jobs.paddleocr.config import PaddleOCRConfig
from bec_orch.jobs.paddleocr.layout_mask import apply_header_footer_mask, removed_code, xywhn_to_xyxy
from bec_orch.jobs.paddleocr.layout_split import (
    crop_labeled_regions,
    max_region_margin_px,
    split_page_columns,
)
from bec_orch.jobs.paddleocr.postprocess import normalize_text
from bec_orch.jobs.paddleocr.preprocess import bytes_to_rgb
from bec_orch.jobs.paddleocr.worker import PaddleOCRJobWorker, _FetchedPage


# Pages chosen for visible headers and/or footers, plus one footnote page
# (footnote must survive). Coordinates come from layout_detection_v1 parquet.
PAGES = [
    {
        "w_id": "W00EGS1016047",
        "i_id": "I1KG80383",
        "filename": "I1KG803830009.jpg",
        "reason": "running header (no footnote)",
        "layout_uri": (
            "s3://bec.bdrc.io/layout_detection_v1/W00EGS1016047/I1KG80383/"
            "e71c17/W00EGS1016047-I1KG80383-e71c17.parquet"
        ),
    },
    {
        "w_id": "W00EGS1016047",
        "i_id": "I1KG80383",
        "filename": "I1KG803830010.jpg",
        "reason": "two header boxes (title + folio)",
        "layout_uri": (
            "s3://bec.bdrc.io/layout_detection_v1/W00EGS1016047/I1KG80383/"
            "e71c17/W00EGS1016047-I1KG80383-e71c17.parquet"
        ),
    },
    {
        "w_id": "W00EGS1015752",
        "i_id": "I1GS140243",
        "filename": "I1GS1402430015.tif",
        "reason": "header + footer folio number",
        "layout_uri": (
            "s3://bec.bdrc.io/layout_detection_v1/W00EGS1015752/I1GS140243/"
            "ff3bb3/W00EGS1015752-I1GS140243-ff3bb3.parquet"
        ),
    },
    {
        "w_id": "W00EGS1016213",
        "i_id": "I1KG176",
        "filename": "I1KG1760009.TIF",
        "reason": "footer + footnotes (footnotes isolated to footnote_text)",
        "layout_uri": (
            "s3://bec.bdrc.io/layout_detection_v1/W00EGS1016213/I1KG176/"
            "90b001/W00EGS1016213-I1KG176-90b001.parquet"
        ),
    },
    {
        "w_id": "W00EGS1016682",
        "i_id": "I01JW56",
        "filename": "I01JW560045.tif",
        "reason": "two-column text-areas (clean pair)",
        "layout_uri": (
            "s3://bec.bdrc.io/layout_detection_v1/W00EGS1016682/I01JW56/"
            "cee74d/W00EGS1016682-I01JW56-cee74d.parquet"
        ),
    },
    {
        "w_id": "W00EGS1016682",
        "i_id": "I01JW56",
        "filename": "I01JW560047.tif",
        "reason": "two columns + a full-width text-area (pair only)",
        "layout_uri": (
            "s3://bec.bdrc.io/layout_detection_v1/W00EGS1016682/I01JW56/"
            "cee74d/W00EGS1016682-I01JW56-cee74d.parquet"
        ),
    },
]


def _s3():
    return boto3.client("s3", region_name=os.environ.get("BEC_REGION", "us-east-1"))


def _parse_s3(uri: str) -> tuple[str, str]:
    assert uri.startswith("s3://")
    rest = uri[5:]
    bucket, _, key = rest.partition("/")
    return bucket, key


def _bdrc_prefix(w_id: str, i_id: str) -> str:
    from bec_orch.core.worker_runtime import get_s3_folder_prefix

    return get_s3_folder_prefix(w_id, i_id)


def _load_boxes(layout_uri: str, filename: str) -> list[dict]:
    import pyarrow.parquet as pq
    import s3fs

    path = layout_uri.replace("s3://", "")
    fs = s3fs.S3FileSystem()
    with fs.open(path, "rb") as f:
        table = pq.read_table(f, columns=["img_file_name", "boxes"])
    names = table.column("img_file_name").to_pylist()
    boxes_col = table.column("boxes").to_pylist()
    for name, boxes in zip(names, boxes_col):
        if name == filename:
            return [b for b in (boxes or []) if isinstance(b, dict)]
    raise KeyError(f"{filename} not in {layout_uri}")


def _fetch_bytes(w_id: str, i_id: str, filename: str) -> bytes:
    bucket = os.environ.get("BEC_SOURCE_S3_BUCKET", "archive.tbrc.org")
    key = f"{_bdrc_prefix(w_id, i_id)}{filename}"
    resp = _s3().get_object(Bucket=bucket, Key=key)
    return resp["Body"].read()


def _annotate(img: Image.Image, boxes: list[dict], *, labels: set[str]) -> Image.Image:
    """Draw box outlines on a copy (for the crop contact sheet)."""
    out = img.copy()
    draw = ImageDraw.Draw(out)
    w, h = out.size
    colors = {"header": (220, 40, 40), "footer": (40, 80, 220), "footnote": (40, 180, 40)}
    for b in boxes:
        lab = str(b.get("label") or "")
        if lab not in labels:
            continue
        xyxy = xywhn_to_xyxy(float(b["x"]), float(b["y"]), float(b["w"]), float(b["h"]), w, h)
        if xyxy is None:
            continue
        x1, y1, x2, y2 = xyxy
        draw.rectangle((x1, y1, x2 - 1, y2 - 1), outline=colors.get(lab, (0, 0, 0)), width=3)
    return out


def _crop_sheet(before: Image.Image, after: Image.Image, frac: float = 0.18) -> Image.Image:
    """Top and bottom strips, before | after, for a compact before/after view."""
    w, h = before.size
    band = max(32, int(h * frac))
    def strip(im: Image.Image) -> Image.Image:
        top = im.crop((0, 0, w, band))
        bot = im.crop((0, h - band, w, h))
        canvas = Image.new("RGB", (w, band * 2 + 8), (255, 255, 255))
        canvas.paste(top, (0, 0))
        canvas.paste(bot, (0, band + 8))
        return canvas
    left = strip(before)
    right = strip(after)
    gap = 12
    sheet = Image.new("RGB", (left.width * 2 + gap, left.height), (255, 255, 255))
    sheet.paste(left, (0, 0))
    sheet.paste(right, (left.width + gap, 0))
    return sheet


def _save_jpeg(img: Image.Image, path: Path, max_side: int = 1600) -> None:
    im = img.copy()
    im.thumbnail((max_side, max_side), Image.Resampling.BICUBIC)
    im.save(path, format="JPEG", quality=85)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", type=Path, default=Path("/tmp/paddleocr_v2_ab"))
    args = p.parse_args(argv)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    cfg = PaddleOCRConfig(
        layout_mask_enabled=True,
        layout_split_columns=True,
        layout_isolate_footnotes=True,
        ocr_batch_size=8,
    )
    print(f"loading engine from {cfg.resolved_model_dir()}", flush=True)
    worker = PaddleOCRJobWorker(cfg)

    results = []
    for spec in PAGES:
        fn = spec["filename"]
        stem = Path(fn).stem
        print(f"\n=== {spec['w_id']}/{spec['i_id']} {fn} ({spec['reason']}) ===", flush=True)
        boxes = _load_boxes(spec["layout_uri"], fn)
        labels = [b.get("label") for b in boxes]
        print(f"  layout boxes: {labels}", flush=True)
        data = _fetch_bytes(spec["w_id"], spec["i_id"], fn)
        before_img, res_scale = bytes_to_rgb(data, cfg)
        after_img, n_masked = apply_header_footer_mask(
            before_img,
            boxes,
            mask_labels=cfg.layout_mask_labels,
            protect_labels=cfg.layout_mask_protect_labels,
            pad_px=cfg.layout_mask_pad_px,
        )
        print(f"  painted {n_masked} boxes, res_scale={res_scale}", flush=True)

        outlined = _annotate(
            before_img, boxes, labels={"header", "footer", "footnote", "text-area"}
        )
        _save_jpeg(outlined, args.out_dir / f"{stem}_before.jpg")
        _save_jpeg(after_img, args.out_dir / f"{stem}_after.jpg")
        _save_jpeg(_crop_sheet(outlined, after_img), args.out_dir / f"{stem}_crops.jpg")

        footnote_images = crop_labeled_regions(
            after_img,
            boxes,
            label=cfg.layout_footnote_label,
            margin_frac=cfg.layout_footnote_margin_frac,
            min_px=cfg.layout_footnote_margin_min_px,
            box_margin_frac=cfg.layout_footnote_box_margin_frac,
        )
        n_footnotes = len(footnote_images)
        if footnote_images:
            for i, fn_im in enumerate(footnote_images):
                _save_jpeg(fn_im, args.out_dir / f"{stem}_fn{i}.jpg")
            fn_pad = max(
                cfg.layout_mask_pad_px,
                max_region_margin_px(
                    after_img.width,
                    after_img.height,
                    boxes,
                    label=cfg.layout_footnote_label,
                    margin_frac=cfg.layout_footnote_margin_frac,
                    min_px=cfg.layout_footnote_margin_min_px,
                    box_margin_frac=cfg.layout_footnote_box_margin_frac,
                ),
            )
            after_img, _ = apply_header_footer_mask(
                after_img,
                boxes,
                mask_labels=(cfg.layout_footnote_label,),
                protect_labels=("text-area",),
                pad_px=fn_pad,
            )
            _save_jpeg(after_img, args.out_dir / f"{stem}_after.jpg")
            print(f"  isolated {n_footnotes} footnotes", flush=True)

        column_images = split_page_columns(
            after_img,
            boxes,
            label=cfg.layout_column_label,
            min_vert_overlap=cfg.layout_column_min_vert_overlap,
            max_horiz_overlap=cfg.layout_column_max_horiz_overlap,
            margin_frac=cfg.layout_column_margin_frac,
        )
        n_columns = len(column_images) if column_images else 1
        if column_images:
            for i, col in enumerate(column_images):
                _save_jpeg(col, args.out_dir / f"{stem}_col{i}.jpg")
            print(f"  split into {n_columns} columns", flush=True)

        removed = removed_code(boxes)
        pages = [
            _FetchedPage(filename=fn, etag="", image=before_img, res_scale=res_scale),
            _FetchedPage(
                filename=fn,
                etag="",
                image=after_img,
                res_scale=res_scale,
                layout_masked=n_masked > 0,
                n_masked_boxes=n_masked,
                column_images=column_images,
                n_columns=n_columns,
                footnote_images=footnote_images or None,
                n_footnotes=n_footnotes,
                removed=removed,
            ),
        ]
        ocr = worker._ocr_batch(pages)
        before_text = normalize_text(ocr[0].raw_text)
        after_text = normalize_text(ocr[1].raw_text)
        footnote_text = normalize_text(ocr[1].footnote_text)
        rec = {
            "w_id": spec["w_id"],
            "i_id": spec["i_id"],
            "filename": fn,
            "reason": spec["reason"],
            "n_masked_boxes": n_masked,
            "n_columns": n_columns,
            "n_footnotes": n_footnotes,
            "removed": removed,
            "layout_labels": labels,
            "before_text": before_text,
            "after_text": after_text,
            "footnote_text": footnote_text,
            "before_tokens": ocr[0].output_tokens,
            "after_tokens": ocr[1].output_tokens,
            "unchanged": before_text == after_text,
        }
        results.append(rec)
        print(f"  removed={removed} footnotes={n_footnotes}", flush=True)
        print("  BEFORE:", before_text[:240].replace("\n", " / "), flush=True)
        print("  AFTER: ", after_text[:240].replace("\n", " / "), flush=True)
        if footnote_text:
            print("  FOOTNOTE:", footnote_text[:240].replace("\n", " / "), flush=True)

    out_json = args.out_dir / "results.json"
    out_json.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nwrote {out_json}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
