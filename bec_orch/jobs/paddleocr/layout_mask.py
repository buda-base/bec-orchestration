"""Paint header/footer regions with page background before OCR.

``paddleocr_v2`` consumes the sibling ``layout_detection_v1`` parquet (same
volume + version; only the job-name path segment differs) and fills detected
**header** and **footer** boxes with an estimate of the page background colour
so running titles and folio numbers don't leak into the transcription.

Footnotes are **not** painted in the header/footer pass (they are cropped and
OCR'd separately when isolation is on). Text-area boxes are subtracted from
the mask when they overlap a header/footer, so an oversized header box cannot
wipe body text. If the layout artifact is missing, OCR proceeds on the
unmodified page.
"""

from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING, Any, Iterable

from PIL import Image, ImageDraw

if TYPE_CHECKING:
    from .config import PaddleOCRConfig

logger = logging.getLogger(__name__)

# Default class names as emitted by layout_detection_v1 (checkpoint order
# 0 header, 1 text-area, 2 footnote, 3 footer). Matching is case-insensitive.
DEFAULT_MASK_LABELS = ("header", "footer")
DEFAULT_PROTECT_LABELS = ("text-area", "footnote")

# Fallback when a box has no ``label``: layout_detection_v1 class ids.
_CLS_TO_LABEL = {0: "header", 1: "text-area", 2: "footnote", 3: "footer"}


def sibling_layout_uri(bucket: str, ocr_prefix: str, basename: str, sibling_job: str) -> str:
    """Locate the layout job's parquet for the same volume+version.

    ``ocr_prefix`` looks like ``paddleocr_v2/<W>/<I>/<version>``; we swap the
    leading job-name segment for ``sibling_job``.
    """
    from .filter import sibling_parquet_uri

    return sibling_parquet_uri(bucket, ocr_prefix, basename, sibling_job)


def load_layout_map(
    bucket: str, ocr_prefix: str, basename: str, cfg: PaddleOCRConfig
) -> tuple[dict[str, list[dict[str, Any]]], bool]:
    """Load ``{filename: boxes}`` from the sibling layout parquet.

    Returns ``(layout_map, found)``. Missing / unreadable artifacts yield
    ``({}, False)``; the caller decides whether that's fatal.
    """
    import pyarrow.parquet as pq
    import s3fs

    uri = sibling_layout_uri(bucket, ocr_prefix, basename, cfg.layout_mask_job_name)
    path = uri.replace("s3://", "")
    fs = s3fs.S3FileSystem()

    try:
        exists = fs.exists(path)
    except Exception as e:  # noqa: BLE001 — treat listing errors as "not found"
        logger.warning(f"[paddleocr.layout] could not stat {uri}: {e}")
        return {}, False
    if not exists:
        logger.warning(f"[paddleocr.layout] no layout artifact at {uri}")
        return {}, False

    with fs.open(path, "rb") as f:
        table = pq.read_table(f, columns=["img_file_name", "status", "boxes"])

    names = table.column("img_file_name").to_pylist()
    statuses = table.column("status").to_pylist()
    boxes_col = table.column("boxes").to_pylist()

    layout_map: dict[str, list[dict[str, Any]]] = {}
    n_with_boxes = 0
    for name, status, boxes in zip(names, statuses, boxes_col, strict=False):
        if not name or status != "ok":
            continue
        page_boxes = [b for b in (boxes or []) if isinstance(b, dict)]
        layout_map[name] = page_boxes
        if page_boxes:
            n_with_boxes += 1

    logger.info(
        f"[paddleocr.layout] {uri}: {len(layout_map)} ok pages "
        f"({n_with_boxes} with detections)"
    )
    return layout_map, True


def xywhn_to_xyxy(
    x: float,
    y: float,
    w: float,
    h: float,
    img_w: int,
    img_h: int,
    pad_px: int = 0,
) -> tuple[int, int, int, int] | None:
    """Convert a centre-format normalized box to clamped pixel ``(x1,y1,x2,y2)``.

    ``x``/``y``/``w``/``h`` are Ultralytics ``xywhn`` (centre, width, height in
    ``[0,1]``). Normalized coords are scale-invariant, so ``img_w``/``img_h``
    may be the decoded (possibly downscaled) image size rather than the
    original scan.
    """
    if img_w <= 0 or img_h <= 0 or w <= 0 or h <= 0:
        return None
    pad = max(0, int(pad_px))
    x1 = (float(x) - float(w) / 2.0) * img_w - pad
    y1 = (float(y) - float(h) / 2.0) * img_h - pad
    x2 = (float(x) + float(w) / 2.0) * img_w + pad
    y2 = (float(y) + float(h) / 2.0) * img_h + pad
    ix1 = max(0, min(img_w, int(math.floor(x1))))
    iy1 = max(0, min(img_h, int(math.floor(y1))))
    ix2 = max(0, min(img_w, int(math.ceil(x2))))
    iy2 = max(0, min(img_h, int(math.ceil(y2))))
    if ix2 <= ix1 or iy2 <= iy1:
        return None
    return ix1, iy1, ix2, iy2


def estimate_background_rgb(img: Image.Image) -> tuple[int, int, int]:
    """Estimate page-paper colour from an inset border strip.

    The outer ~1% is skipped (scanner black edges); the next ~4% is sampled.
    Dark pixels (text, scan border leftovers) are dropped by keeping the
    lighter 60% by luminance, then taking the per-channel median. Falls back
    to a light cream if sampling fails.
    """
    import numpy as np

    arr = np.asarray(img.convert("RGB"), dtype=np.uint8)
    h, w = arr.shape[:2]
    if h < 8 or w < 8:
        return (245, 242, 232)

    inset = max(1, int(min(h, w) * 0.01))
    band = max(2, int(min(h, w) * 0.04))
    y0, y1 = inset, min(h, inset + band)
    x0, x1 = inset, min(w, inset + band)
    y2, y3 = max(0, h - inset - band), max(0, h - inset)
    x2, x3 = max(0, w - inset - band), max(0, w - inset)

    strips = [
        arr[y0:y1, x0:w - inset, :],
        arr[y2:y3, x0:w - inset, :],
        arr[y0:h - inset, x0:x1, :],
        arr[y0:h - inset, x2:x3, :],
    ]
    samples = np.concatenate([s.reshape(-1, 3) for s in strips if s.size], axis=0)
    if samples.size == 0:
        return (245, 242, 232)

    lum = samples.astype(np.float32).mean(axis=1)
    floor = float(np.percentile(lum, 40.0))
    kept = samples[lum >= floor]
    if kept.size == 0:
        kept = samples
    med = np.median(kept.astype(np.float32), axis=0)
    return (int(med[0]), int(med[1]), int(med[2]))


def resolved_label(box: dict[str, Any]) -> str:
    """Class name for a layout box (``label``, else checkpoint class id)."""
    raw = box.get("label")
    if isinstance(raw, str) and raw.strip():
        return raw.strip().lower()
    try:
        return _CLS_TO_LABEL.get(int(box.get("cls")), "")
    except (TypeError, ValueError):
        return ""


def removed_code(boxes: Iterable[dict[str, Any]] | None) -> str:
    """What header/footer content was left out of the main OCR.

    Returns one of ``none``, ``h``, ``f``, ``hf`` based on which of those
    classes are present in ``boxes`` (the regions we paint with background).
    Footnotes are tracked separately and do not appear here.
    """
    labels = {resolved_label(b) for b in (boxes or [])}
    has_h = "header" in labels
    has_f = "footer" in labels
    if has_h and has_f:
        return "hf"
    if has_h:
        return "h"
    if has_f:
        return "f"
    return "none"


def _rects_from_boxes(
    boxes: Iterable[dict[str, Any]],
    img_w: int,
    img_h: int,
    pad_px: int,
    pred,
) -> list[tuple[int, int, int, int]]:
    rects: list[tuple[int, int, int, int]] = []
    for box in boxes:
        if not pred(box):
            continue
        try:
            x, y, w, h = float(box["x"]), float(box["y"]), float(box["w"]), float(box["h"])
        except (KeyError, TypeError, ValueError):
            continue
        xyxy = xywhn_to_xyxy(x, y, w, h, img_w, img_h, pad_px=pad_px)
        if xyxy is not None:
            rects.append(xyxy)
    return rects


def apply_header_footer_mask(
    img: Image.Image,
    boxes: list[dict[str, Any]] | None,
    *,
    mask_labels: Iterable[str] = DEFAULT_MASK_LABELS,
    protect_labels: Iterable[str] = DEFAULT_PROTECT_LABELS,
    pad_px: int = 2,
    background: tuple[int, int, int] | None = None,
) -> tuple[Image.Image, int]:
    """Fill header/footer boxes with page background.

    Returns ``(image, n_painted_boxes)``. The input image is copied when any
    box is painted; otherwise the original is returned unchanged. Footnote /
    text-area pixels are restored after the fill so they are never blanked.
    """
    if img.mode != "RGB":
        img = img.convert("RGB")
    if not boxes:
        return img, 0

    mask_set = {s.strip().lower() for s in mask_labels if s}
    protect_set = {s.strip().lower() for s in protect_labels if s}
    w, h = img.size
    paint = _rects_from_boxes(
        boxes, w, h, pad_px, lambda b: resolved_label(b) in mask_set
    )
    if not paint:
        return img, 0
    protect = _rects_from_boxes(
        boxes, w, h, 0, lambda b: resolved_label(b) in protect_set
    )

    bg = background if background is not None else estimate_background_rgb(img)

    if not protect:
        out = img.copy()
        draw = ImageDraw.Draw(out)
        for x1, y1, x2, y2 in paint:
            draw.rectangle((x1, y1, x2 - 1, y2 - 1), fill=bg)
        return out, len(paint)

    import numpy as np

    src = np.asarray(img, dtype=np.uint8)
    dst = src.copy()
    fill = np.array(bg, dtype=np.uint8)
    for x1, y1, x2, y2 in paint:
        dst[y1:y2, x1:x2] = fill
    for x1, y1, x2, y2 in protect:
        dst[y1:y2, x1:x2] = src[y1:y2, x1:x2]
    return Image.fromarray(dst), len(paint)
