"""Split two-column pages from ``layout_detection_v1`` text-area boxes.

A pair of text-area detections is treated as two columns when they overlap
by at least ``min_vert_overlap`` of the shorter box vertically and by less
than ``max_horiz_overlap`` of the narrower box horizontally (defaults 60% /
5%). Matching boxes are pasted onto a canvas with synthetic background
margin (including past the page edge) and OCR'd separately; the worker
concatenates the transcriptions with two line breaks.
"""

from __future__ import annotations

from typing import Any, Iterable, Sequence

from PIL import Image, ImageDraw

from .layout_mask import estimate_background_rgb, resolved_label, xywhn_to_xyxy

Rect = tuple[int, int, int, int]  # x1, y1, x2, y2
DEFAULT_COLUMN_LABEL = "text-area"
DEFAULT_JOIN = "\n\n"


def interval_overlap_frac(a1: float, a2: float, b1: float, b2: float) -> float:
    """Intersection of two intervals as a fraction of the shorter length."""
    inter = max(0.0, min(a2, b2) - max(a1, b1))
    shorter = min(max(0.0, a2 - a1), max(0.0, b2 - b1))
    if shorter <= 0.0:
        return 0.0
    return inter / shorter


def vert_overlap_frac(a: Rect, b: Rect) -> float:
    return interval_overlap_frac(a[1], a[3], b[1], b[3])


def horiz_overlap_frac(a: Rect, b: Rect) -> float:
    return interval_overlap_frac(a[0], a[2], b[0], b[2])


def _area(r: Rect) -> int:
    return max(0, r[2] - r[0]) * max(0, r[3] - r[1])


def fully_inside(inner: Rect, outer: Rect, *, atol: int = 1) -> bool:
    """True if ``inner`` lies entirely in ``outer`` (1px slack for rounding)."""
    return (
        inner[0] >= outer[0] - atol
        and inner[1] >= outer[1] - atol
        and inner[2] <= outer[2] + atol
        and inner[3] <= outer[3] + atol
    )


def coverage_frac(inner: Rect, outer: Rect) -> float:
    """Fraction of ``inner``'s area that lies inside ``outer``."""
    ix = max(0, min(inner[2], outer[2]) - max(inner[0], outer[0]))
    iy = max(0, min(inner[3], outer[3]) - max(inner[1], outer[1]))
    inter = ix * iy
    a_inner = _area(inner)
    if a_inner <= 0:
        return 0.0
    return inter / a_inner


def drop_fully_contained(
    rects: Sequence[Rect], *, atol: int = 1, coverage: float = 0.95
) -> list[Rect]:
    """Drop a box that sits (almost) fully inside another of the same list.

    A box is dropped when it is geometrically inside a sibling (``atol`` px
    slack) *or* at least ``coverage`` of its area is covered by that sibling —
    the latter catches detections that overhang by a few pixels but are really
    the same note. Only the smaller of the pair is dropped; equal duplicates
    keep the first. Used so a footnote (or text-area) drawn inside a larger
    sibling is not OCR'd twice.
    """
    kept: list[Rect] = []
    for i, a in enumerate(rects):
        skip = False
        for j, b in enumerate(rects):
            if i == j:
                continue
            if not (fully_inside(a, b, atol=atol) or coverage_frac(a, b) >= coverage):
                continue
            if _area(a) < _area(b) or (_area(a) == _area(b) and i > j):
                skip = True
                break
        if not skip:
            kept.append(a)
    return kept


def _paint_rect_on_canvas(
    canvas: Image.Image,
    canvas_x1: int,
    canvas_y1: int,
    rect: Rect,
    color: tuple[int, int, int],
) -> None:
    x1, y1, x2, y2 = rect
    bx1 = max(0, x1 - canvas_x1)
    by1 = max(0, y1 - canvas_y1)
    bx2 = min(canvas.width, x2 - canvas_x1)
    by2 = min(canvas.height, y2 - canvas_y1)
    if bx2 <= bx1 or by2 <= by1:
        return
    ImageDraw.Draw(canvas).rectangle([bx1, by1, bx2 - 1, by2 - 1], fill=color)


def text_area_rects(
    boxes: Iterable[dict[str, Any]],
    img_w: int,
    img_h: int,
    *,
    label: str = DEFAULT_COLUMN_LABEL,
) -> list[Rect]:
    """Pixel xyxy rects for ``text-area`` (or ``label``) detections."""
    want = (label or DEFAULT_COLUMN_LABEL).strip().lower()
    rects: list[Rect] = []
    for box in boxes:
        if resolved_label(box) != want:
            continue
        try:
            x, y, w, h = float(box["x"]), float(box["y"]), float(box["w"]), float(box["h"])
        except (KeyError, TypeError, ValueError):
            continue
        xyxy = xywhn_to_xyxy(x, y, w, h, img_w, img_h, pad_px=0)
        if xyxy is not None:
            rects.append(xyxy)
    return rects


def select_column_rects(
    rects: Sequence[Rect],
    *,
    min_vert_overlap: float = 0.60,
    max_horiz_overlap: float = 0.05,
) -> list[Rect] | None:
    """Return left-to-right column rects, or None if the page is not multi-column.

    Builds a graph of rect pairs that satisfy the vertical/horizontal overlap
    heuristic and returns the largest connected component with at least two
    boxes, sorted by x-centre. A full-width text-area sitting on top of two
    columns will not pair with either (horizontal overlap is high), so it is
    ignored — which is what we want.
    """
    n = len(rects)
    if n < 2:
        return None
    adj: list[list[int]] = [[] for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            if vert_overlap_frac(rects[i], rects[j]) < min_vert_overlap:
                continue
            if horiz_overlap_frac(rects[i], rects[j]) >= max_horiz_overlap:
                continue
            adj[i].append(j)
            adj[j].append(i)

    seen: set[int] = set()
    best: list[int] | None = None
    best_area = -1
    for i in range(n):
        if i in seen or not adj[i]:
            continue
        stack = [i]
        seen.add(i)
        comp: list[int] = []
        while stack:
            u = stack.pop()
            comp.append(u)
            for v in adj[u]:
                if v not in seen:
                    seen.add(v)
                    stack.append(v)
        if len(comp) < 2:
            continue
        area = sum(_area(rects[k]) for k in comp)
        if area > best_area:
            best_area = area
            best = comp
    if not best:
        return None
    return sorted((rects[k] for k in best), key=lambda r: (r[0] + r[2]) / 2.0)


def _projection_gaps(intervals: Sequence[tuple[float, float]]) -> list[tuple[float, float]]:
    """Empty gaps between the merged 1-D ``intervals`` (left→right)."""
    if not intervals:
        return []
    ordered = sorted(intervals)
    gaps: list[tuple[float, float]] = []
    cur_end = ordered[0][1]
    for a, b in ordered[1:]:
        if a > cur_end:
            gaps.append((cur_end, a))
            cur_end = b
        else:
            cur_end = max(cur_end, b)
    return gaps


def reading_order_xycut(
    rects: Sequence[Rect], page_w: int, page_h: int
) -> list[Rect]:
    """Order rectangles by recursive XY-cut (standard reading-order heuristic).

    At each step the region is split at the widest whitespace valley — the
    larger of the widest horizontal gap (fraction of page height) and the
    widest vertical gap (fraction of page width). A horizontal cut yields
    top-then-bottom; a vertical cut yields left-then-right. Ties prefer a
    horizontal cut, so a title/heading band is peeled off the top before the
    body underneath is split into columns (which then read top-to-bottom,
    i.e. column-major). Handles modern book pages: single column, two
    columns, a spanning title over two columns, and stacked blocks. When no
    axis has a separating gap (boxes overlap both ways) it falls back to a
    top-to-bottom, left-to-right sort.
    """
    rects = list(rects)
    if len(rects) <= 1:
        return rects
    ph = max(1, int(page_h))
    pw = max(1, int(page_w))
    ygaps = _projection_gaps([(r[1], r[3]) for r in rects])
    xgaps = _projection_gaps([(r[0], r[2]) for r in rects])
    best_y = max(((ge - gs) / ph, (gs + ge) / 2.0) for gs, ge in ygaps) if ygaps else None
    best_x = max(((ge - gs) / pw, (gs + ge) / 2.0) for gs, ge in xgaps) if xgaps else None
    if best_y is None and best_x is None:
        return sorted(rects, key=lambda r: (r[1], r[0]))
    use_y = best_x is None or (best_y is not None and best_y[0] >= best_x[0])
    if use_y:
        pos = best_y[1]  # type: ignore[index]
        first = [r for r in rects if (r[1] + r[3]) / 2.0 < pos]
        second = [r for r in rects if (r[1] + r[3]) / 2.0 >= pos]
    else:
        pos = best_x[1]  # type: ignore[index]
        first = [r for r in rects if (r[0] + r[2]) / 2.0 < pos]
        second = [r for r in rects if (r[0] + r[2]) / 2.0 >= pos]
    return reading_order_xycut(first, pw, ph) + reading_order_xycut(second, pw, ph)


def region_margin_px(
    page_w: int,
    page_h: int,
    rect: Rect | None = None,
    *,
    margin_frac: float,
    min_px: int = 8,
    box_margin_frac: float = 0.0,
) -> int:
    """Pixels of synthetic background padding around a crop.

    Takes the max of ``min_px``, ``margin_frac`` of the shorter page side, and
    (when ``rect`` is given) ``box_margin_frac`` of the box height. Tight
    footnote detections get a taller pad so stacked Tibetan vowels are not
    flush with the crop edge. The pad is painted, not cropped from the page,
    and may extend past the original image bounds.
    """
    page_m = int(round(min(page_w, page_h) * max(0.0, float(margin_frac))))
    box_m = 0
    if rect is not None and box_margin_frac > 0.0:
        box_h = max(1, int(rect[3]) - int(rect[1]))
        box_m = int(round(box_h * max(0.0, float(box_margin_frac))))
    return max(max(0, int(min_px)), page_m, box_m)


def crop_with_background_margin(
    img: Image.Image,
    rect: Rect,
    *,
    margin_px: int,
    background: tuple[int, int, int],
    blank_rects: Sequence[Rect] | None = None,
) -> Image.Image:
    """Paste ``rect`` onto a canvas padded with synthetic ``background``.

    The canvas is always ``rect`` expanded by ``margin_px`` on every side,
    even when that hangs off the page. Only on-page pixels inside ``rect``
    are copied; the entire margin is estimated page background so
    neighbouring text, sibling columns, and scanner edges never appear.

    ``blank_rects`` (e.g. a footer sitting inside a footnote, or a footnote
    overlapping a text-area) are then painted with ``background`` so they
    do not appear in this crop.
    """
    if img.mode != "RGB":
        img = img.convert("RGB")
    x1, y1, x2, y2 = rect
    m = max(0, int(margin_px))
    canvas_x1, canvas_y1 = x1 - m, y1 - m
    canvas_x2, canvas_y2 = x2 + m, y2 + m
    canvas = Image.new(
        "RGB",
        (max(1, canvas_x2 - canvas_x1), max(1, canvas_y2 - canvas_y1)),
        background,
    )
    keep_x1 = max(0, x1)
    keep_y1 = max(0, y1)
    keep_x2 = min(img.width, x2)
    keep_y2 = min(img.height, y2)
    if keep_x2 > keep_x1 and keep_y2 > keep_y1:
        canvas.paste(
            img.crop((keep_x1, keep_y1, keep_x2, keep_y2)),
            (keep_x1 - canvas_x1, keep_y1 - canvas_y1),
        )
    for other in blank_rects or ():
        if other == rect:
            continue
        _paint_rect_on_canvas(canvas, canvas_x1, canvas_y1, other, background)
    return canvas


def crop_body_regions(
    img: Image.Image,
    boxes: list[dict[str, Any]] | None,
    *,
    label: str = DEFAULT_COLUMN_LABEL,
    min_vert_overlap: float = 0.60,
    max_horiz_overlap: float = 0.05,
    margin_frac: float = 0.02,
    min_px: int = 16,
    background: tuple[int, int, int] | None = None,
    blank_labels: tuple[str, ...] = ("footnote", "header", "footer"),
) -> list[Image.Image]:
    """Crop each kept text-area onto a synthetic-margin canvas, in reading order.

    A clean two-column pair (the overlap heuristic) yields left-to-right
    column crops — this keeps a full-width envelope sitting over two columns
    from swallowing the pair. Any other page (one text-area, or three or
    more) crops **every** non-nested text-area and orders them by recursive
    XY-cut (title band first, then columns top-to-bottom). Overlapping
    footnotes / headers / footers inside a crop are painted with background.
    """
    if not boxes:
        return []
    if img.mode != "RGB":
        img = img.convert("RGB")
    w, h = img.size
    rects = text_area_rects(boxes, w, h, label=label)
    if not rects:
        return []
    kept = drop_fully_contained(rects)
    cols = select_column_rects(
        rects,
        min_vert_overlap=min_vert_overlap,
        max_horiz_overlap=max_horiz_overlap,
    )
    if cols is not None and len(cols) == 2 and len(kept) <= 2:
        body = cols
    else:
        body = reading_order_xycut(kept, w, h)
    blank: list[Rect] = []
    for lab in blank_labels:
        blank.extend(drop_fully_contained(text_area_rects(boxes, w, h, label=lab)))
    bg = background if background is not None else estimate_background_rgb(img)
    margin_px = region_margin_px(
        w, h, margin_frac=margin_frac, min_px=min_px, box_margin_frac=0.0
    )
    return [
        crop_with_background_margin(
            img, rect, margin_px=margin_px, background=bg, blank_rects=blank
        )
        for rect in body
    ]


def split_page_columns(
    img: Image.Image,
    boxes: list[dict[str, Any]] | None,
    *,
    label: str = DEFAULT_COLUMN_LABEL,
    min_vert_overlap: float = 0.60,
    max_horiz_overlap: float = 0.05,
    margin_frac: float = 0.02,
    background: tuple[int, int, int] | None = None,
) -> list[Image.Image] | None:
    """Return left-to-right column crops, or None if the page is not multi-column."""
    if not boxes:
        return None
    if img.mode != "RGB":
        img = img.convert("RGB")
    w, h = img.size
    rects = text_area_rects(boxes, w, h, label=label)
    cols = select_column_rects(
        rects,
        min_vert_overlap=min_vert_overlap,
        max_horiz_overlap=max_horiz_overlap,
    )
    if not cols:
        return None
    crops = crop_body_regions(
        img,
        boxes,
        label=label,
        min_vert_overlap=min_vert_overlap,
        max_horiz_overlap=max_horiz_overlap,
        margin_frac=margin_frac,
        background=background,
    )
    return crops if len(crops) >= 2 else None


def crop_labeled_regions(
    img: Image.Image,
    boxes: list[dict[str, Any]] | None,
    *,
    label: str,
    margin_frac: float = 0.05,
    min_px: int = 32,
    box_margin_frac: float = 0.5,
    background: tuple[int, int, int] | None = None,
    blank_labels: tuple[str, ...] = ("header", "footer"),
) -> list[Image.Image]:
    """Cut every non-nested box of ``label`` (top-to-bottom, then left-to-right).

    Used for footnotes: a detection fully inside another footnote is dropped.
    Each remaining box is pasted onto a canvas padded with synthetic page
    background. Headers/footers that sit inside the crop (a folio number
    boxed both as footer and as part of the note) are painted out.
    """
    if not boxes:
        return []
    if img.mode != "RGB":
        img = img.convert("RGB")
    w, h = img.size
    rects = drop_fully_contained(text_area_rects(boxes, w, h, label=label))
    if not rects:
        return []
    rects.sort(key=lambda r: (r[1], r[0]))
    blank: list[Rect] = []
    for lab in blank_labels:
        blank.extend(text_area_rects(boxes, w, h, label=lab))
    bg = background if background is not None else estimate_background_rgb(img)
    crops: list[Image.Image] = []
    for rect in rects:
        margin_px = region_margin_px(
            w,
            h,
            rect,
            margin_frac=margin_frac,
            min_px=min_px,
            box_margin_frac=box_margin_frac,
        )
        crops.append(
            crop_with_background_margin(
                img,
                rect,
                margin_px=margin_px,
                background=bg,
                blank_rects=blank,
            )
        )
    return crops


def max_region_margin_px(
    page_w: int,
    page_h: int,
    boxes: list[dict[str, Any]] | None,
    *,
    label: str,
    margin_frac: float,
    min_px: int = 32,
    box_margin_frac: float = 0.5,
) -> int:
    """Largest crop margin among ``label`` boxes — used when blanking them."""
    if not boxes:
        return max(0, int(min_px))
    rects = text_area_rects(boxes, page_w, page_h, label=label)
    if not rects:
        return max(0, int(min_px))
    return max(
        region_margin_px(
            page_w,
            page_h,
            r,
            margin_frac=margin_frac,
            min_px=min_px,
            box_margin_frac=box_margin_frac,
        )
        for r in rects
    )


def join_column_texts(texts: Sequence[str], sep: str = DEFAULT_JOIN) -> str:
    """Concatenate column/footnote transcriptions, skipping empty crops."""
    parts = [t.strip() for t in texts if t and str(t).strip()]
    return sep.join(parts)
