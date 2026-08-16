"""Image preprocessing + adaptive per-page resolution for the PaddleOCR-VL job.

Two stages:

1. **Decode** raw bytes to an RGB PIL image, capping the longest side to
   ``cfg.max_longest_side`` (cheap upper bound so huge modern scans don't blow
   up memory). libvips is preferred (decode + downscale in one streaming pass);
   PIL is the fallback.

2. **Resolution router** (``cfg.res_mode``). The vision prefill dominates GPU
   time (the LM is only ~0.9B) and vision-token count ~= pixels / 28^2, so the
   per-page pixel budget is the main speed lever. We downsize each page to the
   smallest candidate budget that still keeps the glyph body legible:

   - ``adaptive``: estimate the p75 connected-component height (glyph-body
     proxy) and pick the cheapest ``res_scales`` budget whose downsize keeps
     that height >= ``res_tfloor`` px (~one 28px merged-patch cell). Falls back
     to the full 1x budget when the scale can't be estimated (safe).
   - ``fixed``: apply ``res_budget_scale`` * 1x budget to every page.

Ported from ``bec-ocr-training/deploy/fast_inference/bench.py`` so serving
matches the benchmarked operating point (p75/tfloor=24 => ~free-lunch speedup,
CER <= 1x).

The checkpoint's processor is patched to the 1x budget, so pre-shrinking here
directly reduces the vision tokens vLLM sees.
"""

from __future__ import annotations

import io
import logging
from typing import TYPE_CHECKING

from PIL import Image

if TYPE_CHECKING:
    from .config import PaddleOCRConfig

logger = logging.getLogger(__name__)

try:  # libvips preferred, not required
    import pyvips  # type: ignore[import-not-found]

    _VIPS_OK = True
except ImportError:
    pyvips = None
    _VIPS_OK = False

try:  # OpenCV + numpy only needed for adaptive mode
    import cv2  # type: ignore[import-not-found]
    import numpy as np  # type: ignore[import-not-found]

    _CV_OK = True
except ImportError:
    cv2 = None
    np = None
    _CV_OK = False

# Pechas are legitimately wide; disable PIL's decompression-bomb guard.
Image.MAX_IMAGE_PIXELS = None


class ImageDecodeError(RuntimeError):
    """Raised when the bytes can't be turned into an RGB image."""


def _decode_via_vips(data: bytes, *, max_side: int) -> Image.Image | None:
    """Decode bytes -> RGB PIL image, capping the longest side to ``max_side``.

    Uses ``thumbnail_buffer`` so libvips shrinks during the read (never
    allocates the full-res buffer). Only downsizes (``size="down"``).
    Returns None on failure so the caller falls back to PIL.
    """
    if not _VIPS_OK:
        return None
    try:
        v = pyvips.Image.thumbnail_buffer(data, max_side, height=max_side, size="down")

        if v.bands == 1:
            v = v.colourspace("srgb")
        elif v.bands == 4:
            v = v.flatten()
        elif v.bands != 3:
            v = v.colourspace("srgb")

        if v.format != "uchar":
            v = v.cast("uchar")

        mem = v.write_to_memory()
        return Image.frombytes("RGB", (int(v.width), int(v.height)), mem)
    except Exception as e:  # noqa: BLE001 — many vips loaders raise
        logger.debug(f"[paddleocr] libvips decode failed, will try PIL: {e}")
        return None


def _decode_via_pil(data: bytes, *, max_side: int) -> Image.Image:
    try:
        im = Image.open(io.BytesIO(data))
        im.load()
    except Exception as e:
        raise ImageDecodeError(f"PIL decode failed: {e}") from e
    if im.mode != "RGB":
        im = im.convert("RGB")
    w, h = im.size
    if max(w, h) > max_side:
        im.thumbnail((max_side, max_side), Image.Resampling.BICUBIC)
    return im


# --------------------------------------------------------------------------- #
# Adaptive resolution (ported from bench.py)
# --------------------------------------------------------------------------- #
def glyph_height_pct(img: Image.Image, pct: float = 75.0) -> float | None:
    """``pct``-th percentile of connected-component heights (px) as a glyph-body
    scale proxy. Returns None if OpenCV is missing or too few components.

    Tibetan CC heights are multi-modal (tiny tsheg/vowel marks + main glyph
    bodies); the median underestimates the legibility-critical scale, so we take
    a high percentile (default p75) to track the glyph body.
    """
    if not _CV_OK:
        return None
    try:
        g = np.asarray(img.convert("L"))
        h_img, w_img = g.shape[:2]
        th = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
        n, _, stats, _ = cv2.connectedComponentsWithStats(th, connectivity=8)
        hs = []
        for i in range(1, n):
            w, h, area = int(stats[i, 2]), int(stats[i, 3]), int(stats[i, 4])
            if area < 8 or h < 3:
                continue
            if h > 0.15 * h_img or w > 0.5 * w_img:  # borders, rules, figures
                continue
            hs.append(h)
        if len(hs) < 5:
            return None
        return float(np.percentile(hs, pct))
    except Exception as e:  # noqa: BLE001 — never fail a page on scale estimation
        logger.debug(f"[paddleocr] glyph-height estimation failed: {e}")
        return None


def choose_budget_pixels(
    img: Image.Image,
    max_px: int,
    scales: tuple[float, ...],
    tfloor: float,
    pct: float = 75.0,
) -> tuple[float, float]:
    """Smallest candidate budget whose downsize keeps the glyph-body height
    (``pct``-ile) >= ``tfloor`` px. Falls back to the full budget when scale
    can't be estimated (safe). Returns ``(budget_px, scale_fraction)``."""
    orig = img.width * img.height
    h = glyph_height_pct(img, pct)
    if h is None or h <= 0:
        return float(max_px), scales[-1]
    for s in scales:  # ascending → prefer the cheapest that stays legible
        budget = s * max_px
        factor = min(1.0, (budget / orig) ** 0.5)  # downscale-only
        if h * factor >= tfloor:
            return budget, s
    return float(max_px), scales[-1]


def resize_to_budget(img: Image.Image, budget_px: float) -> Image.Image:
    """Downscale (never upscale) so total pixels <= ``budget_px``, keeping aspect."""
    px = img.width * img.height
    if px <= budget_px:
        return img
    f = (budget_px / px) ** 0.5
    w = max(28, int(round(img.width * f)))
    h = max(28, int(round(img.height * f)))
    return img.resize((w, h), Image.LANCZOS)


def bytes_to_rgb(data: bytes, cfg: PaddleOCRConfig) -> tuple[Image.Image, float]:
    """Decode + resolution-route raw image bytes.

    Returns ``(rgb_image, res_scale)`` where ``res_scale`` is the fraction of
    the 1x pixel budget the page was routed to (1.0 = full budget). The image is
    pre-shrunk to that budget so vLLM sees fewer vision tokens.
    """
    if not data:
        raise ImageDecodeError("empty image bytes")

    im = _decode_via_vips(data, max_side=cfg.max_longest_side)
    if im is None:
        im = _decode_via_pil(data, max_side=cfg.max_longest_side)

    if im.mode != "RGB":
        im = im.convert("RGB")

    w, h = im.size
    if w <= 0 or h <= 0:
        raise ImageDecodeError(f"invalid image dimensions: {w}x{h}")

    max_px = cfg.processor_longest_edge()
    if cfg.res_mode == "fixed":
        scale = cfg.res_budget_scale
        return resize_to_budget(im, scale * max_px), scale

    budget, scale = choose_budget_pixels(
        im, max_px, cfg.res_scales, cfg.res_tfloor, cfg.res_percentile
    )
    return resize_to_budget(im, budget), scale
