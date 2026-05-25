"""Image preprocessing for ``ocr_qwen_v1``.

Decodes raw image bytes (typically TIFF for BDRC pechas, sometimes JPEG/PNG)
into a PIL ``RGB`` image, capping size to fit the vLLM context budget.

Decoding follows the same fast-path layering as ``shared/decoder.py``:
libvips first (fast), then PIL fallback. We do NOT use OpenCV here because
we need RGB rather than grayscale and PIL handles all the formats we see.
"""

from __future__ import annotations

import io
import logging
from typing import TYPE_CHECKING

from PIL import Image

if TYPE_CHECKING:
    from .config import OCRQwenV1Config

logger = logging.getLogger(__name__)


try:  # libvips is preferred for big images, but not required
    import pyvips  # type: ignore[import-not-found]

    _VIPS_OK = True
except ImportError:
    pyvips = None
    _VIPS_OK = False


# Bilevel TIFFs from BDRC come in mode "1" (1 bit per pixel). PIL handles them
# fine but emits DecompressionBombWarning on big TIFFs even when we then
# immediately resize. Suppress that — pechas are legitimately wide.
Image.MAX_IMAGE_PIXELS = None


_RESAMPLE = {
    "nearest": Image.Resampling.NEAREST,
    "bilinear": Image.Resampling.BILINEAR,
    "bicubic": Image.Resampling.BICUBIC,
    "lanczos": Image.Resampling.LANCZOS,
}


class ImageDecodeError(RuntimeError):
    """Raised when the bytes can't be turned into an RGB image."""


def _decode_via_vips(data: bytes) -> Image.Image | None:
    if not _VIPS_OK:
        return None
    try:
        # ``access="sequential"`` lets libvips stream-decode big TIFFs without
        # materialising the whole pixel buffer twice.
        v = pyvips.Image.new_from_buffer(data, "", access="sequential")
        if v.bands == 1:
            v = v.colourspace("srgb")
        elif v.bands == 4:
            v = v.flatten()
        mem = v.write_to_memory()
        return Image.frombytes("RGB", (int(v.width), int(v.height)), mem)
    except Exception as e:  # noqa: BLE001 — many vips loaders raise
        logger.debug(f"libvips decode failed, will try PIL: {e}")
        return None


def _decode_via_pil(data: bytes) -> Image.Image:
    try:
        im = Image.open(io.BytesIO(data))
        im.load()
    except Exception as e:
        raise ImageDecodeError(f"PIL decode failed: {e}") from e
    if im.mode != "RGB":
        im = im.convert("RGB")
    return im


def _fit_to_budget(
    im: Image.Image, *, max_side: int, max_pixels: int, resample_name: str
) -> Image.Image:
    """Shrink ``im`` so that ``max(W,H) ≤ max_side`` and ``W*H ≤ max_pixels``.

    Whichever constraint is tighter wins; aspect ratio is preserved. Never
    upscales.
    """
    w, h = im.size
    if w <= 0 or h <= 0:
        raise ImageDecodeError(f"invalid image dimensions: {w}x{h}")

    long_side = max(w, h)
    side_scale = max_side / long_side if long_side > max_side else 1.0

    pixel_scale = 1.0
    if w * h > max_pixels:
        # solve s for (s*w)*(s*h) == max_pixels  →  s = sqrt(max_pixels/(w*h))
        pixel_scale = (max_pixels / (w * h)) ** 0.5

    scale = min(side_scale, pixel_scale)
    if scale >= 1.0:
        return im

    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    resample = _RESAMPLE.get(resample_name, Image.Resampling.BICUBIC)
    return im.resize((new_w, new_h), resample)


def bytes_to_rgb(data: bytes, cfg: OCRQwenV1Config) -> Image.Image:
    """Decode raw image bytes to a vLLM-ready RGB PIL image.

    Args:
        data: raw bytes from S3.
        cfg: job config (uses ``max_image_side``, ``max_image_pixels``,
            ``downscale_resample``).

    Raises:
        ImageDecodeError: if neither libvips nor PIL can decode the bytes.
    """
    if not data:
        raise ImageDecodeError("empty image bytes")

    im = _decode_via_vips(data)
    if im is None:
        im = _decode_via_pil(data)

    # ``_decode_via_vips`` already returns RGB; ``_decode_via_pil`` also.
    # Belt-and-braces though, for any future decoder lanes.
    if im.mode != "RGB":
        im = im.convert("RGB")

    return _fit_to_budget(
        im,
        max_side=cfg.max_image_side,
        max_pixels=cfg.max_image_pixels,
        resample_name=cfg.downscale_resample,
    )
