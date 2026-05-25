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


def _compute_vips_scale(w: int, h: int, *, max_side: int, max_pixels: int) -> float:
    """Compute a scale factor that satisfies both size caps; never upscales."""
    if w <= 0 or h <= 0:
        return 1.0
    long_side = max(w, h)
    side_scale = max_side / long_side if long_side > max_side else 1.0
    pixel_scale = 1.0
    if w * h > max_pixels:
        pixel_scale = (max_pixels / (w * h)) ** 0.5
    return min(side_scale, pixel_scale, 1.0)


def _decode_via_vips(
    data: bytes, *, max_side: int, max_pixels: int
) -> Image.Image | None:
    """Decode bytes → RGB PIL image, downscaling **during** the libvips read.

    libvips reads the image header first; we use the header dimensions to
    compute a scale factor and pass it to ``Image.resize`` BEFORE
    ``write_to_memory`` materialises the pixel buffer. This is critical for
    pathologically big images (e.g. 10000×10000 modern scans) — we never
    allocate the full-resolution NumPy/PIL buffer.

    Returns ``None`` if libvips can't decode (caller falls back to PIL).
    """
    if not _VIPS_OK:
        return None
    try:
        # ``access="sequential"`` is a big throughput win on large TIFFs.
        v = pyvips.Image.new_from_buffer(data, "", access="sequential")

        # Header-only dimensions are available immediately (no full decode yet).
        w, h = int(v.width), int(v.height)
        scale = _compute_vips_scale(w, h, max_side=max_side, max_pixels=max_pixels)
        if scale < 1.0:
            # ``kernel="linear"`` (bilinear) is the libvips default and a good
            # speed/quality tradeoff for continuous-tone content; the Qwen
            # model handles slight smoothing just fine.
            v = v.resize(scale, kernel="linear")
            logger.debug(
                f"[ocr_qwen_v1] vips downscaled {w}x{h} → {int(v.width)}x{int(v.height)} "
                f"(scale={scale:.3f})"
            )

        # Force RGB (3 bands) so the PIL.frombytes call below is consistent.
        if v.bands == 1:
            v = v.colourspace("srgb")
        elif v.bands == 4:
            v = v.flatten()
        elif v.bands != 3:
            v = v.colourspace("srgb")

        # libvips may return non-uchar formats (e.g. ushort TIFFs); cast.
        if v.format != "uchar":
            v = v.cast("uchar")

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

    # Preferred lane: libvips decodes AND downscales in one streaming pass,
    # so we never materialise the full-res pixel buffer for huge inputs.
    im = _decode_via_vips(
        data, max_side=cfg.max_image_side, max_pixels=cfg.max_image_pixels
    )
    if im is None:
        # PIL fallback: we have to decode at native resolution first, then
        # shrink. Fine for typical-sized images; the libvips lane handles
        # the pathological ones above.
        im = _decode_via_pil(data)

    if im.mode != "RGB":
        im = im.convert("RGB")

    # Safety net: re-enforce the budget for the PIL lane (no-op when vips
    # already resized within tolerance).
    return _fit_to_budget(
        im,
        max_side=cfg.max_image_side,
        max_pixels=cfg.max_image_pixels,
        resample_name=cfg.downscale_resample,
    )
