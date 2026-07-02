import io

# Must import before pyvips (see the try/except below) -- importing pyvips
# before torch has been observed to segfault at model-construction time
# (native dylib / OpenMP conflict). transforms.py doesn't need torch
# directly, but it's imported before vendor/models.py in pipeline.py's
# import order, so this is the one place that reliably guarantees load
# order regardless of caller.
import torch  # noqa: F401

from PIL import Image
from transformers import AutoImageProcessor

from . import config

try:  # libvips is preferred (fused decode+resize, avoids a full-res PIL
    # buffer for huge scans), but not required -- PIL fallback below.
    import pyvips  # type: ignore[import-not-found]

    _VIPS_OK = True
except ImportError:
    pyvips = None
    _VIPS_OK = False

_processor = None


def get_processor():
    global _processor
    if _processor is None:
        _processor = AutoImageProcessor.from_pretrained(config.BACKBONE_ID)
    return _processor


def _resize_short_edge(img: Image.Image, target: int) -> Image.Image:
    w, h = img.size
    if h <= w:
        new_h = target
        new_w = max(1, round(w * target / h))
    else:
        new_w = target
        new_h = max(1, round(h * target / w))
    return img.resize((new_w, new_h), Image.BICUBIC)


def _decode_and_resize_via_vips(data: bytes, target: int) -> Image.Image | None:
    """Decode + resize-short-edge-to-``target`` in one libvips pass.

    Returns ``None`` if libvips can't handle these bytes (caller falls back
    to ``_decode_and_resize_via_pil``).
    """
    if not _VIPS_OK:
        return None
    try:
        v = pyvips.Image.new_from_buffer(data, "", access="sequential")

        w, h = int(v.width), int(v.height)
        scale = target / min(w, h)
        if scale != 1.0:
            # "cubic" matches PIL's BICUBIC above -- kept consistent between
            # both decode lanes since this job's accuracy is more sensitive
            # to resize-kernel choice than e.g. ocr_qwen_v1's OCR pipeline.
            v = v.resize(scale, kernel="cubic")

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
    except Exception:
        return None


def _decode_and_resize_via_pil(data: bytes, target: int) -> Image.Image:
    img = Image.open(io.BytesIO(data))
    img = img.convert("RGB")
    return _resize_short_edge(img, target)


def _raw_orientation_tag(image_bytes: bytes) -> int | None:
    """Raw EXIF/TIFF orientation tag, read but never applied to pixels.

    Deliberately always uses PIL's ``getexif()`` (a cheap header-only
    parse -- ``Image.open`` is lazy, no pixel decode happens here),
    independent of which library decodes the pixels below. libvips's own
    ``orientation`` metadata field is *synthetic*: it defaults to ``1``
    ("normal") whenever no tag is present, rather than being absent --
    confirmed empirically against 180 real sample TIFFs, 9 of which have no
    Orientation tag at all in their raw EXIF dict yet libvips reported
    ``orientation=1`` for all of them. Using PIL here keeps this nullable
    column faithful to the source metadata regardless of the decode lane.
    """
    try:
        return Image.open(io.BytesIO(image_bytes)).getexif().get(0x0112)
    except Exception:
        return None


def decode_and_resize(
    image_bytes: bytes, target: int = config.CROP_SIZE
) -> tuple[Image.Image, int | None]:
    """Decode raw bytes to RGB, resized so the short edge == ``target``, plus
    the raw EXIF/TIFF orientation tag (read but never applied to pixels --
    unchanged behavior from upstream).

    Tries libvips first (decode+resize fused, avoids materializing a
    full-resolution PIL buffer for large scans); falls back to PIL decode +
    ``_resize_short_edge`` on any libvips failure or when pyvips isn't
    installed. The resize happens exactly once here -- callers (see
    ``pipeline.py``) must not resize again, including for the rotated pass:
    for a uniform-scale aspect-preserving resize, 180-degree rotation
    commutes with resize, so rotating this already-small result is pixel-
    identical to rotating the full-resolution image and resizing after, just
    far cheaper.
    """
    exif_tag = _raw_orientation_tag(image_bytes)
    img = _decode_and_resize_via_vips(image_bytes, target)
    if img is None:
        img = _decode_and_resize_via_pil(image_bytes, target)
    return img, exif_tag


def _center_crop(img: Image.Image, size: int = config.CROP_SIZE) -> Image.Image:
    # Assumes `img` already has short edge == size -- resizing now happens
    # exactly once, in decode_and_resize() above, before any rotation. Do
    # NOT resize here; both the original and the 180-degree-rotated small
    # image already satisfy this (rotation preserves dimensions).
    w, h = img.size
    left = max(0, (w - size) // 2)
    top = max(0, (h - size) // 2)
    crop = img.crop((left, top, left + size, top + size))
    if crop.size != (size, size):
        # Deviation from upstream: upstream pastes at (0, 0), anchoring content
        # in the top-left corner with all padding on the bottom/right. Centering
        # the paste keeps content roughly centered, matching the DINOv3 ViT
        # backbone's learned positional embeddings — this path is rare (only
        # hit when _resize_short_edge's rounding leaves the crop short of
        # `size`), so this has no bearing on the checkpoints' training data.
        padded = Image.new("RGB", (size, size), (255, 255, 255))
        offset = ((size - crop.width) // 2, (size - crop.height) // 2)
        padded.paste(crop, offset)
        return padded
    return crop


def preprocess(img: Image.Image):
    img = _center_crop(img)
    # geometry disabled: manual center_crop above already matches training; the
    # processor is only used for its ImageNet normalize step
    pv = get_processor()(
        images=img,
        do_resize=False,
        do_center_crop=False,
        return_tensors="pt",
    )["pixel_values"]
    assert pv.shape[-2:] == (config.CROP_SIZE, config.CROP_SIZE)
    return pv
