"""Image decode + DINOv3 preprocessing for ``script_classification_v2``.

Adapted from ``script_classification/vendor/transforms.py``. The only change
is that the crop size and backbone id are passed in explicitly (from the job
config) instead of being read from a module-level constant, so a single
vendored copy serves classifiers trained at different input sizes.
"""

import io

# Import torch before pyvips: importing pyvips before torch has been observed
# to segfault at model-construction time (native dylib / OpenMP conflict).
import torch  # noqa: F401

from PIL import Image
from transformers import AutoImageProcessor

try:  # libvips is preferred (fused decode+resize), but not required.
    import pyvips  # type: ignore[import-not-found]

    _VIPS_OK = True
except ImportError:
    pyvips = None
    _VIPS_OK = False

_processor = None


def get_processor(backbone_id: str):
    """ImageNet-normalize processor (resize/crop disabled — we crop manually).

    Prefer the canonical ``preprocessor_config.json`` from the (gated)
    backbone repo when reachable, else fall back to transformers' bundled
    ``DINOv3ViTImageProcessor`` (verified to produce byte-identical
    normalization to the Hub config).
    """
    global _processor
    if _processor is None:
        try:
            _processor = AutoImageProcessor.from_pretrained(backbone_id)
        except Exception:
            from transformers import DINOv3ViTImageProcessor

            _processor = DINOv3ViTImageProcessor()
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
    if not _VIPS_OK:
        return None
    try:
        v = pyvips.Image.new_from_buffer(data, "", access="sequential")
        w, h = int(v.width), int(v.height)
        scale = target / min(w, h)
        if scale != 1.0:
            v = v.resize(scale, kernel="cubic")  # matches PIL BICUBIC
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
    except Exception:
        return None


def _decode_and_resize_via_pil(data: bytes, target: int) -> Image.Image:
    img = Image.open(io.BytesIO(data))
    img = img.convert("RGB")
    return _resize_short_edge(img, target)


def _raw_orientation_tag(image_bytes: bytes) -> int | None:
    """Raw EXIF/TIFF orientation tag, read but never applied to pixels."""
    try:
        return Image.open(io.BytesIO(image_bytes)).getexif().get(0x0112)
    except Exception:
        return None


def decode_and_resize(image_bytes: bytes, target: int) -> tuple[Image.Image, int | None]:
    """Decode raw bytes to RGB, short edge == ``target``, plus the raw EXIF tag."""
    exif_tag = _raw_orientation_tag(image_bytes)
    img = _decode_and_resize_via_vips(image_bytes, target)
    if img is None:
        img = _decode_and_resize_via_pil(image_bytes, target)
    return img, exif_tag


def _center_crop(img: Image.Image, size: int) -> Image.Image:
    # Assumes `img` already has short edge == size (from decode_and_resize).
    w, h = img.size
    left = max(0, (w - size) // 2)
    top = max(0, (h - size) // 2)
    crop = img.crop((left, top, left + size, top + size))
    if crop.size != (size, size):
        # Rare: resize rounding left the crop 1px short. Center the paste.
        padded = Image.new("RGB", (size, size), (255, 255, 255))
        offset = ((size - crop.width) // 2, (size - crop.height) // 2)
        padded.paste(crop, offset)
        return padded
    return crop


def decode_resize_crop(
    image_bytes: bytes, target: int
) -> tuple[Image.Image, int | None]:
    """Decode + resize-short-edge + center-crop -> exactly ``target x target``."""
    img, exif_tag = decode_and_resize(image_bytes, target)
    cropped = _center_crop(img, target)
    return cropped, exif_tag


def preprocess_batch(imgs: list[Image.Image], backbone_id: str, crop_size: int):
    """Normalize a list of already-cropped (``crop_size x crop_size``) images
    in a single HF processor call -> one ``[N, 3, crop_size, crop_size]`` tensor.
    """
    pv = get_processor(backbone_id)(
        images=imgs,
        do_resize=False,
        do_center_crop=False,
        return_tensors="pt",
    )["pixel_values"]
    assert pv.shape[0] == len(imgs)
    assert pv.shape[-2:] == (crop_size, crop_size)
    return pv
