from PIL import Image
from transformers import AutoImageProcessor

from . import config

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


def _center_crop(img: Image.Image, size: int = config.CROP_SIZE) -> Image.Image:
    img = _resize_short_edge(img, size)
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
