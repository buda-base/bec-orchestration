import numpy as np
from PIL import Image

from . import config


def is_blank(img: Image.Image, std_thresh: float = config.BLANK_STD_THRESH) -> bool:
    gray = img.convert("L")
    w, h = gray.size
    long_side = max(w, h)
    if long_side > config.BLANK_MAX_SIDE:
        scale = config.BLANK_MAX_SIDE / long_side
        gray = gray.resize((max(1, round(w * scale)), max(1, round(h * scale))), Image.BICUBIC)

    std = np.asarray(gray, dtype=np.float32).std()
    return std < std_thresh
