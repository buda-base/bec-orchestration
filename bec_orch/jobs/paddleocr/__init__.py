"""Version-agnostic PaddleOCR-VL inference job.

This package implements a full-page Tibetan OCR worker built on the **vLLM**
offline engine (native PaddleOCR-VL support) with the DRY loop guard and
adaptive per-page resolution. It is intentionally **not** tied to a specific
model version: the checkpoint, prompt and all generation / preprocessing
parameters live in :class:`PaddleOCRConfig`, so a new job (e.g. ``paddleocr_v2``)
that only swaps the fine-tuned checkpoint can reuse this exact code — it just
registers another factory with a different default ``checkpoint_s3_uri`` (see
``core/registry.py``).
"""

from .config import PaddleOCRConfig
from .worker import PaddleOCRJobWorker

__all__ = ["PaddleOCRConfig", "PaddleOCRJobWorker"]
