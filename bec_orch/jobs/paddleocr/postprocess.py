"""Post-processing for PaddleOCR-VL (grow26-ep2) predictions.

The ``grow26-ep2`` checkpoint is trained on a **warmed Tibetan unicode-stack
tokenizer**, so it emits **Tibetan Unicode directly** — there is NO Wylie/EWTS
conversion (that was the previous, Wylie-trained checkpoint). Post-processing is
just the canonical Unicode normalization used by the training/eval scorer
(:func:`normalize_unicode_text`: NFD reorder + graphical fold), matching CER
scoring exactly.

We also compute a cheap loop self-flag ``rep_score`` = fraction of repeated
n-grams over the prediction's syllable tokens (split on tsheg / whitespace);
pages at/above the configured threshold are flagged ``likely_loop`` for review.
DRY already removes hard loops at decode time, so this is a belt-and-suspenders
review signal.
"""

from __future__ import annotations

import logging
import re

from .tibetan_normalize import normalize_unicode_text

logger = logging.getLogger(__name__)

# Split into syllable-ish tokens for the repetition score: on Tibetan tsheg
# (U+0F0B), shad (U+0F0D), and any whitespace. Keeps the n-gram signal
# meaningful on Unicode output (Tibetan rarely uses spaces).
_TOKEN_SPLIT = re.compile(r"[\u0f0b\u0f0d\s]+")


def normalize_text(pred: str) -> str:
    """Normalize a raw Unicode prediction to the canonical scoring form.

    **Hardened: never raises.** A single malformed/adversarial prediction (e.g.
    an unguarded loop hallucination) must never fail the whole volume; on any
    normalization error we log and fall back to the raw text.
    """
    if not pred:
        return ""
    try:
        return normalize_unicode_text(pred)
    except Exception:  # noqa: BLE001 — normalization must never crash the pipeline
        logger.warning(
            "[paddleocr.post] unicode normalization failed; returning raw text "
            "(len=%d, head=%r)",
            len(pred),
            pred[:80],
            exc_info=True,
        )
        return pred


def rep_score(pred: str, n: int = 20) -> float:
    """Repetition score in ``[0, 1]``: ``1 - unique(n-grams)/total(n-grams)``.

    Computed on syllable tokens (split on tsheg/shad/whitespace). ``0.0`` when
    there are fewer than ``n`` tokens (not enough context). Higher => more
    repetition; near 1.0 is a strong loop-hallucination signal.
    """
    tokens = [t for t in _TOKEN_SPLIT.split(pred) if t]
    if len(tokens) < n:
        return 0.0
    grams = [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]
    if not grams:
        return 0.0
    return 1.0 - (len(set(grams)) / len(grams))
