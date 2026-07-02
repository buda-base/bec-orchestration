"""Vendored port of https://github.com/OpenPecha/tibetan-manuscript-classifier.

Kept as a 1:1 diffable subpackage against upstream. Adaptations from
upstream are limited to:
  - pipeline.py: the blank-page pre-filter call is commented out (untested,
    no promising results), and the `blank`/`flip_applied` fields are
    dropped from the returned row (see pipeline.py for details).

Everything else (config.py, transforms.py, models.py, blank.py, loader.py)
is reproduced verbatim from upstream.
"""

from .loader import get_pipeline
from .pipeline import Pipeline

__all__ = ["Pipeline", "get_pipeline"]
