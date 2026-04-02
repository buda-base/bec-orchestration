from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class HFFConfig:
    """Configuration for the HFF-Remover Surya pipeline.

    All fields have sensible defaults; override via the job's ``config`` column
    in the ``jobs`` table (merged on top of defaults in ``HFFv1JobWorker``).
    """

    # ── S3 ────────────────────────────────────────────────────────────────────
    s3_source_bucket: str = "archive.tbrc.org"
    s3_region: str = "us-east-1"

    # ── Surya / detection ─────────────────────────────────────────────────────
    # Minimum confidence to keep a detection (0–1).
    confidence_threshold: float = 0.5
    # Extra white pixels added around each masked region.
    margin: int = 0
    # When True only header/footer/footnote detections are returned by the
    # detector; set to False to keep all Surya layout classes (for debugging).
    filter_to_hff_only: bool = True

    # ── Output ────────────────────────────────────────────────────────────────
    # JPEG quality for the masked output images written back to S3 (0–95).
    output_quality: int = 95
    # Image format written to S3.  "JPEG" or "PNG".
    output_format: str = "JPEG"

    # ── Concurrency ──────────────────────────────────────────────────────────
    # Number of S3 GETs that may be in flight at the same time.
    s3_concurrency: int = 32
    # Timeout (seconds) for a single S3 GET.
    s3_get_timeout_s: int = 60

    # ── Debug ────────────────────────────────────────────────────────────────
    debug_mode: bool = False
