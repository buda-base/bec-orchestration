"""HFFv1 job worker — fetch images from S3 and detect header/footer/footnote
regions using the Surya layout model.

Job name: ``hffv1``

Environment variables
---------------------
BEC_REGION          AWS region (default: us-east-1)
"""
from __future__ import annotations

import io
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Tuple

import boto3
import numpy as np
from botocore.config import Config as BotoConfig
from PIL import Image

from bec_orch.core.models import TaskResult
from bec_orch.errors import RetryableTaskError, TerminalTaskError
from bec_orch.jobs.base import JobContext
from bec_orch.jobs.hffv1.config import HFFConfig

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _bytes_to_bgr(raw: bytes) -> np.ndarray:
    """Decode raw image bytes → BGR numpy array (OpenCV / HFF convention)."""
    pil = Image.open(io.BytesIO(raw)).convert("RGB")
    return np.array(pil)[:, :, ::-1]


def _s3_key_from_uri(uri: str) -> Tuple[str, str]:
    """Parse ``s3://bucket/key`` → ``(bucket, key)``."""
    parsed = urlparse(uri)
    return parsed.netloc, parsed.path.lstrip("/")


def _get_s3_folder_prefix(w_id: str, i_id: str) -> str:
    """Return the S3 key prefix for a BDRC volume's images.

    Mirrors ``get_s3_folder_prefix`` from ``bec_orch.core.worker_runtime``.
    """
    try:
        from bec_orch.core.worker_runtime import get_s3_folder_prefix
        return get_s3_folder_prefix(w_id, i_id)
    except ImportError:
        # Fallback: construct the prefix directly
        return f"Works/{w_id[:2]}/{w_id}/{i_id}/"


# ──────────────────────────────────────────────────────────────────────────────
# Core processing
# ──────────────────────────────────────────────────────────────────────────────

class _ImageProcessor:
    """Holds shared, reusable objects (S3 client and Surya detector)."""

    def __init__(self, cfg: HFFConfig) -> None:
        from hff_remover.detector import SuryaLayoutDetector

        logger.info("Loading Surya layout model …")
        self.detector = SuryaLayoutDetector(
            confidence_threshold=cfg.confidence_threshold,
        )
        logger.info("Surya layout model loaded.")

        boto_cfg = BotoConfig(
            max_pool_connections=cfg.s3_concurrency,
            retries={"max_attempts": 3, "mode": "adaptive"},
        )
        self.s3 = boto3.client(
            "s3",
            region_name=cfg.s3_region,
            config=boto_cfg,
        )
        self.cfg = cfg

    def process_image(
        self,
        source_bucket: str,
        source_key: str,
    ) -> Dict[str, Any]:
        """Fetch one image, run HFF detection, and return results.

        Returns a dict with ``filename``, ``detections``, ``duration_ms``,
        and ``error`` (None on success).
        """
        filename = source_key.split("/")[-1]
        t0 = time.time()
        try:
            # ── fetch ──────────────────────────────────────────────────────
            body = self.s3.get_object(Bucket=source_bucket, Key=source_key)[
                "Body"
            ].read()

            # ── detect ─────────────────────────────────────────────────────
            bgr = _bytes_to_bgr(body)
            dets = self.detector.detect(
                bgr,
                filter_to_hff_only=self.cfg.filter_to_hff_only,
            )

            duration_ms = (time.time() - t0) * 1000
            if self.cfg.debug_mode:
                logger.debug(
                    "[HFFv1] %s — %d detection(s) — %.0f ms",
                    filename,
                    len(dets),
                    duration_ms,
                )
            return {
                "filename": filename,
                "detections": len(dets),
                "duration_ms": duration_ms,
                "error": None,
            }

        except Exception as exc:
            duration_ms = (time.time() - t0) * 1000
            logger.warning(
                "[HFFv1] Failed to process %s: %s", filename, exc, exc_info=True
            )
            return {
                "filename": filename,
                "detections": 0,
                "duration_ms": duration_ms,
                "error": str(exc),
            }


# ──────────────────────────────────────────────────────────────────────────────
# JobWorker implementation
# ──────────────────────────────────────────────────────────────────────────────

class HFFv1JobWorker:
    """BEC orchestration job worker for HFF detection using Surya.

    The Surya model is loaded **once** in ``__init__`` and reused across every
    volume assigned to this worker process, matching the same pattern as
    ``LDV1JobWorker``.

    Registers as job name ``hffv1``.
    """

    def __init__(self) -> None:
        # Defer heavy imports / model load until first use so that the worker
        # process can start quickly and only pays the cost when it has work.
        self._proc: Optional[_ImageProcessor] = None

    # ── helpers ───────────────────────────────────────────────────────────────

    def _get_processor(self, cfg: HFFConfig) -> _ImageProcessor:
        """Lazily create (and cache) the shared processor."""
        if self._proc is None:
            self._proc = _ImageProcessor(cfg)
        return self._proc

    def _build_config(self, ctx: JobContext) -> HFFConfig:
        """Merge defaults with per-job overrides from the DB ``jobs.config``."""
        defaults: Dict[str, Any] = {
            "s3_source_bucket": "archive.tbrc.org",
            "s3_region": os.environ.get("BEC_REGION", "us-east-1"),
        }
        merged = {**defaults, **ctx.job_config}
        # Keep only fields that HFFConfig knows about to avoid dataclass errors.
        valid_fields = HFFConfig.__dataclass_fields__.keys()
        filtered = {k: v for k, v in merged.items() if k in valid_fields}
        return HFFConfig(**filtered)

    def _build_image_list(
        self,
        ctx: JobContext,
        cfg: HFFConfig,
    ) -> List[Tuple[str, str]]:
        """Build ``[(source_bucket, source_key), ...]``."""
        vol_prefix = _get_s3_folder_prefix(ctx.volume.w_id, ctx.volume.i_id)

        items: List[Tuple[str, str]] = []
        for item in ctx.volume_manifest.manifest:
            filename = item.get("filename")
            if not filename:
                continue
            source_key = f"{vol_prefix}{filename}"
            items.append((cfg.s3_source_bucket, source_key))
        return items

    # ── JobWorker protocol ────────────────────────────────────────────────────

    def run(self, ctx: JobContext) -> TaskResult:
        """Process one volume: detect HFF regions in every image."""
        logger.info(
            "[HFFv1] Starting volume %s/%s (%d images)",
            ctx.volume.w_id,
            ctx.volume.i_id,
            len(ctx.volume_manifest.manifest),
        )
        start = time.time()

        cfg = self._build_config(ctx)
        proc = self._get_processor(cfg)
        image_list = self._build_image_list(ctx, cfg)

        if not image_list:
            logger.warning("[HFFv1] No images found in manifest — nothing to do.")
            return TaskResult(
                total_images=0,
                nb_errors=0,
                total_duration_ms=0.0,
                avg_duration_per_page_ms=0.0,
            )

        results: List[Dict[str, Any]] = []

        # Process images in parallel using a thread pool (Surya is CPU-bound;
        # threads let S3 I/O overlap with inference on other images).
        with ThreadPoolExecutor(
            max_workers=cfg.s3_concurrency, thread_name_prefix="hffv1"
        ) as pool:
            futures = {
                pool.submit(proc.process_image, src_b, src_k): (
                    src_k.split("/")[-1]
                )
                for src_b, src_k in image_list
            }
            for future in as_completed(futures):
                try:
                    results.append(future.result())
                except Exception as exc:
                    filename = futures[future]
                    logger.error("[HFFv1] Unhandled error for %s: %s", filename, exc)
                    results.append(
                        {
                            "filename": filename,
                            "detections": 0,
                            "duration_ms": 0.0,
                            "error": str(exc),
                        }
                    )

        # ── aggregate metrics ─────────────────────────────────────────────────
        errors = [r for r in results if r["error"] is not None]
        successes = [r for r in results if r["error"] is None]
        durations = [r["duration_ms"] for r in successes]

        total_images = len(results)
        nb_errors = len(errors)
        total_duration_ms = (time.time() - start) * 1000
        avg_duration_per_page_ms = (
            sum(durations) / len(durations) if durations else total_duration_ms
        )

        logger.info(
            "[HFFv1] Finished volume %s/%s — %d ok / %d errors in %.1f s",
            ctx.volume.w_id,
            ctx.volume.i_id,
            len(successes),
            nb_errors,
            total_duration_ms / 1000,
        )

        if nb_errors == total_images and total_images > 0:
            # Every single image failed — something systemic, make it retryable.
            raise RetryableTaskError(
                f"All {total_images} images failed. First error: {errors[0]['error']}"
            )

        if nb_errors > 0:
            # Some images failed — treat as terminal (bad source data).
            raise TerminalTaskError(
                f"{nb_errors}/{total_images} images failed. "
                f"First error: {errors[0]['error']}"
            )

        return TaskResult(
            total_images=total_images,
            nb_errors=nb_errors,
            total_duration_ms=total_duration_ms,
            avg_duration_per_page_ms=avg_duration_per_page_ms,
        )
