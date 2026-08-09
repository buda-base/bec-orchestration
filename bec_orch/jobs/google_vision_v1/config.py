"""Configuration for the ``google_vision_v1`` job.

Unlike the ML jobs, this worker does no GPU inference. Per volume it:

1. stream-copies every page image from S3 (``archive.tbrc.org``) to a GCS
   staging area (Google Vision's async batch API can only read from GCS),
2. submits one or more async ``DOCUMENT_TEXT_DETECTION`` batches to the Vision
   API (one batch sequence per *lane* -- ``images`` for jpg/png/jp2 and
   ``files`` for tif/tiff, mirroring the standalone buda-scripts pipeline),
3. polls the returned long-running operations to completion (all state kept in
   memory for the duration of the volume -- there is no external DB), then
4. reads the Vision JSON back from GCS and writes a per-volume parquet +
   ``jsonl.zst`` to the destination S3 bucket.

Every field has a default, so an empty job config is valid; any key present in
the DB ``jobs.config`` JSON overrides the matching default (see the factory in
``bec_orch.core.registry``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

# Google Vision async image annotation hard limit (requests per batch).
VISION_MAX_BATCH_SIZE = 2000


@dataclass
class GoogleVisionV1Config:
    """Tunable parameters for ``google_vision_v1``."""

    # ------------------------------------------------------------------
    # Google credentials / project
    # ------------------------------------------------------------------
    # Path to a Google service-account JSON key used for BOTH the Vision API
    # and GCS. If ``None``/empty, Application Default Credentials are used
    # (e.g. ``GOOGLE_APPLICATION_CREDENTIALS`` or an attached GCP identity).
    google_credentials_path: Optional[str] = None

    # Optional explicit GCP project id (usually inferred from the credentials).
    gcp_project: Optional[str] = None

    # ------------------------------------------------------------------
    # GCS staging (S3 -> GCS image copy). Images are KEPT (no cleanup).
    # Staged at gs://{staging_bucket}/{staging_prefix}{w_id}/{i_id}/{version}/{filename}
    # ------------------------------------------------------------------
    staging_bucket: str = "archive-mirror.tbrc.org"
    staging_prefix: str = "google_vision_v1_staging/"

    # If the staging blob already exists (same path), skip re-copying it. Makes
    # reprocessing / retries cheap and idempotent.
    skip_existing_gcs: bool = True

    # ThreadPoolExecutor size for the parallel S3 GET -> GCS PUT copy.
    transfer_concurrency: int = 32

    # ------------------------------------------------------------------
    # GCS Vision output. Written by the Vision API to
    # gs://{output_bucket}/{output_prefix}{w_id}/{i_id}/{version}/{lane}/{batch_id}/
    # and KEPT (no cleanup).
    # ------------------------------------------------------------------
    output_bucket: str = "bec.bdrc.io"
    output_prefix: str = "google_vision_v1_vision-json/"

    # Responses per GCS output file (Vision ``OutputConfig.batch_size``).
    output_shard_size: int = 100

    # ------------------------------------------------------------------
    # Vision request shape
    # ------------------------------------------------------------------
    feature_type: str = "DOCUMENT_TEXT_DETECTION"
    model: str = "builtin/weekly"

    # Images per submitted async batch (<= VISION_MAX_BATCH_SIZE). One batch
    # sequence is produced per lane.
    batch_size: int = 500

    # Max async operations kept in flight simultaneously for a single volume.
    max_concurrent_ops: int = 8

    # ------------------------------------------------------------------
    # Polling / waiting
    # ------------------------------------------------------------------
    # Seconds between poll cycles over the in-flight operations.
    poll_interval_s: int = 30

    # Hard ceiling on how long one volume may wait for all operations to finish
    # before it is reported as a retryable failure. Keep this comfortably below
    # the runtime's ``visibility_max_total_seconds`` (default 4h).
    volume_timeout_s: float = 10800.0

    # ------------------------------------------------------------------
    # Quota (HTTP 429 / RESOURCE_EXHAUSTED) backoff on submit
    # ------------------------------------------------------------------
    quota_pause_s: float = 60.0
    quota_pause_multiplier: float = 1.5
    max_quota_pause_s: float = 900.0
    max_submit_retries: int = 5

    # ------------------------------------------------------------------
    # Result download / output writing
    # ------------------------------------------------------------------
    # Parallel GCS blob downloads when reading Vision output back.
    gcs_download_concurrency: int = 8

    # Per-GCS-blob download timeout (seconds).
    gcs_download_timeout_s: int = 180

    # zstd is used for the raw jsonl sidecar (matches export_volume_ocr.py).
    jsonl_zstd_level: int = 1

    # Parquet codec. ``snappy`` matches the existing google-vision exports so
    # downstream tooling reads these files unchanged.
    parquet_compression: Literal["snappy", "zstd", "gzip", "none"] = "snappy"

    # Append ``-gv`` to artifact basenames to match the standalone pipeline's
    # ``{w}-{i}-{version}-gv.parquet`` / ``.jsonl.zst`` naming.
    #
    # NOTE: the top-level output directory (default ``{job_name}/...``) is
    # controlled by the runtime, not this job. Set ``"artifact_prefix": "gv"`` in
    # the DB ``jobs.config`` to make the whole artifact location (parquet +
    # jsonl.zst + success.json) land under ``s3://{dest}/gv/{w}/{i}/{version}/``.
    artifact_suffix: str = "-gv"

    # ------------------------------------------------------------------
    # S3 (source images + destination artifacts)
    # ------------------------------------------------------------------
    s3_get_timeout_s: int = 60
    s3_max_attempts: int = 3

    # ------------------------------------------------------------------
    # Failure handling
    # ------------------------------------------------------------------
    # Fraction of pages that may fail (transfer, submit, or missing OCR
    # response) before the whole volume is reported as a retryable failure.
    max_page_failure_rate: float = 0.05

    def __post_init__(self) -> None:
        if not self.staging_bucket:
            raise ValueError("staging_bucket must be set")
        if not self.output_bucket:
            raise ValueError("output_bucket must be set")
        if self.batch_size < 1 or self.batch_size > VISION_MAX_BATCH_SIZE:
            raise ValueError(
                f"batch_size must be in [1, {VISION_MAX_BATCH_SIZE}], got {self.batch_size}"
            )
        if self.output_shard_size < 1:
            raise ValueError(f"output_shard_size must be >= 1, got {self.output_shard_size}")
        if self.max_concurrent_ops < 1:
            raise ValueError(f"max_concurrent_ops must be >= 1, got {self.max_concurrent_ops}")
        if self.transfer_concurrency < 1:
            raise ValueError(f"transfer_concurrency must be >= 1, got {self.transfer_concurrency}")
        if self.gcs_download_concurrency < 1:
            raise ValueError(f"gcs_download_concurrency must be >= 1, got {self.gcs_download_concurrency}")
        if self.poll_interval_s < 1:
            raise ValueError(f"poll_interval_s must be >= 1, got {self.poll_interval_s}")
        if self.volume_timeout_s <= 0:
            raise ValueError(f"volume_timeout_s must be > 0, got {self.volume_timeout_s}")
        if self.max_submit_retries < 1:
            raise ValueError(f"max_submit_retries must be >= 1, got {self.max_submit_retries}")
        if not 0.0 <= self.max_page_failure_rate <= 1.0:
            raise ValueError(f"max_page_failure_rate out of [0,1]: {self.max_page_failure_rate}")

        # Normalize prefixes to end with exactly one "/".
        self.staging_prefix = self._norm_prefix(self.staging_prefix)
        self.output_prefix = self._norm_prefix(self.output_prefix)

    @staticmethod
    def _norm_prefix(prefix: str) -> str:
        prefix = (prefix or "").lstrip("/")
        if prefix and not prefix.endswith("/"):
            prefix += "/"
        return prefix
