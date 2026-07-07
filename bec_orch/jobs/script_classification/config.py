"""Configuration for ``script_classification`` — BDRC Tibetan page classifier.

Wraps the vendored ``tibetan-manuscript-classifier`` pipeline (orientation +
6-class script classification; the blank-page pre-filter is disabled, see
``vendor/pipeline.py``). Override any field at job-creation time via the
JSON config file or ``--config-text`` flag of ``bec jobs create``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass
class ScriptClassificationConfig:
    """Tunable parameters for ``script_classification``.

    NOTE: both models (orientation + 6-class) are downloaded from the
    HuggingFace Hub at first run into the standard HF cache (set ``HF_HOME``
    to relocate). Both HF repos, plus the shared DINOv3 backbone, are
    gated — an ``HF_TOKEN`` env var (or ``huggingface-cli login``) must be
    present before the worker process starts.
    """

    # ------------------------------------------------------------------
    # I/O pipeline
    # ------------------------------------------------------------------
    # Pages fetched from S3 (raw bytes only — the vendored pipeline decodes
    # internally) per fetch -> classify -> write cycle. Bounds how many raw
    # image byte-strings are resident in the parent process at once. As of
    # batched inference (vendor/pipeline.py::Pipeline.run_batch), this ALSO
    # doubles as the model's forward-pass batch size -- the whole fetched
    # batch is classified in one run_batch() call, one model call per
    # batch rather than one per image. There is no separate model-batch-size
    # cap: ViT-S/16 activation memory at N=64, 448x448 is trivial on either
    # CPU or GPU. If this is ever raised substantially for unrelated
    # S3-throughput reasons, be aware it also raises the model batch size.
    classify_batch_size: int = 64

    # ThreadPoolExecutor size for parallel S3 GET. This is sized for S3
    # throughput and is independent of the CPU-side decode parallelism (see
    # ``decode_workers`` below) and the batched model forward passes.
    s3_fetch_concurrency: int = 32

    # Per-page S3 GET timeout (boto3 read/connect timeout).
    s3_get_timeout_s: int = 30

    # Per-page S3 retry attempts (boto3 standard retry mode handles backoff).
    s3_max_attempts: int = 3

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------
    # ThreadPoolExecutor size for parallel per-image decode+resize+crop
    # *within* a single classify batch (see
    # vendor/pipeline.py::Pipeline.run_batch's self._decode_pool). This is
    # where the CPU-side parallelism lives -- each of the two model forward
    # passes is a single batched call per classify batch, not one call per
    # image, so this knob controls decode/crop throughput, NOT model-call
    # concurrency (hence the name: decode workers, not inference workers).
    # Default is a placeholder sized by analogy to ldv1's tile_workers (24)
    # -- NOT derived from real fleet data, profile on the actual target
    # instance type before trusting it in production. See
    # bec-script-classification.service's MemoryHigh/MemoryMax, which were
    # sized for the OLD one-image-at-a-time decode profile and may need
    # re-tuning now that up to this many decodes can be in flight
    # concurrently.
    decode_workers: int = 16

    # When True (default), both classifiers run on CUDA if available --
    # falls back to CPU with a logged warning if CUDA isn't present (never
    # a hard failure). See vendor/pipeline.py::Pipeline.__init__.
    use_gpu: bool = True

    # ------------------------------------------------------------------
    # Output writer
    # ------------------------------------------------------------------
    # Parquet flush threshold (records buffered before each ``write_table``).
    parquet_flush_every: int = 256

    # zstd is the project standard (also used by LDV1 / OCRV1 / ocr_qwen_v1).
    parquet_compression: Literal["zstd", "snappy", "gzip", "none"] = "zstd"

    # Also write a sidecar ``<basename>-errors.jsonl`` with per-page failure
    # details.
    write_errors_jsonl: bool = True

    # ------------------------------------------------------------------
    # Failure handling
    # ------------------------------------------------------------------
    # Volume-level timeout. A single DINOv3 ViT-S/16 forward pass on CPU is
    # cheap relative to OCR generation; 30 min mirrors the other jobs'
    # default and should be revisited after a real smoke test (see README).
    volume_timeout_s: float = 1800.0

    # Fraction of pages that may fail (S3 fetch or classify) before the
    # whole volume is reported as a terminal failure.
    max_page_failure_rate: float = 0.05

    def __post_init__(self) -> None:
        # Cheap validation to catch typos before the model-load pass.
        if self.classify_batch_size < 1:
            raise ValueError(f"classify_batch_size must be >= 1, got {self.classify_batch_size}")
        if self.s3_fetch_concurrency < 1:
            raise ValueError(f"s3_fetch_concurrency must be >= 1, got {self.s3_fetch_concurrency}")
        if self.s3_get_timeout_s < 1:
            raise ValueError(f"s3_get_timeout_s must be >= 1, got {self.s3_get_timeout_s}")
        if self.s3_max_attempts < 1:
            raise ValueError(f"s3_max_attempts must be >= 1, got {self.s3_max_attempts}")
        if self.decode_workers < 1:
            raise ValueError(f"decode_workers must be >= 1, got {self.decode_workers}")
        if self.parquet_flush_every < 1:
            raise ValueError(f"parquet_flush_every must be >= 1, got {self.parquet_flush_every}")
        if self.volume_timeout_s <= 0:
            raise ValueError(f"volume_timeout_s must be > 0, got {self.volume_timeout_s}")
        if not 0.0 <= self.max_page_failure_rate <= 1.0:
            raise ValueError(f"max_page_failure_rate out of [0,1]: {self.max_page_failure_rate}")
