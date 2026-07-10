"""Configuration for ``script_classification_v2`` — BDRC 8-class page classifier.

Runs a single fine-tuned DINOv3 image classifier
(``BDRC/8-class-tibetan-page-classifier``) over each page and records the full
softmax vector. Override any field at job-creation time via the JSON config
file or ``--config-text`` flag of ``bec jobs create``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass
class ScriptClassificationV2Config:
    """Tunable parameters for ``script_classification_v2``.

    NOTE: the fine-tuned checkpoint is downloaded from the HuggingFace Hub at
    first run into the standard HF cache (set ``HF_HOME`` to relocate). The
    BDRC repo is gated, so an ``HF_TOKEN`` env var (or ``huggingface-cli
    login``) with access to it must be present before the worker process
    starts. The DINOv3 backbone itself is NOT needed from the Hub: its
    pretrained weights are fully overwritten by the fine-tuned checkpoint, and
    its architecture/normalization are reconstructed from transformers'
    bundled ``DINOv3ViT`` classes (see ``vendor/models.py`` /
    ``vendor/transforms.py``), so gated access to ``facebook/dinov3-*`` is not
    required.
    """

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    # HuggingFace repo id of the fine-tuned classifier checkpoint.
    model_repo_id: str = "BDRC/8-class-tibetan-page-classifier"

    # Checkpoint filename inside the repo. All BDRC DINOv3 classifiers ship
    # ``final_model.pt`` (a dict with ``model_state_dict`` + ``idx_to_label``
    # [+ optional ``pooling``]).
    checkpoint_filename: str = "final_model.pt"

    # DINOv3 backbone architecture id. Only the architecture + normalization
    # constants are used; the weights come from the checkpoint. The 8-class
    # checkpoint is a ViT-B/16 (hidden 768, ``cls_mean_std`` pooling) —
    # confirmed from the checkpoint's own ``model_id`` field. Kept overridable
    # for future checkpoints.
    backbone_id: str = "facebook/dinov3-vitb16-pretrain-lvd1689m"

    # Square center-crop size (px) fed to the backbone. The BDRC classifiers
    # are trained with a fixed short-edge-resize + center-crop; this MUST
    # match the checkpoint's training size or accuracy degrades. The 8-class
    # page classifier trains at 448 (``dinov3_8way_vitb16_center_crop``:
    # short-edge resize to 448 then center crop — confirmed from the repo's
    # README / results.json), same as the 6-class script model.
    crop_size: int = 448

    # ------------------------------------------------------------------
    # I/O pipeline
    # ------------------------------------------------------------------
    # Pages fetched from S3 (raw bytes only — the pipeline decodes internally)
    # per fetch -> classify -> write cycle. Also the model forward-pass batch
    # size (the whole fetched batch is classified in one ``run_batch`` call).
    classify_batch_size: int = 64

    # ThreadPoolExecutor size for parallel S3 GET.
    s3_fetch_concurrency: int = 32

    # Batches the background prefetch thread may fetch ahead of the classifier
    # (bounded queue depth). >=1 overlaps S3 fetch with CPU decode + GPU
    # forward so neither the network nor the GPU sits idle.
    prefetch_batches: int = 1

    # Per-page S3 GET timeout (boto3 read/connect timeout).
    s3_get_timeout_s: int = 30

    # Per-page S3 retry attempts (boto3 standard retry mode handles backoff).
    s3_max_attempts: int = 3

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------
    # ThreadPoolExecutor size for parallel per-image decode+resize+crop within
    # a single classify batch. Controls decode/crop throughput, NOT model-call
    # concurrency (the forward pass is a single batched call per batch).
    decode_workers: int = 16

    # When True (default), the classifier runs on CUDA if available — falls
    # back to CPU with a logged warning if CUDA isn't present (never a hard
    # failure).
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
    # Volume-level timeout. A single DINOv3 ViT-S/16 forward pass is cheap;
    # 30 min mirrors the other jobs' default.
    volume_timeout_s: float = 1800.0

    # Fraction of pages that may fail (S3 fetch or classify) before the whole
    # volume is reported as a terminal failure.
    max_page_failure_rate: float = 0.05

    def __post_init__(self) -> None:
        if self.crop_size < 16 or self.crop_size % 16 != 0:
            raise ValueError(
                f"crop_size must be a positive multiple of 16 (DINOv3 patch size), "
                f"got {self.crop_size}"
            )
        if self.classify_batch_size < 1:
            raise ValueError(f"classify_batch_size must be >= 1, got {self.classify_batch_size}")
        if self.s3_fetch_concurrency < 1:
            raise ValueError(f"s3_fetch_concurrency must be >= 1, got {self.s3_fetch_concurrency}")
        if self.prefetch_batches < 1:
            raise ValueError(f"prefetch_batches must be >= 1, got {self.prefetch_batches}")
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
