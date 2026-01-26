"""
Async OCR worker using the async pipeline.

Uses asyncio for high-concurrency S3 prefetching with backpressure.
"""

import asyncio
import json
import logging
import os
import time
from pathlib import Path
from typing import TYPE_CHECKING

import boto3
import pyarrow.parquet as pq
from botocore.config import Config as BotoConfig

if TYPE_CHECKING:
    from bec_orch.core.models import TaskResult
    from bec_orch.jobs.base import JobContext

from .ctc_decoder import CTCDecoder, DEFAULT_WORD_DELIMITERS, SPACE_ONLY_DELIMITERS, TIBETAN_WORD_DELIMITERS
from .model import OCRModel
from .pipeline_async import AsyncOCRPipeline

logger = logging.getLogger(__name__)


class OCRV1JobWorkerAsync:
    """
    Async OCR worker with pipeline architecture.

    Uses asyncio for S3 prefetching with backpressure, thread pools for
    CPU-bound image processing and CTC decoding.
    """

    def __init__(self, word_delimiters: frozenset[str] | None = None):
        """Initialize the OCR worker.
        
        Args:
            word_delimiters: Characters that trigger word boundaries for CTC decoding.
                           - None (default): Use TIBETAN_WORD_DELIMITERS for syllable-level decoding
                           - SPACE_ONLY_DELIMITERS: Original behavior for backward compatibility
                           - TIBETAN_WORD_DELIMITERS: Explicit syllable-level decoding
        """
        model_dir = os.environ.get("BEC_OCR_MODEL_DIR")
        if not model_dir:
            raise ValueError("BEC_OCR_MODEL_DIR environment variable not set.")

        model_dir = model_dir.strip("\"'")
        model_dir_path = Path(model_dir)
        if not model_dir_path.exists():
            raise FileNotFoundError(f"OCR model directory not found: {model_dir}")

        config_path = model_dir_path / "model_config.json"
        if not config_path.exists():
            raise FileNotFoundError(f"model_config.json not found in {model_dir}")

        logger.info(f"Loading OCR model config from: {config_path}")

        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)

        onnx_model_file = model_dir_path / config["onnx-model"]
        if not onnx_model_file.exists():
            raise FileNotFoundError(f"ONNX model file not found: {onnx_model_file}")

        self._input_width = config["input_width"]
        self._input_height = config["input_height"]
        charset = config["charset"]
        squeeze_channel = config["squeeze_channel_dim"] == "yes"
        swap_hw = config["swap_hw"] == "yes"
        add_blank = config["add_blank"] == "yes"

        logger.info(f"Loading OCR model: {onnx_model_file}")
        logger.info(
            f"  Architecture: {config.get('architecture', 'unknown')}, Version: {config.get('version', 'unknown')}"
        )
        logger.info(f"  Input: {self._input_width}x{self._input_height}, Charset length: {len(charset)}")

        self.ocr_model = OCRModel(
            model_file=str(onnx_model_file),
            input_layer=config["input_layer"],
            output_layer=config["output_layer"],
            squeeze_channel=squeeze_channel,
            swap_hw=swap_hw,
        )

        # Store word_delimiters for reference
        self.word_delimiters = word_delimiters if word_delimiters is not None else DEFAULT_WORD_DELIMITERS
        logger.info(f"  Word delimiters: {len(self.word_delimiters)} chars ({'Tibetan syllable' if self.word_delimiters == TIBETAN_WORD_DELIMITERS else 'space-only'})")
        
        self.ctc_decoder = CTCDecoder(charset=charset, add_blank=add_blank, word_delimiters=self.word_delimiters)

        self.ld_bucket = os.environ.get("BEC_LD_BUCKET", "bec.bdrc.io")
        self.source_image_bucket = os.environ.get("BEC_SOURCE_IMAGE_BUCKET", "archive.tbrc.org")

        # Create boto3 client for S3
        boto_config = BotoConfig(
            max_pool_connections=200,
            retries={"max_attempts": 3, "mode": "adaptive"},
        )
        self._s3_client = boto3.client("s3", config=boto_config)

        # Pipeline config
        self.prefetch_concurrency = 64
        self.image_processor_workers = 16
        self.ctc_workers = 8
        self.gpu_batch_size = 16
        self.beam_width: int | None = None  # None = use module default
        self.token_min_logp: float | None = None  # None = use module default
        self.vocab_prune_threshold: float | None = None  # None = use module default
        self.vocab_prune_mode: str | None = None  # None = use module default
        self.use_greedy_decode: bool = False  # Use fast greedy decode instead of beam search
        self.use_hybrid_decode: bool = True  # Greedy + beam search fallback for low-confidence lines
        self.greedy_confidence_threshold: float | None = None  # Confidence threshold for hybrid decode (None = module default -0.5)
        self.use_nemo_decoder: bool = False  # Use NeMo GPU decoder instead of pyctcdecode
        self.use_sequential_pipeline: bool = False  # Run GPU inference first, then CTC decode
        self.kenlm_path: str | None = None  # Path to KenLM language model for NeMo decoder
        self.debug_output_dir: str | None = None  # Directory to save preprocessed line images for debugging

        logger.info("OCR model loaded successfully")

    def run(self, ctx: "JobContext") -> "TaskResult":
        """Run OCR on a volume using async pipeline."""
        return asyncio.run(self._run_async(ctx))

    async def _run_async(self, ctx: "JobContext") -> "TaskResult":
        from bec_orch.core.models import TaskResult
        from bec_orch.errors import TerminalTaskError
        from bec_orch.jobs.shared.memory_monitor import log_memory_snapshot

        logger.info(f"Starting OCRV1 async job for volume {ctx.volume.w_id}/{ctx.volume.i_id}")
        start_time = time.time()

        log_memory_snapshot(f"[OCRV1] Start volume {ctx.volume.w_id}/{ctx.volume.i_id}")

        # Check LD success
        ld_success_uri = self._get_ld_success_uri(ctx)
        if not self._check_s3_exists(ld_success_uri):
            raise TerminalTaskError(
                f"Line detection not completed for volume {ctx.volume.w_id}/{ctx.volume.i_id}. "
                f"Expected success marker at: {ld_success_uri}"
            )
        logger.info(f"LD success marker found: {ld_success_uri}")

        # Read LD parquet
        ld_parquet_uri = self._get_ld_parquet_uri(ctx)
        output_parquet_uri = self._get_output_parquet_uri(ctx)

        logger.info(f"Reading LD parquet from {ld_parquet_uri}")
        input_table = pq.read_table(ld_parquet_uri)
        logger.info(f"Read {len(input_table)} rows from parquet")

        parquet_rows_by_filename: dict[str, dict] = {row["img_file_name"]: row for row in input_table.to_pylist()}
        logger.info(f"Indexed {len(parquet_rows_by_filename)} files from parquet")

        manifest_filenames: set[str] = {
            str(item["filename"]) for item in ctx.volume_manifest.manifest if item.get("filename")
        }

        missing_in_parquet = manifest_filenames - set(parquet_rows_by_filename.keys())
        if missing_in_parquet:
            raise TerminalTaskError(
                f"LD parquet missing {len(missing_in_parquet)} images from manifest: "
                f"{sorted(missing_in_parquet)[:5]}{'...' if len(missing_in_parquet) > 5 else ''}"
            )

        total_images = len(manifest_filenames)
        sorted_filenames = sorted(manifest_filenames)

        # Build page list
        pages = [
            (page_idx, filename, parquet_rows_by_filename[filename])
            for page_idx, filename in enumerate(sorted_filenames)
        ]

        # Get volume prefix for S3
        volume_prefix = self._get_volume_prefix(ctx)

        # Create and run pipeline
        pipeline = AsyncOCRPipeline(
            ocr_model=self.ocr_model,
            ctc_decoder=self.ctc_decoder,
            input_width=self._input_width,
            input_height=self._input_height,
            s3_client=self._s3_client,
            source_image_bucket=self.source_image_bucket,
            volume_prefix=volume_prefix,
            prefetch_concurrency=self.prefetch_concurrency,
            image_processor_workers=self.image_processor_workers,
            ctc_workers=self.ctc_workers,
            gpu_batch_size=self.gpu_batch_size,
            beam_width=self.beam_width,
            token_min_logp=self.token_min_logp,
            use_greedy_decode=self.use_greedy_decode,
            use_hybrid_decode=self.use_hybrid_decode,
            greedy_confidence_threshold=self.greedy_confidence_threshold,
            use_nemo_decoder=self.use_nemo_decoder,
            kenlm_path=self.kenlm_path,
            debug_output_dir=self.debug_output_dir,
        )
        pipeline.vocab_prune_threshold = self.vocab_prune_threshold
        pipeline.vocab_prune_mode = self.vocab_prune_mode

        try:
            if self.use_sequential_pipeline:
                stats = await pipeline.run_sequential(pages, output_parquet_uri)
            else:
                stats = await pipeline.run(pages, output_parquet_uri)
        finally:
            await pipeline.close()

        elapsed_ms = (time.time() - start_time) * 1000
        avg_duration_per_page_ms = elapsed_ms / max(1, total_images)

        log_memory_snapshot(f"[OCRV1] End volume {ctx.volume.w_id}/{ctx.volume.i_id}")

        logger.info(
            f"OCRV1 async job completed: {total_images} images, {stats['errors']} errors, "
            f"avg {avg_duration_per_page_ms:.2f}ms/page"
        )

        return TaskResult(
            total_images=total_images,
            nb_errors=stats["errors"],
            total_duration_ms=elapsed_ms,
            avg_duration_per_page_ms=avg_duration_per_page_ms,
            nb_dropped_records=0,
            errors_by_stage=None,
        )

    def _get_ld_success_uri(self, ctx: "JobContext") -> str:
        version = ctx.artifacts_location.basename.split("-")[-1]
        return f"s3://{self.ld_bucket}/ldv1/{ctx.volume.w_id}/{ctx.volume.i_id}/{version}/success.json"

    def _get_ld_parquet_uri(self, ctx: "JobContext") -> str:
        version = ctx.artifacts_location.basename.split("-")[-1]
        basename = f"{ctx.volume.w_id}-{ctx.volume.i_id}-{version}.parquet"
        return f"s3://{self.ld_bucket}/ldv1/{ctx.volume.w_id}/{ctx.volume.i_id}/{version}/{basename}"

    def _get_output_parquet_uri(self, ctx: "JobContext") -> str:
        return f"s3://{ctx.artifacts_location.bucket}/{ctx.artifacts_location.prefix}/{ctx.artifacts_location.basename}.parquet"

    def _get_volume_prefix(self, ctx: "JobContext") -> str:
        import hashlib

        w_prefix = hashlib.md5(ctx.volume.w_id.encode()).hexdigest()[:2]
        return f"Works/{w_prefix}/{ctx.volume.w_id}/images/{ctx.volume.w_id}-{ctx.volume.i_id}"

    def _check_s3_exists(self, uri: str) -> bool:
        import s3fs

        fs = s3fs.S3FileSystem()
        path = uri.replace("s3://", "")
        return fs.exists(path)
