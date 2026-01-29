"""
Async OCR worker using the async pipeline.

Uses asyncio for high-concurrency S3 prefetching with backpressure.
"""

import asyncio
import hashlib
import json
import logging
import os
import time
from pathlib import Path
from typing import TYPE_CHECKING

import boto3
import s3fs
from botocore.config import Config as BotoConfig

from bec_orch.core.models import TaskResult
from bec_orch.errors import TerminalTaskError
from bec_orch.jobs.shared.memory_monitor import log_memory_snapshot

if TYPE_CHECKING:
    from bec_orch.jobs.base import JobContext

from .config import TIBETAN_WORD_DELIMITERS, OCRV1Config
from .ctc_decoder import CTCDecoder
from .data_structures import ImageTask
from .model import OCRModel
from .pipeline_async import AsyncOCRPipeline

logger = logging.getLogger(__name__)


class OCRV1JobWorkerAsync:
    """
    Async OCR worker with pipeline architecture.

    Uses asyncio for S3 prefetching with backpressure, thread pools for
    CPU-bound image processing and CTC decoding.
    """

    def __init__(self, cfg: OCRV1Config | None = None) -> None:
        """Initialize the OCR worker.

        Args:
            cfg: OCRV1Config with all configuration options.
                 If None, a default config is created from model dimensions.
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

        with config_path.open(encoding="utf-8") as f:
            model_config = json.load(f)

        onnx_model_file = model_dir_path / model_config["onnx-model"]
        if not onnx_model_file.exists():
            raise FileNotFoundError(f"ONNX model file not found: {onnx_model_file}")

        input_width = model_config["input_width"]
        input_height = model_config["input_height"]
        charset = model_config["charset"]
        squeeze_channel = model_config["squeeze_channel_dim"] == "yes"
        swap_hw = model_config["swap_hw"] == "yes"
        add_blank = model_config["add_blank"] == "yes"

        # Create config if not provided
        if cfg is None:
            cfg = OCRV1Config(input_width=input_width, input_height=input_height)
        else:
            # Update config with model dimensions
            cfg.input_width = input_width
            cfg.input_height = input_height

        self.cfg = cfg

        logger.info(f"Loading OCR model: {onnx_model_file}")
        logger.info(
            f"  Architecture: {model_config.get('architecture', 'unknown')}, "
            f"Version: {model_config.get('version', 'unknown')}"
        )
        logger.info(f"  Input: {input_width}x{input_height}, Charset length: {len(charset)}")

        self.ocr_model = OCRModel(
            model_file=str(onnx_model_file),
            input_layer=model_config["input_layer"],
            output_layer=model_config["output_layer"],
            squeeze_channel=squeeze_channel,
            swap_hw=swap_hw,
            apply_log_softmax=cfg.apply_log_softmax,
            vocab_prune_threshold=cfg.vocab_prune_threshold,
        )

        logger.info(
            f"  Word delimiters: {len(cfg.word_delimiters)} chars "
            f"({'Tibetan syllable' if cfg.word_delimiters == TIBETAN_WORD_DELIMITERS else 'space-only'})"
        )

        self.ctc_decoder = CTCDecoder(charset=charset, add_blank=add_blank, word_delimiters=cfg.word_delimiters)

        self.ld_bucket = os.environ.get("BEC_LD_BUCKET", "bec.bdrc.io")
        self.source_image_bucket = os.environ.get("BEC_SOURCE_IMAGE_BUCKET", "archive.tbrc.org")

        # Create boto3 client for S3
        boto_config = BotoConfig(
            max_pool_connections=200,
            retries={"max_attempts": 3, "mode": "adaptive"},
        )
        self._s3_client = boto3.client("s3", config=boto_config)

        logger.info("OCR model loaded successfully")

    def run(self, ctx: "JobContext") -> "TaskResult":
        """Run OCR on a volume using async pipeline."""
        return asyncio.run(self._run_async(ctx))

    async def _run_async(self, ctx: "JobContext") -> "TaskResult":
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

        # Get URIs (parquet will be loaded async by pipeline)
        ld_parquet_uri = self._get_ld_parquet_uri(ctx)
        output_parquet_uri = self._get_output_parquet_uri(ctx)

        # Get manifest filenames
        manifest_filenames: set[str] = {
            str(item["filename"]) for item in ctx.volume_manifest.manifest if item.get("filename")
        }

        total_images = len(manifest_filenames)
        sorted_filenames = sorted(manifest_filenames)

        # Build ImageTask list (parquet loaded async by pipeline)
        tasks = [ImageTask(page_idx=page_idx, filename=filename) for page_idx, filename in enumerate(sorted_filenames)]

        logger.info(f"Starting pipeline for {total_images} images, parquet: {ld_parquet_uri}")

        # Get volume prefix for S3
        volume_prefix = self._get_volume_prefix(ctx)

        # Create and run pipeline
        pipeline = AsyncOCRPipeline(
            cfg=self.cfg,
            ocr_model=self.ocr_model,
            ctc_decoder=self.ctc_decoder,
            s3_client=self._s3_client,
            source_image_bucket=self.source_image_bucket,
            volume_prefix=volume_prefix,
        )

        try:
            if self.cfg.use_sequential_pipeline:
                logger.info(f"Using SEQUENTIAL pipeline mode for {total_images} images")
                stats = await pipeline.run_sequential(tasks, ld_parquet_uri, output_parquet_uri)
            else:
                logger.info(f"Using PARALLEL pipeline mode for {total_images} images")
                stats = await pipeline.run(tasks, ld_parquet_uri, output_parquet_uri)
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
        return f"s3://{ctx.artifacts_location.bucket}/{ctx.artifacts_location.prefix}/{ctx.artifacts_location.basename}_ocrv1.parquet"

    def _get_volume_prefix(self, ctx: "JobContext") -> str:
        w_prefix = hashlib.md5(ctx.volume.w_id.encode()).hexdigest()[:2]  # noqa: S324
        return f"Works/{w_prefix}/{ctx.volume.w_id}/images/{ctx.volume.w_id}-{ctx.volume.i_id}"

    def _check_s3_exists(self, uri: str) -> bool:
        fs = s3fs.S3FileSystem()
        path = uri.replace("s3://", "")
        return fs.exists(path)
