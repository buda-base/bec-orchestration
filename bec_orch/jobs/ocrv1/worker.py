"""
Synchronous OCR worker for BEC orchestration integration.

Implements the JobWorker protocol expected by BECWorkerRuntime.
The model is loaded once in __init__ and reused across all volumes.
"""

import json
import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING

import boto3
from botocore.config import Config

from bec_orch.errors import RetryableTaskError, TerminalTaskError
from bec_orch.jobs.shared.memory_monitor import log_memory_snapshot

if TYPE_CHECKING:
    from bec_orch.core.models import TaskResult
    from bec_orch.jobs.base import JobContext

from .config import TIBETAN_WORD_DELIMITERS, OCRV1Config
from .ctc_decoder import CTCDecoder
from .model import OCRModel
from .worker_async import OCRV1JobWorkerAsync

logger = logging.getLogger(__name__)


class OCRV1JobWorker:
    """
    Synchronous adapter for OCRV1JobWorkerAsync to integrate with BEC orchestration.

    Implements the JobWorker protocol expected by BECWorkerRuntime.

    The model is loaded once in __init__ and reused across all volumes to avoid
    reloading overhead. The model stays loaded between volumes.
    """

    def __init__(self, cfg: OCRV1Config) -> None:
        """Initialize the job worker and load the model."""
        self.cfg = cfg

        # Validate model is specified
        if not cfg.model or cfg.model.strip() == "":
            raise ValueError(
                "model is required in OCRV1Config. Available models: Woodblock, Ume_Druma, Ume_Petsuk, Modern, etc."
            )

        # Get base model directory from environment (required for ocrv1)
        base_model_dir = os.environ.get("BEC_OCR_MODEL_DIR")
        if not base_model_dir:
            raise ValueError(
                "BEC_OCR_MODEL_DIR environment variable not set. "
                "Set it to the base directory containing model subdirectories."
            )

        # Strip quotes if present (common in .env files)
        base_model_dir = base_model_dir.strip("\"'")

        # Construct full model path using model from config
        model_name = cfg.model
        model_dir = Path(base_model_dir) / model_name

        logger.info(f"Using model: {model_name}")
        logger.info(f"Full model path: {model_dir}")

        # Validate model directory exists
        if not model_dir.exists():
            raise FileNotFoundError(
                f"OCR model directory not found: {model_dir}\n"
                f"Available models in {base_model_dir}: {[d.name for d in Path(base_model_dir).iterdir() if d.is_dir()]}\n"
                f"Please check that BEC_OCR_MODEL_DIR is set correctly and model '{model_name}' exists."
            )

        config_path = model_dir / "model_config.json"
        if not config_path.exists():
            raise FileNotFoundError(f"model_config.json not found in {model_dir}")

        logger.info(f"Loading OCR model config from: {config_path}")

        # Load model configuration
        with config_path.open(encoding="utf-8") as f:
            model_config = json.load(f)

        onnx_model_file = model_dir / model_config["onnx-model"]
        if not onnx_model_file.exists():
            raise FileNotFoundError(f"ONNX model file not found: {onnx_model_file}")

        # Extract model configuration
        input_width = model_config["input_width"]
        input_height = model_config["input_height"]
        charset = model_config["charset"]
        squeeze_channel = model_config["squeeze_channel_dim"] == "yes"
        swap_hw = model_config["swap_hw"] == "yes"
        add_blank = model_config["add_blank"] == "yes"

        logger.debug(f"Loading OCR model: {onnx_model_file}")
        logger.debug(
            f"  Architecture: {model_config.get('architecture', 'unknown')}, "
            f"Version: {model_config.get('version', 'unknown')}"
        )
        logger.debug(f"  Input: {input_width}x{input_height}, Charset length: {len(charset)}")

        # Load OCR model
        self.ocr_model = OCRModel(
            model_file=str(onnx_model_file),
            input_layer=model_config["input_layer"],
            output_layer=model_config["output_layer"],
            input_width=input_width,
            input_height=input_height,
            squeeze_channel=squeeze_channel,
            swap_hw=swap_hw,
            apply_log_softmax=cfg.apply_log_softmax,
            use_gpu_operations=True,
            vocab_prune_threshold=cfg.vocab_prune_threshold,
        )

        logger.debug(
            f"  Word delimiters: {len(cfg.word_delimiters)} chars "
            f"({'Tibetan syllable' if cfg.word_delimiters == TIBETAN_WORD_DELIMITERS else 'space-only'})"
        )

        self.ctc_decoder = CTCDecoder(charset=charset, add_blank=add_blank, word_delimiters=cfg.word_delimiters)

        self.ld_bucket = os.environ.get("BEC_LD_BUCKET", "bec.bdrc.io")
        self.source_image_bucket = os.environ.get("BEC_SOURCE_IMAGE_BUCKET", "archive.tbrc.org")

        boto_config = Config(
            max_pool_connections=200,
            retries={"max_attempts": 3, "mode": "adaptive"},
        )
        self._s3_client = boto3.client("s3", config=boto_config)

        logger.debug("OCR model loaded successfully")

        # Create async worker with the loaded model and config (model stays loaded between volumes, same as ldv1)
        self.async_worker = OCRV1JobWorkerAsync(cfg, ocr_model=self.ocr_model, ctc_decoder=self.ctc_decoder)

    def run(self, ctx: "JobContext") -> "TaskResult":
        """
        Run OCR on a volume.

        Args:
            ctx: Job context with volume info, config, and artifact location

        Returns:
            TaskResult with metrics
        """
        logger.info(f"Starting OCRV1 job for volume {ctx.volume.w_id}/{ctx.volume.i_id}")

        # Log memory at start of volume processing
        log_memory_snapshot(f"[OCRV1] Start volume {ctx.volume.w_id}/{ctx.volume.i_id}")

        # Track metrics
        metrics = {
            "total_images": len(ctx.volume_manifest.manifest),
            "nb_errors": 0,
        }

        try:
            # Run async pipeline and collect results
            result = self.async_worker.run(ctx)

            # Update metrics from async result
            metrics["nb_errors"] = result.nb_errors

            log_memory_snapshot(f"[OCRV1] End volume {ctx.volume.w_id}/{ctx.volume.i_id}")

            logger.info(
                f"OCRV1 job completed: {result.total_images} images, {result.nb_errors} errors, "
                f"avg {result.avg_duration_per_page_ms:.2f}ms/page"
            )

        except (TerminalTaskError, RetryableTaskError):
            raise

        except Exception as e:
            # Log memory state at failure (helps diagnose OOM)
            log_memory_snapshot(f"[OCRV1] FAILED volume {ctx.volume.w_id}/{ctx.volume.i_id}", level=logging.ERROR)

            # Log detailed error information
            error_type = type(e).__name__
            logger.exception(f"OCRV1 pipeline failed for volume {ctx.volume.w_id}/{ctx.volume.i_id}: {error_type}")
            # Re-classify exceptions for orchestration layer
            # Classify based on error type
            if "CUDA out of memory" in str(e) or "OOM" in str(e):
                logger.exception("GPU out of memory error detected - task will be retried")
                raise RetryableTaskError(f"GPU OOM error: {e}") from e

            if "NotFound" in error_type or "404" in str(e):
                logger.exception("Resource not found - task is terminal")
                raise TerminalTaskError(f"Resource not found: {e}") from e

            if error_type == "FileNotFoundError":
                logger.exception("File not found - task is terminal")
                raise TerminalTaskError(f"File not found: {e}") from e

            # Default: retryable
            logger.warning(f"Unclassified error ({error_type}) - task will be retried")
            raise RetryableTaskError(f"Pipeline error ({error_type}): {e}") from e

        else:
            if result.nb_errors > 0:
                raise TerminalTaskError(
                    f"Volume had {result.nb_errors} processing errors out of {result.total_images} images"
                )
            return result
