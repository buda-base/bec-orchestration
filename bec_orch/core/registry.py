from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

from bec_orch.jobs.base import JobWorker

logger = logging.getLogger(__name__)

WorkerFactory = Callable[[dict[str, Any] | None], JobWorker]


# Registry mapping job names to worker factory functions
_REGISTRY: dict[str, WorkerFactory] = {}


def register_job_worker(job_name: str, factory: WorkerFactory) -> None:
    """
    Register a job worker factory.

    Args:
        job_name: Job name (e.g., "ldv1", "ocr")
        factory: Factory function that returns a JobWorker instance
    """
    _REGISTRY[job_name] = factory


def get_job_worker_factory(job_name: str) -> WorkerFactory:
    """
    Get job worker factory for a given job name.

    Args:
        job_name: Job name or prefix (e.g., "ldv1", "ld", "ocr")

    Returns:
        Factory function that creates a JobWorker

    Raises:
        ValueError: If no worker found for job name
    """
    # Try exact match first
    if job_name in _REGISTRY:
        return _REGISTRY[job_name]

    # Try prefix match (e.g., "ld_v1" matches "ld")
    for registered_name, factory in _REGISTRY.items():
        if job_name.startswith(registered_name):
            return factory

    # No match found
    available = ", ".join(_REGISTRY.keys())
    raise ValueError(f"No job worker registered for '{job_name}'. Available: {available}")


# Auto-register known job workers on import
def _auto_register() -> None:
    """Auto-register job workers from known modules."""

    # Try to import and register ldv1 worker
    try:
        from bec_orch.jobs.ldv1.worker import LDV1JobWorker

        def ldv1_factory(job_config: dict[str, Any] | None) -> JobWorker:
            """Factory for LDV1JobWorker that ignores job_config (uses env vars)."""
            return LDV1JobWorker()

        register_job_worker("ldv1", ldv1_factory)
    except ImportError as e:
        logger.warning(
            f"Failed to auto-register ldv1 worker: {e}. "
            f"Dependencies may not be installed. Worker will be lazy-loaded when needed."
        )
    except AttributeError as e:
        logger.warning(f"Failed to auto-register ldv1 worker: {e}. Worker class may not exist in module.")

    # Try to import and register ocrv1 worker
    try:
        from dataclasses import fields

        from bec_orch.jobs.ocrv1.config import OCRV1Config
        from bec_orch.jobs.ocrv1.worker import OCRV1JobWorker

        def ocrv1_factory(job_config: dict[str, Any] | None) -> JobWorker:
            """Factory for OCRV1JobWorker that creates config from job_config."""
            # Create OCRV1Config from job_config
            if not job_config:
                raise ValueError("OCRV1 job requires job_config with 'model' field")

            # Extract required model field
            model = job_config.get("model")
            if not model:
                raise ValueError("OCRV1 job config must contain 'model' field")

            # Get model directory from environment
            import os
            from pathlib import Path

            base_model_dir = os.environ.get("BEC_OCR_MODEL_DIR")
            if not base_model_dir:
                raise ValueError("BEC_OCR_MODEL_DIR environment variable not set")

            model_dir = Path(base_model_dir) / model
            if not model_dir.exists():
                raise FileNotFoundError(f"Model directory not found: {model_dir}")

            # Load model_config.json to get required dimensions
            config_path = model_dir / "model_config.json"
            if not config_path.exists():
                raise FileNotFoundError(f"model_config.json not found in {model_dir}")

            import json

            with config_path.open(encoding="utf-8") as f:
                model_config = json.load(f)

            # Create base config with required fields from model_config.json
            config_kwargs = {
                "model": model,
            }

            # Get all OCRV1Config fields (excluding required ones we already set)
            config_fields = {f.name for f in fields(OCRV1Config) if f.name != "model"}

            # Add optional fields from job_config if present
            for field in config_fields:
                if field in job_config:
                    config_kwargs[field] = job_config[field]

            cfg = OCRV1Config(**config_kwargs)
            return OCRV1JobWorker(cfg)

        register_job_worker("ocrv1", ocrv1_factory)
    except ImportError as e:
        logger.warning(
            f"Failed to auto-register ocrv1 worker: {e}. "
            f"Dependencies may not be installed. Worker will be lazy-loaded when needed."
        )
    except AttributeError as e:
        logger.warning(f"Failed to auto-register ocrv1 worker: {e}. Worker class may not exist in module.")

    # Try to import and register ocr_qwen_v1 worker
    try:
        from dataclasses import fields

        from bec_orch.jobs.ocr_qwen_v1.config import OCRQwenV1Config
        from bec_orch.jobs.ocr_qwen_v1.worker import OCRQwenV1JobWorker

        def ocr_qwen_v1_factory(job_config: dict[str, Any] | None) -> JobWorker:
            """Factory for OCRQwenV1JobWorker — builds config from job_config.

            Every ``OCRQwenV1Config`` field has a sensible default (from the
            benchmark sweeps in ``scratch/findings.md``), so an empty job
            config is valid; any keys present in ``job_config`` override the
            defaults.
            """
            cfg_kwargs: dict[str, Any] = {}
            if job_config:
                allowed = {f.name for f in fields(OCRQwenV1Config)}
                for k, v in job_config.items():
                    if k in allowed:
                        cfg_kwargs[k] = v
                    else:
                        logger.warning(
                            f"[ocr_qwen_v1] ignoring unknown config key '{k}' "
                            f"(allowed: {sorted(allowed)})"
                        )
            cfg = OCRQwenV1Config(**cfg_kwargs)
            return OCRQwenV1JobWorker(cfg)

        register_job_worker("ocr_qwen_v1", ocr_qwen_v1_factory)
    except ImportError as e:
        logger.warning(
            f"Failed to auto-register ocr_qwen_v1 worker: {e}. "
            f"Dependencies (vllm, boto3, Pillow, pyarrow) may not be installed."
        )
    except AttributeError as e:
        logger.warning(
            f"Failed to auto-register ocr_qwen_v1 worker: {e}. Worker class may not exist in module."
        )

    # Try to import and register script_classification worker
    try:
        from dataclasses import fields

        from bec_orch.jobs.script_classification.config import ScriptClassificationConfig
        from bec_orch.jobs.script_classification.worker import ScriptClassificationJobWorker

        def script_classification_factory(job_config: dict[str, Any] | None) -> JobWorker:
            """Factory for ScriptClassificationJobWorker — builds config from job_config.

            Every ``ScriptClassificationConfig`` field has a default, so an
            empty job config is valid; any keys present in ``job_config``
            override the defaults.
            """
            cfg_kwargs: dict[str, Any] = {}
            if job_config:
                allowed = {f.name for f in fields(ScriptClassificationConfig)}
                for k, v in job_config.items():
                    if k in allowed:
                        cfg_kwargs[k] = v
                    else:
                        logger.warning(
                            f"[script_classification] ignoring unknown config key '{k}' "
                            f"(allowed: {sorted(allowed)})"
                        )
            cfg = ScriptClassificationConfig(**cfg_kwargs)
            return ScriptClassificationJobWorker(cfg)

        register_job_worker("script_classification", script_classification_factory)
    except ImportError as e:
        logger.warning(
            f"Failed to auto-register script_classification worker: {e}. "
            f"Dependencies (torch, transformers, huggingface_hub, Pillow, pyarrow) may not be installed."
        )
    except AttributeError as e:
        logger.warning(
            f"Failed to auto-register script_classification worker: {e}. Worker class may not exist in module."
        )

    # Try to import and register script_classification_v2 worker
    try:
        from dataclasses import fields

        from bec_orch.jobs.script_classification_v2.config import ScriptClassificationV2Config
        from bec_orch.jobs.script_classification_v2.worker import (
            ScriptClassificationV2JobWorker,
        )

        def script_classification_v2_factory(job_config: dict[str, Any] | None) -> JobWorker:
            """Factory for ScriptClassificationV2JobWorker — builds config from job_config.

            Every ``ScriptClassificationV2Config`` field has a default, so an
            empty job config is valid; any keys present in ``job_config``
            override the defaults.
            """
            cfg_kwargs: dict[str, Any] = {}
            if job_config:
                allowed = {f.name for f in fields(ScriptClassificationV2Config)}
                for k, v in job_config.items():
                    if k in allowed:
                        cfg_kwargs[k] = v
                    else:
                        logger.warning(
                            f"[script_classification_v2] ignoring unknown config key '{k}' "
                            f"(allowed: {sorted(allowed)})"
                        )
            cfg = ScriptClassificationV2Config(**cfg_kwargs)
            return ScriptClassificationV2JobWorker(cfg)

        register_job_worker("script_classification_v2", script_classification_v2_factory)
    except ImportError as e:
        logger.warning(
            f"Failed to auto-register script_classification_v2 worker: {e}. "
            f"Dependencies (torch, transformers, huggingface_hub, Pillow, pyarrow) may not be installed."
        )
    except AttributeError as e:
        logger.warning(
            f"Failed to auto-register script_classification_v2 worker: {e}. Worker class may not exist in module."
        )

    # Try to import and register layout_detection_v1 worker
    try:
        from dataclasses import fields

        from bec_orch.jobs.layout_detection_v1.config import LayoutDetectionV1Config
        from bec_orch.jobs.layout_detection_v1.worker import LayoutDetectionV1JobWorker

        def layout_detection_v1_factory(job_config: dict[str, Any] | None) -> JobWorker:
            """Factory for LayoutDetectionV1JobWorker — builds config from job_config.

            Every ``LayoutDetectionV1Config`` field has a default, so an empty
            job config is valid; any keys present in ``job_config`` override the
            defaults.
            """
            cfg_kwargs: dict[str, Any] = {}
            if job_config:
                allowed = {f.name for f in fields(LayoutDetectionV1Config)}
                for k, v in job_config.items():
                    if k in allowed:
                        cfg_kwargs[k] = v
                    else:
                        logger.warning(
                            f"[layout_detection_v1] ignoring unknown config key '{k}' "
                            f"(allowed: {sorted(allowed)})"
                        )
            cfg = LayoutDetectionV1Config(**cfg_kwargs)
            return LayoutDetectionV1JobWorker(cfg)

        register_job_worker("layout_detection_v1", layout_detection_v1_factory)
    except ImportError as e:
        logger.warning(
            f"Failed to auto-register layout_detection_v1 worker: {e}. "
            f"Dependencies (torch, ultralytics, huggingface_hub, pyvips, pyarrow) may not be installed."
        )
    except AttributeError as e:
        logger.warning(
            f"Failed to auto-register layout_detection_v1 worker: {e}. Worker class may not exist in module."
        )

    # Try to import and register PaddleOCR-VL worker(s).
    #
    # The implementation in ``bec_orch.jobs.paddleocr`` is version-agnostic:
    # everything model-specific (checkpoint, prompt, generation knobs) lives in
    # ``PaddleOCRConfig``. To add ``paddleocr_v2`` when its checkpoint is ready,
    # register another job name with different ``base_defaults`` — no new code:
    #
    #     register_job_worker(
    #         "paddleocr_v2",
    #         _make_paddleocr_factory({
    #             "checkpoint_s3_uri": "s3://bec.bdrc.io/checkpoints/PaddleOCR/<v2>/epoch_0/",
    #         }),
    #     )
    #
    # (Job-creation config still overrides these per-job at runtime.)
    try:
        from dataclasses import fields

        from bec_orch.jobs.paddleocr.config import PaddleOCRConfig
        from bec_orch.jobs.paddleocr.worker import PaddleOCRJobWorker

        def _make_paddleocr_factory(base_defaults: dict[str, Any] | None = None) -> WorkerFactory:
            """Build a factory for a PaddleOCR-VL job.

            ``base_defaults`` are job-specific overrides (e.g. a different
            checkpoint for v2). The DB ``job_config`` overrides those in turn.
            Unknown keys are ignored with a warning.
            """
            base_defaults = base_defaults or {}

            def factory(job_config: dict[str, Any] | None) -> JobWorker:
                allowed = {f.name for f in fields(PaddleOCRConfig)}
                merged: dict[str, Any] = {}
                for src in (base_defaults, job_config or {}):
                    for k, v in src.items():
                        if k in allowed:
                            merged[k] = v
                        else:
                            logger.warning(
                                f"[paddleocr] ignoring unknown config key '{k}' "
                                f"(allowed: {sorted(allowed)})"
                            )
                cfg = PaddleOCRConfig(**merged)
                return PaddleOCRJobWorker(cfg)

            return factory

        # v1 uses the defaults baked into PaddleOCRConfig verbatim.
        register_job_worker("paddleocr_v1", _make_paddleocr_factory())
    except ImportError as e:
        logger.warning(
            f"Failed to auto-register paddleocr worker: {e}. "
            f"Dependencies (vllm, transformers, opencv-python-headless, numpy, pyvips, pyarrow) may not be installed."
        )
    except AttributeError as e:
        logger.warning(
            f"Failed to auto-register paddleocr worker: {e}. Worker class may not exist in module."
        )

    # Try to import and register google_vision_v1 worker (no GPU; Google Vision).
    try:
        from dataclasses import fields

        from bec_orch.jobs.google_vision_v1.config import GoogleVisionV1Config
        from bec_orch.jobs.google_vision_v1.worker import GoogleVisionV1JobWorker

        def google_vision_v1_factory(job_config: dict[str, Any] | None) -> JobWorker:
            """Factory for GoogleVisionV1JobWorker — builds config from job_config.

            Every ``GoogleVisionV1Config`` field has a default, so an empty job
            config is valid; any keys present in ``job_config`` override the
            defaults. (In practice ``google_credentials_path`` is usually set.)
            """
            cfg_kwargs: dict[str, Any] = {}
            if job_config:
                allowed = {f.name for f in fields(GoogleVisionV1Config)}
                for k, v in job_config.items():
                    if k in allowed:
                        cfg_kwargs[k] = v
                    else:
                        logger.warning(
                            f"[google_vision_v1] ignoring unknown config key '{k}' "
                            f"(allowed: {sorted(allowed)})"
                        )
            cfg = GoogleVisionV1Config(**cfg_kwargs)
            return GoogleVisionV1JobWorker(cfg)

        register_job_worker("google_vision_v1", google_vision_v1_factory)
    except ImportError as e:
        logger.warning(
            f"Failed to auto-register google_vision_v1 worker: {e}. "
            f"Dependencies (google-cloud-vision, google-cloud-storage, pyarrow, zstandard) "
            f"may not be installed."
        )
    except AttributeError as e:
        logger.warning(
            f"Failed to auto-register google_vision_v1 worker: {e}. Worker class may not exist in module."
        )


# Run auto-registration on module import
_auto_register()
