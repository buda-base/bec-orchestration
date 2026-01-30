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


# Run auto-registration on module import
_auto_register()
