from __future__ import annotations
import logging
from typing import Callable

from bec_orch.jobs.base import JobWorker

logger = logging.getLogger(__name__)

WorkerFactory = Callable[[], JobWorker]


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
    available = ', '.join(_REGISTRY.keys())
    raise ValueError(
        f"No job worker registered for '{job_name}'. "
        f"Available: {available}"
    )


# Auto-register known job workers on import
def _auto_register() -> None:
    """Auto-register job workers from known modules."""
    
    # Try to import and register ldv1 worker
    try:
        from bec_orch.jobs.ldv1.worker import LDV1JobWorker
        register_job_worker("ldv1", LDV1JobWorker)
    except ImportError as e:
        logger.warning(
            f"Failed to auto-register ldv1 worker: {e}. "
            f"Dependencies may not be installed. Worker will be lazy-loaded when needed."
        )
    except AttributeError as e:
        logger.warning(
            f"Failed to auto-register ldv1 worker: {e}. "
            f"Worker class may not exist in module."
        )


# Run auto-registration on module import
_auto_register()
