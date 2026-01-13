from __future__ import annotations
from typing import Callable, Dict
from bec_orch.jobs.base import JobWorker

WorkerFactory = Callable[[], JobWorker]

def get_job_worker_factory(job_name: str) -> WorkerFactory:
    """
    Map job_name (or prefix) to a job worker implementation.
    e.g. 'ld' or 'ld_v1' -> LDVolumeWorker.
    """
    ...
