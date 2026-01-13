from __future__ import annotations
from typing import Any, Dict, Optional
import psycopg

from bec_orch.core.models import JobRecord

class DBClient:
    def __init__(self, dsn: str): ...

    def connect(self) -> psycopg.Connection: ...

    # --- workers ---
    def register_worker(
        self,
        conn: psycopg.Connection,
        instance_id: str,
        hostname: str,
        tags: Dict[str, Any],
    ) -> int: ...

    def heartbeat(self, conn: psycopg.Connection, worker_id: int) -> None: ...

    def mark_worker_stopped(self, conn: psycopg.Connection, worker_id: int) -> None: ...

    # --- jobs ---
    def fetch_job(self, conn: psycopg.Connection, job_id: int) -> JobRecord: ...

    # --- volumes ---
    def get_volume_id(self, conn: psycopg.Connection, w_id: str, i_id: str) -> int: ...

    # --- task executions / idempotency ---
    def claim_task_execution(
        self,
        conn: psycopg.Connection,
        job_id: int,
        volume_id: int,
        s3_etag_bytes: bytes,
        worker_id: int,
    ) -> Optional[int]:
        """
        Atomically claim a (job_id, volume_id, s3_etag).
        Returns task_execution.id if inserted, else None if already exists.
        Requires UNIQUE(job_id, volume_id, s3_etag).
        """
        ...

    def mark_task_done(
        self,
        conn: psycopg.Connection,
        task_execution_id: int,
        total_images: int,
        nb_errors: int,
        total_duration_ms: float,
        avg_duration_per_page_ms: float,
    ) -> None: ...

    def mark_task_failed(
        self,
        conn: psycopg.Connection,
        task_execution_id: int,
        retryable: bool,
    ) -> None: ...
