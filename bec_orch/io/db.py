from __future__ import annotations
import json
from typing import Any, Dict, Optional
import psycopg
from psycopg.rows import dict_row

from bec_orch.core.models import JobRecord


class DBClient:
    """PostgreSQL client for task tracking and worker registration."""

    def __init__(self, dsn: str):
        """
        Initialize DB client with connection string.
        
        Args:
            dsn: PostgreSQL connection string (e.g., "postgresql://user:pass@host:port/dbname")
        """
        self.dsn = dsn

    def connect(self) -> psycopg.Connection:
        """
        Establish and return a database connection.
        
        Returns:
            psycopg.Connection configured with dict_row factory
        """
        conn = psycopg.connect(self.dsn, row_factory=dict_row, autocommit=False)
        return conn

    # --- workers ---
    def register_worker(
        self,
        conn: psycopg.Connection,
        instance_id: str,
        hostname: str,
        tags: Dict[str, Any],
    ) -> int:
        """
        Register this worker instance in the database (idempotent).
        
        If a worker with the same instance_id already exists and is not stopped,
        returns its worker_id. Otherwise, creates a new worker entry.
        
        Args:
            conn: Database connection
            instance_id: EC2 instance ID or unique worker identifier
            hostname: Worker hostname
            tags: Optional metadata tags
            
        Returns:
            worker_id: The registered worker ID
        """
        with conn.cursor() as cur:
            # First check if worker already exists and is active (not stopped)
            cur.execute(
                """
                SELECT worker_id FROM workers
                WHERE worker_name = %s AND stopped_at IS NULL
                ORDER BY worker_id DESC
                LIMIT 1
                """,
                (instance_id,)
            )
            existing = cur.fetchone()
            
            if existing is not None:
                # Worker already registered and active
                worker_id = existing['worker_id']
                # Update heartbeat to show it's alive
                cur.execute(
                    "UPDATE workers SET last_heartbeat_at = now() WHERE worker_id = %s",
                    (worker_id,)
                )
                conn.commit()
                return worker_id
            
            # Create new worker entry
            cur.execute(
                """
                INSERT INTO workers (worker_name, hostname, tags, started_at, last_heartbeat_at)
                VALUES (%s, %s, %s, now(), now())
                RETURNING worker_id
                """,
                (instance_id, hostname, json.dumps(tags) if tags else None)
            )
            result = cur.fetchone()
            conn.commit()
            return result['worker_id']

    def heartbeat(self, conn: psycopg.Connection, worker_id: int) -> None:
        """
        Update worker heartbeat timestamp.
        
        Args:
            conn: Database connection
            worker_id: Worker ID
        """
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE workers SET last_heartbeat_at = now() WHERE worker_id = %s",
                (worker_id,)
            )
            conn.commit()

    def mark_worker_stopped(self, conn: psycopg.Connection, worker_id: int) -> None:
        """
        Mark worker as stopped.
        
        Args:
            conn: Database connection
            worker_id: Worker ID
        """
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE workers SET stopped_at = now() WHERE worker_id = %s",
                (worker_id,)
            )
            conn.commit()

    # --- jobs ---
    def fetch_job(self, conn: psycopg.Connection, job_id: int) -> JobRecord:
        """
        Fetch job record by ID.
        
        Args:
            conn: Database connection
            job_id: Job ID
            
        Returns:
            JobRecord with id, name, config_text, queue_url, and dlq_url
        """
        with conn.cursor() as cur:
            cur.execute(
                "SELECT id, name, config, queue_url, dlq_url FROM jobs WHERE id = %s",
                (job_id,)
            )
            row = cur.fetchone()
            if row is None:
                raise ValueError(f"Job {job_id} not found")
            
            return JobRecord(
                id=row['id'],
                name=row['name'],
                config_text=row['config'],
                queue_url=row['queue_url'],
                dlq_url=row['dlq_url']
            )
    
    def fetch_job_by_name(self, conn: psycopg.Connection, job_name: str) -> JobRecord:
        """
        Fetch job record by name.
        
        Args:
            conn: Database connection
            job_name: Job name
            
        Returns:
            JobRecord
        """
        with conn.cursor() as cur:
            cur.execute(
                "SELECT id, name, config, queue_url, dlq_url FROM jobs WHERE name = %s",
                (job_name,)
            )
            row = cur.fetchone()
            if row is None:
                raise ValueError(f"Job '{job_name}' not found")
            
            return JobRecord(
                id=row['id'],
                name=row['name'],
                config_text=row['config'],
                queue_url=row['queue_url'],
                dlq_url=row['dlq_url']
            )

    # --- volumes ---
    def get_volume_id(self, conn: psycopg.Connection, w_id: str, i_id: str) -> int:
        """
        Get volume ID by w_id and i_id.
        
        Args:
            conn: Database connection
            w_id: Work ID (BDRC work identifier)
            i_id: Image group ID
            
        Returns:
            volume_id: Database ID for the volume
            
        Raises:
            ValueError: If volume not found
        """
        with conn.cursor() as cur:
            cur.execute(
                "SELECT id FROM volumes WHERE bdrc_w_id = %s AND bdrc_i_id = %s",
                (w_id, i_id)
            )
            row = cur.fetchone()
            if row is None:
                raise ValueError(f"Volume {w_id}/{i_id} not found")
            return row['id']

    def ensure_volume(
        self,
        conn: psycopg.Connection,
        w_id: str,
        i_id: str,
        s3_etag_bytes: bytes,
        last_modified_iso: str,
        nb_images: int,
        nb_images_intro: int = 0
    ) -> int:
        """
        Ensure volume exists in database, insert or update as needed.
        
        Args:
            conn: Database connection
            w_id: Work ID
            i_id: Image group ID
            s3_etag_bytes: S3 etag as bytes (16 bytes MD5)
            last_modified_iso: ISO timestamp of last modification
            nb_images: Total number of images
            nb_images_intro: Number of intro images to skip
            
        Returns:
            volume_id: Database ID for the volume
        """
        with conn.cursor() as cur:
            # Upsert: insert or update if w_id/i_id exists
            cur.execute(
                """
                INSERT INTO volumes (bdrc_w_id, bdrc_i_id, last_s3_etag, last_modified_at, nb_images, nb_images_intro)
                VALUES (%s, %s, %s, %s::timestamptz, %s, %s)
                ON CONFLICT (bdrc_w_id, bdrc_i_id) DO UPDATE
                SET last_s3_etag = EXCLUDED.last_s3_etag,
                    last_modified_at = EXCLUDED.last_modified_at,
                    nb_images = EXCLUDED.nb_images
                RETURNING id
                """,
                (w_id, i_id, s3_etag_bytes, last_modified_iso, nb_images, nb_images_intro)
            )
            result = cur.fetchone()
            conn.commit()
            return result['id']

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
        
        Args:
            conn: Database connection
            job_id: Job ID
            volume_id: Volume ID
            s3_etag_bytes: S3 etag as bytes (16 bytes MD5)
            worker_id: Worker ID
            
        Returns:
            task_execution_id or None if already claimed
        """
        with conn.cursor() as cur:
            # Try to find existing task execution
            cur.execute(
                """
                SELECT id, status FROM task_executions
                WHERE job_id = %s AND volume_id = %s AND s3_etag = %s
                """,
                (job_id, volume_id, s3_etag_bytes)
            )
            row = cur.fetchone()
            
            if row is not None:
                # Task already exists
                # If it's done, we might want to return None (idempotent)
                # If it's running, also return None (someone else is working on it)
                return None
            
            # Create new task execution
            try:
                cur.execute(
                    """
                    INSERT INTO task_executions (job_id, volume_id, s3_etag, status, worker_id, started_at, attempt)
                    VALUES (%s, %s, %s, 'running', %s, now(), 1)
                    RETURNING id
                    """,
                    (job_id, volume_id, s3_etag_bytes, worker_id)
                )
                result = cur.fetchone()
                conn.commit()
                return result['id']
            except psycopg.errors.UniqueViolation:
                # Race condition: another worker claimed it
                conn.rollback()
                return None

    def mark_task_done(
        self,
        conn: psycopg.Connection,
        task_execution_id: int,
        total_images: int,
        nb_errors: int,
        total_duration_ms: float,
        avg_duration_per_page_ms: float,
    ) -> None:
        """
        Mark task as successfully completed.
        
        Args:
            conn: Database connection
            task_execution_id: Task execution ID
            total_images: Total number of images processed
            nb_errors: Number of errors encountered
            total_duration_ms: Total processing duration in milliseconds
            avg_duration_per_page_ms: Average duration per page in milliseconds
        """
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE task_executions
                SET status = 'done',
                    done_at = now(),
                    total_images = %s,
                    nb_errors = %s,
                    total_duration_ms = %s,
                    avg_duration_per_page_ms = %s
                WHERE id = %s
                """,
                (
                    total_images,
                    nb_errors,
                    total_duration_ms,
                    avg_duration_per_page_ms,
                    task_execution_id
                )
            )
            conn.commit()

    def mark_task_failed(
        self,
        conn: psycopg.Connection,
        task_execution_id: int,
        retryable: bool,
    ) -> None:
        """
        Mark task as failed.
        
        Args:
            conn: Database connection
            task_execution_id: Task execution ID
            retryable: Whether the failure is retryable
        """
        status = 'retryable_failed' if retryable else 'terminal_failed'
        
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE task_executions
                SET status = %s::task_status,
                    done_at = now()
                WHERE id = %s
                """,
                (status, task_execution_id)
            )
            conn.commit()


def etag_to_bytes(etag: str) -> bytes:
    """
    Convert S3 etag string to 16-byte MD5 hash.
    
    S3 etags are quoted hex strings like '"a1b2c3d4..."'
    We need to extract the hex and convert to bytes.
    
    Args:
        etag: S3 etag string (may be quoted)
        
    Returns:
        16-byte MD5 hash
    """
    # Remove quotes if present
    etag = etag.strip('"')
    
    # For multipart uploads, etag has format: hash-partcount
    # We only use the hash part
    if '-' in etag:
        etag = etag.split('-')[0]
    
    # Convert hex to bytes (should be 16 bytes for MD5)
    etag_bytes = bytes.fromhex(etag)
    
    if len(etag_bytes) != 16:
        raise ValueError(f"Etag must be 16 bytes (MD5), got {len(etag_bytes)} bytes from '{etag}'")
    
    return etag_bytes
