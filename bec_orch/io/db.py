from __future__ import annotations
import json
import os
from typing import Any, Dict, Optional
from urllib.parse import quote_plus
import psycopg
from psycopg.rows import dict_row

from bec_orch.core.models import JobRecord


def build_dsn_from_env() -> str:
    """
    Build PostgreSQL DSN from environment variables.
    
    Required environment variables:
    - BEC_SQL_HOST: PostgreSQL host
    - BEC_SQL_USER: PostgreSQL user
    - BEC_SQL_PASSWORD: PostgreSQL password
    
    Optional environment variables:
    - BEC_SQL_PORT: PostgreSQL port (default: 5432)
    - BEC_SQL_DATABASE: Database name (default: pipeline_v1)
    
    Returns:
        PostgreSQL DSN string
        
    Raises:
        ValueError: If required environment variables are missing
    """
    sql_host = os.environ.get('BEC_SQL_HOST')
    sql_port = os.environ.get('BEC_SQL_PORT', '5432')
    sql_user = os.environ.get('BEC_SQL_USER')
    sql_password = os.environ.get('BEC_SQL_PASSWORD')
    sql_database = os.environ.get('BEC_SQL_DATABASE', 'pipeline_v1')
    
    if not all([sql_host, sql_user, sql_password]):
        raise ValueError(
            "Missing required SQL environment variables. "
            "Required: BEC_SQL_HOST, BEC_SQL_USER, BEC_SQL_PASSWORD"
        )
    
    # URL-encode password to handle special characters
    encoded_password = quote_plus(sql_password)
    # Add SSL requirement for secure connections
    return f"postgresql://{sql_user}:{encoded_password}@{sql_host}:{sql_port}/{sql_database}?sslmode=require"


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
    ) -> tuple[Optional[int], Optional[str], Optional[Any]]:
        """
        Atomically claim a (job_id, volume_id, s3_etag).
        Returns (task_execution.id, None, None) if inserted, else (None, existing_status, started_at) if already exists.
        Requires UNIQUE(job_id, volume_id, s3_etag).
        
        Args:
            conn: Database connection
            job_id: Job ID
            volume_id: Volume ID
            s3_etag_bytes: S3 etag as bytes (16 bytes MD5)
            worker_id: Worker ID
            
        Returns:
            (task_execution_id, None, None) if successfully claimed, or (None, existing_status, started_at) if already exists
        """
        with conn.cursor() as cur:
            # Try to find existing task execution
            cur.execute(
                """
                SELECT id, status, started_at FROM task_executions
                WHERE job_id = %s AND volume_id = %s AND s3_etag = %s
                """,
                (job_id, volume_id, s3_etag_bytes)
            )
            row = cur.fetchone()
            
            if row is not None:
                # Task already exists - return None with the existing status and started_at
                existing_status = row['status']
                started_at = row['started_at']
                return None, existing_status, started_at
            
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
                return result['id'], None, None
            except psycopg.errors.UniqueViolation:
                # Race condition: another worker claimed it between our SELECT and INSERT
                conn.rollback()
                # Fetch the status and started_at of the task that was just claimed
                cur.execute(
                    """
                    SELECT status, started_at FROM task_executions
                    WHERE job_id = %s AND volume_id = %s AND s3_etag = %s
                    """,
                    (job_id, volume_id, s3_etag_bytes)
                )
                row = cur.fetchone()
                existing_status = row['status'] if row else 'running'
                started_at = row['started_at'] if row else None
                return None, existing_status, started_at

    def claim_stale_task_execution(
        self,
        conn: psycopg.Connection,
        job_id: int,
        volume_id: int,
        s3_etag_bytes: bytes,
        worker_id: int,
    ) -> Optional[int]:
        """
        Claim a stale task execution (any status, started_at > 5 minutes ago).
        Updates the existing record to claim it for this worker.
        Note: success.json is the source of truth, so even 'done' status can be stale if success.json is missing.
        
        Args:
            conn: Database connection
            job_id: Job ID
            volume_id: Volume ID
            s3_etag_bytes: S3 etag as bytes (16 bytes MD5)
            worker_id: Worker ID
            
        Returns:
            task_execution_id if successfully claimed, None if not stale or doesn't exist
        """
        with conn.cursor() as cur:
            # Update stale task (any status) to claim it for this worker
            cur.execute(
                """
                UPDATE task_executions
                SET worker_id = %s, started_at = now(), attempt = attempt + 1, status = 'running'
                WHERE job_id = %s AND volume_id = %s AND s3_etag = %s
                  AND started_at < now() - INTERVAL '5 minutes'
                RETURNING id
                """,
                (worker_id, job_id, volume_id, s3_etag_bytes)
            )
            row = cur.fetchone()
            if row is not None:
                conn.commit()
                return row['id']
            conn.rollback()
            return None

    def force_claim_task_execution(
        self,
        conn: psycopg.Connection,
        job_id: int,
        volume_id: int,
        s3_etag_bytes: bytes,
        worker_id: int,
    ) -> Optional[int]:
        """
        Forcefully claim a task execution regardless of status or age.
        Updates the existing record to claim it for this worker, or creates new if doesn't exist.
        This is used when --force flag is specified.
        
        Args:
            conn: Database connection
            job_id: Job ID
            volume_id: Volume ID
            s3_etag_bytes: S3 etag as bytes (16 bytes MD5)
            worker_id: Worker ID
            
        Returns:
            task_execution_id (always succeeds)
        """
        with conn.cursor() as cur:
            # Try to update existing task (any status, any age)
            cur.execute(
                """
                UPDATE task_executions
                SET worker_id = %s, started_at = now(), attempt = attempt + 1, status = 'running'
                WHERE job_id = %s AND volume_id = %s AND s3_etag = %s
                RETURNING id
                """,
                (worker_id, job_id, volume_id, s3_etag_bytes)
            )
            row = cur.fetchone()
            
            if row is not None:
                # Successfully updated existing task
                conn.commit()
                return row['id']
            
            # No existing task, create new one
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
                # Race condition: task was created between UPDATE and INSERT
                # Roll back and try UPDATE again
                conn.rollback()
                cur.execute(
                    """
                    UPDATE task_executions
                    SET worker_id = %s, started_at = now(), attempt = attempt + 1, status = 'running'
                    WHERE job_id = %s AND volume_id = %s AND s3_etag = %s
                    RETURNING id
                    """,
                    (worker_id, job_id, volume_id, s3_etag_bytes)
                )
                row = cur.fetchone()
                if row is not None:
                    conn.commit()
                    return row['id']
                # Should never happen, but handle gracefully
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

    def is_volume_done_on_latest_version(
        self,
        conn: psycopg.Connection,
        job_id: int,
        w_id: str,
        i_id: str,
    ) -> bool:
        """
        Check if a volume has been successfully processed on its latest version.
        
        Returns True if:
        - The volume exists in the database
        - There's a task execution with status='done' for this job and volume
        - The s3_etag of the done task matches the latest s3_etag in volumes table
        
        Args:
            conn: Database connection
            job_id: Job ID
            w_id: Work ID
            i_id: Image group ID
            
        Returns:
            True if volume is done on latest version, False otherwise
        """
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT EXISTS (
                    SELECT 1
                    FROM task_executions te
                    JOIN volumes v ON te.volume_id = v.id
                    WHERE v.bdrc_w_id = %s
                      AND v.bdrc_i_id = %s
                      AND te.job_id = %s
                      AND te.status = 'done'
                      AND te.s3_etag = v.last_s3_etag
                ) AS is_done
                """,
                (w_id, i_id, job_id)
            )
            row = cur.fetchone()
            return row['is_done'] if row else False

    def get_volumes_done_on_latest_version(
        self,
        conn: psycopg.Connection,
        job_id: int,
    ) -> set[tuple[str, str]]:
        """
        Get all volumes that have been successfully processed on their latest version for a job.
        
        This is optimized for bulk checking - fetch all done volumes at once rather than
        querying one by one.
        
        Args:
            conn: Database connection
            job_id: Job ID
            
        Returns:
            Set of (w_id, i_id) tuples for volumes that are done on latest version
        """
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT v.bdrc_w_id, v.bdrc_i_id
                FROM task_executions te
                JOIN volumes v ON te.volume_id = v.id
                WHERE te.job_id = %s
                  AND te.status = 'done'
                  AND te.s3_etag = v.last_s3_etag
                """,
                (job_id,)
            )
            rows = cur.fetchall()
            return {(row['bdrc_w_id'], row['bdrc_i_id']) for row in rows}


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
