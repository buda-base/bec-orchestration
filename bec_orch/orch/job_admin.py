from __future__ import annotations
from typing import List, Optional
import psycopg
from psycopg.rows import dict_row

from bec_orch.core.models import JobRecord


def create_job(
    conn: psycopg.Connection,
    name: str,
    queue_url: str,
    config_text: str = "{}",
    dlq_url: Optional[str] = None,
) -> int:
    """
    Create a new job.
    
    Args:
        conn: Database connection
        name: Job name (e.g., "ldv1", "ocr") - must be unique
        queue_url: SQS queue URL for tasks
        config_text: Configuration text (typically JSON, default: {})
        dlq_url: Optional dead-letter queue URL
        
    Returns:
        job_id: The created job ID
    """
    with conn.cursor() as cur:
        cur.execute(
            "INSERT INTO jobs (name, config, queue_url, dlq_url) VALUES (%s, %s, %s, %s) RETURNING id",
            (name, config_text, queue_url, dlq_url)
        )
        result = cur.fetchone()
        conn.commit()
        return result['id']


def get_job(conn: psycopg.Connection, job_id: int) -> JobRecord:
    """
    Get job by ID.
    
    Args:
        conn: Database connection
        job_id: Job ID
        
    Returns:
        JobRecord
        
    Raises:
        ValueError: If job not found
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


def get_job_by_name(conn: psycopg.Connection, job_name: str) -> JobRecord:
    """
    Get job by name.
    
    Args:
        conn: Database connection
        job_name: Job name
        
    Returns:
        JobRecord
        
    Raises:
        ValueError: If job not found
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


def list_jobs(conn: psycopg.Connection) -> List[JobRecord]:
    """
    List all jobs.
    
    Args:
        conn: Database connection
        
    Returns:
        List of JobRecord
    """
    with conn.cursor() as cur:
        cur.execute("SELECT id, name, config, queue_url, dlq_url FROM jobs ORDER BY id")
        rows = cur.fetchall()
        
        return [
            JobRecord(
                id=row['id'],
                name=row['name'],
                config_text=row['config'],
                queue_url=row['queue_url'],
                dlq_url=row['dlq_url']
            )
            for row in rows
        ]


def update_job_config(conn: psycopg.Connection, job_id: int, config_text: str) -> None:
    """
    Update job configuration.
    
    Args:
        conn: Database connection
        job_id: Job ID
        config_text: New configuration text
    """
    with conn.cursor() as cur:
        cur.execute(
            "UPDATE jobs SET config = %s, updated_at = now() WHERE id = %s",
            (config_text, job_id)
        )
        if cur.rowcount == 0:
            raise ValueError(f"Job {job_id} not found")
        conn.commit()


def update_job_queues(
    conn: psycopg.Connection,
    job_id: int,
    queue_url: Optional[str] = None,
    dlq_url: Optional[str] = None
) -> None:
    """
    Update job queue URLs.
    
    Args:
        conn: Database connection
        job_id: Job ID
        queue_url: New queue URL (if provided)
        dlq_url: New DLQ URL (if provided)
    """
    updates = []
    params = []
    
    if queue_url is not None:
        updates.append("queue_url = %s")
        params.append(queue_url)
    
    if dlq_url is not None:
        updates.append("dlq_url = %s")
        params.append(dlq_url)
    
    if not updates:
        return
    
    updates.append("updated_at = now()")
    params.append(job_id)
    
    with conn.cursor() as cur:
        cur.execute(
            f"UPDATE jobs SET {', '.join(updates)} WHERE id = %s",
            params
        )
        if cur.rowcount == 0:
            raise ValueError(f"Job {job_id} not found")
        conn.commit()


def delete_job(conn: psycopg.Connection, job_id: int) -> None:
    """
    Delete a job.
    
    Args:
        conn: Database connection
        job_id: Job ID
        
    Note: This will cascade delete all task_executions for this job
    """
    with conn.cursor() as cur:
        cur.execute("DELETE FROM jobs WHERE id = %s", (job_id,))
        if cur.rowcount == 0:
            raise ValueError(f"Job {job_id} not found")
        conn.commit()
