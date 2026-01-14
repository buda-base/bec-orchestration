from __future__ import annotations
import gzip
import hashlib
import io
import json
import logging
import socket
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import boto3
from botocore.exceptions import ClientError

from bec_orch.config import OrchestrationConfig
from bec_orch.core.models import (
    ArtifactLocation,
    JobRecord,
    SqsTaskMessage,
    TaskResult,
    VolumeManifest,
    VolumeRef,
)
from bec_orch.core.registry import get_job_worker_factory
from bec_orch.errors import RetryableTaskError, TerminalTaskError
from bec_orch.io.db import DBClient, etag_to_bytes
from bec_orch.io.sqs import SQSClient
from bec_orch.jobs.base import JobContext, JobWorker

# Use "bec" namespace so logs appear at INFO level (not WARNING from root)
logger = logging.getLogger("bec.core.worker_runtime")

# Common image file extensions (case-insensitive)
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.tif', '.tiff'}


class BECWorkerRuntime:
    """
    Main worker runtime that orchestrates task processing.
    
    Flow:
    1. Initialize: connect to DB, register worker, load job config, instantiate job worker
    2. Poll SQS for tasks
    3. For each task:
       - Fetch volume manifest from S3
       - Ensure volume exists in DB
       - Claim task execution in DB (idempotent)
       - Check if already done (success.json exists)
       - Run job worker
       - Write success.json
       - Update DB with results
       - Delete SQS message
    4. Shutdown: mark worker stopped, close connections
    """

    def __init__(
        self,
        cfg: OrchestrationConfig,
        db: DBClient,
        sqs: SQSClient,
        s3_source_bucket: str = "archive.tbrc.org",
        s3_dest_bucket: Optional[str] = None,
    ):
        """
        Initialize worker runtime.
        
        Args:
            cfg: Orchestration configuration
            db: Database client
            sqs: SQS client
            s3_source_bucket: Source S3 bucket for images (default: archive.tbrc.org)
            s3_dest_bucket: Destination S3 bucket for artifacts (default: from env or cfg)
        """
        self.cfg = cfg
        self.db = db
        self.sqs = sqs
        self.s3_source_bucket = s3_source_bucket
        self.s3_dest_bucket = s3_dest_bucket
        
        # State
        self.conn: Optional[Any] = None  # psycopg.Connection
        self.worker_id: Optional[int] = None
        self.job_record: Optional[JobRecord] = None
        self.job_config: Optional[Dict[str, Any]] = None
        self.job_worker: Optional[JobWorker] = None
        self.queue_url: Optional[str] = None
        self.dlq_url: Optional[str] = None
        
        # S3 client
        self.s3 = boto3.client('s3', region_name=cfg.aws_region)
        
        # Heartbeat tracking
        self._last_heartbeat: float = 0
        self._heartbeat_interval: float = 30.0  # seconds
        
        # Empty poll tracking
        self._empty_polls: int = 0

    def initialize(self) -> None:
        """
        Initialize worker:
        - Open DB connection
        - Register worker in DB
        - Fetch job record (by name) and parse config
        - Get queue URLs from job record
        - Load job worker implementation via registry
        """
        logger.info("Initializing BECWorkerRuntime")
        
        # Connect to database
        self.conn = self.db.connect()
        logger.info("Connected to database")
        
        # Fetch job record by name
        self.job_record = self.db.fetch_job_by_name(self.conn, self.cfg.job_name)
        logger.info(f"Loaded job: {self.job_record.name} (id={self.job_record.id})")
        
        # Get queue URLs from job record
        self.queue_url = self.job_record.queue_url
        self.dlq_url = self.job_record.dlq_url
        logger.info(f"Queue URL: {self.queue_url}")
        if self.dlq_url:
            logger.info(f"DLQ URL: {self.dlq_url}")
        
        # Register worker
        instance_id = self._get_instance_id()
        hostname = socket.gethostname()
        tags = {"job_id": self.job_record.id, "job_name": self.job_record.name}
        
        self.worker_id = self.db.register_worker(self.conn, instance_id, hostname, tags)
        logger.info(f"Registered worker: worker_id={self.worker_id}, instance_id={instance_id}")
        
        # Parse job config
        try:
            self.job_config = json.loads(self.job_record.config_text) if self.job_record.config_text else {}
        except json.JSONDecodeError:
            logger.warning(f"Job config is not valid JSON, using as raw string")
            self.job_config = {}
        
        # Load job worker
        factory = get_job_worker_factory(self.job_record.name)
        self.job_worker = factory()
        logger.info(f"Loaded job worker for '{self.job_record.name}'")

    def run_forever(self) -> None:
        """
        Main loop: poll SQS, process messages, exit after N empty polls.
        """
        logger.info(f"Starting main loop (shutdown after {self.cfg.shutdown_after_empty_polls} empty polls)")
        
        while True:
            # Update heartbeat if needed
            self._maybe_heartbeat()
            
            # Poll SQS for message
            msg = self.sqs.receive_one(
                queue_url=self.queue_url,
                wait_seconds=self.cfg.poll_wait_seconds,
                visibility_timeout=self.cfg.visibility_timeout_seconds,
            )
            
            if msg is None:
                self._empty_polls += 1
                logger.info(f"No messages received ({self._empty_polls}/{self.cfg.shutdown_after_empty_polls})")
                
                if self._empty_polls >= self.cfg.shutdown_after_empty_polls:
                    logger.info("Shutdown threshold reached, exiting")
                    break
                
                continue
            
            # Reset empty poll counter
            self._empty_polls = 0
            
            # Log message receipt
            logger.info(
                f"Received message from SQS: {msg.message_id} for volume {msg.volume.w_id}/{msg.volume.i_id}"
            )
            
            # Process message
            try:
                self._process_message(msg)
                logger.info(f"Successfully processed message: {msg.message_id}")
            except Exception as e:
                logger.error(f"Failed to process message: {e}", exc_info=True)
                # Don't delete message - let it go back to queue or DLQ
                # Could implement visibility timeout extension here

    def shutdown(self) -> None:
        """Mark worker stopped and close DB connection."""
        logger.info("Shutting down BECWorkerRuntime")
        
        if self.worker_id and self.conn:
            try:
                self.db.mark_worker_stopped(self.conn, self.worker_id)
            except Exception as e:
                logger.error(f"Failed to mark worker stopped: {e}")
        
        if self.conn:
            try:
                self.conn.close()
            except Exception as e:
                logger.error(f"Failed to close DB connection: {e}")

    # ---- internals ----
    
    def _process_message(self, msg: SqsTaskMessage) -> None:
        """
        High-level flow:
        - Resolve manifest info + etag
        - Ensure volume exists in DB
        - Compute artifact location
        - Check success.json first (source of truth) - if exists, skip
        - Try to claim task in DB
        - If claim fails:
          - If stale (any status, > 5 min old): claim stale task and proceed
          - If not stale: skip with warning (another worker likely active)
        - Run job worker
        - Write success.json
        - Update DB
        - Delete SQS message when safe
        """
        start_time = time.time()
        volume = msg.volume
        
        logger.info(
            f"Starting volume processing: {volume.w_id}/{volume.i_id} "
            f"(message_id={msg.message_id})"
        )
        
        # Get volume manifest from S3
        manifest = self._get_volume_manifest(volume)
        logger.info(f"Loaded manifest: {len(manifest.manifest)} files, etag={manifest.s3_etag}")
        
        # Ensure volume exists in DB
        etag_bytes = etag_to_bytes(manifest.s3_etag)
        volume_id = self.db.ensure_volume(
            self.conn,
            volume.w_id,
            volume.i_id,
            etag_bytes,
            manifest.last_modified_iso,
            len(manifest.manifest),
            nb_images_intro=0
        )
        logger.info(f"Volume ID: {volume_id}")
        
        # Compute artifact location (needed for success.json check)
        artifacts_location = self._get_artifact_location(volume, manifest.s3_etag)
        logger.info(f"Artifacts location: s3://{artifacts_location.bucket}/{artifacts_location.prefix}")
        
        # Check success.json first (source of truth)
        if self._check_success_marker(artifacts_location):
            logger.info("Success marker (success.json) already exists, task already completed, skipping")
            # Still delete the message to avoid reprocessing
            self.sqs.delete(self.queue_url, msg.receipt_handle)
            return
        
        # Try to claim task execution in DB
        task_execution_id, existing_status, existing_started_at = self.db.claim_task_execution(
            self.conn,
            self.job_record.id,
            volume_id,
            etag_bytes,
            self.worker_id
        )
        
        if task_execution_id is None:
            # Task already exists in database - check if it's stale
            # success.json is the source of truth, so if it doesn't exist, the task is not done
            # regardless of DB status. A task is considered stale if: started_at > 5 minutes ago
            is_stale = False
            if existing_started_at is not None:
                from datetime import datetime, timezone, timedelta
                if isinstance(existing_started_at, str):
                    # Parse ISO format string if needed
                    existing_started_at = datetime.fromisoformat(existing_started_at.replace('Z', '+00:00'))
                if existing_started_at.tzinfo is None:
                    # Assume UTC if timezone-naive
                    existing_started_at = existing_started_at.replace(tzinfo=timezone.utc)
                
                age = datetime.now(timezone.utc) - existing_started_at
                is_stale = age > timedelta(minutes=5)
            
            if is_stale:
                # Stale record - likely from a crashed worker, claim it and proceed
                status_msg = f"status='{existing_status}'" if existing_status else "unknown status"
                logger.warning(
                    f"Task already exists in database with stale {status_msg} "
                    f"(started_at={existing_started_at}, age > 5 minutes) but no success.json found. "
                    f"Assuming previous worker crashed. Claiming stale task and proceeding. "
                    f"job_id={self.job_record.id}, volume_id={volume_id}"
                )
                # Try to claim the stale task
                task_execution_id = self.db.claim_stale_task_execution(
                    self.conn,
                    self.job_record.id,
                    volume_id,
                    etag_bytes,
                    self.worker_id
                )
                if task_execution_id is None:
                    # Race condition: another worker claimed it first, skip
                    logger.warning(
                        f"Failed to claim stale task (another worker may have claimed it). Skipping."
                    )
                    self.sqs.delete(self.queue_url, msg.receipt_handle)
                    return
                logger.info(f"Claimed stale task execution: {task_execution_id}")
            else:
                # Not stale - another worker is likely active, skip with warning
                status_msg = f"status='{existing_status}'" if existing_status else "unknown status"
                etag_preview = manifest.s3_etag[:16] if len(manifest.s3_etag) > 16 else manifest.s3_etag
                logger.warning(
                    f"Task already exists in database ({status_msg}) but no success.json found. "
                    f"job_id={self.job_record.id}, volume_id={volume_id}, etag={etag_preview}..., "
                    f"started_at={existing_started_at}. "
                    f"Assuming another worker is processing. Skipping."
                )
                # Still delete the message to avoid reprocessing
                self.sqs.delete(self.queue_url, msg.receipt_handle)
                return
        else:
            # Successfully claimed new task
            logger.info(f"Claimed new task execution: {task_execution_id}")
        
        # Run job worker
        try:
            ctx = JobContext(
                job_id=self.job_record.id,
                volume=volume,
                job_name=self.job_record.name,
                job_config=self.job_config,
                config_str=self.job_record.config_text,
                volume_manifest=manifest,
                artifacts_location=artifacts_location,
            )
            
            logger.info(f"Running job worker for volume {volume.w_id}/{volume.i_id}")
            result = self.job_worker.run(ctx)
            logger.info(
                f"Job worker completed for volume {volume.w_id}/{volume.i_id}: "
                f"{result.total_images} images, {result.nb_errors} errors, "
                f"avg {result.avg_duration_per_page_ms:.1f}ms/page"
            )
            
            # Write success marker
            elapsed_ms = (time.time() - start_time) * 1000
            success_payload = {
                'job_id': self.job_record.id,
                'job_name': self.job_record.name,
                'volume': {'w_id': volume.w_id, 'i_id': volume.i_id},
                'worker_id': self.worker_id,
                'task_execution_id': task_execution_id,
                'total_images': result.total_images,
                'nb_errors': result.nb_errors,
                'total_duration_ms': result.total_duration_ms,
                'avg_duration_per_page_ms': result.avg_duration_per_page_ms,
                'wall_clock_duration_ms': elapsed_ms,
                'timestamp': time.time(),
            }
            self._write_success_marker(artifacts_location, success_payload)
            logger.info("Wrote success marker")
            
            # Update DB
            self.db.mark_task_done(
                self.conn,
                task_execution_id,
                result.total_images,
                result.nb_errors,
                result.total_duration_ms,
                result.avg_duration_per_page_ms,
            )
            logger.info("Updated DB with task completion")
            
            # Delete SQS message
            self.sqs.delete(self.queue_url, msg.receipt_handle)
            
            # Log volume completion
            total_time = (time.time() - start_time)
            logger.info(
                f"Volume {volume.w_id}/{volume.i_id} completed successfully: "
                f"{result.total_images} images, {result.nb_errors} errors, "
                f"wall_clock={total_time:.1f}s, avg={result.avg_duration_per_page_ms:.1f}ms/page"
            )
            
        except Exception as exc:
            # Classify exception
            retryable, reason = self._classify_exception(exc)
            logger.error(f"Task failed (retryable={retryable}): {reason}", exc_info=True)
            
            # Update DB
            self.db.mark_task_failed(self.conn, task_execution_id, retryable)
            
            # If terminal error, delete message to avoid reprocessing
            if not retryable:
                self.sqs.delete(self.queue_url, msg.receipt_handle)
            
            # Re-raise to let caller handle
            raise

    def _get_volume_manifest(self, volume: VolumeRef) -> VolumeManifest:
        """
        Fetch volume manifest from S3.
        
        The manifest is stored at: Works/{hash}/{w_id}/images/{w_id}-{suffix}/dimensions.json
        It's gzipped JSON with format: [{"filename": "I123.jpg", ...}, ...]
        """
        prefix = get_s3_folder_prefix(volume.w_id, volume.i_id)
        manifest_key = f"{prefix}dimensions.json"
        
        try:
            # Get object with metadata
            response = self.s3.get_object(Bucket=self.s3_source_bucket, Key=manifest_key)
            etag = response['ETag'].strip('"')
            last_modified = response['LastModified'].isoformat()
            
            # Read and decompress
            body_bytes = response['Body'].read()
            uncompressed = gzip.decompress(body_bytes)
            data = json.loads(uncompressed.decode('utf-8'))
            
            # Filter to image files only
            manifest = []
            for item in data:
                filename = item.get('filename')
                if not filename:
                    continue
                
                ext = Path(filename).suffix.lower()
                if ext in IMAGE_EXTENSIONS:
                    manifest.append(item)
            
            return VolumeManifest(
                manifest=manifest,
                s3_etag=etag,
                last_modified_iso=last_modified,
            )
            
        except ClientError as e:
            if e.response['Error']['Code'] == '404':
                raise TerminalTaskError(f"Manifest not found: s3://{self.s3_source_bucket}/{manifest_key}")
            raise RetryableTaskError(f"Failed to fetch manifest: {e}")

    def _get_artifact_location(self, volume: VolumeRef, s3_etag: str) -> ArtifactLocation:
        """
        Compute artifact location for this job/volume/version.
        
        Format: {job_name}/{w_id}/{i_id}/{version}/
        Where version is first 6 chars of etag (without quotes).
        """
        version = s3_etag.replace('"', '').split('-')[0][:6]
        prefix = f"{self.job_record.name}/{volume.w_id}/{volume.i_id}/{version}"
        basename = f"{volume.w_id}-{volume.i_id}-{version}"
        
        return ArtifactLocation(
            bucket=self.s3_dest_bucket,
            prefix=prefix,
            basename=basename,
        )

    def _check_success_marker(self, artifacts: ArtifactLocation) -> bool:
        """Check if success.json exists."""
        try:
            self.s3.head_object(Bucket=artifacts.bucket, Key=artifacts.success_key)
            return True
        except ClientError as e:
            if e.response['Error']['Code'] == '404':
                return False
            raise

    def _read_success_marker(self, artifacts: ArtifactLocation) -> Dict[str, Any]:
        """Read success.json content."""
        try:
            response = self.s3.get_object(Bucket=artifacts.bucket, Key=artifacts.success_key)
            data = json.loads(response['Body'].read().decode('utf-8'))
            return data
        except Exception as e:
            logger.warning(f"Failed to read success marker: {e}")
            return {}

    def _write_success_marker(
        self,
        artifacts: ArtifactLocation,
        payload: Dict[str, Any],
    ) -> None:
        """Write success.json to S3."""
        content = json.dumps(payload, indent=2)
        
        self.s3.put_object(
            Bucket=artifacts.bucket,
            Key=artifacts.success_key,
            Body=content.encode('utf-8'),
            ContentType='application/json',
        )

    def _classify_exception(self, exc: Exception) -> Tuple[bool, str]:
        """
        Returns (retryable, reason).
        Default: retryable=True unless it's a TerminalTaskError.
        """
        if isinstance(exc, TerminalTaskError):
            return False, str(exc)
        elif isinstance(exc, RetryableTaskError):
            return True, str(exc)
        else:
            # Default: treat as retryable
            return True, f"{type(exc).__name__}: {exc}"

    def _maybe_heartbeat(self) -> None:
        """Update heartbeat if enough time has passed."""
        now = time.time()
        if now - self._last_heartbeat >= self._heartbeat_interval:
            try:
                self.db.heartbeat(self.conn, self.worker_id)
                self._last_heartbeat = now
            except Exception as e:
                logger.warning(f"Failed to update heartbeat: {e}")

    def _get_instance_id(self) -> str:
        """
        Get EC2 instance ID, or use hostname as fallback.
        """
        try:
            from ec2_metadata import ec2_metadata
            if ec2_metadata.instance_id:
                return ec2_metadata.instance_id
        except Exception:
            pass
        
        # Fall back to hostname if not on EC2 or library not available
        return socket.gethostname()


def get_s3_folder_prefix(w_id: str, i_id: str) -> str:
    """
    Compute the S3 prefix (~folder) for a volume.
    
    Format: Works/{hash}/{w_id}/images/{w_id}-{suffix}/
    
    Example:
       - w_id=W22084, i_id=I0886
       - result = "Works/60/W22084/images/W22084-0886/"
    
    Where:
       - hash is first 2 chars of MD5 of w_id
       - suffix is i_id without "I" prefix if it's I + 4 digits, else full i_id
    """
    md5_hash = hashlib.md5(w_id.encode()).hexdigest()[:2]
    
    # Compute suffix
    if i_id.startswith('I') and i_id[1:].isdigit() and len(i_id) == 5:
        suffix = i_id[1:]  # Remove 'I' prefix
    else:
        suffix = i_id
    
    return f"Works/{md5_hash}/{w_id}/images/{w_id}-{suffix}/"
