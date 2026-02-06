from __future__ import annotations

import gzip
import hashlib
import json
import logging
import signal
import socket
import time
from pathlib import Path
from typing import Any

import boto3
import psycopg
from botocore.exceptions import ClientError

from bec_orch.config import OrchestrationConfig
from bec_orch.core.models import (
    ArtifactLocation,
    JobRecord,
    SqsTaskMessage,
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
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}


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
        s3_dest_bucket: str | None = None,
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
        self.conn: Any | None = None  # psycopg.Connection
        self.worker_id: int | None = None
        self.job_record: JobRecord | None = None
        self.job_config: dict[str, Any] | None = None
        self.job_worker: JobWorker | None = None
        self.queue_url: str | None = None
        self.dlq_url: str | None = None

        # S3 client
        self.s3 = boto3.client("s3", region_name=cfg.aws_region)

        # Heartbeat tracking
        self._last_heartbeat: float = 0
        self._heartbeat_interval: float = 30.0  # seconds

        # Empty poll tracking
        self._empty_polls: int = 0

        # DB reconnection settings
        self._max_db_retries: int = 3
        self._db_retry_delay: float = 1.0  # seconds

        # Graceful shutdown flag
        self._shutdown_requested: bool = False
        self._processing_message: bool = False

    def initialize(self) -> None:
        """
        Initialize worker:
        - Open DB connection
        - Register worker in DB
        - Fetch job record (by name) and parse config
        - Get queue URLs from job record
        - Load job worker implementation via registry
        - Install signal handlers for graceful shutdown
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
            logger.warning("Job config is not valid JSON, using as raw string")
            self.job_config = {}

        # Load job worker
        factory = get_job_worker_factory(self.job_record.name)
        self.job_worker = factory(self.job_config)  # Pass job_config to factory
        logger.info(f"Loaded job worker for '{self.job_record.name}'")

        # Install signal handlers for graceful shutdown
        self._install_signal_handlers()

    def run_forever(self) -> None:
        """
        Main loop: poll SQS, process messages, exit after N empty polls.
        If shutdown_after_empty_polls <= 0, runs indefinitely (daemon mode for systemd).
        Handles graceful shutdown on SIGTERM/SIGINT.
        """
        if self.cfg.shutdown_after_empty_polls > 0:
            logger.info(f"Starting main loop (shutdown after {self.cfg.shutdown_after_empty_polls} empty polls)")
        else:
            logger.info("Starting main loop (running indefinitely in daemon mode)")

        while True:
            # Check for shutdown request
            if self._shutdown_requested:
                if self._processing_message:
                    logger.info("Shutdown requested, waiting for current job to complete")
                    # Continue to allow current message to finish
                else:
                    logger.info("Shutdown requested, stopping SQS polling")
                    break

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

                # Only check shutdown threshold if configured (> 0)
                if self.cfg.shutdown_after_empty_polls > 0:
                    logger.info(f"No messages received ({self._empty_polls}/{self.cfg.shutdown_after_empty_polls})")

                    if self._empty_polls >= self.cfg.shutdown_after_empty_polls:
                        logger.info("Shutdown threshold reached, exiting")
                        break
                # In daemon mode, log less frequently
                elif self._empty_polls % 10 == 1:
                    logger.debug(
                        f"No messages, continuing to poll (daemon mode, {self._empty_polls} empty polls so far)"
                    )

                continue

            # Check again for shutdown before processing new message
            if self._shutdown_requested:
                logger.info("Shutdown requested before processing message, stopping without processing")
                logger.info(f"Message {msg.message_id} will be reprocessed after visibility timeout expires")
                break

            # Reset empty poll counter
            self._empty_polls = 0

            # Log message receipt
            logger.info(f"Received message from SQS: {msg.message_id} for volume {msg.volume.w_id}/{msg.volume.i_id}")

            # Mark that we're processing a message
            self._processing_message = True

            # Process message
            try:
                self._process_message(msg)
            except Exception as e:
                logger.error(f"Failed to process message: {e}", exc_info=True)
                # Don't delete message - let it go back to queue or DLQ
                # Could implement visibility timeout extension here
            finally:
                self._processing_message = False

    def shutdown(self) -> None:
        """Mark worker stopped and close DB connection."""
        logger.info("Shutting down BECWorkerRuntime")

        if self.worker_id:
            try:
                # Try to mark worker stopped with reconnection if needed
                self._ensure_connection()
                self.db.mark_worker_stopped(self.conn, self.worker_id)
            except Exception as e:
                logger.error(f"Failed to mark worker stopped: {e}")

        if self.conn:
            try:
                self.conn.close()
            except Exception as e:
                logger.error(f"Failed to close DB connection: {e}")

    def process_volume_directly(self, w_id: str, i_id: str, force: bool = False) -> None:
        """
        Process a single volume directly without pulling from SQS.

        This simulates receiving an SQS message but operates directly on the volume.
        Useful for manual processing, testing, or reprocessing failed volumes.

        Args:
            w_id: Work ID (e.g., "W22084")
            i_id: Image group ID (e.g., "I0886")
            force: If True, bypass success.json check AND claim task even if another
                   worker has it (useful for reprocessing or recovering from stuck tasks)

        Raises:
            KeyboardInterrupt: If user cancels, re-raised with message about --force
        """
        volume = VolumeRef(w_id=w_id, i_id=i_id)

        logger.info(f"Processing volume directly: {w_id}/{i_id} (force={force})")

        # Create a fake SQS message for compatibility with existing flow
        fake_msg = SqsTaskMessage(
            message_id=f"direct-{w_id}-{i_id}",
            receipt_handle="",  # No receipt handle for direct processing
            body="",
            volume=volume,
        )

        try:
            self._process_volume(fake_msg, force=force, is_direct=True)
            logger.info(f"Volume {w_id}/{i_id} processed successfully")
        except KeyboardInterrupt:
            logger.warning(f"Volume processing cancelled by user: {w_id}/{i_id}")
            # Check if success.json exists to guide user
            try:
                manifest = self._get_volume_manifest(volume)
                artifacts = self._get_artifact_location(volume, manifest.s3_etag)
                has_success = self._check_success_marker(artifacts)

                if has_success and not force:
                    logger.info(
                        f"Volume {w_id}/{i_id} was already completed. To reprocess, run again with --force flag."
                    )
            except Exception:
                pass  # Ignore errors during cleanup

            raise

    # ---- internals ----

    def _process_message(self, msg: SqsTaskMessage) -> None:
        """
        High-level flow for SQS message processing.
        Wraps _process_volume and handles SQS-specific logic (message deletion).
        """
        try:
            self._process_volume(msg, force=False, is_direct=False)
            logger.info(f"Successfully processed message: {msg.message_id}")
        except Exception:
            # Don't delete message - let it go back to queue or DLQ
            raise

    def _process_volume(self, msg: SqsTaskMessage, force: bool = False, is_direct: bool = False) -> None:
        """
        Core volume processing logic.

        High-level flow:
        - Resolve manifest info + etag
        - Ensure volume exists in DB
        - Compute artifact location
        - Check success.json first (source of truth) - if exists, skip (unless force=True)
        - Try to claim task in DB
        - If claim fails:
          - If force=True: forcefully claim task (UPDATE) regardless of status/age
          - If not force but stale (> 5 min old): claim stale task and proceed
          - If not force and not stale: skip with warning (another worker likely active)
        - Run job worker
        - Write success.json
        - Update DB
        - Delete SQS message when safe (if not direct)

        Args:
            msg: SQS message (or fake message for direct processing)
            force: If True, bypass success.json check AND forcefully claim task (no restrictions)
            is_direct: If True, this is direct processing (not from SQS)
        """
        start_time = time.time()
        volume = msg.volume

        logger.info(
            f"Starting volume processing: {volume.w_id}/{volume.i_id} "
            f"(message_id={msg.message_id}, force={force}, direct={is_direct})"
        )

        # Get volume manifest from S3
        manifest = self._get_volume_manifest(volume)
        logger.info(f"Loaded manifest: {len(manifest.manifest)} files, etag={manifest.s3_etag}")

        # Ensure volume exists in DB (with retry on connection errors)
        etag_bytes = etag_to_bytes(manifest.s3_etag)
        volume_id = self._execute_with_retry(
            "ensure_volume",
            self.db.ensure_volume,
            volume.w_id,
            volume.i_id,
            etag_bytes,
            manifest.last_modified_iso,
            len(manifest.manifest),
            nb_images_intro=0,
        )
        logger.info(f"Volume ID: {volume_id}")

        # Compute artifact location (needed for success.json check)
        artifacts_location = self._get_artifact_location(volume, manifest.s3_etag)
        logger.info(f"Artifacts location: s3://{artifacts_location.bucket}/{artifacts_location.prefix}")

        # Check success.json first (source of truth)
        if self._check_success_marker(artifacts_location):
            if force:
                logger.warning(
                    f"Success marker (success.json) already exists for {volume.w_id}/{volume.i_id}, "
                    "but force=True, so reprocessing anyway"
                )
            else:
                logger.info("Success marker (success.json) already exists, task already completed, skipping")
                # Still delete the message to avoid reprocessing (if not direct)
                if not is_direct and msg.receipt_handle:
                    self.sqs.delete(self.queue_url, msg.receipt_handle)
                return

        # Try to claim task execution in DB (with retry on connection errors)
        task_execution_id, existing_status, existing_started_at = self._execute_with_retry(
            "claim_task_execution",
            self.db.claim_task_execution,
            self.job_record.id,
            volume_id,
            etag_bytes,
            self.worker_id,
        )

        if task_execution_id is None:
            # Task already exists in database
            status_msg = f"status='{existing_status}'" if existing_status else "unknown status"

            if force:
                # Force mode: claim task regardless of status or age
                logger.warning(
                    f"Task already exists in database with {status_msg} "
                    f"(started_at={existing_started_at}) but force=True. "
                    f"Forcefully claiming task and reprocessing. "
                    f"job_id={self.job_record.id}, volume_id={volume_id}"
                )
                # Use force claim which always succeeds (with retry on connection errors)
                task_execution_id = self._execute_with_retry(
                    "force_claim_task_execution",
                    self.db.force_claim_task_execution,
                    self.job_record.id,
                    volume_id,
                    etag_bytes,
                    self.worker_id,
                )
                if task_execution_id is None:
                    # Should never happen with force claim, but handle gracefully
                    logger.error("Failed to force claim task - unexpected error")
                    if not is_direct and msg.receipt_handle:
                        self.sqs.delete(self.queue_url, msg.receipt_handle)
                    return
                logger.info(f"Force claimed task execution: {task_execution_id}")
            else:
                # Normal mode: check if task is stale (> 5 minutes old)
                is_stale = False
                if existing_started_at is not None:
                    from datetime import datetime, timedelta, timezone

                    if isinstance(existing_started_at, str):
                        # Parse ISO format string if needed
                        existing_started_at = datetime.fromisoformat(existing_started_at.replace("Z", "+00:00"))
                    if existing_started_at.tzinfo is None:
                        # Assume UTC if timezone-naive
                        existing_started_at = existing_started_at.replace(tzinfo=timezone.utc)

                    age = datetime.now(timezone.utc) - existing_started_at
                    is_stale = age > timedelta(minutes=5)

                if is_stale:
                    # Stale record - likely from a crashed worker, claim it and proceed
                    logger.warning(
                        f"Task already exists in database with stale {status_msg} "
                        f"(started_at={existing_started_at}, age > 5 minutes) but no success.json found. "
                        f"Assuming previous worker crashed. Claiming stale task and proceeding. "
                        f"job_id={self.job_record.id}, volume_id={volume_id}"
                    )
                    # Try to claim the stale task (with retry on connection errors)
                    task_execution_id = self._execute_with_retry(
                        "claim_stale_task_execution",
                        self.db.claim_stale_task_execution,
                        self.job_record.id,
                        volume_id,
                        etag_bytes,
                        self.worker_id,
                    )
                    if task_execution_id is None:
                        # Race condition: another worker claimed it first, skip
                        logger.warning("Failed to claim stale task (another worker may have claimed it). Skipping.")
                        if not is_direct and msg.receipt_handle:
                            self.sqs.delete(self.queue_url, msg.receipt_handle)
                        return
                    logger.info(f"Claimed stale task execution: {task_execution_id}")
                else:
                    # Not stale - another worker is likely active, skip with warning
                    etag_preview = manifest.s3_etag[:16] if len(manifest.s3_etag) > 16 else manifest.s3_etag
                    logger.warning(
                        f"Task already exists in database ({status_msg}) but no success.json found. "
                        f"job_id={self.job_record.id}, volume_id={volume_id}, etag={etag_preview}..., "
                        f"started_at={existing_started_at}. "
                        f"Assuming another worker is processing. Skipping."
                    )
                    # Still delete the message to avoid reprocessing (if not direct)
                    if not is_direct and msg.receipt_handle:
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
                "job_id": self.job_record.id,
                "job_name": self.job_record.name,
                "volume": {"w_id": volume.w_id, "i_id": volume.i_id},
                "worker_id": self.worker_id,
                "task_execution_id": task_execution_id,
                "total_images": result.total_images,
                "nb_errors": result.nb_errors,
                "total_duration_ms": result.total_duration_ms,
                "avg_duration_per_page_ms": result.avg_duration_per_page_ms,
                "wall_clock_duration_ms": elapsed_ms,
                "timestamp": time.time(),
            }
            self._write_success_marker(artifacts_location, success_payload)
            logger.info("Wrote success marker")

            # Update DB with retry logic for connection errors
            self._execute_with_retry(
                "mark_task_done",
                self.db.mark_task_done,
                task_execution_id,
                result.total_images,
                result.nb_errors,
                result.total_duration_ms,
                result.avg_duration_per_page_ms,
            )
            logger.info("Updated DB with task completion")

            # Delete SQS message (if not direct)
            if not is_direct and msg.receipt_handle:
                self.sqs.delete(self.queue_url, msg.receipt_handle)

            # Log volume completion
            total_time = time.time() - start_time
            logger.info(
                f"Volume {volume.w_id}/{volume.i_id} completed successfully: "
                f"{result.total_images} images, {result.nb_errors} errors, "
                f"wall_clock={total_time:.1f}s, avg={result.avg_duration_per_page_ms:.1f}ms/page"
            )

        except Exception as exc:
            # Classify exception
            retryable, reason = self._classify_exception(exc)
            logger.error(f"Task failed (retryable={retryable}): {reason}", exc_info=True)

            # Update DB (with retry on connection errors)
            try:
                self._execute_with_retry(
                    "mark_task_failed",
                    self.db.mark_task_failed,
                    task_execution_id,
                    retryable,
                )
            except Exception as db_err:
                # Log but don't fail - the original exception is more important
                logger.error(f"Failed to mark task as failed in DB: {db_err}")

            # If terminal error, delete message to avoid reprocessing (if not direct)
            if not retryable and not is_direct and msg.receipt_handle:
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
        logger.debug(f"Fetching manifest from s3://{self.s3_source_bucket}/{manifest_key}")

        try:
            # Get object with metadata
            response = self.s3.get_object(Bucket=self.s3_source_bucket, Key=manifest_key)
            etag = response["ETag"].strip('"')
            last_modified = response["LastModified"].isoformat()

            # Read and decompress
            body_bytes = response["Body"].read()
            uncompressed = gzip.decompress(body_bytes)
            data = json.loads(uncompressed.decode("utf-8"))

            # Filter to image files only
            manifest = []
            for item in data:
                filename = item.get("filename")
                if not filename:
                    continue
                if "." not in filename or "/" in filename:
                    # should always have a dot and no /
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
            if e.response["Error"]["Code"] == "404":
                raise TerminalTaskError(f"Manifest not found: s3://{self.s3_source_bucket}/{manifest_key}")
            raise RetryableTaskError(f"Failed to fetch manifest: {e}")

    def _get_artifact_location(self, volume: VolumeRef, s3_etag: str) -> ArtifactLocation:
        """
        Compute artifact location for this job/volume/version.

        Format: {job_name}/{w_id}/{i_id}/{version}/
        Where version is first 6 chars of etag (without quotes).
        """
        version = s3_etag.replace('"', "").split("-")[0][:6]
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
            if e.response["Error"]["Code"] == "404":
                return False
            raise

    def _read_success_marker(self, artifacts: ArtifactLocation) -> dict[str, Any]:
        """Read success.json content."""
        try:
            response = self.s3.get_object(Bucket=artifacts.bucket, Key=artifacts.success_key)
            data = json.loads(response["Body"].read().decode("utf-8"))
            return data
        except Exception as e:
            logger.warning(f"Failed to read success marker: {e}")
            return {}

    def _write_success_marker(
        self,
        artifacts: ArtifactLocation,
        payload: dict[str, Any],
    ) -> None:
        """Write success.json to S3."""
        content = json.dumps(payload, indent=2)

        self.s3.put_object(
            Bucket=artifacts.bucket,
            Key=artifacts.success_key,
            Body=content.encode("utf-8"),
            ContentType="application/json",
        )

    def _classify_exception(self, exc: Exception) -> tuple[bool, str]:
        """
        Returns (retryable, reason).
        Default: retryable=True unless it's a TerminalTaskError.
        """
        if isinstance(exc, TerminalTaskError):
            return False, str(exc)
        if isinstance(exc, RetryableTaskError):
            return True, str(exc)
        # Check for VolumeTimeoutError by name (avoid circular import)
        exc_type = type(exc).__name__
        if exc_type == "VolumeTimeoutError":
            return True, f"Volume timeout: {exc}"
        # Default: treat as retryable
        return True, f"{exc_type}: {exc}"

    def _ensure_connection(self) -> None:
        """
        Ensure the database connection is alive, reconnecting if necessary.

        This method checks if the connection is closed and attempts to reconnect.
        Should be called before critical DB operations.
        """
        if self.conn is None or self.conn.closed:
            logger.warning("Database connection is closed, attempting to reconnect...")
            try:
                self.conn = self.db.connect()
                logger.info("Successfully reconnected to database")
            except Exception as e:
                logger.error(f"Failed to reconnect to database: {e}")
                raise

    def _execute_with_retry(self, operation_name: str, operation_func, *args, **kwargs):
        """
        Execute a database operation with automatic retry on connection errors.

        Args:
            operation_name: Human-readable name for logging
            operation_func: The function to call (should be a DBClient method)
            *args: Positional arguments for the function
            **kwargs: Keyword arguments for the function

        Returns:
            The result of the operation

        Raises:
            The last exception if all retries fail
        """
        last_exception = None

        for attempt in range(self._max_db_retries):
            try:
                # Ensure connection is alive before attempting operation
                self._ensure_connection()
                return operation_func(self.conn, *args, **kwargs)
            except psycopg.OperationalError as e:
                last_exception = e
                error_msg = str(e).lower()
                is_connection_error = (
                    "connection is closed" in error_msg
                    or "timeout" in error_msg
                    or "connection refused" in error_msg
                    or "could not connect" in error_msg
                    or "server closed the connection" in error_msg
                )

                if is_connection_error and attempt < self._max_db_retries - 1:
                    logger.warning(
                        f"DB connection error during {operation_name} (attempt {attempt + 1}/{self._max_db_retries}): {e}"
                    )
                    # Force reconnection on next attempt
                    try:
                        if self.conn and not self.conn.closed:
                            self.conn.close()
                    except Exception:
                        pass
                    self.conn = None
                    time.sleep(self._db_retry_delay * (attempt + 1))  # Exponential backoff
                    continue
                raise
            except Exception as e:
                # Non-connection errors are not retried
                raise

        # All retries exhausted
        raise last_exception

    def _maybe_heartbeat(self) -> None:
        """Update heartbeat if enough time has passed."""
        now = time.time()
        if now - self._last_heartbeat >= self._heartbeat_interval:
            try:
                self._ensure_connection()
                self.db.heartbeat(self.conn, self.worker_id)
                self._last_heartbeat = now
            except psycopg.OperationalError as e:
                # Connection errors during heartbeat - try to reconnect
                logger.warning(f"Failed to update heartbeat (connection error): {e}")
                try:
                    if self.conn and not self.conn.closed:
                        self.conn.close()
                except Exception:
                    pass
                self.conn = None
                # Will reconnect on next operation
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

    def _install_signal_handlers(self) -> None:
        """
        Install signal handlers for graceful shutdown.

        Handles SIGTERM (sent by systemd/AWS during shutdown) and SIGINT (Ctrl+C).
        When a signal is received:
        - Stops polling SQS immediately (no new messages)
        - Allows current job to complete if one is in progress
        - Cleans up and exits
        """

        def signal_handler(signum: int, frame: Any) -> None:
            sig_name = signal.Signals(signum).name
            if not self._shutdown_requested:
                logger.info(
                    f"Received {sig_name} signal. Initiating graceful shutdown. "
                    f"Will stop polling SQS and finish current job if any."
                )
                self._shutdown_requested = True

                # Log current state
                if self._processing_message:
                    logger.info(
                        "Currently processing a job. Will complete it before shutting down. "
                        "If the job doesn't complete within systemd timeout, the message will "
                        "be reprocessed after visibility timeout expires."
                    )
                else:
                    logger.info("No job in progress. Stopping SQS polling immediately.")
            else:
                # Second signal - force exit
                logger.warning(
                    f"Received second {sig_name} signal. Forcing immediate shutdown. "
                    "Current job (if any) will be reprocessed after visibility timeout expires."
                )
                raise KeyboardInterrupt("Forced shutdown by second signal")

        # Install handlers for SIGTERM (systemd/AWS shutdown) and SIGINT (Ctrl+C)
        signal.signal(signal.SIGTERM, signal_handler)
        signal.signal(signal.SIGINT, signal_handler)
        logger.info("Signal handlers installed for graceful shutdown (SIGTERM, SIGINT)")


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
    if i_id.startswith("I") and i_id[1:].isdigit() and len(i_id) == 5:
        suffix = i_id[1:]  # Remove 'I' prefix
    else:
        suffix = i_id

    return f"Works/{md5_hash}/{w_id}/images/{w_id}-{suffix}/"
