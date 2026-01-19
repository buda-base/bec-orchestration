import asyncio
import logging
import time
import traceback
from typing import Iterable, Optional

from pathlib import Path
from typing import List, Optional, Union
from urllib.parse import urlparse, unquote

from ..ldv1.types_common import *

logger = logging.getLogger(__name__)
# Separate logger for timing/performance warnings (ERROR by default)
timings_logger = logging.getLogger("bec_timings")

class BasePrefetcher:
    """
    Shared worker pool + error wrapping.
    Subclasses implement _fetch_impl(task) -> (source_etag, bytes).
    """

    def __init__(self, cfg, volume_task: VolumeTask, q_prefetcher_to_decoder: asyncio.Queue[FetchedBytesMsg]):
        self.cfg = cfg
        self.volume_task = volume_task
        self.image_tasks: List[ImageTask] = list(volume_task.image_tasks)
        self.q_prefetcher_to_decoder = q_prefetcher_to_decoder
        self._per_worker_sem: asyncio.Semaphore

    def _is_retryable(self, exc: Exception) -> Union[bool, float]:
        """Determine if an exception is retryable and return retry delay.
        
        Returns:
            False: Not retryable
            float: Retryable with the given delay in seconds (per attempt, will be multiplied by attempt number)
        """
        return False  # keep simple; classify later if desired

    async def _fetch_impl(self, task: ImageTask) -> tuple[Optional[str], bytes]:
        raise NotImplementedError

    async def _fetch_one(self, task: ImageTask) -> FetchedBytesMsg:
        attempt = 1
        max_attempts = 3

        while True:
            try:
                async with self._per_worker_sem:
                    source_etag, body = await self._fetch_impl(task)
                return FetchedBytes(task=task, source_etag=source_etag, file_bytes=body)

            except Exception as e:
                tb = traceback.format_exc()
                retry_result = self._is_retryable(e)
                
                # retry_result is either False (not retryable) or a float (delay per attempt)
                retryable = retry_result is not False
                base_delay = retry_result if isinstance(retry_result, (int, float)) else 0.2

                err = PipelineError(
                    stage="Prefetcher",
                    task=task,
                    source_etag=None,
                    error_type=type(e).__name__,
                    message=str(e),
                    traceback=tb,
                    retryable=retryable,
                    attempt=attempt,
                )

                if attempt >= max_attempts or not retryable:
                    return err

                attempt += 1
                # Multiply base delay by attempt number for exponential backoff
                delay = base_delay * attempt
                await asyncio.sleep(delay)

    async def run(self) -> None:
        bulk_prefetch = getattr(self.cfg, "bulk_prefetch", False)
        
        if bulk_prefetch:
            await self._run_bulk()
        else:
            await self._run_streaming()

    async def _run_bulk(self) -> None:
        """
        Bulk prefetch mode: high-parallelism streaming with large queue.
        
        Strategy:
        1. High concurrency (128) to maximize S3 throughput
        2. Large queue (1000) so we never block on backpressure
        3. Emit immediately as fetches complete (overlap I/O with processing)
        4. Short warmup in TileBatcher handles batch consistency
        """
        bulk_concurrency = getattr(self.cfg, "bulk_prefetch_concurrency", 128)
        self._per_worker_sem = asyncio.Semaphore(bulk_concurrency)
        n_images = len(self.image_tasks)
        
        run_start = time.perf_counter()
        logger.info(f"[Prefetcher] BULK MODE: {n_images} images, {bulk_concurrency} concurrent, queue={self.q_prefetcher_to_decoder.maxsize}")
        
        # Create all fetch tasks at once
        fetch_tasks = [
            asyncio.create_task(self._fetch_one(task)) 
            for task in self.image_tasks
        ]
        
        # Emit as they complete (streaming, no blocking with large queue)
        fetched = 0
        errors = 0
        total_bytes = 0
        
        for coro in asyncio.as_completed(fetch_tasks):
            msg = await coro
            
            if isinstance(msg, FetchedBytes):
                fetched += 1
                total_bytes += len(msg.file_bytes)
                trace_frame("Prefetcher", "fetched", msg.task.img_filename)
            elif isinstance(msg, PipelineError):
                errors += 1
                trace_frame_error("Prefetcher", msg.task.img_filename if msg.task else "unknown", f"{msg.error_type}: {msg.message}")
            else:
                errors += 1
            
            # With large queue (1000), this should never block
            await self.q_prefetcher_to_decoder.put(msg)
            
            # Progress every 200 images
            if (fetched + errors) % 200 == 0:
                elapsed = time.perf_counter() - run_start
                mb = total_bytes / (1024 * 1024)
                logger.info(f"[Prefetcher] Progress: {fetched + errors}/{n_images} ({mb:.0f}MB in {elapsed:.1f}s)")
        
        total_time = time.perf_counter() - run_start
        mb_fetched = total_bytes / (1024 * 1024)
        throughput = mb_fetched / total_time if total_time > 0 else 0
        
        logger.info(
            f"[Prefetcher] DONE - {fetched} fetched, {errors} errors, "
            f"{mb_fetched:.1f}MB in {total_time:.2f}s ({throughput:.1f}MB/s)"
        )
        
        await self.q_prefetcher_to_decoder.put(
            EndOfStream(stream="prefetched", producer=type(self).__name__)
        )

    async def _run_streaming(self) -> None:
        """Original streaming mode: fetch and emit one-by-one."""
        self._per_worker_sem = asyncio.Semaphore(self.cfg.inflight_per_worker)

        work_q: asyncio.Queue[Optional[ImageTask]] = asyncio.Queue()
        for t in self.image_tasks:
            work_q.put_nowait(t)

        n_workers = min(len(self.image_tasks), max(1, self.cfg.inflight_per_worker))
        n_images = len(self.image_tasks)
        
        # Timing stats (shared across workers)
        stats = {
            "fetched": 0,
            "errors": 0,
            "total_fetch_time": 0.0,
            "total_bytes": 0,
        }
        stats_lock = asyncio.Lock()
        
        run_start = time.perf_counter()
        logger.info(f"[Prefetcher] Starting {n_workers} workers for {n_images} images")

        async def worker() -> None:
            while True:
                task = await work_q.get()
                if task is None:
                    work_q.task_done()
                    return
                
                fetch_start = time.perf_counter()
                msg = await self._fetch_one(task)
                fetch_time = time.perf_counter() - fetch_start
                
                await self.q_prefetcher_to_decoder.put(msg)
                work_q.task_done()
                
                # Update stats
                async with stats_lock:
                    if isinstance(msg, FetchedBytes):
                        stats["fetched"] += 1
                        stats["total_bytes"] += len(msg.file_bytes)
                    else:
                        stats["errors"] += 1
                    stats["total_fetch_time"] += fetch_time
                    
                    # Log slow fetches
                    if fetch_time > 1.0:
                        timings_logger.warning(
                            f"[Prefetcher] Slow fetch: {task.img_filename} took {fetch_time:.2f}s"
                        )

        workers = [asyncio.create_task(worker()) for _ in range(n_workers)]
        for _ in range(n_workers):
            work_q.put_nowait(None)

        await work_q.join()
        for w in workers:
            await w
        
        run_time = time.perf_counter() - run_start
        avg_fetch = stats["total_fetch_time"] / max(1, stats["fetched"])
        mb_fetched = stats["total_bytes"] / (1024 * 1024)
        throughput = mb_fetched / run_time if run_time > 0 else 0
        
        logger.info(
            f"[Prefetcher] DONE - {stats['fetched']} fetched, {stats['errors']} errors, "
            f"{mb_fetched:.1f}MB in {run_time:.2f}s ({throughput:.1f}MB/s), "
            f"avg_fetch={avg_fetch*1000:.1f}ms"
        )

        await self.q_prefetcher_to_decoder.put(
            EndOfStream(stream="prefetched", producer=type(self).__name__)
        )


# --- S3 implementation ---

class S3Prefetcher(BasePrefetcher):
    def __init__(self, cfg, s3ctx, volume_task: VolumeTask, q_prefetcher_to_decoder: asyncio.Queue[FetchedBytesMsg]):
        super().__init__(cfg, volume_task, q_prefetcher_to_decoder)
        self.s3 = s3ctx

    def _is_retryable(self, exc: Exception) -> Union[bool, float]:
        """Mark credential errors as retryable with longer delay.
        
        On EC2 instances with IAM roles, credential retrieval from the instance
        metadata service can intermittently fail under high concurrency due to:
        - Rate limiting on the metadata service
        - Network timeouts
        - Race conditions during credential refresh
        
        These errors are typically transient and should be retried with a longer
        delay to allow the metadata service to respond.
        
        Returns:
            False: Not a credential error, not retryable
            0.5: Credential error, retryable with 0.5s delay per attempt
        """
        error_type = type(exc).__name__
        error_module = type(exc).__module__
        
        # Check for botocore credential errors
        if error_module.startswith("botocore") and error_type in (
            "NoCredentialsError",
            "CredentialRetrievalError",
            "PartialCredentialsError",
        ):
            return 0.5  # 0.5s delay per attempt for credential errors
        
        # Also check by error message as fallback for edge cases
        error_msg = str(exc).lower()
        if "unable to locate credentials" in error_msg:
            return 0.5  # 0.5s delay per attempt for credential errors
        
        return False

    async def _fetch_impl(self, task: ImageTask) -> tuple[Optional[str], bytes]:
        bucket, key = _parse_s3_uri(task.source_uri)

        async with self.s3.global_sem:
            etag, body = await self.s3.get_object(bucket, key)
            return normalize_etag(etag), body


# --- Local implementation ---

class LocalPrefetcher(BasePrefetcher):
    def __init__(self, cfg, volume_task: VolumeTask, q_prefetcher_to_decoder: asyncio.Queue[FetchedBytesMsg]):
        super().__init__(cfg, volume_task, q_prefetcher_to_decoder)

    async def _fetch_impl(self, task: ImageTask) -> tuple[Optional[str], bytes]:
        p = _parse_file_uri(task.source_uri)

        # Offload blocking disk IO to the default thread pool
        data: bytes = await asyncio.to_thread(p.read_bytes)

        # “etag” equivalent: cheap fingerprint for debugging/caching consistency
        st = await asyncio.to_thread(p.stat)
        pseudo_etag = f"local:{st.st_size}:{st.st_mtime_ns}"
        return pseudo_etag, data


def _parse_s3_uri(uri: str) -> tuple[str, str]:
    """
    Accepts s3://bucket/key and returns (bucket, key).
    """
    u = urlparse(uri)
    if u.scheme != "s3":
        raise ValueError(f"Not an s3:// URI: {uri}")
    bucket = u.netloc
    key = u.path.lstrip("/")
    if not bucket or not key:
        raise ValueError(f"Invalid s3:// URI (need bucket and key): {uri}")
    return bucket, key

def _parse_file_uri(uri: str) -> Path:
    """
    Accepts file:///abs/path (or file:/abs/path) and returns a Path.
    """
    u = urlparse(uri)
    if u.scheme != "file":
        raise ValueError(f"Not a file:// URI: {uri}")

    # urlparse on file URIs:
    # - path is percent-encoded; unquote it
    # - netloc is typically "" for local files; we ignore netloc here
    p = unquote(u.path)
    if not p:
        raise ValueError(f"Invalid file:// URI: {uri}")
    return Path(p)

def normalize_etag(etag: str) -> str:
    """Normalize S3 ETag by stripping a single pair of surrounding quotes."""
    if len(etag) >= 2 and etag[0] == '"' and etag[-1] == '"':
        return etag[1:-1]
    return etag