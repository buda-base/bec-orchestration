"""
Async OCR pipeline with backpressure.

Pipeline stages:
1. S3 Prefetcher (async, high concurrency) → FetchedBytes
2. Image Processor (thread pool) → LineTensor batches
3. GPU Forward (single async task, batched) → Logits
4. CTC Decoder (thread pool) → PageResult
5. Parquet Writer (single async task) → S3

All stages connected by bounded asyncio.Queue for backpressure.

IMPORTANT: Deadlock prevention measures:
- All queue.get() calls have timeouts to prevent infinite waits
- ProcessPoolExecutor worker crashes are handled with try/except
- asyncio.CancelledError is caught explicitly
- EndOfStream is ALWAYS sent in finally blocks
"""

import asyncio
import contextlib
import logging
import os
import time
from concurrent.futures import BrokenExecutor, ProcessPoolExecutor, ThreadPoolExecutor
from typing import TYPE_CHECKING

import numpy.typing as npt
from botocore.client import BaseClient

# Timeout for queue/executor operations (10 minutes, less than job timeout)
QUEUE_GET_TIMEOUT = 600.0
# Timeout for individual CTC decode operation (5 minutes per line is very generous)
CTC_DECODE_TIMEOUT = 300.0

from .data_structures import (
    EndOfStream,
    ImageTask,
    InferredPage,
    LineResult,
    PageOCRResult,
    PageResult,
    PipelineError,
)
from .line import BBox

if TYPE_CHECKING:
    from .config import OCRV1Config

from .ctc_decoder import (
    CTCDecoder,
    LineDecodeResult,
    decode_logits_beam_search,
    decode_logits_with_segments,
    init_worker_process,
)

# Re-export for use in this module
__all__ = ["AsyncOCRPipeline"]
from .line_decoder import LineDecoder, ProcessedPage
from .model import OCRModel
from .stages import (
    CTCDecoderStage,
    GPUInferenceStage,
    ImageProcessorStage,
    OutputWriterStage,
    ParquetLoaderStage,
    PrefetcherStage,
)

logger = logging.getLogger(__name__)

# Import for parquet loading

# Default vocabulary size for fallback logits
DEFAULT_VOCAB_SIZE = 84

# =============================================================================
# OCR Result Building Utilities
# =============================================================================


def _build_page_ocr_result(
    processed_page: ProcessedPage,
    line_decode_results: list[LineDecodeResult],
) -> PageOCRResult:
    """Build PageOCRResult from processed page and line decode results.

    Each ProcessedLine is already a complete logical line (line segments are merged
    in line_decoder.py if merge_line_segments is enabled). We create a 1:1 mapping
    from ProcessedLine to LineResult.
    """
    if processed_page.error:
        return PageOCRResult(
            img_file_name=processed_page.filename,
            source_etag=processed_page.source_etag,
            rotation_angle=processed_page.rotation_angle,
            tps_points=processed_page.tps_points,
            lines=[],
            error=processed_page.error,
        )

    # Build LineResult objects - one per ProcessedLine (already grouped/merged)
    # Convert bbox from resized image coordinates to original image coordinates
    scale_factor = processed_page.scale_factor
    inv_scale = 1.0 / scale_factor if scale_factor != 0 else 1.0

    line_results = []
    for line_idx, (processed_line, decode_result) in enumerate(
        zip(processed_page.lines, line_decode_results, strict=False)
    ):
        # Skip empty lines (or lines with only whitespace)
        if not decode_result.text or not decode_result.text.strip():
            continue

        x, y, w, h = processed_line.bbox

        # Scale bbox back to original image coordinates
        orig_x = int(x * inv_scale)
        orig_y = int(y * inv_scale)
        orig_w = int(w * inv_scale)
        orig_h = int(h * inv_scale)

        line_result = LineResult(
            line_idx=line_idx,
            bbox=BBox(x=orig_x, y=orig_y, w=orig_w, h=orig_h),
            text=decode_result.text,
            confidence=decode_result.line_confidence,
            syllables=decode_result.segments,
        )
        line_results.append(line_result)

    return PageOCRResult(
        img_file_name=processed_page.filename,
        source_etag=processed_page.source_etag,
        rotation_angle=processed_page.rotation_angle,
        tps_points=processed_page.tps_points,
        lines=line_results,
    )


# =============================================================================
# CTC Decoding Utilities
# =============================================================================


def _decode_single_line(
    cropped_logits: npt.NDArray,
    vocab: list[str],
    beam_width: int | None = None,
    token_min_logp: float | None = None,
    submit_time: float | None = None,
) -> tuple[str, float, float, int]:
    """
    Decode a single line's pre-cropped logits. Used for parallel line decoding.

    Vocabulary pruning is now handled in the model before IPC.

    Args:
        cropped_logits: Pre-cropped logits in (time, vocab) shape (already pruned by model)
        vocab: Vocabulary list (already pruned by model)
        beam_width: Beam width for decoding (passed to worker)
        token_min_logp: Token min log prob (passed to worker)
        submit_time: Time when task was submitted (for IPC measurement)

    Returns:
        Tuple of (decoded_text, decode_time_ms, ipc_overhead_ms, worker_pid)
    """

    arrival_time = time.perf_counter()
    ipc_in_ms = (arrival_time - submit_time) * 1000 if submit_time else 0.0

    start = time.perf_counter()
    text = decode_logits_beam_search(cropped_logits, vocab, beam_width, token_min_logp)
    elapsed_ms = (time.perf_counter() - start) * 1000
    return text.strip().replace("§", " "), elapsed_ms, ipc_in_ms, os.getpid()


def _decode_single_line_with_segments(
    cropped_logits: npt.NDArray,
    vocab: list[str],
    original_width: int,
    beam_width: int | None = None,
    token_min_logp: float | None = None,
    submit_time: float | None = None,
) -> tuple[LineDecodeResult, float, float, int]:
    """
    Decode a single line's pre-cropped logits with structured segment data.

    Args:
        cropped_logits: Pre-cropped logits in (time, vocab) shape (already pruned by model)
        vocab: Vocabulary list (already pruned by model)
        original_width: Original width of the line image
        beam_width: Beam width for decoding (passed to worker)
        token_min_logp: Token min log prob (passed to worker)
        submit_time: Time when task was submitted (for IPC measurement)

    Returns:
        Tuple of (line_decode_result, decode_time_ms, ipc_overhead_ms, worker_pid)
    """

    arrival_time = time.perf_counter()
    ipc_in_ms = (arrival_time - submit_time) * 1000 if submit_time else 0.0

    start = time.perf_counter()
    decode_result = decode_logits_with_segments(
        cropped_logits,
        vocab,
        original_width,
        beam_width=beam_width,
        token_min_logp=token_min_logp,
    )
    elapsed_ms = (time.perf_counter() - start) * 1000
    return decode_result, elapsed_ms, ipc_in_ms, os.getpid()


def _decode_page_lines(
    cropped_logits_list: list[npt.NDArray],
    vocab: list[str],
    beam_width: int | None = None,
    token_min_logp: float | None = None,
) -> list[str]:
    """
    Decode all lines of a page in a single worker call to reduce IPC overhead.

    Instead of 6 IPC calls per page (one per line), we do 1 IPC call with all lines.
    This reduces overhead from ~6*30ms = 180ms to ~30ms per page.

    Vocabulary pruning is now handled in the model before IPC.
    """
    texts = []
    for cropped_logits in cropped_logits_list:
        text = decode_logits_beam_search(cropped_logits, vocab, beam_width, token_min_logp)
        texts.append(text.strip().replace("§", " "))
    return texts


# FetchedBytes and ProcessedPage are imported from line_decoder
# InferredPage, EndOfStream, PipelineError are imported from data_structures


class AsyncOCRPipeline:
    """
    Async OCR pipeline with bounded queues for backpressure.

    Modeled after LDv1 pipeline pattern.
    """

    def __init__(
        self,
        cfg: "OCRV1Config",
        ocr_model: OCRModel,
        ctc_decoder: CTCDecoder,
        s3_client: BaseClient,
        source_image_bucket: str,
        volume_prefix: str,
    ) -> None:
        self.cfg = cfg
        self.ocr_model = ocr_model
        self.ctc_decoder = ctc_decoder
        self.s3_client = s3_client
        self.source_image_bucket = source_image_bucket
        self.volume_prefix = volume_prefix

        # Create LineDecoder for image processing
        self.line_decoder = LineDecoder(cfg, ocr_model)

        # Bounded queues for backpressure
        self.q_fetched: asyncio.Queue = asyncio.Queue(maxsize=64)
        self.q_processed: asyncio.Queue = asyncio.Queue(maxsize=32)
        self.q_inferred: asyncio.Queue = asyncio.Queue(maxsize=32)
        self.q_results: asyncio.Queue = asyncio.Queue(maxsize=64)

        # Thread pool for image processing (cv2 releases GIL)
        self._image_executor = ThreadPoolExecutor(max_workers=cfg.image_processor_workers, thread_name_prefix="img")
        # Process pool for CTC decoding - IPC overhead (~190ms/call) is acceptable tradeoff
        # to avoid blocking the async event loop (which starves GPU inference)
        self._ctc_executor = ProcessPoolExecutor(
            max_workers=cfg.ctc_workers,
            initializer=init_worker_process,
            initargs=(ctc_decoder.ctc_vocab,),
        )

        logger.debug(
            f"Pipeline config: prefetch={cfg.prefetch_concurrency}, "
            f"img_workers={cfg.image_processor_workers}, "
            f"ctc_workers={cfg.ctc_workers}, gpu_batch={cfg.gpu_batch_size}"
        )

        # Stats (shared across stages)
        self.stats = {
            "fetched": 0,
            "processed": 0,
            "inferred": 0,
            "decoded": 0,
            "errors": 0,
        }

        # Create stage instances
        self._parquet_loader = ParquetLoaderStage(s3_client=s3_client)

        self._prefetcher = PrefetcherStage(
            cfg=cfg,
            s3_client=s3_client,
            source_image_bucket=source_image_bucket,
            volume_prefix=volume_prefix,
            q_out=self.q_fetched,
            stats=self.stats,
        )

        self._image_processor = ImageProcessorStage(
            cfg=cfg,
            line_decoder=self.line_decoder,
            parquet_loader=self._parquet_loader,
            q_in=self.q_fetched,
            q_out=self.q_processed,
            executor=self._image_executor,
            stats=self.stats,
        )

        self._gpu_inference = GPUInferenceStage(
            cfg=cfg,
            ocr_model=ocr_model,
            q_in=self.q_processed,
            q_out=self.q_inferred,
            stats=self.stats,
        )

        self._ctc_decoder_stage = CTCDecoderStage(
            cfg=cfg,
            ctc_decoder=ctc_decoder,
            q_in=self.q_inferred,
            q_out=self.q_results,
            executor=self._ctc_executor,
            stats=self.stats,
        )

        # Task tracking for timeout/cleanup
        self._stage_tasks: list[asyncio.Task] = []
        self._is_running = False

    async def run(
        self,
        tasks: list[ImageTask],
        ld_parquet_uri: str,
        output_parquet_uri: str,
        timeout_s: float | None = None,
    ) -> dict:
        """Run the full pipeline with optional timeout protection.

        Args:
            tasks: List of ImageTask objects identifying pages to process
            ld_parquet_uri: S3 URI of the line detection parquet file
            output_parquet_uri: S3 URI for output parquet file
            timeout_s: Optional timeout in seconds. If None, uses cfg.volume_timeout_s

        Raises:
            asyncio.TimeoutError: If pipeline exceeds timeout (after graceful cleanup)
        """
        timeout = timeout_s or getattr(self.cfg, "volume_timeout_s", 600.0)

        try:
            return await asyncio.wait_for(
                self._run_pipeline(tasks, ld_parquet_uri, output_parquet_uri), timeout=timeout
            )
        except asyncio.TimeoutError:
            # Log diagnostic state before cleanup
            diagnostic_state = self.get_diagnostic_state()
            logger.exception(f"[AsyncOCRPipeline] Timeout after {timeout}s. Diagnostic: {diagnostic_state}")
            # Try graceful cleanup
            await self._graceful_shutdown()
            raise

    async def _run_pipeline(
        self,
        tasks: list[ImageTask],
        ld_parquet_uri: str,
        output_parquet_uri: str,
    ) -> dict:
        """Internal pipeline execution (called by run with timeout)."""
        start_time = time.perf_counter()
        total_pages = len(tasks)
        self._is_running = True

        logger.info(f"[AsyncOCRPipeline] Starting pipeline for {total_pages} pages")

        # Queue depth monitor task
        async def monitor_queues() -> None:
            while True:
                await asyncio.sleep(5)
                logger.debug(
                    f"[QueueDepth] fetched={self.q_fetched.qsize()}, processed={self.q_processed.qsize()}, "
                    f"inferred={self.q_inferred.qsize()}, results={self.q_results.qsize()}"
                )

        monitor_task = asyncio.create_task(monitor_queues(), name="monitor")

        # Create output writer stage for this run
        expected_filenames = [t.filename for t in tasks]
        output_writer = OutputWriterStage(
            parquet_uri=output_parquet_uri,
            q_in=self.q_results,
            expected_filenames=expected_filenames,
            stats=self.stats,
        )

        try:
            # Phase 1: Download parquet first (small file, needs full bandwidth)
            # This avoids competing with the prefetcher's 64 concurrent image fetches
            parquet_task = asyncio.create_task(self._parquet_loader.run(ld_parquet_uri), name="parquet_loader")
            await parquet_task

            if self._parquet_loader.parquet_error:
                logger.error(f"[AsyncOCRPipeline] Parquet loading failed: {self._parquet_loader.parquet_error}")
                # Still need to run stages to emit proper errors

            # Phase 2: Start all other stages (parquet already loaded)
            self._stage_tasks = [
                asyncio.create_task(self._prefetcher.run(tasks), name="prefetcher"),
                asyncio.create_task(self._image_processor.run(), name="image_processor"),
                asyncio.create_task(self._gpu_inference.run(), name="gpu_inference"),
                asyncio.create_task(self._ctc_decoder_stage.run(), name="ctc_decoder"),
                asyncio.create_task(output_writer.run(), name="writer"),
            ]

            # Wait for all tasks
            results = await asyncio.gather(*self._stage_tasks, return_exceptions=True)

            # Check for errors
            stage_names = ["prefetcher", "image_processor", "gpu_inference", "ctc_decoder", "writer"]
            for name, result in zip(stage_names, results, strict=True):
                if isinstance(result, Exception):
                    logger.error(f"Pipeline stage '{name}' failed: {result}", exc_info=result)

        finally:
            monitor_task.cancel()
            self._is_running = False
            self._stage_tasks = []

        elapsed = time.perf_counter() - start_time
        logger.info(
            f"[AsyncOCRPipeline] Completed in {elapsed:.2f}s - "
            f"fetched={self.stats['fetched']}, processed={self.stats['processed']}, "
            f"inferred={self.stats['inferred']}, decoded={self.stats['decoded']}, "
            f"errors={self.stats['errors']}"
        )

        return self.stats

    async def _graceful_shutdown(self) -> None:
        """Gracefully shutdown the pipeline on timeout.

        Tries to let the writer flush before cancelling all tasks.
        """
        if not self._stage_tasks:
            return

        logger.info("[AsyncOCRPipeline] Starting graceful shutdown...")

        # Find writer task
        writer_task = None
        for t in self._stage_tasks:
            if t.get_name() == "writer":
                writer_task = t
            elif not t.done():
                t.cancel()

        # Wait briefly for non-writer tasks to cancel
        non_writer_tasks = [t for t in self._stage_tasks if t.get_name() != "writer"]
        with contextlib.suppress(asyncio.TimeoutError):
            await asyncio.wait_for(asyncio.gather(*non_writer_tasks, return_exceptions=True), timeout=2.0)

        # Try to send EOS to writer to let it flush
        if writer_task and not writer_task.done():
            try:
                await asyncio.wait_for(
                    self.q_results.put(EndOfStream(stream="results", producer="timeout_shutdown")), timeout=1.0
                )
                # Wait briefly for writer to flush
                await asyncio.wait_for(writer_task, timeout=5.0)
                logger.debug("[AsyncOCRPipeline] Writer flushed successfully")
            except asyncio.TimeoutError:
                logger.warning("[AsyncOCRPipeline] Writer did not flush in time, cancelling")
                writer_task.cancel()
                with contextlib.suppress(asyncio.TimeoutError, asyncio.CancelledError):
                    await asyncio.wait_for(writer_task, timeout=1.0)

        self._stage_tasks = []
        self._is_running = False
        logger.info("[AsyncOCRPipeline] Graceful shutdown complete")

    def get_diagnostic_state(self) -> dict:
        """Return diagnostic state for debugging stuck pipelines.

        Called when timeout occurs to help identify where the pipeline is stuck.
        """
        return {
            "is_running": self._is_running,
            "queues": {
                "q_fetched": f"{self.q_fetched.qsize()}/64",
                "q_processed": f"{self.q_processed.qsize()}/32",
                "q_inferred": f"{self.q_inferred.qsize()}/32",
                "q_results": f"{self.q_results.qsize()}/64",
            },
            "stats": dict(self.stats),
            "parquet_ready": self._parquet_loader.is_load_complete,
            "parquet_error": self._parquet_loader.parquet_error,
            "active_tasks": [t.get_name() for t in self._stage_tasks if not t.done()] if self._stage_tasks else [],
        }

    async def run_sequential(
        self,
        tasks: list[ImageTask],
        ld_parquet_uri: str,
        output_parquet_uri: str,
        timeout_s: float | None = None,
    ) -> dict:
        """
        Run pipeline in two phases to eliminate CPU contention:
        Phase 1: Prefetch + Image Processing + GPU Inference (collect all logits in memory)
        Phase 2: CTC Decode all collected logits + Write to Parquet

        Args:
            tasks: List of ImageTask objects identifying pages to process
            ld_parquet_uri: S3 URI of the line detection parquet file
            output_parquet_uri: S3 URI for output parquet file
            timeout_s: Optional timeout in seconds. If None, uses cfg.volume_timeout_s

        Raises:
            asyncio.TimeoutError: If pipeline exceeds timeout (after graceful cleanup)
        """
        timeout = timeout_s or getattr(self.cfg, "volume_timeout_s", 600.0)

        try:
            return await asyncio.wait_for(
                self._run_sequential_pipeline(tasks, ld_parquet_uri, output_parquet_uri),
                timeout=timeout,
            )
        except asyncio.TimeoutError:
            diagnostic_state = self.get_diagnostic_state()
            logger.exception(f"[AsyncOCRPipeline] Sequential timeout after {timeout}s. Diagnostic: {diagnostic_state}")
            await self._graceful_shutdown()
            raise

    async def _run_sequential_pipeline(
        self,
        tasks: list[ImageTask],
        ld_parquet_uri: str,
        output_parquet_uri: str,
    ) -> dict:
        """Internal sequential pipeline execution (called by run_sequential with timeout)."""
        start_time = time.perf_counter()
        total_pages = len(tasks)
        self._is_running = True

        logger.info(f"[AsyncOCRPipeline] Starting SEQUENTIAL pipeline for {total_pages} pages")

        # Phase 1: Collect all inferred pages in memory
        all_inferred: list[InferredPage | PipelineError] = []

        async def collect_inferred() -> None:
            """Collect all inferred pages instead of passing to CTC decoder."""
            while True:
                try:
                    msg = await asyncio.wait_for(self.q_inferred.get(), timeout=QUEUE_GET_TIMEOUT)
                except asyncio.TimeoutError:
                    logger.error("[collect_inferred] Timeout waiting for inferred page - upstream may be stuck")
                    break
                except asyncio.CancelledError:
                    logger.warning("[collect_inferred] Task cancelled")
                    break

                if isinstance(msg, EndOfStream):
                    break
                all_inferred.append(msg)

        # Queue depth monitor
        async def monitor_queues() -> None:
            while True:
                await asyncio.sleep(5)
                logger.debug(
                    f"[QueueDepth] fetched={self.q_fetched.qsize()}, processed={self.q_processed.qsize()}, "
                    f"inferred={self.q_inferred.qsize()}, collected={len(all_inferred)}"
                )

        monitor_task = asyncio.create_task(monitor_queues(), name="monitor")

        try:
            # Download parquet first (small file, needs full bandwidth)
            logger.info("[Phase 0] Downloading parquet...")
            parquet_task = asyncio.create_task(self._parquet_loader.run(ld_parquet_uri), name="parquet_loader")
            await parquet_task

            if self._parquet_loader.parquet_error:
                logger.error(f"[AsyncOCRPipeline] Parquet loading failed: {self._parquet_loader.parquet_error}")

            # Phase 1: Run prefetch + image processing + GPU inference
            logger.info("[Phase 1] Starting GPU inference phase...")
            self._stage_tasks = [
                asyncio.create_task(self._prefetcher.run(tasks), name="prefetcher"),
                asyncio.create_task(self._image_processor.run(), name="image_processor"),
                asyncio.create_task(self._gpu_inference.run(), name="gpu_inference"),
                asyncio.create_task(collect_inferred(), name="collector"),
            ]

            await asyncio.gather(*self._stage_tasks, return_exceptions=True)
            monitor_task.cancel()  # Stop monitor after Phase 1
            phase1_time = time.perf_counter() - start_time
            logger.debug(f"[Phase 1] GPU inference complete in {phase1_time:.2f}s - collected {len(all_inferred)} pages")

            # Phase 2: CTC decode all collected logits at once (max parallelism)
            logger.info("[Phase 2] Starting CTC decode phase...")
            phase2_start = time.perf_counter()

            vocab = self.ctc_decoder.ctc_vocab
            loop = asyncio.get_event_loop()

            # Prepare all lines from all pages for batch submission
            # Each entry: (page_idx, line_idx, cropped_logits, inferred, keep_indices)
            all_decode_tasks: list[tuple[int, int, npt.NDArray, InferredPage, npt.NDArray | None]] = []

            for item in all_inferred:
                # Handle PipelineError objects - just pass them through
                if isinstance(item, PipelineError):
                    await self.q_results.put(item)
                    self.stats["errors"] += 1
                    continue

                # Handle InferredPage objects
                inferred = item
                if inferred.error or not inferred.logits_list:
                    await self.q_results.put(self._create_error_page_ocr(inferred, None))
                    self.stats["errors"] += 1
                    continue

                for line_idx, line_logits in enumerate(inferred.logits_list):
                    # Logits are already cropped to remove padding (done in model before softmax)
                    # Just need to transpose from (vocab, time) -> (time, vocab) if needed
                    actual_vocab_size = line_logits.vocab_size
                    needs_transpose = line_logits.logits.shape[0] == actual_vocab_size
                    cropped = line_logits.logits.T if needs_transpose else line_logits.logits
                    all_decode_tasks.append((inferred.page_idx, line_idx, cropped, inferred, line_logits.keep_indices))

            # Count how many have GPU pruning
            gpu_pruned_count = sum(1 for _, _, _, _, ki in all_decode_tasks if ki is not None)
            if gpu_pruned_count > 0:
                sample_ki = next((ki for _, _, _, _, ki in all_decode_tasks if ki is not None), None)
                logger.debug(
                    f"[Phase 2] Submitting {len(all_decode_tasks)} lines to {self.cfg.ctc_workers} workers... "
                    f"({gpu_pruned_count} GPU-pruned, sample_keep_indices={len(sample_ki) if sample_ki is not None else 'N/A'})"
                )
            else:
                logger.debug(
                    f"[Phase 2] Submitting {len(all_decode_tasks)} lines to {self.cfg.ctc_workers} workers... (0 GPU-pruned)"
                )

            # Submit ALL lines at once to ProcessPoolExecutor
            futures = []
            for page_idx, line_idx, cropped, inferred, keep_indices in all_decode_tasks:
                submit_time = time.perf_counter()
                pruned_vocab = [vocab[i] for i in keep_indices] if keep_indices is not None else vocab

                # Use decode_logits_with_segments for structured output
                # Get line width in original coordinates (not page width!)
                processed_line = inferred.processed_page.lines[line_idx]
                line_bbox_w = processed_line.bbox[2]  # Width in resized coords
                inv_scale = (
                    1.0 / inferred.processed_page.scale_factor if inferred.processed_page.scale_factor != 0 else 1.0
                )
                original_line_width = int(line_bbox_w * inv_scale)  # Scale to original coords

                future = loop.run_in_executor(
                    self._ctc_executor,
                    _decode_single_line_with_segments,
                    cropped,
                    pruned_vocab,
                    original_line_width,  # Pass line width, not page width!
                    self.cfg.beam_width,
                    self.cfg.token_min_logp,
                    submit_time,
                )
                futures.append((page_idx, line_idx, future, inferred, submit_time))

            # Wait for decodes with progress logging
            decode_start = time.perf_counter()
            total_lines = len(futures)
            completed = 0
            failed_lines = 0

            # Wait for all decodes and log each line as it completes
            pending = {
                f[2]: (f[0], f[1], f[3], f[4]) for f in futures
            }  # future -> (page_idx, line_idx, inferred, submit_time)
            results = [None] * total_lines
            total_ipc_in = 0.0
            total_ipc_out = 0.0

            while pending:
                try:
                    # Add timeout to prevent infinite wait if workers crash
                    done, still_pending = await asyncio.wait(
                        pending.keys(),
                        return_when=asyncio.FIRST_COMPLETED,
                        timeout=CTC_DECODE_TIMEOUT,
                    )
                except asyncio.CancelledError:
                    logger.warning("[CTC] Decode wait cancelled, cancelling remaining futures")
                    for f in pending.keys():
                        f.cancel()
                    raise

                # Handle timeout - no futures completed
                if not done:
                    logger.error(f"[CTC] Timeout waiting for {len(pending)} remaining decode tasks")
                    # Create dummy results for remaining and break
                    for future in list(pending.keys()):
                        page_idx, line_idx, inferred, submit_time = pending.pop(future)
                        idx = next(i for i, f in enumerate(futures) if f[2] is future)
                        # Create empty decode result
                        results[idx] = LineDecodeResult(
                            text="",
                            segments=[],
                            line_confidence=float("-inf"),
                            logit_score=0.0,
                        )
                        failed_lines += 1
                        future.cancel()
                    break

                for future in done:
                    page_idx, line_idx, inferred, submit_time = pending.pop(future)
                    result_time = time.perf_counter()

                    try:
                        line_decode_result, decode_ms, ipc_in_ms, worker_pid = future.result()
                    except BrokenExecutor as e:
                        # Worker process crashed
                        logger.error(f"[CTC] Worker process crashed for page={page_idx} line={line_idx}: {e}")
                        line_decode_result = LineDecodeResult(
                            text="",
                            segments=[],
                            line_confidence=float("-inf"),
                            logit_score=0.0,
                        )
                        decode_ms = 0.0
                        ipc_in_ms = 0.0
                        worker_pid = -1
                        failed_lines += 1
                    except Exception as e:
                        # Other error (e.g., decoding error)
                        logger.warning(f"[CTC] Decode failed for page={page_idx} line={line_idx}: {e}")
                        line_decode_result = LineDecodeResult(
                            text="",
                            segments=[],
                            line_confidence=float("-inf"),
                            logit_score=0.0,
                        )
                        decode_ms = 0.0
                        ipc_in_ms = 0.0
                        worker_pid = -1
                        failed_lines += 1

                    ipc_out_ms = (time.perf_counter() - result_time) * 1000  # Time to deserialize result
                    total_roundtrip_ms = (result_time - submit_time) * 1000
                    total_ipc_in += ipc_in_ms
                    total_ipc_out += ipc_out_ms
                    idx = next(i for i, f in enumerate(futures) if f[2] is future)
                    results[idx] = line_decode_result
                    completed += 1
                    logger.debug(
                        f"[CTC] line {completed}/{total_lines} page={page_idx} line={line_idx} worker={worker_pid} "
                        f"decode={decode_ms:.1f}ms ipc_in={ipc_in_ms:.1f}ms roundtrip={total_roundtrip_ms:.1f}ms"
                    )

            if failed_lines > 0:
                logger.warning(f"[CTC] {failed_lines} lines failed to decode")

            all_decode_results = results
            decode_time = time.perf_counter() - decode_start
            num_results = len(all_decode_results) if all_decode_results else 0
            avg_ipc_in = total_ipc_in / num_results if num_results > 0 else 0
            avg_ipc_out = total_ipc_out / num_results if num_results > 0 else 0
            ms_per_line = decode_time * 1000 / num_results if num_results > 0 else 0
            logger.debug(
                f"[Phase 2] All {num_results} lines decoded in {decode_time:.2f}s "
                f"({ms_per_line:.1f}ms/line, avg_ipc_in={avg_ipc_in:.1f}ms, avg_ipc_out={avg_ipc_out:.1f}ms)"
            )

            # Group results by page
            page_results: dict[int, tuple[InferredPage, list[LineDecodeResult]]] = {}
            for (page_idx, _, _, inferred, _), decode_result in zip(futures, all_decode_results, strict=True):
                if page_idx not in page_results:
                    page_results[page_idx] = (inferred, [])
                page_results[page_idx][1].append(decode_result)

            # Build PageOCRResult objects for each page
            results_to_write = []
            for page_idx in sorted(page_results.keys()):
                inferred, line_decode_results = page_results[page_idx]

                # Build PageOCRResult using the structured approach
                page_result = _build_page_ocr_result(inferred.processed_page, line_decode_results)
                results_to_write.append(page_result)
                self.stats["decoded"] += 1

            # Emit results and write to parquet concurrently
            async def emit_results() -> None:
                for result in results_to_write:
                    try:
                        await asyncio.wait_for(self.q_results.put(result), timeout=60.0)
                    except asyncio.TimeoutError:
                        logger.error(f"[emit_results] Timeout putting result for {result.img_file_name}")
                    except asyncio.CancelledError:
                        logger.warning("[emit_results] Task cancelled")
                        raise
                try:
                    await asyncio.wait_for(
                        self.q_results.put(EndOfStream(stream="results", producer="run_sequential")),
                        timeout=60.0,
                    )
                except (asyncio.TimeoutError, asyncio.CancelledError):
                    logger.error("[emit_results] Failed to send EOS")

            # Create output writer stage for this run
            expected_filenames = [t.filename for t in tasks]
            output_writer = OutputWriterStage(
                parquet_uri=output_parquet_uri,
                q_in=self.q_results,
                expected_filenames=expected_filenames,
                stats=self.stats,
            )

            await asyncio.gather(
                emit_results(),
                output_writer.run(),
            )

            phase2_time = time.perf_counter() - phase2_start
            logger.info(
                f"[Phase 2] CTC decode + write complete in {phase2_time:.2f}s - decoded {self.stats['decoded']} pages"
            )

            elapsed = time.perf_counter() - start_time
            logger.info(
                f"[AsyncOCRPipeline] SEQUENTIAL completed in {elapsed:.2f}s - "
                f"fetched={self.stats['fetched']}, processed={self.stats['processed']}, "
                f"inferred={self.stats['inferred']}, decoded={self.stats['decoded']}, "
                f"errors={self.stats['errors']}"
            )

            return self.stats

        finally:
            self._is_running = False
            self._stage_tasks = []

    # =========================================================================
    # Helper methods (used by run_sequential Phase 2)
    # =========================================================================

    def _create_error_page(self, inferred: InferredPage, error: str | None) -> PageResult:
        """Create a PageResult for error cases."""
        return PageResult(
            task=inferred.task,
            source_etag=inferred.source_etag,
            texts=[],
            error=error or inferred.error,
        )

    def _create_error_page_ocr(self, inferred: InferredPage, error: str | None) -> PageOCRResult:
        """Create a PageOCRResult for error cases."""
        return PageOCRResult(
            img_file_name=inferred.filename,
            source_etag=inferred.source_etag,
            rotation_angle=inferred.processed_page.rotation_angle,
            tps_points=inferred.processed_page.tps_points,
            lines=[],
            error=error or inferred.error,
        )

    def _create_inferred_page(self, page: ProcessedPage, logits_list: list) -> InferredPage:
        """Create InferredPage from ProcessedPage and logits."""
        return InferredPage(
            task=page.task,
            source_etag=page.source_etag,
            logits_list=logits_list,
            processed_page=page,
            error=page.error,
        )

    def _prepare_logits_and_vocab(self, inferred: InferredPage) -> tuple[list[npt.NDArray], list[list[str]]]:
        """Prepare cropped logits and corresponding vocabularies for decoding."""
        vocab = self.ctc_decoder.ctc_vocab

        cropped_logits_list = []
        pruned_vocab_list = []

        for line_logits in inferred.logits_list:
            actual_vocab_size = line_logits.vocab_size
            needs_transpose = line_logits.logits.shape[0] == actual_vocab_size
            cropped = line_logits.logits.T if needs_transpose else line_logits.logits

            pruned_vocab = (
                [vocab[i] for i in line_logits.keep_indices] if line_logits.keep_indices is not None else vocab
            )
            pruned_vocab_list.append(pruned_vocab)

            cropped_logits_list.append(cropped)

        return cropped_logits_list, pruned_vocab_list

    async def close(self) -> None:
        """Cleanup resources."""
        self._image_executor.shutdown(wait=False)
        self._ctc_executor.shutdown(wait=False)
