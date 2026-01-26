"""
Async OCR pipeline with backpressure.

Pipeline stages:
1. S3 Prefetcher (async, high concurrency) → FetchedBytes
2. Image Processor (thread pool) → LineTensor batches
3. GPU Forward (single async task, batched) → Logits
4. CTC Decoder (thread pool) → PageResult
5. Parquet Writer (single async task) → S3

All stages connected by bounded asyncio.Queue for backpressure.
"""

import asyncio
import logging
import os
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt
from botocore.client import BaseClient

from .data_structures import (
    LineLogits,
    LineResult,
    PageInFlight,
    PageOCRResult,
    PageResult,
    PendingTensor,
    SegmentResult,
)
from .line import BBox, build_line_data, sort_lines_by_threshold

if TYPE_CHECKING:
    from .config import OCRV1Config

from .ctc_decoder import (
    CTCDecoder,
    LineDecodeResult,
    decode_logits_beam_search,
    decode_logits_with_segments,
    init_worker_process,
)
from .line_decoder import FetchedBytes, LineDecoder, ProcessedPage
from .model import OCRModel
from .output_writer import OutputWriter

logger = logging.getLogger(__name__)

# Default vocabulary size for fallback logits
DEFAULT_VOCAB_SIZE = 84

# =============================================================================
# OCR Result Building Utilities
# =============================================================================


def _build_page_ocr_result(
    processed_page: ProcessedPage,
    line_decode_results: list[LineDecodeResult],
) -> PageOCRResult:
    """Build PageOCRResult from processed page and line decode results."""
    if processed_page.error:
        return PageOCRResult(
            img_file_name=processed_page.filename,
            source_etag=processed_page.source_etag,
            rotation_angle=processed_page.rotation_angle,
            tps_points=processed_page.tps_points,
            lines=[],
            error=processed_page.error,
        )

    # Group line segments into logical lines
    line_objects = [
        build_line_data(processed_line.contours[0] if processed_line.contours else [])
        for i, processed_line in enumerate(processed_page.lines)
        if i < len(line_decode_results)
    ]

    # Sort and group lines
    grouped_lines = sort_lines_by_threshold(line_objects)

    # Build LineResult objects
    line_results = []
    for line_idx, line_group in enumerate(grouped_lines):
        # Combine all segments in this logical line
        segments = []
        line_text_parts = []
        total_weighted_confidence = 0.0
        total_chars = 0

        for line_obj in line_group:
            # Find the corresponding ProcessedLine and decode result
            processed_line_idx = next(
                (i for i, pl in enumerate(processed_page.lines) if pl.contours and line_obj.center == pl.bbox[:2]), 0
            )

            if processed_line_idx < len(line_decode_results):
                decode_result = line_decode_results[processed_line_idx]
                processed_line = processed_page.lines[processed_line_idx]

                # Create SegmentResult
                x, y, w, h = processed_line.bbox
                segment_result = SegmentResult(
                    segment_idx=len(segments),
                    bbox=BBox(x=x, y=y, w=w, h=h),
                    text=decode_result.text,
                    confidence=decode_result.line_confidence,
                    syllables=decode_result.segments,
                )
                segments.append(segment_result)
                line_text_parts.append(decode_result.text)

                # Weight confidence by character count
                char_count = len(decode_result.text)
                total_weighted_confidence += decode_result.line_confidence * char_count
                total_chars += char_count

        # Create LineResult
        line_confidence = total_weighted_confidence / total_chars if total_chars > 0 else 0.0
        line_result = LineResult(
            line_idx=line_idx,
            text=" ".join(line_text_parts),
            confidence=line_confidence,
            segments=segments,
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


@dataclass
class InferredPage:
    page_idx: int
    filename: str
    source_etag: str
    logits_list: list[LineLogits]  # Structured line logits data
    processed_page: ProcessedPage  # Original processed page data for line grouping
    error: str | None = None


class EndOfStream:
    pass


EOS = EndOfStream()


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
        self.line_decoder = LineDecoder(cfg)

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

        logger.info(
            f"Pipeline config: prefetch={cfg.prefetch_concurrency}, "
            f"img_workers={cfg.image_processor_workers}, "
            f"ctc_workers={cfg.ctc_workers}, gpu_batch={cfg.gpu_batch_size}"
        )

        # Semaphore for S3 concurrency
        self._s3_sem = asyncio.Semaphore(cfg.prefetch_concurrency)

        # Stats
        self.stats = {
            "fetched": 0,
            "processed": 0,
            "inferred": 0,
            "decoded": 0,
            "errors": 0,
        }

    async def run(
        self,
        pages: list[tuple[int, str, dict]],  # (page_idx, filename, ld_row)
        output_parquet_uri: str,
    ) -> dict:
        """Run the full pipeline."""
        start_time = time.perf_counter()
        total_pages = len(pages)

        logger.info(f"[AsyncOCRPipeline] Starting pipeline for {total_pages} pages")

        # Queue depth monitor task
        async def monitor_queues() -> None:
            while True:
                await asyncio.sleep(5)
                logger.info(
                    f"[QueueDepth] fetched={self.q_fetched.qsize()}, processed={self.q_processed.qsize()}, "
                    f"inferred={self.q_inferred.qsize()}, results={self.q_results.qsize()}"
                )

        monitor_task = asyncio.create_task(monitor_queues(), name="monitor")

        # Start all stages as concurrent tasks
        tasks = [
            asyncio.create_task(self._prefetcher(pages), name="prefetcher"),
            asyncio.create_task(self._image_processor(), name="image_processor"),
            asyncio.create_task(self._gpu_inference(), name="gpu_inference"),
            asyncio.create_task(self._ctc_decoder_stage(), name="ctc_decoder"),
            asyncio.create_task(self._output_writer(output_parquet_uri, total_pages), name="writer"),
        ]

        # Wait for all tasks
        results = await asyncio.gather(*tasks, return_exceptions=True)
        monitor_task.cancel()

        # Check for errors
        stage_names = ["prefetcher", "image_processor", "gpu_inference", "ctc_decoder", "writer"]
        for name, result in zip(stage_names, results, strict=True):
            if isinstance(result, Exception):
                logger.error(f"Pipeline stage '{name}' failed: {result}", exc_info=result)

        elapsed = time.perf_counter() - start_time
        logger.info(
            f"[AsyncOCRPipeline] Completed in {elapsed:.2f}s - "
            f"fetched={self.stats['fetched']}, processed={self.stats['processed']}, "
            f"inferred={self.stats['inferred']}, decoded={self.stats['decoded']}, "
            f"errors={self.stats['errors']}"
        )

        return self.stats

    async def run_sequential(
        self,
        pages: list[tuple[int, str, dict]],  # (page_idx, filename, ld_row)
        output_parquet_uri: str,
    ) -> dict:
        """
        Run pipeline in two phases to eliminate CPU contention:
        Phase 1: Prefetch + Image Processing + GPU Inference (collect all logits in memory)
        Phase 2: CTC Decode all collected logits + Write to Parquet
        """
        start_time = time.perf_counter()
        total_pages = len(pages)

        logger.info(f"[AsyncOCRPipeline] Starting SEQUENTIAL pipeline for {total_pages} pages")

        # Phase 1: Collect all inferred pages in memory
        all_inferred: list[InferredPage] = []

        async def collect_inferred() -> None:
            """Collect all inferred pages instead of passing to CTC decoder."""
            while True:
                msg = await self.q_inferred.get()
                if isinstance(msg, EndOfStream):
                    break
                all_inferred.append(msg)

        # Queue depth monitor
        async def monitor_queues() -> None:
            while True:
                await asyncio.sleep(5)
                logger.info(
                    f"[QueueDepth] fetched={self.q_fetched.qsize()}, processed={self.q_processed.qsize()}, "
                    f"inferred={self.q_inferred.qsize()}, collected={len(all_inferred)}"
                )

        monitor_task = asyncio.create_task(monitor_queues(), name="monitor")

        # Phase 1: Run prefetch + image processing + GPU inference
        logger.info("[Phase 1] Starting GPU inference phase...")
        phase1_tasks = [
            asyncio.create_task(self._prefetcher(pages), name="prefetcher"),
            asyncio.create_task(self._image_processor(), name="image_processor"),
            asyncio.create_task(self._gpu_inference(), name="gpu_inference"),
            asyncio.create_task(collect_inferred(), name="collector"),
        ]

        await asyncio.gather(*phase1_tasks, return_exceptions=True)
        monitor_task.cancel()  # Stop monitor after Phase 1
        phase1_time = time.perf_counter() - start_time
        logger.info(f"[Phase 1] GPU inference complete in {phase1_time:.2f}s - collected {len(all_inferred)} pages")

        # Phase 2: CTC decode all collected logits at once (max parallelism)
        logger.info("[Phase 2] Starting CTC decode phase...")
        phase2_start = time.perf_counter()

        vocab = self.ctc_decoder.ctc_vocab
        loop = asyncio.get_event_loop()

        # Prepare all lines from all pages for batch submission
        # Each entry: (page_idx, line_idx, cropped_logits, inferred, keep_indices)
        all_decode_tasks: list[tuple[int, int, npt.NDArray, InferredPage, npt.NDArray | None]] = []

        for inferred in all_inferred:
            if inferred.error or not inferred.logits_list:
                await self.q_results.put(self._create_error_page(inferred, None))
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
            logger.info(
                f"[Phase 2] Submitting {len(all_decode_tasks)} lines to {self.cfg.ctc_workers} workers... "
                f"({gpu_pruned_count} GPU-pruned, sample_keep_indices={len(sample_ki) if sample_ki is not None else 'N/A'})"
            )
        else:
            logger.info(
                f"[Phase 2] Submitting {len(all_decode_tasks)} lines to {self.cfg.ctc_workers} workers... (0 GPU-pruned)"
            )

        # Submit ALL lines at once to ProcessPoolExecutor
        futures = []
        for page_idx, line_idx, cropped, inferred, keep_indices in all_decode_tasks:
            submit_time = time.perf_counter()
            pruned_vocab = [vocab[i] for i in keep_indices] if keep_indices is not None else vocab

            # Use decode_logits_with_segments for structured output
            original_width = inferred.processed_page.orig_width
            future = loop.run_in_executor(
                self._ctc_executor,
                _decode_single_line_with_segments,
                cropped,
                pruned_vocab,
                original_width,
                self.cfg.beam_width,
                self.cfg.token_min_logp,
                submit_time,
            )
            futures.append((page_idx, line_idx, future, inferred, submit_time))

        # Wait for decodes with progress logging
        decode_start = time.perf_counter()
        total_lines = len(futures)
        completed = 0

        # Wait for all decodes and log each line as it completes
        pending = {
            f[2]: (f[0], f[1], f[3], f[4]) for f in futures
        }  # future -> (page_idx, line_idx, inferred, submit_time)
        results = [None] * total_lines
        total_ipc_in = 0.0
        total_ipc_out = 0.0

        while pending:
            done, _ = await asyncio.wait(pending.keys(), return_when=asyncio.FIRST_COMPLETED)
            for future in done:
                page_idx, line_idx, inferred, submit_time = pending.pop(future)
                result_time = time.perf_counter()
                line_decode_result, decode_ms, ipc_in_ms, worker_pid = future.result()
                ipc_out_ms = (time.perf_counter() - result_time) * 1000  # Time to deserialize result
                total_roundtrip_ms = (result_time - submit_time) * 1000
                total_ipc_in += ipc_in_ms
                total_ipc_out += ipc_out_ms
                idx = next(i for i, f in enumerate(futures) if f[2] is future)
                results[idx] = line_decode_result
                completed += 1
                logger.info(
                    f"[CTC] line {completed}/{total_lines} page={page_idx} line={line_idx} worker={worker_pid} "
                    f"decode={decode_ms:.1f}ms ipc_in={ipc_in_ms:.1f}ms roundtrip={total_roundtrip_ms:.1f}ms"
                )

        all_decode_results = results
        decode_time = time.perf_counter() - decode_start
        num_results = len(all_decode_results) if all_decode_results else 0
        avg_ipc_in = total_ipc_in / num_results if num_results > 0 else 0
        avg_ipc_out = total_ipc_out / num_results if num_results > 0 else 0
        ms_per_line = decode_time * 1000 / num_results if num_results > 0 else 0
        logger.info(
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
                await self.q_results.put(result)
            await self.q_results.put(EOS)

        await asyncio.gather(
            emit_results(),
            self._output_writer(output_parquet_uri, total_pages),
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

    async def _prefetcher(self, pages: list[tuple[int, str, dict]]) -> None:
        """Fetch images from S3 with high concurrency."""
        logger.info(f"[Prefetcher] Starting with {self.cfg.prefetch_concurrency} concurrency")

        async def fetch_one(page_idx: int, filename: str, ld_row: dict) -> None:
            async with self._s3_sem:
                try:
                    key = f"{self.volume_prefix}/{filename}"
                    loop = asyncio.get_event_loop()

                    def _fetch() -> tuple[str, bytes]:
                        response = self.s3_client.get_object(Bucket=self.source_image_bucket, Key=key)
                        return response.get("ETag", "").strip('"'), response["Body"].read()

                    source_etag, file_bytes = await loop.run_in_executor(None, _fetch)

                    await self.q_fetched.put(
                        self._create_fetched_bytes(page_idx, filename, ld_row, file_bytes, source_etag)
                    )
                    self.stats["fetched"] += 1

                    if self.stats["fetched"] % 20 == 0:
                        logger.info(f"[Prefetcher] Fetched {self.stats['fetched']} pages")

                except Exception as e:
                    logger.warning(f"[Prefetcher] Failed to fetch {filename}: {e}")
                    await self.q_fetched.put(self._create_fetched_bytes(page_idx, filename, ld_row))
                    self.stats["errors"] += 1

        # Create all fetch tasks
        fetch_tasks = [
            asyncio.create_task(fetch_one(page_idx, filename, ld_row)) for page_idx, filename, ld_row in pages
        ]

        # Wait for all fetches
        await asyncio.gather(*fetch_tasks)

        # Signal end of stream
        await self.q_fetched.put(EOS)
        logger.info(f"[Prefetcher] Done, fetched {self.stats['fetched']} pages")

    async def _image_processor(self) -> None:
        """Process images concurrently using semaphore for backpressure."""
        logger.info(f"[ImageProcessor] Starting with {self.cfg.image_processor_workers} concurrent workers")
        loop = asyncio.get_event_loop()

        # Semaphore limits concurrent processing
        sem = asyncio.Semaphore(self.cfg.image_processor_workers)
        pending_tasks: set[asyncio.Task] = set()

        async def process_one(fetched: FetchedBytes) -> None:
            async with sem:
                if not fetched.file_bytes:
                    await self.q_processed.put(self._create_processed_page(fetched, "Failed to fetch image"))
                    return

                try:
                    processed = await loop.run_in_executor(
                        self._image_executor,
                        self.line_decoder.process,
                        fetched,
                    )
                    await self.q_processed.put(processed)
                    self.stats["processed"] += 1

                    if processed.error:
                        self.stats["errors"] += 1

                except Exception as e:
                    logger.warning(f"[ImageProcessor] Failed to process {fetched.filename}: {e}")
                    await self.q_processed.put(self._create_processed_page(fetched, str(e)))
                    self.stats["errors"] += 1

        while True:
            msg = await self.q_fetched.get()
            if isinstance(msg, EndOfStream):
                # Wait for all pending tasks
                if pending_tasks:
                    await asyncio.gather(*pending_tasks, return_exceptions=True)
                await self.q_processed.put(EOS)
                break

            fetched: FetchedBytes = msg
            task = asyncio.create_task(process_one(fetched))
            pending_tasks.add(task)
            task.add_done_callback(pending_tasks.discard)

        logger.info(f"[ImageProcessor] Done, processed {self.stats['processed']} pages")

    async def _gpu_inference(self) -> None:
        """Run GPU inference with batching - emit pages as soon as all lines are done."""
        logger.info(f"[GPUInference] Starting with batch_size={self.cfg.gpu_batch_size}")

        # Track pages waiting for their lines to be processed
        pages_in_flight: dict[int, PageInFlight] = {}

        # Batch of tensors waiting to be processed
        pending_tensors: list[PendingTensor] = []

        lines_processed = 0
        pages_emitted = 0
        pages_received = 0

        async def flush_batch() -> None:
            nonlocal lines_processed
            if not pending_tensors:
                return

            batch_size = len(pending_tensors)
            start_time = time.perf_counter()

            # Stack tensors and collect content widths and left pad widths
            tensors = np.concatenate([pt.tensor for pt in pending_tensors], axis=0)
            content_widths = [pt.content_width for pt in pending_tensors]
            left_pad_widths = [pt.left_pad_width for pt in pending_tensors]

            # Run inference with content widths and left pad widths for proper cropping
            # Model crops time dimension BEFORE softmax/pruning to save computation
            # Returns LineLogits objects for each line
            loop = asyncio.get_event_loop()
            line_logits_list = await loop.run_in_executor(
                None, lambda: self.ocr_model.predict(tensors, content_widths, left_pad_widths, self.cfg.input_width)
            )

            # line_logits_list is now a list of LineLogits objects (one per item)
            # Distribute results back to pages
            for pending_tensor, line_logits in zip(pending_tensors, line_logits_list, strict=True):
                if pending_tensor.page_idx in pages_in_flight:
                    page_flight = pages_in_flight[pending_tensor.page_idx]
                    # Store LineLogits object
                    page_flight.line_logits[pending_tensor.line_idx] = line_logits
                lines_processed += 1

            pending_tensors.clear()

            batch_ms = (time.perf_counter() - start_time) * 1000
            logger.info(
                f"[GPUInference] Batch {batch_size} lines in {batch_ms:.0f}ms ({batch_ms / batch_size:.0f}ms/line). Total: {lines_processed} lines, {pages_emitted} pages emitted"
            )

        async def try_emit_completed_pages() -> None:
            nonlocal pages_emitted
            # Check if any pages have all their lines done
            completed = []
            for page_idx, page_flight in pages_in_flight.items():
                if len(page_flight.line_logits) >= page_flight.expected_lines:
                    completed.append(page_idx)

            for page_idx in completed:
                page_flight = pages_in_flight.pop(page_idx)

                # Build logits list in order
                logits_list = [
                    page_flight.line_logits[i]
                    if i in page_flight.line_logits
                    else LineLogits(
                        logits=np.zeros((1, DEFAULT_VOCAB_SIZE), dtype=np.float32),
                        content_width=1,
                        left_pad_width=0,
                        keep_indices=None,
                    )
                    for i in range(page_flight.expected_lines)
                ]
                await self.q_inferred.put(self._create_inferred_page(page_flight.processed_page, logits_list))
                pages_emitted += 1
                self.stats["inferred"] += 1

        while True:
            msg = await self.q_processed.get()
            if isinstance(msg, EndOfStream):
                logger.info(f"[GPUInference] Received EOS, flushing {len(pending_tensors)} pending tensors")
                # Flush remaining batch
                await flush_batch()
                await try_emit_completed_pages()

                # Emit any remaining pages (shouldn't happen normally)
                if pages_in_flight:
                    logger.info(f"[GPUInference] Emitting {len(pages_in_flight)} remaining pages")
                for page_flight in pages_in_flight.values():
                    logits_list = [
                        page_flight.line_logits[i]
                        if i in page_flight.line_logits
                        else LineLogits(
                            logits=np.zeros((1, DEFAULT_VOCAB_SIZE), dtype=np.float32),
                            content_width=1,
                            left_pad_width=0,
                            keep_indices=None,
                        )
                        for i in range(page_flight.expected_lines)
                    ]
                    await self.q_inferred.put(self._create_inferred_page(page_flight.processed_page, logits_list))
                    self.stats["inferred"] += 1

                await self.q_inferred.put(EOS)
                break

            processed: ProcessedPage = msg
            pages_received += 1

            if pages_received % 20 == 0:
                logger.info(
                    f"[GPUInference] Received {pages_received} pages, {len(pending_tensors)} tensors pending, {len(pages_in_flight)} in flight"
                )

            # Handle error/empty pages immediately
            if processed.error or not processed.lines:
                await self.q_inferred.put(self._create_inferred_page(processed, []))
                pages_emitted += 1
                self.stats["inferred"] += 1
                continue

            # Register page
            pages_in_flight[processed.page_idx] = PageInFlight(
                processed_page=processed, expected_lines=len(processed.lines), line_logits={}
            )

            # Add tensors to batch
            for line_idx, line in enumerate(processed.lines):
                pending_tensors.append(
                    PendingTensor(
                        page_idx=processed.page_idx,
                        line_idx=line_idx,
                        tensor=line.tensor,
                        content_width=line.content_width,
                        left_pad_width=line.left_pad_width,
                    )
                )

                if len(pending_tensors) >= self.cfg.gpu_batch_size:
                    await flush_batch()
                    await try_emit_completed_pages()

        logger.info(f"[GPUInference] Done, processed {lines_processed} lines, emitted {pages_emitted} pages")

    async def _ctc_decoder_stage(self) -> None:
        """Decode logits to text concurrently using ProcessPoolExecutor."""
        logger.info(f"[CTCDecoder] Starting with {self.cfg.ctc_workers} workers")
        pending_tasks: set[asyncio.Task] = set()
        pages_received = 0

        # Limit concurrent page decodes to balance parallelism vs resource contention
        # With N workers and ~6 lines/page, allow N/2 pages to avoid memory pressure
        max_concurrent_pages = max(2, self.cfg.ctc_workers // 2)
        page_sem = asyncio.Semaphore(max_concurrent_pages)
        logger.info(f"[CTCDecoder] Max concurrent page decodes: {max_concurrent_pages}")

        async def decode_one(inferred: InferredPage) -> None:
            if inferred.error or not inferred.logits_list:
                await self.q_results.put(self._create_error_page_ocr(inferred, None))
                return

            async with page_sem:
                try:
                    start_time = time.perf_counter()
                    num_lines = len(inferred.logits_list)

                    # Logits are already cropped to remove padding (done in model before softmax)
                    # Just need to transpose from (vocab, time) -> (time, vocab) if needed
                    # NOTE: Each line may have different keep_indices if they were in different GPU batches
                    cropped_logits_list, pruned_vocab_list = self._prepare_logits_and_vocab(inferred)

                    # Use decode_logits_with_segments for structured output
                    line_decode_results = []
                    for cropped, vocab in zip(cropped_logits_list, pruned_vocab_list, strict=True):
                        # Get original width for this line from the processed page
                        original_width = inferred.processed_page.orig_width
                        decode_result = decode_logits_with_segments(
                            cropped,
                            vocab,
                            original_width,
                            beam_width=self.cfg.beam_width,
                            token_min_logp=self.cfg.token_min_logp,
                        )
                        line_decode_results.append(decode_result)

                    decode_ms = (time.perf_counter() - start_time) * 1000
                    logger.info(
                        f"[CTCDecoder] Page {inferred.page_idx} decoded {num_lines} lines in {decode_ms:.0f}ms ({decode_ms / max(1, num_lines):.0f}ms/line)"
                    )

                    # Build PageOCRResult using the structured approach
                    page_result = _build_page_ocr_result(inferred.processed_page, line_decode_results)
                    await self.q_results.put(page_result)
                    self.stats["decoded"] += 1

                except Exception as e:
                    logger.warning(f"[CTCDecoder] Failed to decode {inferred.filename}: {e}")
                    await self.q_results.put(self._create_error_page_ocr(inferred, str(e)))
                    self.stats["errors"] += 1

        while True:
            msg = await self.q_inferred.get()
            if isinstance(msg, EndOfStream):
                if pending_tasks:
                    await asyncio.gather(*pending_tasks, return_exceptions=True)
                await self.q_results.put(EOS)
                break

            inferred: InferredPage = msg
            pages_received += 1

            if pages_received % 20 == 0:
                logger.info(f"[CTCDecoder] Received {pages_received} pages")

            task = asyncio.create_task(decode_one(inferred))
            pending_tasks.add(task)
            task.add_done_callback(pending_tasks.discard)

        logger.info(f"[CTCDecoder] Done, decoded {self.stats['decoded']} pages")

    async def _output_writer(self, output_prefix: str, total_pages: int) -> None:
        """Write results to both Parquet and JSONL.gz using OutputWriter."""
        logger.info(f"[OutputWriter] Starting, expecting {total_pages} pages")

        # Extract volume_id from output_prefix (assuming format like s3://bucket/path/volume_id/)
        volume_id = output_prefix.split("/")[-2] if "/" in output_prefix else "unknown"

        writer = OutputWriter(volume_id, output_prefix)
        pages_written = 0

        try:
            while True:
                msg = await self.q_results.get()
                if isinstance(msg, EndOfStream):
                    break

                result: PageOCRResult = msg
                writer.write_page(result)
                pages_written += 1

                if pages_written % 100 == 0:
                    logger.info(f"[OutputWriter] Progress: {pages_written}/{total_pages}")
        finally:
            writer.close()

        logger.info(f"[OutputWriter] Done, wrote {pages_written} pages")

    def _create_error_page(self, inferred: InferredPage, error: str | None) -> PageResult:
        """Create a PageResult for error cases."""
        return PageResult(
            page_idx=inferred.page_idx,
            filename=inferred.filename,
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

    def _create_processed_page(self, fetched: FetchedBytes, error: str | None) -> ProcessedPage:
        """Create ProcessedPage with error handling."""
        return ProcessedPage(
            page_idx=fetched.page_idx,
            filename=fetched.filename,
            source_etag=fetched.source_etag,
            lines=[],
            orig_width=0,
            orig_height=0,
            transformed_width=0,
            transformed_height=0,
            error=error,
        )

    def _create_inferred_page(self, page: ProcessedPage, logits_list: list) -> InferredPage:
        """Create InferredPage from ProcessedPage and logits."""
        return InferredPage(
            page_idx=page.page_idx,
            filename=page.filename,
            source_etag=page.source_etag,
            logits_list=logits_list,
            processed_page=page,
            error=page.error,
        )

    def _create_fetched_bytes(
        self,
        page_idx: int,
        filename: str,
        ld_row: dict,
        file_bytes: bytes = b"",
        source_etag: str = "",
    ) -> FetchedBytes:
        """Create FetchedBytes with error handling."""
        return FetchedBytes(
            page_idx=page_idx,
            filename=filename,
            source_etag=source_etag,
            file_bytes=file_bytes,
            ld_row=ld_row,
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
