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
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import cv2
import numpy as np
import numpy.typing as npt

if TYPE_CHECKING:
    pass

from ..ldv1.img_helpers import adaptive_binarize, apply_transform_1
from ..shared.decoder import bytes_to_frame
from .ctc_decoder import (
    CTCDecoder,
    decode_logits_beam_search,
    decode_logits_greedy,
    decode_logits_hybrid_global,
    init_worker_process,
)
from .line import get_line_image
from .model import OCRModel
from .parquet_writer import StreamingParquetWriter

logger = logging.getLogger(__name__)


# =============================================================================
# Line Segment Merging Utilities
# =============================================================================

@dataclass
class LineSegment:
    """Represents a line segment with contour and bounding box info."""
    contour: npt.NDArray  # numpy contour array
    bbox: tuple[int, int, int, int]  # (x, y, w, h)
    center: tuple[int, int]  # (x_center, y_center)


def _build_line_segment(contour: npt.NDArray) -> LineSegment:
    """Create a LineSegment from a contour."""
    # Ensure contour is in standard OpenCV format (N, 1, 2)
    if contour.ndim == 2:
        contour = contour.reshape(-1, 1, 2)
    x, y, w, h = cv2.boundingRect(contour)
    x_center = x + (w // 2)
    y_center = y + (h // 2)
    return LineSegment(contour=contour, bbox=(x, y, w, h), center=(x_center, y_center))


def _sort_bbox_centers(
    bbox_centers: list[tuple[int, int]], line_threshold: float = 20
) -> list[list[tuple[int, int]]]:
    """
    Sort bounding box centers into horizontal lines based on y-position.
    
    Args:
        bbox_centers: List of (x, y) center coordinates
        line_threshold: Vertical distance threshold for grouping
        
    Returns:
        List of lists, each containing centers on the same line
    """
    if not bbox_centers:
        return []

    sorted_bbox_centers = []
    tmp_line = []

    for i in range(len(bbox_centers)):
        if len(tmp_line) > 0:
            # Use mean y of current line group for comparison
            ys = [y[1] for y in tmp_line]
            mean_y = np.mean(ys)
            y_diff = abs(mean_y - bbox_centers[i][1])

            if y_diff > line_threshold:
                tmp_line.sort(key=lambda x: x[0])
                sorted_bbox_centers.append(tmp_line.copy())
                tmp_line.clear()

            tmp_line.append(bbox_centers[i])
        else:
            tmp_line.append(bbox_centers[i])

    # Add the last tmp_line if not empty
    if tmp_line:
        sorted_bbox_centers.append(tmp_line)

    # Sort each line by x-coordinate
    for line in sorted_bbox_centers:
        line.sort(key=lambda x: x[0])

    sorted_bbox_centers.reverse()
    return sorted_bbox_centers


def _group_line_segments(
    sorted_bbox_centers: list[list[tuple[int, int]]], segments: list[LineSegment]
) -> list[LineSegment]:
    """
    Group line segments that belong to the same horizontal line into unified segments.
    
    Args:
        sorted_bbox_centers: Sorted bounding box centers by lines
        segments: Original LineSegment objects
        
    Returns:
        List of merged LineSegment objects
    """
    new_segments = []
    
    for bbox_centers in sorted_bbox_centers:
        if len(bbox_centers) > 1:
            # Multiple segments on same line - merge them
            contour_stack = []
            
            for box_center in bbox_centers:
                for segment in segments:
                    if box_center == segment.center:
                        contour_stack.append(segment.contour)
                        break
            
            if contour_stack:
                # Stack and create convex hull
                stacked_contour = np.vstack(contour_stack)
                stacked_contour = cv2.convexHull(stacked_contour)
                new_segments.append(_build_line_segment(stacked_contour))
        else:
            # Single segment - keep as is
            for bcenter in bbox_centers:
                for segment in segments:
                    if bcenter == segment.center:
                        new_segments.append(segment)
                        break
    
    return new_segments


def _estimate_line_threshold(segments: list[LineSegment]) -> float:
    """
    Estimate a reasonable line threshold based on segment heights.
    
    Uses median height of segments as a baseline for grouping threshold.
    """
    if not segments:
        return 20.0
    
    heights = [seg.bbox[3] for seg in segments]  # bbox[3] is height
    median_height = float(np.median(heights))
    # Use half the median height as threshold
    return max(10.0, median_height * 0.5)


def _merge_line_segments(
    contours: list[list[dict]], line_threshold: float | None = None
) -> list[list[dict]]:
    """
    Merge line segment contours that belong to the same horizontal line.
    
    Args:
        contours: List of contours, where each contour is a list of {"x": x, "y": y} dicts
        line_threshold: Vertical threshold for grouping (auto-calculated if None)
        
    Returns:
        List of merged contours in the same format
    """
    if not contours:
        return []
    
    # Convert dict format to numpy contours and build LineSegments
    segments = []
    for contour_points in contours:
        if not contour_points:
            continue
        pts = np.array([[p["x"], p["y"]] for p in contour_points], dtype=np.int32)
        segments.append(_build_line_segment(pts))
    
    if not segments:
        return []
    
    # Calculate threshold if not provided
    if line_threshold is None:
        line_threshold = _estimate_line_threshold(segments)
    
    # Get centers and sort into lines
    bbox_centers = [seg.center for seg in segments]
    
    # Sort by y-coordinate first (top to bottom)
    sorted_indices = sorted(range(len(bbox_centers)), key=lambda i: bbox_centers[i][1])
    bbox_centers_sorted = [bbox_centers[i] for i in sorted_indices]
    
    sorted_bbox_centers = _sort_bbox_centers(bbox_centers_sorted, line_threshold=line_threshold)
    
    # Group segments
    merged_segments = _group_line_segments(sorted_bbox_centers, segments)
    
    # Convert back to dict format
    merged_contours = []
    for segment in merged_segments:
        contour_dicts = [{"x": int(pt[0][0]), "y": int(pt[0][1])} for pt in segment.contour]
        merged_contours.append(contour_dicts)
    
    return merged_contours


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
    import os

    arrival_time = time.perf_counter()
    ipc_in_ms = (arrival_time - submit_time) * 1000 if submit_time else 0.0

    start = time.perf_counter()
    text = decode_logits_beam_search(cropped_logits, vocab, beam_width, token_min_logp)
    elapsed_ms = (time.perf_counter() - start) * 1000
    return text.strip().replace("§", " "), elapsed_ms, ipc_in_ms, os.getpid()


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


@dataclass
class FetchedBytes:
    page_idx: int
    filename: str
    source_etag: str
    file_bytes: bytes
    ld_row: dict


@dataclass
class ProcessedPage:
    page_idx: int
    filename: str
    source_etag: str
    line_tensors: list[tuple[npt.NDArray, int, int]]  # (tensor, content_width, left_pad_width)
    error: Optional[str] = None


@dataclass
class InferredPage:
    page_idx: int
    filename: str
    source_etag: str
    logits_list: list[tuple[npt.NDArray, int, int, npt.NDArray | None]]  # (logits, content_width, left_pad_width, keep_indices)
    error: Optional[str] = None


@dataclass
class PageResult:
    page_idx: int
    filename: str
    source_etag: str
    texts: list[str]
    error: Optional[str] = None


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
        ocr_model: OCRModel,
        ctc_decoder: CTCDecoder,
        input_width: int,
        input_height: int,
        s3_client,
        source_image_bucket: str,
        volume_prefix: str,
        prefetch_concurrency: int = 32,
        image_processor_workers: int = 4,
        ctc_workers: int = 4,
        gpu_batch_size: int = 16,
        beam_width: int | None = None,
        token_min_logp: float | None = None,
        use_greedy_decode: bool = False,
        use_hybrid_decode: bool = True,
        greedy_confidence_threshold: float | None = None,
        use_nemo_decoder: bool = False,
        kenlm_path: str | None = None,
        use_line_prepadding: bool = True,
        merge_line_segments: bool = True,
        line_merge_threshold: float | None = None,
        debug_output_dir: str | None = None,
    ):
        self.ocr_model = ocr_model
        self.ctc_decoder = ctc_decoder
        self.input_width = input_width
        self.input_height = input_height
        self.s3_client = s3_client
        self.source_image_bucket = source_image_bucket
        self.volume_prefix = volume_prefix

        self.prefetch_concurrency = prefetch_concurrency
        self.image_processor_workers = image_processor_workers
        self.ctc_workers = ctc_workers
        self.gpu_batch_size = gpu_batch_size
        self.beam_width = beam_width
        self.token_min_logp = token_min_logp
        self.use_greedy_decode = use_greedy_decode
        self.use_hybrid_decode = use_hybrid_decode
        self.greedy_confidence_threshold = greedy_confidence_threshold
        self.use_nemo_decoder = use_nemo_decoder
        self.merge_line_segments = merge_line_segments
        self.line_merge_threshold = line_merge_threshold
        self.kenlm_path = kenlm_path
        self.use_line_prepadding = use_line_prepadding
        self.debug_output_dir = debug_output_dir
        self._nemo_decoder = None  # Lazy init when needed

        # Bounded queues for backpressure
        self.q_fetched: asyncio.Queue = asyncio.Queue(maxsize=64)
        self.q_processed: asyncio.Queue = asyncio.Queue(maxsize=32)
        self.q_inferred: asyncio.Queue = asyncio.Queue(maxsize=32)
        self.q_results: asyncio.Queue = asyncio.Queue(maxsize=64)

        # Thread pool for image processing (cv2 releases GIL)
        self._image_executor = ThreadPoolExecutor(max_workers=image_processor_workers, thread_name_prefix="img")
        # Process pool for CTC decoding - IPC overhead (~190ms/call) is acceptable tradeoff
        # to avoid blocking the async event loop (which starves GPU inference)
        self._ctc_executor = ProcessPoolExecutor(
            max_workers=ctc_workers,
            initializer=init_worker_process,
            initargs=(ctc_decoder.ctc_vocab,),
        )

        logger.info(
            f"Pipeline config: prefetch={prefetch_concurrency}, img_workers={image_processor_workers}, ctc_workers={ctc_workers}, gpu_batch={gpu_batch_size}"
        )

        # Semaphore for S3 concurrency
        self._s3_sem = asyncio.Semaphore(prefetch_concurrency)

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
        async def monitor_queues():
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
            asyncio.create_task(self._parquet_writer(output_parquet_uri, total_pages), name="writer"),
        ]

        # Wait for all tasks
        results = await asyncio.gather(*tasks, return_exceptions=True)
        monitor_task.cancel()

        # Check for errors
        stage_names = ["prefetcher", "image_processor", "gpu_inference", "ctc_decoder", "writer"]
        for name, result in zip(stage_names, results):
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

        async def collect_inferred():
            """Collect all inferred pages instead of passing to CTC decoder."""
            while True:
                msg = await self.q_inferred.get()
                if isinstance(msg, EndOfStream):
                    break
                all_inferred.append(msg)

        # Queue depth monitor
        async def monitor_queues():
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
        vocab_size = len(vocab)
        loop = asyncio.get_event_loop()

        # Prepare all lines from all pages for batch submission
        # Each entry: (page_idx, line_idx, cropped_logits, inferred)
        all_decode_tasks: list[tuple[int, int, npt.NDArray, InferredPage]] = []

        for inferred in all_inferred:
            if inferred.error or not inferred.logits_list:
                # Handle error pages immediately
                await self.q_results.put(
                    PageResult(
                        page_idx=inferred.page_idx,
                        filename=inferred.filename,
                        source_etag=inferred.source_etag,
                        texts=[],
                        error=inferred.error,
                    )
                )
                continue

            for line_idx, (logits, content_w, left_pad_w, keep_indices) in enumerate(inferred.logits_list):
                # Logits are already cropped to remove padding (done in model before softmax)
                # Just need to transpose from (vocab, time) -> (time, vocab) if needed
                actual_vocab_size = vocab_size if keep_indices is None else len(keep_indices)
                needs_transpose = logits.shape[0] == actual_vocab_size
                if needs_transpose:
                    cropped = logits.T
                else:
                    cropped = logits
                all_decode_tasks.append((inferred.page_idx, line_idx, cropped, inferred, keep_indices))

        # Count how many have GPU pruning
        gpu_pruned_count = sum(1 for _, _, _, _, ki in all_decode_tasks if ki is not None)
        if gpu_pruned_count > 0:
            sample_ki = next((ki for _, _, _, _, ki in all_decode_tasks if ki is not None), None)
            logger.info(
                f"[Phase 2] Submitting {len(all_decode_tasks)} lines to {self.ctc_workers} workers... "
                f"({gpu_pruned_count} GPU-pruned, sample_keep_indices={len(sample_ki) if sample_ki is not None else 'N/A'})"
            )
        else:
            logger.info(f"[Phase 2] Submitting {len(all_decode_tasks)} lines to {self.ctc_workers} workers... (0 GPU-pruned)")

        # Submit ALL lines at once to ProcessPoolExecutor
        futures = []
        for page_idx, line_idx, cropped, inferred, keep_indices in all_decode_tasks:
            submit_time = time.perf_counter()
            
            # Model already pruned vocabulary - use pruned vocab list
            if keep_indices is not None:
                pruned_vocab = [vocab[i] for i in keep_indices]
            else:
                pruned_vocab = vocab
            
            future = loop.run_in_executor(
                self._ctc_executor,
                _decode_single_line,
                cropped,
                pruned_vocab,
                self.beam_width,
                self.token_min_logp,
                submit_time,
            )
            futures.append((page_idx, line_idx, future, inferred, submit_time))

        # Wait for decodes with progress logging
        decode_start = time.perf_counter()
        total_lines = len(futures)
        completed = 0
        all_texts = []

        # Wait for all decodes and log each line as it completes
        pending = {f[2]: (f[0], f[1], f[3], f[4]) for f in futures}  # future -> (page_idx, line_idx, inferred, submit_time)
        results = [None] * total_lines
        total_ipc_in = 0.0
        total_ipc_out = 0.0

        while pending:
            done, _ = await asyncio.wait(pending.keys(), return_when=asyncio.FIRST_COMPLETED)
            for future in done:
                page_idx, line_idx, inferred, submit_time = pending.pop(future)
                result_time = time.perf_counter()
                text, decode_ms, ipc_in_ms, worker_pid = future.result()
                ipc_out_ms = (time.perf_counter() - result_time) * 1000  # Time to deserialize result
                total_roundtrip_ms = (result_time - submit_time) * 1000
                total_ipc_in += ipc_in_ms
                total_ipc_out += ipc_out_ms
                idx = next(i for i, f in enumerate(futures) if f[2] is future)
                results[idx] = text
                completed += 1
                logger.info(
                    f"[CTC] line {completed}/{total_lines} page={page_idx} line={line_idx} worker={worker_pid} "
                    f"decode={decode_ms:.1f}ms ipc_in={ipc_in_ms:.1f}ms roundtrip={total_roundtrip_ms:.1f}ms"
                )

        all_texts = results
        decode_time = time.perf_counter() - decode_start
        num_texts = len(all_texts) if all_texts else 0
        avg_ipc_in = total_ipc_in / num_texts if num_texts > 0 else 0
        avg_ipc_out = total_ipc_out / num_texts if num_texts > 0 else 0
        ms_per_line = decode_time * 1000 / num_texts if num_texts > 0 else 0
        logger.info(
            f"[Phase 2] All {num_texts} lines decoded in {decode_time:.2f}s "
            f"({ms_per_line:.1f}ms/line, avg_ipc_in={avg_ipc_in:.1f}ms, avg_ipc_out={avg_ipc_out:.1f}ms)"
        )

        # Group results by page
        page_results: dict[int, tuple[InferredPage, list[tuple[int, str]]]] = {}
        for (page_idx, line_idx, _, inferred, _), text in zip(futures, all_texts):
            if page_idx not in page_results:
                page_results[page_idx] = (inferred, [])
            page_results[page_idx][1].append((line_idx, text))

        # Sort lines within each page and build PageResult objects
        results_to_write = []
        for page_idx in sorted(page_results.keys()):
            inferred, line_texts = page_results[page_idx]
            line_texts.sort(key=lambda x: x[0])
            texts = [t for _, t in line_texts]
            results_to_write.append(
                PageResult(
                    page_idx=page_idx,
                    filename=inferred.filename,
                    source_etag=inferred.source_etag,
                    texts=texts,
                    error=None,
                )
            )
            self.stats["decoded"] += 1

        # Emit results and write to parquet concurrently
        async def emit_results():
            for result in results_to_write:
                await self.q_results.put(result)
            await self.q_results.put(EOS)

        await asyncio.gather(
            emit_results(),
            self._parquet_writer(output_parquet_uri, total_pages),
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
        logger.info(f"[Prefetcher] Starting with {self.prefetch_concurrency} concurrency")

        async def fetch_one(page_idx: int, filename: str, ld_row: dict) -> None:
            async with self._s3_sem:
                try:
                    key = f"{self.volume_prefix}/{filename}"
                    loop = asyncio.get_event_loop()

                    def _fetch():
                        response = self.s3_client.get_object(Bucket=self.source_image_bucket, Key=key)
                        return response.get("ETag", "").strip('"'), response["Body"].read()

                    source_etag, file_bytes = await loop.run_in_executor(None, _fetch)

                    await self.q_fetched.put(
                        FetchedBytes(
                            page_idx=page_idx,
                            filename=filename,
                            source_etag=source_etag,
                            file_bytes=file_bytes,
                            ld_row=ld_row,
                        )
                    )
                    self.stats["fetched"] += 1

                    if self.stats["fetched"] % 20 == 0:
                        logger.info(f"[Prefetcher] Fetched {self.stats['fetched']} pages")

                except Exception as e:
                    logger.warning(f"[Prefetcher] Failed to fetch {filename}: {e}")
                    await self.q_fetched.put(
                        FetchedBytes(
                            page_idx=page_idx,
                            filename=filename,
                            source_etag="",
                            file_bytes=b"",
                            ld_row=ld_row,
                        )
                    )
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
        logger.info(f"[ImageProcessor] Starting with {self.image_processor_workers} concurrent workers")
        loop = asyncio.get_event_loop()

        # Semaphore limits concurrent processing
        sem = asyncio.Semaphore(self.image_processor_workers)
        pending_tasks: set[asyncio.Task] = set()

        async def process_one(fetched: FetchedBytes) -> None:
            async with sem:
                if not fetched.file_bytes:
                    await self.q_processed.put(
                        ProcessedPage(
                            page_idx=fetched.page_idx,
                            filename=fetched.filename,
                            source_etag=fetched.source_etag,
                            line_tensors=[],
                            error="Failed to fetch image",
                        )
                    )
                    return

                try:
                    processed = await loop.run_in_executor(
                        self._image_executor,
                        self._process_image_sync,
                        fetched,
                    )
                    await self.q_processed.put(processed)
                    self.stats["processed"] += 1

                except Exception as e:
                    logger.warning(f"[ImageProcessor] Failed to process {fetched.filename}: {e}")
                    await self.q_processed.put(
                        ProcessedPage(
                            page_idx=fetched.page_idx,
                            filename=fetched.filename,
                            source_etag=fetched.source_etag,
                            line_tensors=[],
                            error=str(e),
                        )
                    )
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

    def _process_image_sync(self, fetched: FetchedBytes) -> ProcessedPage:
        """Synchronous image processing (runs in thread pool)."""
        ld_row = fetched.ld_row

        # Decode image using shared decoder (returns grayscale)
        # Parameters: max_width=6000, max_height=3000, patch_size=6000 (no patching)
        image, is_binary, orig_h, orig_w = bytes_to_frame(
            fetched.filename,
            fetched.file_bytes,
            max_width=6000,
            max_height=3000,
            patch_size=6000,
            linearize=True,
        )

        # Check if downscaling occurred and compute scale factor
        actual_h, actual_w = image.shape[:2]
        scale_factor = 1.0
        if orig_h > 0 and orig_w > 0:
            scale_h = actual_h / orig_h
            scale_w = actual_w / orig_w
            # Use the smaller scale factor (they should be equal for uniform scaling)
            if abs(scale_h - 1.0) > 0.001 or abs(scale_w - 1.0) > 0.001:
                scale_factor = min(scale_h, scale_w)
                logger.error(
                    f"Image {fetched.filename} was downscaled: {orig_w}x{orig_h} -> {actual_w}x{actual_h} (scale={scale_factor:.3f})"
                )

        # Apply transforms (rotation + TPS) using shared helper
        rotation_angle = ld_row.get("rotation_angle", 0.0) or 0.0
        tps_points = ld_row.get("tps_points")
        tps_alpha = ld_row.get("tps_alpha", 0.5)

        # Extract and scale TPS points if present
        tps_input_pts = None
        tps_output_pts = None
        if tps_points:
            tps_input_pts, tps_output_pts = tps_points
            # Scale TPS points if image was downscaled
            if scale_factor != 1.0:
                if tps_input_pts is not None:
                    tps_input_pts = [[p[0] * scale_factor, p[1] * scale_factor] for p in tps_input_pts]
                if tps_output_pts is not None:
                    tps_output_pts = [[p[0] * scale_factor, p[1] * scale_factor] for p in tps_output_pts]

        # Apply rotation and TPS in one call (grayscale)
        image = apply_transform_1(image, rotation_angle, tps_input_pts, tps_output_pts, tps_alpha)

        # Track if any transformation was applied (affects binarization decision)
        was_transformed = (
            scale_factor != 1.0 or  # resized/downscaled
            (rotation_angle is not None and abs(rotation_angle) > 0.01) or  # rotated
            tps_input_pts is not None  # TPS applied
        )

        # Extract lines
        contours = ld_row.get("contours", [])
        if not contours:
            return ProcessedPage(
                page_idx=fetched.page_idx,
                filename=fetched.filename,
                source_etag=fetched.source_etag,
                line_tensors=[],
            )

        # Scale contours if image was downscaled
        if scale_factor != 1.0:
            scaled_contours = []
            for contour_points in contours:
                scaled_points = [
                    {"x": int(p["x"] * scale_factor), "y": int(p["y"] * scale_factor)}
                    for p in contour_points
                ]
                scaled_contours.append(scaled_points)
            contours = scaled_contours

        # Merge line segments that belong to the same horizontal line
        if self.merge_line_segments:
            original_count = len(contours)
            contours = _merge_line_segments(contours, line_threshold=self.line_merge_threshold)
            if len(contours) != original_count:
                logger.debug(
                    f"[ImageProcessor] {fetched.filename}: merged {original_count} segments into {len(contours)} lines"
                )

        line_tensors = []
        mask_buffer = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
        current_k = 1.7

        # Determine if we should skip binarization (already binary and not transformed)
        # weirdly, not rebinarizing already binary images gives different results...
        skip_binarization = False # is_binary and not was_transformed

        for line_idx, contour_points in enumerate(contours):
            line_img, current_k = self._extract_line(image, contour_points, current_k, mask_buffer)
            if line_img is None:
                line_tensors.append((np.zeros((1, self.input_height, self.input_width), dtype=np.float32), 1, 0))
                continue

            tensor, content_width, left_pad_width = self._preprocess_line(
                line_img,
                skip_binarization=skip_binarization,
                use_prepadding=self.use_line_prepadding,
                debug_filename=fetched.filename if self.debug_output_dir else None,
                debug_line_idx=line_idx if self.debug_output_dir else None,
            )
            line_tensors.append((tensor, content_width, left_pad_width))

        return ProcessedPage(
            page_idx=fetched.page_idx,
            filename=fetched.filename,
            source_etag=fetched.source_etag,
            line_tensors=line_tensors,
        )

    def _extract_line(
        self, image: npt.NDArray, contour_points: list[dict], k_factor: float, mask_buffer: npt.NDArray
    ) -> tuple[Optional[npt.NDArray], float]:
        """Extract line image from contour."""
        if not contour_points:
            return None, k_factor

        pts = np.array([[p["x"], p["y"]] for p in contour_points], dtype=np.int32)
        _, _, _, bbox_h = cv2.boundingRect(pts)
        if bbox_h <= 0:
            return None, k_factor

        mask_buffer.fill(0)
        cv2.drawContours(mask_buffer, [pts], -1, 255, -1)

        line_img, adapted_k = get_line_image(image, mask_buffer, bbox_h, bbox_tolerance=3.0, k_factor=k_factor)

        if line_img.size == 0:
            return None, adapted_k

        return line_img, adapted_k

    def _preprocess_line(
        self,
        image: npt.NDArray,
        skip_binarization: bool = False,
        use_prepadding: bool = True,
        debug_filename: str | None = None,
        debug_line_idx: int | None = None,
    ) -> tuple[npt.NDArray, int, int]:
        """Preprocess line image to tensor.
        
        Args:
            image: Line image (grayscale, may be 2D or 3D with 1 channel)
            skip_binarization: If True, skip binarization (image is already binary and untransformed)
            use_prepadding: If True, add h pixels of padding on left and right before resizing
            debug_filename: If provided (along with debug_output_dir), save debug images
            debug_line_idx: Line index for debug filename
            
        Returns:
            Tuple of (tensor, content_width, left_pad_width) where:
            - tensor: preprocessed image tensor
            - content_width: width of actual content in the final padded image
            - left_pad_width: width of left padding in the final padded image
        """
        # Ensure we have a 2D grayscale image
        if image.ndim == 3:
            # get_line_image returns (H, W, 1) for grayscale input
            image = image.squeeze(axis=-1)

        h, w = image.shape[:2]
        target_h = self.input_height
        target_w = self.input_width

        if use_prepadding:
            # Add square padding (h x h) on left and right before resizing
            left_pad = h
            right_pad = h
            padded_w = w + left_pad + right_pad
            
            # Create padded image with white (255) padding
            with_lr_pad = np.ones((h, padded_w), dtype=np.uint8) * 255
            with_lr_pad[:, left_pad:left_pad + w] = image

            # Calculate resize dimensions to fit target while maintaining aspect ratio
            aspect = padded_w / h
            if aspect > (target_w / target_h):
                new_w = target_w
                new_h = max(1, int(target_w / aspect))
            else:
                new_h = target_h
                new_w = max(1, int(target_h * aspect))

            resized = cv2.resize(with_lr_pad, (new_w, new_h), interpolation=cv2.INTER_AREA)

            # Calculate content and left_pad widths in resized image coordinates
            scale = new_w / padded_w
            left_pad_resized = int(left_pad * scale)
            content_width_resized = int(w * scale)
        else:
            # No prepadding - just resize to fit target
            left_pad_resized = 0
            aspect = w / h
            if aspect > (target_w / target_h):
                new_w = target_w
                new_h = max(1, int(target_w / aspect))
            else:
                new_h = target_h
                new_w = max(1, int(target_h * aspect))

            resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
            content_width_resized = new_w

        # Pad to target size with white (255)
        padded = np.ones((target_h, target_w), dtype=np.uint8) * 255
        y_offset = (target_h - new_h) // 2
        x_offset = 0
        padded[y_offset : y_offset + new_h, x_offset : x_offset + new_w] = resized

        # Binarize unless image was already binary and untransformed
        if skip_binarization:
            binary = padded
        else:
            binary = adaptive_binarize(padded)

        # Debug: save preprocessed line images
        if self.debug_output_dir and debug_filename is not None and debug_line_idx is not None:
            debug_dir = Path(self.debug_output_dir)
            debug_dir.mkdir(parents=True, exist_ok=True)
            # Create a safe filename from the original
            safe_name = Path(debug_filename).stem.replace("/", "_").replace("\\", "_")
            # Save the preprocessed (binarized) line
            out_path = debug_dir / f"{safe_name}_L{debug_line_idx:02d}.png"
            cv2.imwrite(str(out_path), binary)

        # Normalize
        tensor = binary.reshape((1, target_h, target_w)).astype(np.float32)
        tensor = (tensor / 127.5) - 1.0

        return tensor, content_width_resized, left_pad_resized

    def _apply_tps(self, image: npt.NDArray, tps_points: tuple, alpha: float) -> npt.NDArray:
        """Apply TPS transform (simplified)."""
        try:
            input_pts, output_pts = tps_points
            if input_pts is None or output_pts is None:
                return image

            input_pts = np.array(input_pts, dtype=np.float32)
            output_pts = np.array(output_pts, dtype=np.float32)

            tps = cv2.createThinPlateSplineShapeTransformer()  # type: ignore[attr-defined]
            tps.estimateTransformation(
                output_pts.reshape(1, -1, 2),
                input_pts.reshape(1, -1, 2),
                list(range(len(input_pts))),
            )

            h, w = image.shape[:2]
            result = tps.warpImage(image)
            return result if result is not None else image

        except Exception:
            return image

    async def _gpu_inference(self) -> None:
        """Run GPU inference with batching - emit pages as soon as all lines are done."""
        logger.info(f"[GPUInference] Starting with batch_size={self.gpu_batch_size}")

        # Track pages waiting for their lines to be processed
        # page_idx -> (ProcessedPage, expected_lines, {line_idx: (logits, orig_w, keep_indices)})
        pages_in_flight: dict[int, tuple[ProcessedPage, int, dict[int, tuple]]] = {}

        # Batch of tensors waiting to be processed
        # (page_idx, line_idx, tensor, content_w, left_pad_w)
        pending_tensors: list[tuple[int, int, npt.NDArray, int, int]] = []

        lines_processed = 0
        pages_emitted = 0
        pages_received = 0

        async def flush_batch():
            nonlocal lines_processed
            if not pending_tensors:
                return

            batch_size = len(pending_tensors)
            start_time = time.perf_counter()

            # Stack tensors and collect content widths and left pad widths
            tensors = np.concatenate([t[2] for t in pending_tensors], axis=0)
            content_widths = [t[3] for t in pending_tensors]
            left_pad_widths = [t[4] for t in pending_tensors]

            # Run inference with content widths and left pad widths for proper cropping
            # Model crops time dimension BEFORE softmax/pruning to save computation
            # Returns per-line keep_indices for deterministic pruning
            loop = asyncio.get_event_loop()
            batch_logits, keep_indices_list = await loop.run_in_executor(
                None, 
                lambda: self.ocr_model.predict(tensors, content_widths, left_pad_widths, self.input_width)
            )

            # batch_logits is now a list of arrays (one per item), already cropped
            # keep_indices_list is a list of keep_indices (one per item), or None per item
            # Distribute results back to pages
            for (page_idx, line_idx, _, content_w, left_pad_w), logits, keep_indices in zip(pending_tensors, batch_logits, keep_indices_list):
                if page_idx in pages_in_flight:
                    _, _, logits_dict = pages_in_flight[page_idx]
                    # Store logits along with per-line keep_indices
                    # Note: logits are already cropped to remove left and right padding
                    logits_dict[line_idx] = (logits, content_w, left_pad_w, keep_indices)
                lines_processed += 1

            pending_tensors.clear()

            batch_ms = (time.perf_counter() - start_time) * 1000
            logger.info(
                f"[GPUInference] Batch {batch_size} lines in {batch_ms:.0f}ms ({batch_ms / batch_size:.0f}ms/line). Total: {lines_processed} lines, {pages_emitted} pages emitted"
            )

        async def try_emit_completed_pages():
            nonlocal pages_emitted
            # Check if any pages have all their lines done
            completed = []
            for page_idx, (page, expected, logits_dict) in pages_in_flight.items():
                if len(logits_dict) >= expected:
                    completed.append(page_idx)

            for page_idx in completed:
                page, expected, logits_dict = pages_in_flight.pop(page_idx)

                # Build logits list in order
                logits_list = []
                for i in range(expected):
                    if i in logits_dict:
                        logits_list.append(logits_dict[i])
                    else:
                        logits_list.append((np.zeros((1, 84), dtype=np.float32), 1, 0, None))

                await self.q_inferred.put(
                    InferredPage(
                        page_idx=page.page_idx,
                        filename=page.filename,
                        source_etag=page.source_etag,
                        logits_list=logits_list,
                        error=page.error,
                    )
                )
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
                for page_idx, (page, expected, logits_dict) in pages_in_flight.items():
                    logits_list = []
                    for i in range(expected):
                        if i in logits_dict:
                            logits_list.append(logits_dict[i])
                        else:
                            logits_list.append((np.zeros((1, 84), dtype=np.float32), 1, 0, None))

                    await self.q_inferred.put(
                        InferredPage(
                            page_idx=page.page_idx,
                            filename=page.filename,
                            source_etag=page.source_etag,
                            logits_list=logits_list,
                            error=page.error,
                        )
                    )
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
            if processed.error or not processed.line_tensors:
                await self.q_inferred.put(
                    InferredPage(
                        page_idx=processed.page_idx,
                        filename=processed.filename,
                        source_etag=processed.source_etag,
                        logits_list=[],
                        error=processed.error,
                    )
                )
                pages_emitted += 1
                self.stats["inferred"] += 1
                continue

            # Register page
            expected_lines = len(processed.line_tensors)
            pages_in_flight[processed.page_idx] = (processed, expected_lines, {})

            # Add tensors to batch
            for line_idx, (tensor, content_w, left_pad_w) in enumerate(processed.line_tensors):
                pending_tensors.append((processed.page_idx, line_idx, tensor, content_w, left_pad_w))

                if len(pending_tensors) >= self.gpu_batch_size:
                    await flush_batch()
                    await try_emit_completed_pages()

        logger.info(f"[GPUInference] Done, processed {lines_processed} lines, emitted {pages_emitted} pages")

    async def _ctc_decoder_stage(self) -> None:
        """Decode logits to text concurrently using ProcessPoolExecutor."""
        logger.info(f"[CTCDecoder] Starting with {self.ctc_workers} workers")
        loop = asyncio.get_event_loop()
        pending_tasks: set[asyncio.Task] = set()
        pages_received = 0

        # Limit concurrent page decodes to balance parallelism vs resource contention
        # With N workers and ~6 lines/page, allow N/2 pages to avoid memory pressure
        max_concurrent_pages = max(2, self.ctc_workers // 2)
        page_sem = asyncio.Semaphore(max_concurrent_pages)
        logger.info(f"[CTCDecoder] Max concurrent page decodes: {max_concurrent_pages}")

        async def decode_one(inferred: InferredPage) -> None:
            if inferred.error or not inferred.logits_list:
                await self.q_results.put(
                    PageResult(
                        page_idx=inferred.page_idx,
                        filename=inferred.filename,
                        source_etag=inferred.source_etag,
                        texts=[],
                        error=inferred.error,
                    )
                )
                return

            async with page_sem:
                try:
                    start_time = time.perf_counter()
                    vocab = self.ctc_decoder.ctc_vocab
                    vocab_size = len(vocab)
                    num_lines = len(inferred.logits_list)

                    # Logits are already cropped to remove padding (done in model before softmax)
                    # Just need to transpose from (vocab, time) -> (time, vocab) if needed
                    # NOTE: Each line may have different keep_indices if they were in different GPU batches
                    cropped_logits_list = []
                    keep_indices_list = []
                    for logits, content_w, left_pad_w, _keep_indices in inferred.logits_list:
                        actual_vocab_size = vocab_size if _keep_indices is None else len(_keep_indices)
                        needs_transpose = logits.shape[0] == actual_vocab_size
                        if needs_transpose:
                            cropped = logits.T
                        else:
                            cropped = logits
                        cropped_logits_list.append(cropped)
                        keep_indices_list.append(_keep_indices)

                    if self.use_nemo_decoder:
                        # NeMo GPU decoder - batch decode all lines on GPU
                        if self._nemo_decoder is None:
                            from .ctc_decoder_nemo import CTCDecoderNemo

                            self._nemo_decoder = CTCDecoderNemo(
                                self.ctc_decoder.charset,
                                add_blank=True,
                                device="cuda",
                                beam_width=self.beam_width or 10,
                                kenlm_path=self.kenlm_path,
                            )
                        texts = self._nemo_decoder.decode_batch(cropped_logits_list)
                    elif self.use_greedy_decode:
                        # Greedy decode is fast (~0.6ms/line), run directly without ProcessPoolExecutor
                        texts = []
                        for cropped, keep_indices in zip(cropped_logits_list, keep_indices_list):
                            pruned_vocab = [vocab[i] for i in keep_indices] if keep_indices is not None else vocab
                            text = decode_logits_greedy(cropped, pruned_vocab)
                            texts.append(text.strip().replace("§", " "))
                    elif self.use_hybrid_decode:
                        # Hybrid decode: greedy first, beam search fallback for low-confidence lines
                        texts = []
                        for cropped, keep_indices in zip(cropped_logits_list, keep_indices_list):
                            pruned_vocab = [vocab[i] for i in keep_indices] if keep_indices is not None else vocab
                            text = decode_logits_hybrid_global(
                                cropped, pruned_vocab,
                                confidence_threshold=self.greedy_confidence_threshold,
                                beam_width=self.beam_width,
                                token_min_logp=self.token_min_logp,
                            )
                            texts.append(text.strip().replace("§", " "))
                    else:
                        # Beam search via ProcessPoolExecutor - submit each line for parallel decode
                        futures = []
                        for cropped, keep_indices in zip(cropped_logits_list, keep_indices_list):
                            pruned_vocab = [vocab[i] for i in keep_indices] if keep_indices is not None else vocab
                            future = loop.run_in_executor(
                                self._ctc_executor,
                                _decode_single_line,
                                cropped,
                                pruned_vocab,
                                self.beam_width,
                                self.token_min_logp,
                                None,  # submit_time
                            )
                            futures.append(future)

                        # Wait for all lines to complete in parallel
                        # _decode_single_line returns (text, decode_ms, ipc_in_ms, worker_pid)
                        results = list(await asyncio.gather(*futures))
                        texts = [r[0] for r in results]

                    decode_ms = (time.perf_counter() - start_time) * 1000
                    logger.info(
                        f"[CTCDecoder] Page {inferred.page_idx} decoded {num_lines} lines in {decode_ms:.0f}ms ({decode_ms / max(1, num_lines):.0f}ms/line)"
                    )

                    await self.q_results.put(
                        PageResult(
                            page_idx=inferred.page_idx,
                            filename=inferred.filename,
                            source_etag=inferred.source_etag,
                            texts=texts,
                        )
                    )
                    self.stats["decoded"] += 1

                except Exception as e:
                    logger.warning(f"[CTCDecoder] Failed to decode {inferred.filename}: {e}")
                    await self.q_results.put(
                        PageResult(
                            page_idx=inferred.page_idx,
                            filename=inferred.filename,
                            source_etag=inferred.source_etag,
                            texts=[],
                            error=str(e),
                        )
                    )
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

    async def _parquet_writer(self, output_uri: str, total_pages: int) -> None:
        """Write results to parquet using streaming writer."""
        logger.info(f"[ParquetWriter] Starting, expecting {total_pages} pages")

        writer = StreamingParquetWriter(output_uri, batch_size=100)
        pages_written = 0

        try:
            while True:
                msg = await self.q_results.get()
                if isinstance(msg, EndOfStream):
                    break

                result: PageResult = msg
                writer.write_record(
                    filename=result.filename,
                    source_etag=result.source_etag,
                    texts=result.texts,
                    error=result.error,
                )
                pages_written += 1

                if pages_written % 100 == 0:
                    logger.info(f"[ParquetWriter] Progress: {pages_written}/{total_pages}")
        finally:
            writer.close()

        logger.info(f"[ParquetWriter] Done, wrote {pages_written} pages")

    async def close(self) -> None:
        """Cleanup resources."""
        self._image_executor.shutdown(wait=False)
        self._ctc_executor.shutdown(wait=False)
