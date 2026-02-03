"""
Pipeline stages for OCR processing.

Each stage is a separate class with:
- __init__: Takes config, queues, and dependencies
- run(): Async method that processes items from input queue and puts results to output queue

This follows the LDv1 pipeline pattern for modular, testable stages.
"""

from __future__ import annotations

import asyncio
import io
import logging
import time
from typing import TYPE_CHECKING, Any

import numpy as np
import pyarrow.parquet as pq
from pyarrow import fs

from .ctc_decoder import decode_logits_with_segments
from .data_structures import (
    EndOfStream,
    ImageTask,
    InferredPage,
    LineLogits,
    LineResult,
    PageInFlight,
    PageOCRResult,
    PendingTensor,
    PipelineError,
)
from .line import BBox
from .line_decoder import FetchedBytes, LineDecoder, PrefetchedBytes, ProcessedPage
from .output_writer import OutputWriter

if TYPE_CHECKING:
    from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

    from botocore.client import BaseClient

    from .config import OCRV1Config
    from .ctc_decoder import CTCDecoder
    from .model import OCRModel

logger = logging.getLogger(__name__)

# Default vocabulary size for fallback logits
DEFAULT_VOCAB_SIZE = 84


class ParquetLoaderStage:
    """Loads LD parquet file asynchronously with progressive row availability.

    Downloads parquet via boto3 (faster than PyArrow's S3 filesystem),
    then reads from memory. Makes rows available progressively so
    ImageProcessor can start before all rows are parsed.
    """

    def __init__(self, s3_client: BaseClient | None = None) -> None:
        self.s3_client = s3_client
        self.parquet_data: dict[str, dict] = {}
        self._row_events: dict[str, asyncio.Event] = {}
        self._load_complete = asyncio.Event()
        self.parquet_error: str | None = None

    async def get_row(self, filename: str) -> dict | None:
        """Get row by filename, waiting if not yet loaded.

        Returns immediately if the row is already in memory.
        Otherwise waits until the row's row group is loaded.
        Returns None if the row doesn't exist in the parquet file.
        """
        # Fast path: already in memory
        if filename in self.parquet_data:
            return self.parquet_data[filename]

        # Load finished, row doesn't exist
        if self._load_complete.is_set():
            return None

        # Register interest in this filename
        if filename not in self._row_events:
            self._row_events[filename] = asyncio.Event()

        # Wait for either this row to be loaded or load completion
        row_event = self._row_events[filename]
        load_complete_task = asyncio.create_task(self._load_complete.wait())
        row_event_task = asyncio.create_task(row_event.wait())

        try:
            _, pending = await asyncio.wait(
                {load_complete_task, row_event_task},
                return_when=asyncio.FIRST_COMPLETED,
            )
            for task in pending:
                task.cancel()

            if filename in self.parquet_data:
                return self.parquet_data[filename]
        except Exception:
            return None
        else:
            return None

    @property
    def is_load_complete(self) -> bool:
        """Check if parquet loading has completed."""
        return self._load_complete.is_set()

    def _parse_s3_uri(self, uri: str) -> tuple[str, str]:
        """Parse s3://bucket/key into (bucket, key)."""
        if not uri.startswith("s3://"):
            raise ValueError(f"Expected s3:// URI, got: {uri}")
        path = uri[5:]  # Remove "s3://"
        bucket, _, key = path.partition("/")
        return bucket, key

    async def run(self, ld_parquet_uri: str) -> None:
        """Load parquet file and make rows available progressively.

        Downloads via boto3 (much faster than PyArrow's S3 filesystem),
        then reads from memory. Processes rows in batches with yields
        to the event loop so images can start processing as soon as
        their row is parsed.
        """
        logger.info(f"[ParquetLoader] Starting to load {ld_parquet_uri}")
        start_time = time.perf_counter()
        loop = asyncio.get_event_loop()

        try:
            # Download via boto3 (faster than PyArrow's S3 filesystem)
            if self.s3_client is not None and ld_parquet_uri.startswith("s3://"):
                bucket, key = self._parse_s3_uri(ld_parquet_uri)

                # Define the function with the client as a parameter
                def _download_and_parse(client: BaseClient) -> tuple[Any, int]:
                    response = client.get_object(Bucket=bucket, Key=key)
                    file_bytes = response["Body"].read()
                    file_size_bytes = len(file_bytes)

                    # Read parquet from memory buffer
                    buffer = io.BytesIO(file_bytes)
                    table = pq.read_table(buffer)
                    return table.to_pylist(), file_size_bytes

                all_rows, file_size_bytes = await loop.run_in_executor(None, _download_and_parse, self.s3_client)
            else:
                # Fallback to PyArrow's S3 filesystem for non-S3 URIs or if no client
                def _load_parquet() -> tuple[Any, int]:
                    filesystem, path = fs.FileSystem.from_uri(ld_parquet_uri)
                    file_info = filesystem.get_file_info(path)
                    file_size_bytes = file_info.size

                    table = pq.read_table(ld_parquet_uri)
                    return table.to_pylist(), file_size_bytes

                all_rows, file_size_bytes = await loop.run_in_executor(None, _load_parquet)

            download_time = time.perf_counter() - start_time

            # Format file size for logging
            if file_size_bytes >= 1024 * 1024:
                size_str = f"{file_size_bytes / (1024 * 1024):.2f} MB"
            elif file_size_bytes >= 1024:
                size_str = f"{file_size_bytes / 1024:.1f} KB"
            else:
                size_str = f"{file_size_bytes} bytes"

            speed_mbps = (file_size_bytes / (1024 * 1024)) / download_time if download_time > 0 else 0
            logger.debug(
                f"[ParquetLoader] Downloaded {len(all_rows)} rows ({size_str}) in {download_time:.2f}s ({speed_mbps:.1f} MB/s), processing..."
            )

            # Process rows in batches, yielding to event loop between batches
            # This allows images to start processing as soon as their row is available
            batch_size = 100
            for i in range(0, len(all_rows), batch_size):
                batch = all_rows[i : i + batch_size]
                for row in batch:
                    filename = row["img_file_name"]
                    self.parquet_data[filename] = row
                    if filename in self._row_events:
                        self._row_events[filename].set()

                # Yield to event loop so waiting tasks can proceed
                await asyncio.sleep(0)

            elapsed = time.perf_counter() - start_time
            logger.debug(f"[ParquetLoader] Loaded {len(self.parquet_data)} rows in {elapsed:.2f}s")

        except Exception as e:
            elapsed = time.perf_counter() - start_time
            self.parquet_error = f"{type(e).__name__}: {e}"
            logger.exception(f"[ParquetLoader] Failed in {elapsed:.2f}s: {self.parquet_error}")
        finally:
            self._load_complete.set()
            # Signal all remaining waiters (in case they're waiting for rows that don't exist)
            for event in self._row_events.values():
                event.set()


class PrefetcherStage:
    """Fetches images from S3 with high concurrency.

    Outputs PrefetchedBytes (raw bytes without LD metadata).
    """

    def __init__(
        self,
        cfg: OCRV1Config,
        s3_client: BaseClient,
        source_image_bucket: str,
        volume_prefix: str,
        q_out: asyncio.Queue,
        stats: dict[str, int],
    ) -> None:
        self.cfg = cfg
        self.s3_client = s3_client
        self.source_image_bucket = source_image_bucket
        self.volume_prefix = volume_prefix
        self.q_out = q_out
        self.stats = stats
        self._s3_sem = asyncio.Semaphore(cfg.prefetch_concurrency)

    async def run(self, tasks: list[ImageTask]) -> None:
        """Fetch all images from S3."""
        logger.info(f"[Prefetcher] Starting with {self.cfg.prefetch_concurrency} concurrency")

        async def fetch_one(task: ImageTask) -> None:
            async with self._s3_sem:
                try:
                    key = f"{self.volume_prefix.rstrip('/')}/{task.filename}"
                    loop = asyncio.get_event_loop()

                    def _fetch() -> tuple[str, bytes]:
                        response = self.s3_client.get_object(Bucket=self.source_image_bucket, Key=key)
                        return response.get("ETag", "").strip('"'), response["Body"].read()

                    source_etag, file_bytes = await loop.run_in_executor(None, _fetch)

                    await self.q_out.put(
                        PrefetchedBytes(
                            task=task,
                            source_etag=source_etag,
                            file_bytes=file_bytes,
                        )
                    )
                    self.stats["fetched"] += 1

                    if self.stats["fetched"] % 20 == 0:
                        logger.debug(f"[Prefetcher] Fetched {self.stats['fetched']} pages")

                except Exception as e:
                    key = f"{self.volume_prefix.rstrip('/')}/{task.filename}"
                    logger.warning(f"[Prefetcher] Failed to fetch {task.filename} (s3://{self.source_image_bucket}/{key}): {e}")
                    await self.q_out.put(
                        PipelineError(
                            stage="Prefetcher",
                            task=task,
                            source_etag=None,
                            error_type=type(e).__name__,
                            message=f"Failed to fetch s3://{self.source_image_bucket}/{key}: {e}",
                        )
                    )
                    self.stats["errors"] += 1

        fetch_tasks = [asyncio.create_task(fetch_one(task)) for task in tasks]
        await asyncio.gather(*fetch_tasks)

        await self.q_out.put(EndOfStream(stream="fetched", producer="Prefetcher"))
        logger.info(f"[Prefetcher] Done, fetched {self.stats['fetched']} pages")


class ImageProcessorStage:
    """Processes images using LineDecoder.

    Waits for parquet to be loaded, combines PrefetchedBytes with LD row data,
    and runs LineDecoder to extract lines.
    """

    def __init__(
        self,
        cfg: OCRV1Config,
        line_decoder: LineDecoder,
        parquet_loader: ParquetLoaderStage,
        q_in: asyncio.Queue,
        q_out: asyncio.Queue,
        executor: ThreadPoolExecutor,
        stats: dict[str, int],
    ) -> None:
        self.cfg = cfg
        self.line_decoder = line_decoder
        self.parquet_loader = parquet_loader
        self.q_in = q_in
        self.q_out = q_out
        self.executor = executor
        self.stats = stats

    async def run(self) -> None:
        """Process images concurrently.

        Uses streaming parquet loading - each image waits only for its
        specific row to be loaded, not the entire parquet file.
        """
        logger.info(f"[ImageProcessor] Starting with {self.cfg.image_processor_workers} workers")

        loop = asyncio.get_event_loop()
        sem = asyncio.Semaphore(self.cfg.image_processor_workers)
        pending_tasks: set[asyncio.Task] = set()

        async def process_one(prefetched: PrefetchedBytes) -> None:
            async with sem:
                try:
                    # Wait for this specific row (may return immediately if already loaded)
                    ld_row = await self.parquet_loader.get_row(prefetched.filename)

                    # Check for parquet error
                    if self.parquet_loader.parquet_error:
                        await self.q_out.put(
                            PipelineError(
                                stage="ImageProcessor",
                                task=prefetched.task,
                                source_etag=prefetched.source_etag,
                                error_type="ParquetLoadError",
                                message=f"LD parquet loading failed: {self.parquet_loader.parquet_error}",
                            )
                        )
                        self.stats["errors"] += 1
                        return

                    if ld_row is None:
                        await self.q_out.put(
                            PipelineError(
                                stage="ImageProcessor",
                                task=prefetched.task,
                                source_etag=prefetched.source_etag,
                                error_type="MissingLDData",
                                message=f"No LD data found for {prefetched.filename} in parquet",
                            )
                        )
                        self.stats["errors"] += 1
                        return

                    fetched = FetchedBytes(
                        task=prefetched.task,
                        source_etag=prefetched.source_etag,
                        file_bytes=prefetched.file_bytes,
                        ld_row=ld_row,
                    )

                    processed = await loop.run_in_executor(
                        self.executor,
                        self.line_decoder.process,
                        fetched,
                    )
                    await self.q_out.put(processed)
                    self.stats["processed"] += 1
                    if processed.error:
                        self.stats["errors"] += 1
                except Exception as e:
                    logger.warning(f"[ImageProcessor] Failed to process {prefetched.filename}: {e}")
                    await self.q_out.put(
                        PipelineError(
                            stage="ImageProcessor",
                            task=prefetched.task,
                            source_etag=prefetched.source_etag,
                            error_type=type(e).__name__,
                            message=str(e),
                        )
                    )
                    self.stats["errors"] += 1

        while True:
            msg = await self.q_in.get()
            if isinstance(msg, EndOfStream):
                if pending_tasks:
                    await asyncio.gather(*pending_tasks, return_exceptions=True)
                await self.q_out.put(EndOfStream(stream="processed", producer="ImageProcessor"))
                break

            if isinstance(msg, PipelineError):
                await self.q_out.put(msg)
                continue

            prefetched: PrefetchedBytes = msg
            task = asyncio.create_task(process_one(prefetched))
            pending_tasks.add(task)
            task.add_done_callback(pending_tasks.discard)

        logger.info(f"[ImageProcessor] Done, processed {self.stats['processed']} pages")


class GPUInferenceStage:
    """Runs GPU inference with batching.

    Collects line tensors into batches, runs inference, and distributes results.
    """

    def __init__(
        self,
        cfg: OCRV1Config,
        ocr_model: OCRModel,
        q_in: asyncio.Queue,
        q_out: asyncio.Queue,
        stats: dict[str, int],
    ) -> None:
        self.cfg = cfg
        self.ocr_model = ocr_model
        self.q_in = q_in
        self.q_out = q_out
        self.stats = stats

    async def run(self) -> None:
        """Run GPU inference with batching."""

        logger.info(f"[GPUInference] Starting with batch_size={self.cfg.gpu_batch_size}")

        pending_tensors: list[PendingTensor] = []
        pages_in_flight: dict[int, PageInFlight] = {}
        pages_received = 0
        pages_emitted = 0

        async def flush_batch() -> None:
            nonlocal pages_emitted
            if not pending_tensors:
                return

            batch_size = len(pending_tensors)
            start_time = time.perf_counter()

            # Stack tensors and collect content widths and left pad widths
            tensors = np.concatenate([pt.tensor for pt in pending_tensors], axis=0)
            content_widths = [pt.content_width for pt in pending_tensors]
            left_pad_widths = [pt.left_pad_width for pt in pending_tensors]

            # Run inference in executor to not block event loop
            # Model crops time dimension BEFORE softmax/pruning to save computation
            loop = asyncio.get_event_loop()
            line_logits_list = await loop.run_in_executor(
                None,
                lambda: self.ocr_model.predict(tensors, content_widths, left_pad_widths, self.ocr_model.input_width),
            )

            # Distribute results back to pages
            for pt, line_logits in zip(pending_tensors, line_logits_list, strict=True):
                if pt.page_idx in pages_in_flight:
                    pages_in_flight[pt.page_idx].line_logits[pt.line_idx] = line_logits

            pending_tensors.clear()

            batch_ms = (time.perf_counter() - start_time) * 1000
            logger.debug(
                f"[GPUInference] Batch {batch_size} lines in {batch_ms:.0f}ms "
                f"({batch_ms / batch_size:.0f}ms/line). {pages_emitted} pages emitted"
            )

        async def try_emit_completed_pages() -> None:
            nonlocal pages_emitted
            completed = [pid for pid, pf in pages_in_flight.items() if len(pf.line_logits) >= pf.expected_lines]

            for pid in completed:
                page_flight = pages_in_flight.pop(pid)
                logits_list = [
                    page_flight.line_logits.get(
                        i,
                        LineLogits(
                            logits=np.zeros((1, DEFAULT_VOCAB_SIZE), dtype=np.float32),
                            content_width=1,
                            left_pad_width=0,
                            keep_indices=None,
                        ),
                    )
                    for i in range(page_flight.expected_lines)
                ]
                await self.q_out.put(self._create_inferred_page(page_flight.processed_page, logits_list))
                pages_emitted += 1
                self.stats["inferred"] += 1

        while True:
            msg = await self.q_in.get()
            if isinstance(msg, EndOfStream):
                logger.debug(f"[GPUInference] Received EOS, flushing {len(pending_tensors)} pending tensors")
                await flush_batch()
                await try_emit_completed_pages()

                for page_flight in pages_in_flight.values():
                    logits_list = [
                        page_flight.line_logits.get(
                            i,
                            LineLogits(
                                logits=np.zeros((1, DEFAULT_VOCAB_SIZE), dtype=np.float32),
                                content_width=1,
                                left_pad_width=0,
                                keep_indices=None,
                            ),
                        )
                        for i in range(page_flight.expected_lines)
                    ]
                    await self.q_out.put(self._create_inferred_page(page_flight.processed_page, logits_list))
                    self.stats["inferred"] += 1

                await self.q_out.put(EndOfStream(stream="inferred", producer="GPUInference"))
                break

            if isinstance(msg, PipelineError):
                await self.q_out.put(msg)
                continue

            processed: ProcessedPage = msg
            pages_received += 1

            if pages_received % 20 == 0:
                logger.debug(
                    f"[GPUInference] Received {pages_received} pages, "
                    f"{len(pending_tensors)} tensors pending, {len(pages_in_flight)} in flight"
                )

            if processed.error or not processed.lines:
                await self.q_out.put(self._create_inferred_page(processed, []))
                pages_emitted += 1
                self.stats["inferred"] += 1
                continue

            pages_in_flight[processed.page_idx] = PageInFlight(
                processed_page=processed,
                expected_lines=len(processed.lines),
                line_logits={},
            )

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

        logger.info(f"[GPUInference] Done, inferred {self.stats['inferred']} pages")

    def _create_inferred_page(self, page: ProcessedPage, logits_list: list) -> InferredPage:
        return InferredPage(
            task=page.task,
            source_etag=page.source_etag,
            logits_list=logits_list,
            processed_page=page,
            error=page.error,
        )


class CTCDecoderStage:
    """Decodes logits to text using CTC beam search.

    Runs decoding in a process pool for parallelism.
    """

    def __init__(
        self,
        cfg: OCRV1Config,
        ctc_decoder: CTCDecoder,
        q_in: asyncio.Queue,
        q_out: asyncio.Queue,
        executor: ProcessPoolExecutor,
        stats: dict[str, int],
    ) -> None:
        self.cfg = cfg
        self.ctc_decoder = ctc_decoder
        self.q_in = q_in
        self.q_out = q_out
        self.executor = executor
        self.stats = stats

    async def run(self) -> None:
        """Decode pages from GPU inference."""

        logger.info("[CTCDecoder] Starting")

        sem = asyncio.Semaphore(self.cfg.ctc_workers * 2)
        pending_tasks: set[asyncio.Task] = set()
        pages_received = 0

        async def decode_one(inferred: InferredPage) -> None:
            async with sem:
                try:
                    vocab = self.ctc_decoder.ctc_vocab
                    line_decode_results = []

                    num_lines = len(inferred.logits_list)
                    start_time = time.perf_counter()

                    for line_idx, line_logits in enumerate(inferred.logits_list):
                        actual_vocab_size = line_logits.vocab_size
                        needs_transpose = line_logits.logits.shape[0] == actual_vocab_size
                        cropped = line_logits.logits.T if needs_transpose else line_logits.logits

                        pruned_vocab = (
                            [vocab[i] for i in line_logits.keep_indices]
                            if line_logits.keep_indices is not None
                            else vocab
                        )

                        # Get line width in original coordinates (not page width!)
                        processed_line = inferred.processed_page.lines[line_idx]
                        line_bbox_w = processed_line.bbox[2]  # Width in resized coords
                        inv_scale = (
                            1.0 / inferred.processed_page.scale_factor
                            if inferred.processed_page.scale_factor != 0
                            else 1.0
                        )
                        original_line_width = int(line_bbox_w * inv_scale)  # Scale to original coords

                        decode_result = decode_logits_with_segments(
                            cropped,
                            pruned_vocab,
                            original_line_width,  # Pass line width, not page width!
                            beam_width=self.cfg.beam_width,
                            token_min_logp=self.cfg.token_min_logp,
                        )
                        line_decode_results.append(decode_result)

                    decode_ms = (time.perf_counter() - start_time) * 1000
                    logger.debug(
                        f"[CTCDecoder] Page {inferred.page_idx} decoded {num_lines} lines "
                        f"in {decode_ms:.0f}ms ({decode_ms / max(1, num_lines):.0f}ms/line)"
                    )

                    page_result = self._build_page_ocr_result(inferred.processed_page, line_decode_results)
                    await self.q_out.put(page_result)
                    self.stats["decoded"] += 1

                except Exception as e:
                    logger.warning(f"[CTCDecoder] Failed to decode {inferred.filename}: {e}")
                    await self.q_out.put(
                        PipelineError(
                            stage="CTCDecoder",
                            task=inferred.task,
                            source_etag=inferred.source_etag,
                            error_type=type(e).__name__,
                            message=str(e),
                        )
                    )
                    self.stats["errors"] += 1

        while True:
            msg = await self.q_in.get()
            if isinstance(msg, EndOfStream):
                if pending_tasks:
                    await asyncio.gather(*pending_tasks, return_exceptions=True)
                await self.q_out.put(EndOfStream(stream="results", producer="CTCDecoder"))
                break

            if isinstance(msg, PipelineError):
                await self.q_out.put(msg)
                continue

            inferred: InferredPage = msg
            pages_received += 1

            if pages_received % 20 == 0:
                logger.debug(f"[CTCDecoder] Received {pages_received} pages")

            task = asyncio.create_task(decode_one(inferred))
            pending_tasks.add(task)
            task.add_done_callback(pending_tasks.discard)

        logger.info(f"[CTCDecoder] Done, decoded {self.stats['decoded']} pages")

    def _build_page_ocr_result(self, processed_page: ProcessedPage, line_decode_results: list) -> PageOCRResult:
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


class OutputWriterStage:
    """Writes results to Parquet output file.

    Tracks expected vs received pages and creates error records for any
    missing pages at EndOfStream (like ldv1's ParquetWriter).
    """

    def __init__(
        self,
        parquet_uri: str,
        q_in: asyncio.Queue,
        expected_filenames: list[str],
        stats: dict[str, int],
    ) -> None:
        self.parquet_uri = parquet_uri
        self.q_in = q_in
        self.expected_filenames = set(expected_filenames)
        self.total_pages = len(expected_filenames)
        self.stats = stats

        # Track received filenames for missing page detection
        self._received_filenames: set[str] = set()

    async def run(self) -> None:
        """Write results to Parquet output file."""

        logger.info(f"[OutputWriter] Starting, expecting {self.total_pages} pages")

        writer = OutputWriter(self.parquet_uri)
        pages_written = 0
        errors_received = 0

        try:
            while True:
                msg = await self.q_in.get()
                if isinstance(msg, EndOfStream):
                    break

                if isinstance(msg, PipelineError):
                    # Track received filename
                    if msg.filename:
                        self._received_filenames.add(msg.filename)
                    writer.write_error(msg)
                    errors_received += 1
                    continue

                result: PageOCRResult = msg
                # Track received filename (PageOCRResult has img_file_name, PageResult has filename)
                filename = getattr(result, "img_file_name", None) or getattr(result, "filename", None)
                if filename:
                    self._received_filenames.add(filename)
                writer.write_page(result)
                pages_written += 1

                if pages_written % 100 == 0:
                    logger.debug(f"[OutputWriter] Progress: {pages_written}/{self.total_pages}")

            # Fill missing pages with errors before closing
            dropped_count = self._fill_missing_pages(writer)
            if dropped_count > 0:
                errors_received += dropped_count
                self.stats["errors"] += dropped_count

        finally:
            writer.close()

        logger.info(f"[OutputWriter] Done, wrote {pages_written} pages, {errors_received} errors")

    def _fill_missing_pages(self, writer: OutputWriter) -> int:
        """Create error records for any expected pages that were never received.

        Returns:
            Number of missing pages that were filled.
        """

        missing = self.expected_filenames - self._received_filenames
        if not missing:
            return 0

        # Log summary with sample of missing files
        missing_sorted = sorted(missing)
        sample = missing_sorted[:5]
        suffix = f" ... (+{len(missing_sorted) - 5} more)" if len(missing_sorted) > 5 else ""
        logger.warning(
            f"[OutputWriter] {len(missing)} pages never received. Creating error records for: {sample}{suffix}"
        )

        for filename in missing_sorted:
            # Create synthetic error for missing page
            synthetic_error = PipelineError(
                stage="Pipeline",
                task=ImageTask(page_idx=-1, filename=filename),
                source_etag=None,
                error_type="DroppedByPipeline",
                message="Page never received by output writer (lost in pipeline)",
            )
            writer.write_error(synthetic_error)

        return len(missing)
