"""
Pipeline stages for OCR processing.

Each stage is a separate class with:
- __init__: Takes config, queues, and dependencies
- run(): Async method that processes items from input queue and puts results to output queue

This follows the LDv1 pipeline pattern for modular, testable stages.
"""

from __future__ import annotations

import asyncio
import logging
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt
import pyarrow.parquet as pq
from botocore.client import BaseClient

from .data_structures import (
    EndOfStream,
    ImageTask,
    InferredPage,
    LineLogits,
    PageInFlight,
    PipelineError,
)
from .line_decoder import FetchedBytes, LineDecoder, PrefetchedBytes, ProcessedPage

if TYPE_CHECKING:
    from .config import OCRV1Config
    from .ctc_decoder import CTCDecoder
    from .model import OCRModel

logger = logging.getLogger(__name__)

# Default vocabulary size for fallback logits
DEFAULT_VOCAB_SIZE = 84


class ParquetLoaderStage:
    """Loads LD parquet file asynchronously.
    
    Runs in parallel with prefetcher so image fetching can start
    before parquet is fully loaded.
    """

    def __init__(self, parquet_ready: asyncio.Event) -> None:
        self.parquet_ready = parquet_ready
        self.parquet_data: dict[str, dict] | None = None
        self.parquet_error: str | None = None

    async def run(self, ld_parquet_uri: str) -> None:
        """Load parquet file asynchronously."""
        logger.info(f"[ParquetLoader] Starting to load {ld_parquet_uri}")
        start_time = time.perf_counter()
        loop = asyncio.get_event_loop()

        def _load_parquet() -> dict[str, dict]:
            table = pq.read_table(ld_parquet_uri)
            return {row["img_file_name"]: row for row in table.to_pylist()}

        try:
            self.parquet_data = await loop.run_in_executor(None, _load_parquet)
            elapsed = time.perf_counter() - start_time
            logger.info(f"[ParquetLoader] Loaded {len(self.parquet_data)} rows in {elapsed:.2f}s")
        except Exception as e:
            elapsed = time.perf_counter() - start_time
            self.parquet_error = f"{type(e).__name__}: {e}"
            logger.error(f"[ParquetLoader] Failed in {elapsed:.2f}s: {self.parquet_error}")
        finally:
            self.parquet_ready.set()


class PrefetcherStage:
    """Fetches images from S3 with high concurrency.
    
    Outputs PrefetchedBytes (raw bytes without LD metadata).
    """

    def __init__(
        self,
        cfg: "OCRV1Config",
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
                    key = f"{self.volume_prefix}/{task.filename}"
                    loop = asyncio.get_event_loop()

                    def _fetch() -> tuple[str, bytes]:
                        response = self.s3_client.get_object(
                            Bucket=self.source_image_bucket, Key=key
                        )
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
                        logger.info(f"[Prefetcher] Fetched {self.stats['fetched']} pages")

                except Exception as e:
                    logger.warning(f"[Prefetcher] Failed to fetch {task.filename}: {e}")
                    await self.q_out.put(
                        PipelineError(
                            stage="Prefetcher",
                            task=task,
                            source_etag=None,
                            error_type=type(e).__name__,
                            message=str(e),
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
        cfg: "OCRV1Config",
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
        """Process images concurrently."""
        logger.info(f"[ImageProcessor] Starting with {self.cfg.image_processor_workers} workers")

        # Wait for parquet to be loaded
        logger.info("[ImageProcessor] Waiting for parquet data...")
        await self.parquet_loader.parquet_ready.wait()

        if self.parquet_loader.parquet_error:
            logger.error(f"[ImageProcessor] Parquet loading failed: {self.parquet_loader.parquet_error}")
            await self._drain_on_parquet_error()
            return

        logger.info(f"[ImageProcessor] Parquet ready with {len(self.parquet_loader.parquet_data or {})} rows")
        loop = asyncio.get_event_loop()
        sem = asyncio.Semaphore(self.cfg.image_processor_workers)
        pending_tasks: set[asyncio.Task] = set()

        async def process_one(fetched: FetchedBytes) -> None:
            async with sem:
                try:
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
                    logger.warning(f"[ImageProcessor] Failed to process {fetched.filename}: {e}")
                    await self.q_out.put(
                        PipelineError(
                            stage="ImageProcessor",
                            task=fetched.task,
                            source_etag=fetched.source_etag,
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
            ld_row = self.parquet_loader.parquet_data.get(prefetched.filename) if self.parquet_loader.parquet_data else None

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
                continue

            fetched = FetchedBytes(
                task=prefetched.task,
                source_etag=prefetched.source_etag,
                file_bytes=prefetched.file_bytes,
                ld_row=ld_row,
            )

            task = asyncio.create_task(process_one(fetched))
            pending_tasks.add(task)
            task.add_done_callback(pending_tasks.discard)

        logger.info(f"[ImageProcessor] Done, processed {self.stats['processed']} pages")

    async def _drain_on_parquet_error(self) -> None:
        """Drain input queue on parquet error, emitting errors for each page."""
        while True:
            msg = await self.q_in.get()
            if isinstance(msg, EndOfStream):
                break
            if isinstance(msg, PipelineError):
                await self.q_out.put(msg)
                continue
            prefetched: PrefetchedBytes = msg
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
        await self.q_out.put(EndOfStream(stream="processed", producer="ImageProcessor"))
        logger.info(f"[ImageProcessor] Aborted due to parquet error, {self.stats['errors']} errors")


class GPUInferenceStage:
    """Runs GPU inference with batching.
    
    Collects line tensors into batches, runs inference, and distributes results.
    """

    def __init__(
        self,
        cfg: "OCRV1Config",
        ocr_model: "OCRModel",
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
        from .data_structures import PendingTensor
        
        logger.info(f"[GPUInference] Starting with batch_size={self.cfg.gpu_batch_size}")

        pending_tensors: list[PendingTensor] = []
        pages_in_flight: dict[int, PageInFlight] = {}
        pages_received = 0
        pages_emitted = 0

        async def flush_batch() -> None:
            nonlocal pages_emitted
            if not pending_tensors:
                return

            batch_tensors = [pt.tensor for pt in pending_tensors]
            batch_stack = np.stack(batch_tensors, axis=0)

            logits_batch = self.ocr_model.run_batch(batch_stack)

            for i, pt in enumerate(pending_tensors):
                logits = logits_batch[i]
                line_logits = LineLogits(
                    logits=logits,
                    content_width=pt.content_width,
                    left_pad_width=pt.left_pad_width,
                    keep_indices=None,
                )

                if pt.page_idx in pages_in_flight:
                    pages_in_flight[pt.page_idx].line_logits[pt.line_idx] = line_logits

            pending_tensors.clear()

        async def try_emit_completed_pages() -> None:
            nonlocal pages_emitted
            completed = [
                pid for pid, pf in pages_in_flight.items()
                if len(pf.line_logits) >= pf.expected_lines
            ]

            for pid in completed:
                page_flight = pages_in_flight.pop(pid)
                logits_list = [
                    page_flight.line_logits.get(i, LineLogits(
                        logits=np.zeros((1, DEFAULT_VOCAB_SIZE), dtype=np.float32),
                        content_width=1,
                        left_pad_width=0,
                        keep_indices=None,
                    ))
                    for i in range(page_flight.expected_lines)
                ]
                await self.q_out.put(self._create_inferred_page(page_flight.processed_page, logits_list))
                pages_emitted += 1
                self.stats["inferred"] += 1

        while True:
            msg = await self.q_in.get()
            if isinstance(msg, EndOfStream):
                logger.info(f"[GPUInference] Received EOS, flushing {len(pending_tensors)} pending tensors")
                await flush_batch()
                await try_emit_completed_pages()

                for page_flight in pages_in_flight.values():
                    logits_list = [
                        page_flight.line_logits.get(i, LineLogits(
                            logits=np.zeros((1, DEFAULT_VOCAB_SIZE), dtype=np.float32),
                            content_width=1,
                            left_pad_width=0,
                            keep_indices=None,
                        ))
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
                logger.info(
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
                from .data_structures import PendingTensor
                pending_tensors.append(PendingTensor(
                    page_idx=processed.page_idx,
                    line_idx=line_idx,
                    tensor=line.tensor,
                    content_width=line.content_width,
                    left_pad_width=line.left_pad_width,
                ))

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
