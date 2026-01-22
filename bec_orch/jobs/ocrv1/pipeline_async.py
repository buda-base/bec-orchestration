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
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

import cv2
import numpy as np
import numpy.typing as npt

if TYPE_CHECKING:
    pass

from .ctc_decoder import CTCDecoder, decode_logits_beam_search, init_worker_process
from .line import get_line_image
from .model import OCRModel
from .parquet_writer import StreamingParquetWriter

logger = logging.getLogger(__name__)


def _decode_page_for_process_pool(
    logits_list: list[tuple[npt.NDArray, int]],
    vocab: list[str],
    input_width: int,
) -> list[str]:
    """
    Module-level decode function for ProcessPoolExecutor.

    This function can be pickled and sent to worker processes.
    """
    texts = []
    vocab_size = len(vocab)

    for logits, orig_w in logits_list:
        # Crop logits
        time_axis = 1 if logits.shape[0] == vocab_size else 0
        total_timesteps = logits.shape[time_axis]
        crop_timesteps = max(1, int(total_timesteps * orig_w / input_width))

        if time_axis == 1:
            cropped = logits[:, :crop_timesteps]
        else:
            cropped = logits[:crop_timesteps, :]

        # Decode using module-level function
        text = decode_logits_beam_search(cropped, vocab)
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
    line_tensors: list[tuple[npt.NDArray, int]]  # (tensor, original_width)
    error: Optional[str] = None


@dataclass
class InferredPage:
    page_idx: int
    filename: str
    source_etag: str
    logits_list: list[tuple[npt.NDArray, int]]  # (logits, original_width)
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

        # Bounded queues for backpressure
        self.q_fetched: asyncio.Queue = asyncio.Queue(maxsize=64)
        self.q_processed: asyncio.Queue = asyncio.Queue(maxsize=32)
        self.q_inferred: asyncio.Queue = asyncio.Queue(maxsize=32)
        self.q_results: asyncio.Queue = asyncio.Queue(maxsize=64)

        # Thread pool for image processing (cv2 releases GIL)
        self._image_executor = ThreadPoolExecutor(max_workers=image_processor_workers, thread_name_prefix="img")
        # Process pool for CTC decoding (bypasses GIL for true parallelism)
        # Use initializer to build decoder once per worker process
        # maxtasksperchild=50 restarts workers periodically to free memory
        self._ctc_executor = ProcessPoolExecutor(
            max_workers=ctc_workers,
            initializer=init_worker_process,
            initargs=(ctc_decoder.ctc_vocab,),
            max_tasks_per_child=50,
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
        start_time = time.perf_counter()
        ld_row = fetched.ld_row

        # Decode image
        img_array = np.frombuffer(fetched.file_bytes, dtype=np.uint8)
        image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("Failed to decode image")

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Apply transforms
        rotation_angle = ld_row.get("rotation_angle", 0.0)
        tps_points = ld_row.get("tps_points")
        tps_alpha = ld_row.get("tps_alpha", 0.5)

        if rotation_angle and abs(rotation_angle) > 0.01:
            h, w = image.shape[:2]
            center = (w / 2, h / 2)
            M = cv2.getRotationMatrix2D(center, rotation_angle, 1.0)
            image = cv2.warpAffine(image, M, (w, h), borderMode=cv2.BORDER_REPLICATE)

        if tps_points:
            image = self._apply_tps(image, tps_points, tps_alpha)

        # Extract lines
        contours = ld_row.get("contours", [])
        if not contours:
            return ProcessedPage(
                page_idx=fetched.page_idx,
                filename=fetched.filename,
                source_etag=fetched.source_etag,
                line_tensors=[],
            )

        line_tensors = []
        mask_buffer = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
        current_k = 1.7

        for contour_points in contours:
            line_img, current_k = self._extract_line(image, contour_points, current_k, mask_buffer)
            if line_img is None:
                line_tensors.append((np.zeros((1, self.input_height, self.input_width), dtype=np.float32), 1))
                continue

            original_width = line_img.shape[1]
            tensor = self._preprocess_line(line_img)
            line_tensors.append((tensor, original_width))

        proc_ms = (time.perf_counter() - start_time) * 1000
        logger.info(f"[ImageProcessor] Page {fetched.page_idx} processed {len(line_tensors)} lines in {proc_ms:.0f}ms")

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

    def _preprocess_line(self, image: npt.NDArray) -> npt.NDArray:
        """Preprocess line image to tensor."""
        # Pad to model size
        h, w = image.shape[:2]
        target_h = self.input_height
        target_w = self.input_width
        aspect = w / h

        if aspect > (target_w / target_h):
            new_w = target_w
            new_h = max(1, int(target_w / aspect))
        else:
            new_h = target_h
            new_w = max(1, int(target_h * aspect))

        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

        padded = np.ones((target_h, target_w, 3), dtype=np.uint8) * 255
        y_offset = (target_h - new_h) // 2
        x_offset = 0
        padded[y_offset : y_offset + new_h, x_offset : x_offset + new_w] = resized

        # Binarize
        gray = cv2.cvtColor(padded, cv2.COLOR_RGB2GRAY)
        binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 15)

        # Normalize
        tensor = binary.reshape((1, target_h, target_w)).astype(np.float32)
        tensor = (tensor / 127.5) - 1.0

        return tensor

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
        # page_idx -> (ProcessedPage, expected_lines, {line_idx: (logits, orig_w)})
        pages_in_flight: dict[int, tuple[ProcessedPage, int, dict[int, tuple]]] = {}

        # Batch of tensors waiting to be processed
        # (page_idx, line_idx, tensor, orig_w)
        pending_tensors: list[tuple[int, int, npt.NDArray, int]] = []

        lines_processed = 0
        pages_emitted = 0
        pages_received = 0

        async def flush_batch():
            nonlocal lines_processed
            if not pending_tensors:
                return

            batch_size = len(pending_tensors)
            start_time = time.perf_counter()

            # Stack tensors
            tensors = np.concatenate([t[2] for t in pending_tensors], axis=0)

            # Run inference
            loop = asyncio.get_event_loop()
            batch_logits = await loop.run_in_executor(None, self.ocr_model.predict, tensors)

            # Handle single item
            if len(pending_tensors) == 1:
                batch_logits = [batch_logits]

            # Distribute results back to pages
            for (page_idx, line_idx, _, orig_w), logits in zip(pending_tensors, batch_logits):
                if page_idx in pages_in_flight:
                    _, _, logits_dict = pages_in_flight[page_idx]
                    logits_dict[line_idx] = (logits, orig_w)
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
                        logits_list.append((np.zeros((1, 84), dtype=np.float32), 1))

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
                            logits_list.append((np.zeros((1, 84), dtype=np.float32), 1))

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
            for line_idx, (tensor, orig_w) in enumerate(processed.line_tensors):
                pending_tensors.append((processed.page_idx, line_idx, tensor, orig_w))

                if len(pending_tensors) >= self.gpu_batch_size:
                    await flush_batch()
                    await try_emit_completed_pages()

        logger.info(f"[GPUInference] Done, processed {lines_processed} lines, emitted {pages_emitted} pages")

    async def _ctc_decoder_stage(self) -> None:
        """Decode logits to text concurrently using semaphore for backpressure."""
        logger.info(f"[CTCDecoder] Starting with {self.ctc_workers} concurrent workers")
        loop = asyncio.get_event_loop()

        sem = asyncio.Semaphore(self.ctc_workers)
        pending_tasks: set[asyncio.Task] = set()
        pages_received = 0

        async def decode_one(inferred: InferredPage) -> None:
            async with sem:
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

                try:
                    # Use module-level function for ProcessPoolExecutor
                    num_lines = len(inferred.logits_list)
                    start_time = time.perf_counter()

                    texts = await loop.run_in_executor(
                        self._ctc_executor,
                        _decode_page_for_process_pool,
                        inferred.logits_list,
                        self.ctc_decoder.ctc_vocab,
                        self.input_width,
                    )

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
