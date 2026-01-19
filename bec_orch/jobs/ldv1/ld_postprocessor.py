
import asyncio
import traceback
from .types_common import DecodedFrame, Record, InferredFrame, PipelineError, EndOfStream, trace_frame, trace_frame_error
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
from .img_helpers import apply_transform_1
from .debug_helpers import should_debug_image, save_debug_image, save_debug_contours

import cv2
import numpy as np
import numpy.typing as npt

# -----------------------------
# Packed-bits mask transport (GPU->CPU bandwidth optimization)
# -----------------------------
def _decode_line_mask(line_mask):
    """Accepts either:
    - a regular uint8 HxW numpy mask in {0,255}
    - or a tagged tuple ("packedbits", packed_u8[H,W8], H, W, pad)
      where packed bits are little-endian along width.

    Returns:
        uint8 HxW mask in {0,255}
    """
    if isinstance(line_mask, tuple) and len(line_mask) == 5 and line_mask[0] == "packedbits":
        _tag, packed, h, w, pad = line_mask
        
        # Validate input type
        if not isinstance(packed, np.ndarray):
            raise TypeError(
                f"Expected numpy.ndarray for packed bits, got type: {type(packed)}. "
                f"If you received a torch.Tensor, it should be converted to numpy in the GPU batcher."
            )
        
        # Validate dtype before unpacking (np.unpackbits requires uint8)
        if packed.dtype != np.uint8:
            raise TypeError(
                f"Expected uint8 dtype for packed bits, got dtype: {packed.dtype}, "
                f"shape: {packed.shape}, type: {type(packed)}. "
                f"Array should be converted to uint8 in the GPU batcher."
            )
        
        # Validate shape
        if packed.ndim != 2:
            raise ValueError(
                f"Expected 2D packed bits array, got shape: {packed.shape}, ndim: {packed.ndim}, "
                f"dtype: {packed.dtype}, type: {type(packed)}"
            )
        
        # Unpack little-endian to match packing in GPU stage.
        try:
            unpacked01 = np.unpackbits(packed, axis=1, bitorder="little")
        except TypeError as e:
            raise TypeError(
                f"np.unpackbits failed. Array dtype: {packed.dtype}, shape: {packed.shape}, "
                f"ndim: {packed.ndim}, type: {type(packed)}, "
                f"is_contiguous: {packed.flags['C_CONTIGUOUS']}, "
                f"original error: {e}"
            ) from e
        
        if pad:
            unpacked01 = unpacked01[:, :w]
        # Convert {0,1} -> {0,255} for existing invariants.
        return (unpacked01.astype(np.uint8) * 255)
    return line_mask

class LDPostProcessor:
    """
    Consumes InferredFrame from GPU pass 1 and pass 2, decides rotation/TPS, routes.

    Inputs:
      - q_gpu_pass_1_to_post_processor:  InferredFrameMsg (gpu_pass_1)
      - q_gpu_pass_2_to_post_processor: InferredFrameMsg (gpu_pass_2)
    Outputs:
      - q_post_processor_to_gpu_pass_2: DecodedFrameMsg (decoded stream)
      - q_post_processor_to_writer: RecordMsg (record stream)

    Records are produced ONLY here (per your note).
    """

    def __init__(
        self,
        cfg,
        q_gpu_pass_1_to_post_processor: asyncio.Queue,
        q_gpu_pass_2_to_post_processor: asyncio.Queue,
        q_post_processor_to_gpu_pass_2: asyncio.Queue,
        q_post_processor_to_writer: asyncio.Queue,
    ):
        self.cfg = cfg
        self.q_first = q_gpu_pass_1_to_post_processor
        self.q_second = q_gpu_pass_2_to_post_processor
        self.q_post_processor_to_gpu_pass_2 = q_post_processor_to_gpu_pass_2
        self.q_post_processor_to_writer = q_post_processor_to_writer

        self._p1_done = False
        self._p2_done = False
        
        # Concurrent transform support: fire-and-forget transforms with tracking
        self._pending_transforms: List[asyncio.Task] = []
        self._max_concurrent_transforms = getattr(cfg, "max_concurrent_transforms", 8)
        self._transform_semaphore = asyncio.Semaphore(self._max_concurrent_transforms)
        
        # Timeout for waiting on pending transforms at EOS
        self._transform_gather_timeout_s = getattr(cfg, "transform_gather_timeout_s", 60.0)
        
        # Timeout for queue put operations
        self._queue_put_timeout_s = getattr(cfg, "queue_put_timeout_s", 30.0)
    
    async def _put_with_timeout(
        self,
        queue: asyncio.Queue,
        item: Any,
        queue_name: str,
        item_desc: str,
    ) -> bool:
        """
        Put item to queue with timeout. Returns True if successful.
        
        On timeout, logs error with queue state to help diagnose deadlocks.
        """
        import logging
        logger = logging.getLogger(__name__)
        try:
            await asyncio.wait_for(queue.put(item), timeout=self._queue_put_timeout_s)
            return True
        except asyncio.TimeoutError:
            logger.error(
                f"[PostProcessor] Timeout ({self._queue_put_timeout_s}s) putting {item_desc} to {queue_name} "
                f"(qsize={queue.qsize()}/{queue.maxsize}). Pipeline may be deadlocked."
            )
            return False
    
    def _handle_transform_exception(self, task: asyncio.Task) -> None:
        """Callback to log exceptions from fire-and-forget transform tasks."""
        import logging
        logger = logging.getLogger(__name__)
        if task.cancelled():
            return
        try:
            exc = task.exception()
            if exc is not None:
                logger.error(f"[PostProcessor] Transform task failed: {exc}", exc_info=exc)
        except asyncio.InvalidStateError:
            pass  # Task not done yet (shouldn't happen in done callback)

    async def _emit_pipeline_error(
        self,
        *,
        internal_stage: str,
        exc: BaseException,
        task: Any,
        source_etag: Optional[str],
        retryable: bool = False,
        attempt: int = 1,
    ) -> None:
        tb = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
        err = PipelineError(
            stage="LDPostProcessor",
            task=task,
            source_etag=source_etag,
            error_type=type(exc).__name__,
            message=f"[{internal_stage}] {exc}",
            traceback=tb,
            retryable=bool(retryable),
            attempt=int(attempt),
        )
        await self.q_post_processor_to_writer.put(err)

    def _try_get_nowait(self, q: asyncio.Queue):
        """Non-blocking queue get. Returns None if empty."""
        try:
            return q.get_nowait()
        except asyncio.QueueEmpty:
            return None

    async def _pop_one(self, q: asyncio.Queue, timeout_s: float):
        try:
            return await asyncio.wait_for(q.get(), timeout=timeout_s)
        except asyncio.TimeoutError:
            return None  # no item right now

    async def run(self):
        import time
        import logging
        logger = logging.getLogger(__name__)
        
        # Pass2 is cheap; prioritize it, but don't starve pass1 entirely.
        p2_budget = getattr(self.cfg, "reprocess_budget", 3)
        p1_budget = 16  # Process up to 16 frames per loop (batch-sized chunks)
        timeout_s = getattr(self.cfg, "controller_poll_ms", 5) / 1000.0
        
        # Timing stats
        run_start = time.perf_counter()
        p1_count = 0
        p2_count = 0
        total_p1_time = 0.0
        total_p2_time = 0.0
        
        # NEW: Detailed timing for diagnosing overhead
        total_p1_wait_time = 0.0  # Time waiting for pass 1 queue
        total_p2_wait_time = 0.0  # Time waiting for pass 2 queue
        total_idle_time = 0.0     # Time in sleep(0) / no work available
        pass2_submitted = 0       # Count of frames sent to pass 2
        transform_time = 0.0      # Time spent in apply_transform_1
        p1_eos_time = None        # When we received pass 1 EOS
        p2_eos_time = None        # When we received pass 2 EOS

        while True:
            loop_start = time.perf_counter()
            
            # Terminate only after both GPU streams have ended.
            if self._p1_done and self._p2_done:
                run_time = time.perf_counter() - run_start
                avg_p1 = total_p1_time / max(1, p1_count)
                avg_p2 = total_p2_time / max(1, p2_count)
                # Calculate time between EOS events
                eos_gap = (p2_eos_time - p1_eos_time) if (p1_eos_time and p2_eos_time) else 0.0
                logger.info(
                    f"[PostProcessor] DONE - p1={p1_count} ({total_p1_time:.2f}s, avg={avg_p1*1000:.1f}ms), "
                    f"p2={p2_count} ({total_p2_time:.2f}s, avg={avg_p2*1000:.1f}ms), "
                    f"run_time={run_time:.2f}s"
                )
                # Note: transform_time is 0 when using concurrent transforms (fire-and-forget)
                transform_info = f"transform={transform_time:.2f}s, " if transform_time > 0 else "transforms=concurrent, "
                logger.info(
                    f"[PostProcessor] TIMING BREAKDOWN: "
                    f"p1_wait={total_p1_wait_time:.2f}s, p2_wait={total_p2_wait_time:.2f}s, "
                    f"idle={total_idle_time:.2f}s, {transform_info}"
                    f"pass2_submitted={pass2_submitted}, eos_gap={eos_gap:.2f}s"
                )
                await self.q_post_processor_to_writer.put(EndOfStream(stream="record", producer="LDPostProcessor"))
                return

            took = False

            # --- Prefer gpu_pass_2 results (non-blocking check since often empty) ---
            if not self._p2_done:
                for _ in range(p2_budget):
                    wait_start = time.perf_counter()
                    msg = self._try_get_nowait(self.q_second)
                    # Note: _try_get_nowait doesn't actually wait, but track for consistency
                    if msg is None:
                        break
                    took = True

                    if isinstance(msg, EndOfStream) and msg.stream == "gpu_pass_2":
                        self._p2_done = True
                        p2_eos_time = time.perf_counter() - run_start
                        logger.info(f"[PostProcessor] Received gpu_pass_2 EOS at t={p2_eos_time:.2f}s")
                        
                        # CRITICAL: Drain any remaining pass-2 frames from the queue before breaking.
                        queue_size_p2 = self.q_second.qsize()
                        if queue_size_p2 > 0:
                            logger.info(f"[PostProcessor] Queue has {queue_size_p2} pass-2 items to drain after EOS")
                        
                        p2_drained = 0
                        while True:
                            remaining = self._try_get_nowait(self.q_second)
                            if remaining is None:
                                break
                            if isinstance(remaining, PipelineError):
                                await self.q_post_processor_to_writer.put(remaining)
                                continue
                            if isinstance(remaining, EndOfStream):
                                continue  # Ignore additional EOS
                            if isinstance(remaining, InferredFrame):
                                trace_frame("PostProcessor", "received_pass2", remaining.task.img_filename)
                                try:
                                    t0 = time.perf_counter()
                                    await self._finalize_record(remaining)
                                    trace_frame("PostProcessor", "finalized_pass2", remaining.task.img_filename)
                                    total_p2_time += time.perf_counter() - t0
                                    p2_count += 1
                                    p2_drained += 1
                                except Exception as e:
                                    trace_frame_error("PostProcessor", remaining.task.img_filename, f"finalize_pass2_drain: {e}")
                                    await self._emit_pipeline_error(
                                        internal_stage="run.finalize_pass2_drain",
                                        exc=e,
                                        task=remaining.task,
                                        source_etag=remaining.source_etag,
                                        retryable=False,
                                        attempt=1,
                                    )
                        
                        if p2_drained > 0:
                            logger.info(f"[PostProcessor] Drained {p2_drained} remaining pass-2 frames after EOS")
                        
                        break

                    if isinstance(msg, PipelineError):
                        await self.q_post_processor_to_writer.put(msg)
                        continue

                    frame: InferredFrame = msg
                    trace_frame("PostProcessor", "received_pass2", frame.task.img_filename)
                    try:
                        t0 = time.perf_counter()
                        await self._finalize_record(frame)  # cheap path
                        trace_frame("PostProcessor", "finalized_pass2", frame.task.img_filename)
                        total_p2_time += time.perf_counter() - t0
                        p2_count += 1
                    except Exception as e:
                        trace_frame_error("PostProcessor", frame.task.img_filename, f"finalize_pass2: {e}")
                        await self._emit_pipeline_error(
                            internal_stage="run.finalize_pass2",
                            exc=e,
                            task=frame.task,
                            source_etag=frame.source_etag,
                            retryable=False,
                            attempt=1,
                        )

            if took:
                continue

            # --- Then gpu_pass_1 results (non-blocking first, then blocking) ---
            if not self._p1_done:
                for i in range(p1_budget):
                    # First try non-blocking, then one blocking attempt at the end
                    wait_start = time.perf_counter()
                    if i < p1_budget - 1:
                        msg = self._try_get_nowait(self.q_first)
                    else:
                        # Last iteration: use blocking to avoid busy-spin
                        msg = await self._pop_one(self.q_first, timeout_s)
                        total_p1_wait_time += time.perf_counter() - wait_start
                    
                    if msg is None:
                        break
                    took = True

                    if isinstance(msg, EndOfStream) and msg.stream == "gpu_pass_1":
                        self._p1_done = True
                        p1_eos_time = time.perf_counter() - run_start
                        logger.info(f"[PostProcessor] Received gpu_pass_1 EOS at t={p1_eos_time:.2f}s")
                        
                        # Wait for all pending transforms to complete before sending EOS.
                        # CRITICAL: We must also drain pass-2 during this wait to prevent deadlock!
                        # The old blocking wait caused deadlock: InferenceRunner blocked on full q_pass2_to_post
                        # while PostProcessor was stuck waiting for transforms.
                        wait_start = time.perf_counter()
                        wait_timeout = self._transform_gather_timeout_s
                        
                        while True:
                            # FIRST: Drain any available pass-2 frames to prevent queue backup
                            for _ in range(p2_budget):
                                p2_msg = self._try_get_nowait(self.q_second)
                                if p2_msg is None:
                                    break
                                if isinstance(p2_msg, EndOfStream) and p2_msg.stream == "gpu_pass_2":
                                    # Pass-2 EOS arrived during wait - this is fine, we'll handle it in main loop
                                    self._p2_done = True
                                    p2_eos_time = time.perf_counter() - run_start
                                    logger.info(f"[PostProcessor] Received gpu_pass_2 EOS during transform wait at t={p2_eos_time:.2f}s")
                                    break
                                if isinstance(p2_msg, PipelineError):
                                    await self.q_post_processor_to_writer.put(p2_msg)
                                    continue
                                if isinstance(p2_msg, InferredFrame):
                                    try:
                                        t0 = time.perf_counter()
                                        await self._finalize_record(p2_msg)
                                        total_p2_time += time.perf_counter() - t0
                                        p2_count += 1
                                    except Exception as e:
                                        await self._emit_pipeline_error(
                                            internal_stage="run.finalize_pass2_during_wait",
                                            exc=e,
                                            task=p2_msg.task,
                                            source_etag=p2_msg.source_etag,
                                            retryable=False,
                                            attempt=1,
                                        )
                            
                            # Check if all transforms are done
                            pending = [t for t in self._pending_transforms if not t.done()]
                            if not pending:
                                logger.debug("[PostProcessor] All transforms completed")
                                break
                            
                            # Check timeout
                            elapsed = time.perf_counter() - wait_start
                            if elapsed > wait_timeout:
                                # Log which transforms are being cancelled
                                cancelled_count = len(pending)
                                logger.error(
                                    f"[PostProcessor] Timeout waiting for {cancelled_count} transforms after {elapsed:.1f}s. "
                                    f"Cancelling remaining - errors will be emitted via CancelledError handlers."
                                )
                                for t in pending:
                                    t.cancel()
                                # Give cancelled tasks a chance to run their CancelledError handlers
                                # which will emit PipelineError via put_nowait
                                results = await asyncio.gather(*pending, return_exceptions=True)
                                # Log summary of cancellation results
                                n_cancelled = sum(1 for r in results if isinstance(r, asyncio.CancelledError))
                                n_errors = sum(1 for r in results if isinstance(r, Exception) and not isinstance(r, asyncio.CancelledError))
                                logger.info(
                                    f"[PostProcessor] Transform cancellation complete: "
                                    f"{n_cancelled} cancelled, {n_errors} other errors"
                                )
                                break
                            
                            # Wait a bit before checking again (but not too long)
                            await asyncio.sleep(0.05)
                        
                        self._pending_transforms.clear()
                        
                        # CRITICAL: Drain any remaining pass-1 frames from the queue before breaking.
                        # The EOS might have arrived before we consumed all queued frames.
                        queue_size_before_drain = self.q_first.qsize()
                        if queue_size_before_drain > 0:
                            logger.info(f"[PostProcessor] Queue has {queue_size_before_drain} pass-1 items to drain after EOS")
                        
                        drained_count = 0
                        while True:
                            remaining_msg = self._try_get_nowait(self.q_first)
                            if remaining_msg is None:
                                break
                            if isinstance(remaining_msg, PipelineError):
                                await self.q_post_processor_to_writer.put(remaining_msg)
                                continue
                            if isinstance(remaining_msg, EndOfStream):
                                # Another EOS? Shouldn't happen, but ignore
                                continue
                            if isinstance(remaining_msg, InferredFrame):
                                trace_frame("PostProcessor", "received_pass1", remaining_msg.task.img_filename)
                                try:
                                    t0 = time.perf_counter()
                                    needs_pass2, frame_transform_time = await self._handle_pass1(remaining_msg)
                                    elapsed = time.perf_counter() - t0
                                    total_p1_time += elapsed
                                    p1_count += 1
                                    drained_count += 1
                                    if needs_pass2:
                                        trace_frame("PostProcessor", "needs_pass2", remaining_msg.task.img_filename)
                                        pass2_submitted += 1
                                        transform_time += frame_transform_time
                                    else:
                                        trace_frame("PostProcessor", "finalized_pass1", remaining_msg.task.img_filename)
                                except Exception as e:
                                    trace_frame_error("PostProcessor", remaining_msg.task.img_filename, f"handle_pass1_drain: {e}")
                                    await self._emit_pipeline_error(
                                        internal_stage="run.handle_pass1_drain",
                                        exc=e,
                                        task=remaining_msg.task,
                                        source_etag=remaining_msg.source_etag,
                                        retryable=False,
                                        attempt=1,
                                    )
                        
                        if drained_count > 0:
                            logger.info(f"[PostProcessor] Drained {drained_count} remaining pass-1 frames after EOS")
                        
                        # Now wait for any NEW transforms we just created during draining
                        pending_after_drain = [t for t in self._pending_transforms if not t.done()]
                        if pending_after_drain:
                            logger.debug(f"[PostProcessor] Waiting for {len(pending_after_drain)} transforms created during drain")
                            try:
                                await asyncio.wait_for(
                                    asyncio.gather(*pending_after_drain, return_exceptions=True),
                                    timeout=self._transform_gather_timeout_s
                                )
                            except asyncio.TimeoutError:
                                logger.warning(f"[PostProcessor] Timeout waiting for {len(pending_after_drain)} drain transforms")
                                for t in pending_after_drain:
                                    if not t.done():
                                        t.cancel()
                                await asyncio.gather(*pending_after_drain, return_exceptions=True)
                        
                        self._pending_transforms.clear()
                        
                        # No more pass1 frames => no more reprocess frames will be generated.
                        await self.q_post_processor_to_gpu_pass_2.put(
                            EndOfStream(stream="transformed_pass_1", producer="LDPostProcessor")
                        )
                        break

                    if isinstance(msg, PipelineError):
                        await self.q_post_processor_to_writer.put(msg)
                        continue

                    frame: InferredFrame = msg
                    trace_frame("PostProcessor", "received_pass1", frame.task.img_filename)
                    try:
                        t0 = time.perf_counter()
                        needs_pass2, frame_transform_time = await self._handle_pass1(frame)
                        elapsed = time.perf_counter() - t0
                        total_p1_time += elapsed
                        p1_count += 1
                        if needs_pass2:
                            trace_frame("PostProcessor", "needs_pass2", frame.task.img_filename)
                            pass2_submitted += 1
                            transform_time += frame_transform_time
                        else:
                            trace_frame("PostProcessor", "finalized_pass1", frame.task.img_filename)
                    except Exception as e:
                        trace_frame_error("PostProcessor", frame.task.img_filename, f"handle_pass1: {e}")
                        await self._emit_pipeline_error(
                            internal_stage="run.handle_pass1",
                            exc=e,
                            task=frame.task,
                            source_etag=frame.source_etag,
                            retryable=False,
                            attempt=1,
                        )

            if not took:
                idle_start = time.perf_counter()
                await asyncio.sleep(0)
                total_idle_time += time.perf_counter() - idle_start

    async def _handle_pass1(self, inf_frame: InferredFrame) -> Tuple[bool, float]:
        """
        Handle a pass 1 frame: analyze contours, decide if pass 2 is needed.
        
        Returns:
            Tuple of (needs_pass2, transform_time_seconds):
            - needs_pass2: True if frame was sent to pass 2, False if finalized directly
            - transform_time_seconds: Time spent in apply_transform_1 (0.0 if no transform)
        """
        import time
        # Decode packed-bit masks if GPU stage sent them
        line_mask = _decode_line_mask(inf_frame.line_mask)
        # Check debug mode once and reuse
        debug_this_image = should_debug_image(self.cfg, inf_frame.task.img_filename)
        
        # Debug helper (fire-and-forget, errors won't affect pipeline)
        async def _save_debug_safe(func, *args):
            try:
                await func(*args)
            except Exception:
                pass  # Silently ignore debug save errors
        
        if debug_this_image:
            asyncio.create_task(_save_debug_safe(save_debug_image, self.cfg, inf_frame.task.img_filename, "10_pass1_line_mask", line_mask))
        
        # 1. detect contours
        contours = get_filtered_contours(line_mask)

        # Debug: save pass1 contours
        if debug_this_image:
            asyncio.create_task(_save_debug_safe(save_debug_contours, self.cfg, inf_frame.task.img_filename, "11_pass1_contours", inf_frame.frame, contours))

        # 2. rotation angle
        h, w = line_mask.shape
        rotation_angle = get_rotation_angle(
            contours, h, w,
            max_angle_deg=self.cfg.max_angle_deg,
            min_angle_deg=self.cfg.min_angle_deg,
        )
        if rotation_angle != 0.0:
            contours = rotate_contours(contours, rotation_angle, h, w)
            
            # Debug: save rotation image
            if debug_this_image:
                # Import private helper for rotation (same package, so it's fine)
                from .img_helpers import _apply_rotation_1
                rotated_frame = _apply_rotation_1(inf_frame.frame, rotation_angle)
                asyncio.create_task(_save_debug_safe(save_debug_image, self.cfg, inf_frame.task.img_filename, "12_rotation", rotated_frame))

        # 3. TPS points (may be None)
        tps_points = get_tps_points(
            contours, h, w,
            legacy_tps_detect=self.cfg.legacy_tps_detect,
            alpha=self.cfg.tps_alpha,
            add_corners=self.cfg.add_corners,
        )

        # 4. either finalize or enqueue decoded frame to reprocess
        # Skip pass-2 if: no TPS needed AND (no rotation OR small rotation < 3 deg)
        # For small rotations, we just rotate the contours (already done above) without re-running GPU
        skip_pass2_threshold = getattr(self.cfg, "skip_pass2_rotation_threshold", 3.0)
        small_rotation = abs(rotation_angle) < skip_pass2_threshold
        
        if tps_points is None and (rotation_angle == 0.0 or small_rotation):
            # scale contours to original image dimensions (inf_frame.orig_h, inf_frame.orig_w)
            contours = scale_contours(contours, line_mask.shape[0], line_mask.shape[1], inf_frame.orig_h, inf_frame.orig_w)
            contours_bboxes = get_contour_bboxes(contours)
            rec = Record(
                task=inf_frame.task,
                source_etag=inf_frame.source_etag,
                rotation_angle=rotation_angle if rotation_angle != 0.0 else None,
                tps_data=None,
                contours=contours,
                nb_contours=len(contours),
                contours_bboxes=contours_bboxes,
            )
            await self.q_post_processor_to_writer.put(rec)
            return (False, 0.0)  # Finalized directly, no pass 2, no transform time

        input_pts = output_pts = None
        alpha = None
        if tps_points is not None:
            input_pts, output_pts = tps_points
            alpha = self.cfg.tps_alpha

        # Fire-and-forget transform: don't wait for it to complete.
        # This allows PostProcessor to continue processing other frames while
        # transforms run concurrently in thread pool, preventing queue backpressure.
        # The semaphore limits concurrent transforms to avoid overwhelming the pool.
        task = asyncio.create_task(
            self._do_transform_and_enqueue(
                inf_frame, rotation_angle, input_pts, output_pts, alpha,
                debug_this_image, _save_debug_safe
            )
        )
        # Add callback to log any exceptions (prevents silent failures)
        task.add_done_callback(self._handle_transform_exception)
        self._pending_transforms.append(task)
        
        # Clean up completed tasks periodically (don't let the list grow unbounded)
        self._pending_transforms = [t for t in self._pending_transforms if not t.done()]
        
        return (True, 0.0)  # Return immediately, transform runs in background
    
    async def _do_transform_and_enqueue(
        self,
        inf_frame: InferredFrame,
        rotation_angle: float,
        input_pts: Optional[np.ndarray],
        output_pts: Optional[np.ndarray],
        alpha: Optional[float],
        debug_this_image: bool,
        _save_debug_safe,
    ) -> None:
        """
        Run transform in thread pool and enqueue result for pass 2.
        Uses semaphore to limit concurrent transforms.
        """
        try:
            async with self._transform_semaphore:
                loop = asyncio.get_event_loop()
                transformed_frame = await loop.run_in_executor(
                    None,  # Use default thread pool
                    apply_transform_1,
                    inf_frame.frame,
                    rotation_angle,
                    input_pts,
                    output_pts,
                    alpha,
                )
            
            # Debug: save TPS image (if TPS was applied)
            if debug_this_image and input_pts is not None:
                try:
                    from .img_helpers import _apply_rotation_1, _apply_tps_1
                    if rotation_angle != 0.0:
                        rotated_frame = _apply_rotation_1(inf_frame.frame, rotation_angle)
                        tps_frame = _apply_tps_1(rotated_frame, input_pts, output_pts, alpha)
                    else:
                        tps_frame = _apply_tps_1(inf_frame.frame, input_pts, output_pts, alpha)
                    asyncio.create_task(_save_debug_safe(save_debug_image, self.cfg, inf_frame.task.img_filename, "13_tps", tps_frame))
                except Exception:
                    pass  # Debug failures shouldn't affect pipeline
            
            trace_frame("PostProcessor", "transform_done", inf_frame.task.img_filename)
            success = await self._put_with_timeout(
                self.q_post_processor_to_gpu_pass_2,
                DecodedFrame(
                    task=inf_frame.task,
                    source_etag=inf_frame.source_etag,
                    frame=transformed_frame,
                    orig_w=inf_frame.orig_w,
                    orig_h=inf_frame.orig_h,
                    is_binary=False,
                    first_pass=False,
                    rotation_angle=rotation_angle,
                    tps_data=(input_pts, output_pts, alpha),
                ),
                "q_post_processor_to_gpu_pass_2",
                f"DecodedFrame({inf_frame.task.img_filename}, pass2)",
            )
            if not success:
                trace_frame_error("PostProcessor", inf_frame.task.img_filename, "transform_queue_timeout")
                # Could not enqueue for pass 2, emit error instead
                await self._emit_pipeline_error(
                    internal_stage="_do_transform_and_enqueue.queue_timeout",
                    exc=TimeoutError(f"Could not enqueue for pass 2 after {self._queue_put_timeout_s}s"),
                    task=inf_frame.task,
                    source_etag=inf_frame.source_etag,
                    retryable=False,
                    attempt=1,
                )
        except asyncio.CancelledError:
            # Task was cancelled (e.g., due to timeout during EOS handling).
            # CRITICAL: Emit error BEFORE re-raising so the frame is tracked, not silently lost!
            # 
            # NOTE: We use put_nowait to avoid async operations during cancellation,
            # which can be unreliable. If the queue is full, we fall back to logging.
            import logging
            logger = logging.getLogger(__name__)
            logger.warning(
                f"[PostProcessor] Transform cancelled for {inf_frame.task.img_filename} "
                "(likely timeout during EOS). Emitting error."
            )
            err = PipelineError(
                stage="LDPostProcessor",
                task=inf_frame.task,
                source_etag=inf_frame.source_etag,
                error_type="CancelledError",
                message=f"Transform cancelled (timeout) for {inf_frame.task.img_filename}",
                traceback=None,
                retryable=True,  # Retryable since it was just a timeout
                attempt=1,
            )
            try:
                # Use non-blocking put to avoid async issues during cancellation
                self.q_post_processor_to_writer.put_nowait(err)
            except asyncio.QueueFull:
                # Queue is full - log prominently so we know the frame was lost
                logger.error(
                    f"[PostProcessor] DROPPED: Could not emit error for cancelled transform "
                    f"(queue full): {inf_frame.task.img_filename}"
                )
            raise  # Still re-raise to let asyncio handle cancellation properly
        except Exception as e:
            # Transform failed - emit error to writer so this image is tracked
            await self._emit_pipeline_error(
                internal_stage="_do_transform_and_enqueue",
                exc=e,
                task=inf_frame.task,
                source_etag=inf_frame.source_etag,
                retryable=False,
                attempt=1,
            )

    async def _finalize_record(self, inf_frame: InferredFrame) -> None:
        # Decode packed-bit masks if GPU stage sent them
        line_mask = _decode_line_mask(inf_frame.line_mask)
        # Debug: save pass2 line mask (fire-and-forget, errors won't affect pipeline)
        async def _save_debug_safe(func, *args):
            try:
                await func(*args)
            except Exception:
                pass  # Silently ignore debug save errors
        
        if should_debug_image(self.cfg, inf_frame.task.img_filename):
            asyncio.create_task(_save_debug_safe(save_debug_image, self.cfg, inf_frame.task.img_filename, "20_pass2_line_mask", line_mask))
        
        # Cheap pass2 finalization: just contours
        contours = get_filtered_contours(line_mask)
        
        # Debug: save pass2 contours (on the transformed frame if available)
        if should_debug_image(self.cfg, inf_frame.task.img_filename):
            # Use the frame from inf_frame (which is the transformed frame after rotation/TPS)
            asyncio.create_task(_save_debug_safe(save_debug_contours, self.cfg, inf_frame.task.img_filename, "21_pass2_contours", inf_frame.frame, contours))
        
        # scale contours to original image dimensions (frame.orig_h, frame.orig_w)
        contours = scale_contours(contours, line_mask.shape[0], line_mask.shape[1], inf_frame.orig_h, inf_frame.orig_w)
        # scale tps_data to original image dimensions
        tps_data = inf_frame.tps_data
        if tps_data and tps_data[0] is not None and tps_data[1] is not None:
            # Ensure input and output points are numpy arrays before scaling
            input_pts = np.asarray(tps_data[0], dtype=np.float64) if not isinstance(tps_data[0], np.ndarray) else tps_data[0]
            output_pts = np.asarray(tps_data[1], dtype=np.float64) if not isinstance(tps_data[1], np.ndarray) else tps_data[1]
            scaled_tps_points = scale_tps_points(input_pts, output_pts, line_mask.shape[0], line_mask.shape[1], inf_frame.orig_h, inf_frame.orig_w)
            tps_data = (scaled_tps_points[0], scaled_tps_points[1], tps_data[2])
        contours_bboxes = get_contour_bboxes(contours)
        h, w = line_mask.shape[:2]
        rec = Record(
            task=inf_frame.task,
            source_etag=inf_frame.source_etag,
            rotation_angle=inf_frame.rotation_angle,
            tps_data=tps_data,
            contours=contours,
            nb_contours=len(contours),
            contours_bboxes=contours_bboxes,
        )
        await self.q_post_processor_to_writer.put(rec)


# -----------------------------
# Defaults (tweakable)
# -----------------------------
MAX_ANGLE_DEG_DEFAULT = 5.0
MIN_ANGLE_DEG_DEFAULT = 0.2

# Contour filtering defaults (for "line-ish" contours in a binary mask)
MIN_AREA_FRAC_DEFAULT = 0.001   # fraction of image area
MIN_W_FRAC_DEFAULT = 0.01       # fraction of image width
MIN_H_PX_DEFAULT = 10           # minimum bbox height in px

# TPS defaults
TPS_SLICE_WIDTH_DEFAULT = 40
TPS_ALPHA_DEFAULT = 0.5
TPS_ADD_CORNERS_DEFAULT = True

def scale_contours(
    contours: Iterable[np.ndarray],
    src_h: int,
    src_w: int,
    dst_h: int,
    dst_w: int,
) -> List[np.ndarray]:
    """
    Scales contours produced by cv2.findContours from a (src_h, src_w) frame
    into a (dst_h, dst_w) frame.

    Expected contour formats:
      - (N, 1, 2)  [OpenCV standard]
      - (N, 2)

    Coordinates are assumed to be (x, y).

    Returns:
        A list of scaled contours with the same shape as the input contours,
        dtype float32.
    """
    if src_h == dst_h and src_w == dst_w:
        return contours

    if src_h <= 0 or src_w <= 0 or dst_h <= 0 or dst_w <= 0:
        raise ValueError(
            f"Invalid dimensions: src=({src_h},{src_w}) dst=({dst_h},{dst_w})"
        )

    sx = dst_w / float(src_w)
    sy = dst_h / float(src_h)

    scaled_contours: List[np.ndarray] = []

    for contour in contours:
        if not isinstance(contour, np.ndarray):
            raise TypeError("Each contour must be a numpy array")

        # Work on a float copy to avoid integer truncation during scaling
        pts = contour.astype(np.float64, copy=True)

        if pts.ndim == 3 and pts.shape[1:] == (1, 2):
            # (N, 1, 2)
            pts[:, 0, 0] *= sx  # x
            pts[:, 0, 1] *= sy  # y

        elif pts.ndim == 2 and pts.shape[1] == 2:
            # (N, 2)
            pts[:, 0] *= sx  # x
            pts[:, 1] *= sy  # y

        else:
            raise ValueError(
                f"Unsupported contour shape {contour.shape}; "
                "expected (N,1,2) or (N,2)"
            )

        # Round and cast back to int32
        pts = np.round(pts).astype(np.int32)

        scaled_contours.append(pts)

    return scaled_contours

def scale_tps_points(
    tps_input_points: np.ndarray,
    tps_output_points: np.ndarray,
    src_h: int,
    src_w: int,
    dst_h: int,
    dst_w: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Same scaling as scale_contours, but for TPS point arrays.

    Assumptions:
      - tps_input_points and tps_output_points are float64 numpy arrays.
      - Shape is (N, 2) where each row is (x, y).
      - Returns new arrays (does not modify inputs).

    Returns:
        (scaled_input_points, scaled_output_points) as float64 arrays.
    """
    if src_h == dst_h and src_w == dst_w:
        return tps_input_points, tps_output_points

    if src_h <= 0 or src_w <= 0 or dst_h <= 0 or dst_w <= 0:
        raise ValueError(
            f"Invalid dimensions: src=({src_h},{src_w}) dst=({dst_h},{dst_w})"
        )

    if not isinstance(tps_input_points, np.ndarray) or not isinstance(
        tps_output_points, np.ndarray
    ):
        raise TypeError("tps_input_points and tps_output_points must be numpy arrays")

    sx = dst_w / float(src_w)
    sy = dst_h / float(src_h)

    scaled_in = tps_input_points.copy()
    scaled_out = tps_output_points.copy()

    scaled_in[:, 0] *= sx
    scaled_in[:, 1] *= sy

    scaled_out[:, 0] *= sx
    scaled_out[:, 1] *= sy

    return scaled_in, scaled_out



# -----------------------------
# Validation helpers
# -----------------------------
def _assert_mask_uint8_binary_0_255(line_mask: npt.NDArray[np.uint8]) -> None:
    if line_mask.dtype != np.uint8:
        raise TypeError(f"line_mask must be uint8, got {line_mask.dtype}")
    if line_mask.ndim != 2:
        raise ValueError(f"line_mask must be 2D (H,W), got shape={line_mask.shape}")

    # strict binary check (your invariant for line masks)
    u = np.unique(line_mask)
    if not (u.size <= 2 and set(map(int, u)).issubset({0, 255})):
        raise ValueError(f"line_mask must be binarized in {{0,255}}, got unique={u[:10]}")

def get_contour_bboxes(contours):
    """
    Returns axis-aligned (x, y, w, h) bboxes,
    computed from minAreaRect for robustness.
    """
    bboxes = []

    for cnt in contours:
        if cnt is None or len(cnt) < 3:
            continue

        rect = cv2.minAreaRect(cnt)
        box = cv2.boxPoints(rect)      # 4x2
        box = np.int32(np.round(box))

        x, y, w, h = cv2.boundingRect(box)
        bboxes.append((x, y, w, h))

    return bboxes

# -----------------------------
# Public API 1: contours
# -----------------------------
def get_filtered_contours(
    line_mask: npt.NDArray[np.uint8],
    *,
    min_area_frac: float = MIN_AREA_FRAC_DEFAULT,
    min_w_frac: float = MIN_W_FRAC_DEFAULT,
    min_h_px: int = MIN_H_PX_DEFAULT,
    retrieval_mode: int = cv2.RETR_LIST,
    approx_mode: int = cv2.CHAIN_APPROX_SIMPLE,
) -> List[npt.NDArray[np.int32]]:
    """
    Find contours in a uint8 binary mask in {0,255}, and filter to keep line-like shapes.
    """
    _assert_mask_uint8_binary_0_255(line_mask)

    contours, _ = cv2.findContours(line_mask, retrieval_mode, approx_mode)
    if not contours:
        return []

    h, w = line_mask.shape
    img_area = float(h * w)
    min_area = img_area * float(min_area_frac)
    min_w = float(w) * float(min_w_frac)

    out: List[npt.NDArray[np.int32]] = []
    for c in contours:
        area = cv2.contourArea(c)
        if area <= min_area:
            continue
        x, y, bw, bh = cv2.boundingRect(c)
        if bw < min_w or bh < int(min_h_px):
            continue
        out.append(c)

    return out

# -----------------------------
# Angle estimation helpers
# -----------------------------
def _rect_angle_to_skew_deg(angle_deg: float) -> float:
    """
    Normalize cv2.minAreaRect angle to a skew angle near 0 deg.
    Output roughly in (-45, 45].
    """
    a = float(angle_deg)

    # Handle [0, 90) convention
    if 0.0 <= a <= 90.0:
        if a > 45.0:
            a -= 90.0
        return a

    # Handle (-90, 0] convention
    if a < -45.0:
        a += 90.0
    return a


def _robust_centered_angle_deg(
    angles_deg: npt.NDArray[np.float32],
    weights: Optional[npt.NDArray[np.float32]] = None,
) -> float:
    """
    Robust aggregation: MAD-trim then weighted mean.
    """
    if angles_deg.size == 0:
        return 0.0

    med = float(np.median(angles_deg))
    mad = float(np.median(np.abs(angles_deg - med)))
    if mad == 0.0:
        if weights is None:
            return float(np.mean(angles_deg))
        wsum = float(np.sum(weights))
        return float(np.sum(angles_deg * weights) / wsum) if wsum > 0 else float(np.mean(angles_deg))

    keep = np.abs(angles_deg - med) <= (3.0 * mad)
    kept = angles_deg[keep]
    if kept.size == 0:
        return med

    if weights is None:
        return float(np.mean(kept))

    kept_w = weights[keep]
    wsum = float(np.sum(kept_w))
    return float(np.sum(kept * kept_w) / wsum) if wsum > 0 else float(np.mean(kept))


def _rotation_matrix(angle_deg: float, h: int, w: int) -> npt.NDArray[np.float32]:
    center = (w / 2.0, h / 2.0)
    return cv2.getRotationMatrix2D(center, float(angle_deg), 1.0).astype(np.float32)


# -----------------------------
# Public API 2: angle
# -----------------------------
def get_rotation_angle(
    contours: Sequence[npt.NDArray[np.int32]],
    h: int,
    w: int,
    *,
    max_angle_deg: float = MAX_ANGLE_DEG_DEFAULT,
    min_angle_deg: float = MIN_ANGLE_DEG_DEFAULT,
    use_area_weights: bool = True,
) -> float:
    """
    Returns the deskew rotation angle in degrees (positive = CCW).
    Rules:
      - If |angle| > max_angle_deg => raises ValueError
      - If |angle| < min_angle_deg => returns 0.0
    """
    if not contours:
        return 0.0

    angles: List[float] = []
    weights: List[float] = []

    for c in contours:
        rect = cv2.minAreaRect(c)
        (rw, rh) = rect[1]
        raw_angle = rect[2]
        skew = _rect_angle_to_skew_deg(raw_angle)

        if abs(skew) < 1e-6:
            continue
        if abs(skew) > float(max_angle_deg):
            continue

        angles.append(skew)
        if use_area_weights:
            weights.append(max(float(rw * rh), 1.0))

    if not angles:
        return 0.0

    a = np.asarray(angles, dtype=np.float32)
    wts = np.asarray(weights, dtype=np.float32) if (use_area_weights and len(weights) == len(angles)) else None
    angle = float(_robust_centered_angle_deg(a, wts))

    if abs(angle) > float(max_angle_deg):
        raise ValueError(f"Detected angle {angle:.3f}° exceeds max {max_angle_deg:.3f}°")

    if abs(angle) < float(min_angle_deg):
        return 0.0

    return angle


# -----------------------------
# Public API 3: rotate contours
# -----------------------------
def rotate_contours(
    contours: Sequence[npt.NDArray[np.int32]],
    angle_deg: float,
    h: int,
    w: int,
) -> List[npt.NDArray[np.int32]]:
    """
    Rotate contours using the same affine transform you should use for the image/mask.
    Output contours are int32 with rounding-to-nearest.
    """
    if not contours or abs(float(angle_deg)) < 1e-12:
        return [c.copy() for c in contours]

    M = _rotation_matrix(angle_deg, h, w)

    out: List[npt.NDArray[np.int32]] = []
    for c in contours:
        rc = cv2.transform(c, M)
        out.append(np.rint(rc).astype(np.int32))
    return out


# -----------------------------
# TPS: optimized detection using small local mask (matches legacy accuracy)
# -----------------------------
def _check_line_tps_optimized(
    contour: npt.NDArray[np.int32],
    h: int,
    w: int,
    *,
    slice_width: int,
) -> Tuple[bool, Optional[List[List[int]]], Optional[List[List[int]]], float]:
    """
    Optimized TPS detection that matches legacy accuracy but avoids:
    - Full (h,w) mask allocation (uses bbox-sized mask instead)
    - Multiple findContours calls (computes stats directly on mask)
    """
    x, y, bw, bh = cv2.boundingRect(contour)
    
    # Create small mask just for the bounding box (not full image)
    local_mask = np.zeros((bh, bw), dtype=np.uint8)
    
    # Shift contour to local coordinates
    local_contour = contour.copy()
    local_contour = local_contour.reshape(-1, 2)
    local_contour[:, 0] -= x
    local_contour[:, 1] -= y
    local_contour = local_contour.reshape(-1, 1, 2)
    
    cv2.drawContours(local_mask, [local_contour], contourIdx=0, color=255, thickness=-1)
    
    def clamp(a: int, lo: int, hi: int) -> int:
        return max(lo, min(hi, a))
    
    # 5 slice x-ranges (in global coordinates, then convert to local)
    slices_global = [
        (x, x + slice_width),
        (x + bw // 4 - slice_width, x + bw // 4),
        (x + bw // 2, x + bw // 2 + slice_width),
        (x + bw // 2 + bw // 4, x + bw // 2 + (bw // 4) + slice_width),
        (x + bw - slice_width, x + bw),
    ]
    slices_global = [(clamp(a, x, x + bw), clamp(b, x, x + bw)) for (a, b) in slices_global]
    
    centers_xy: List[Tuple[int, int]] = []
    thicknesses: List[int] = []
    
    for (gx0, gx1) in slices_global:
        # Convert to local mask coordinates
        lx0, lx1 = gx0 - x, gx1 - x
        if lx1 <= lx0:
            # Degenerate slice
            centers_xy.append((gx0 + slice_width // 2, y + bh // 2))
            thicknesses.append(bh)
            continue
        
        slice_col = local_mask[:, lx0:lx1]
        
        # Find rows with any white pixels in this slice
        row_has_content = np.any(slice_col > 0, axis=1)
        rows_with_content = np.where(row_has_content)[0]
        
        if len(rows_with_content) == 0:
            # No content in slice - use fallback
            centers_xy.append((gx0 + (lx1 - lx0) // 2, y + bh // 2))
            thicknesses.append(bh)
            continue
        
        # Compute center and thickness from the filled region
        y_min_local = int(rows_with_content[0])
        y_max_local = int(rows_with_content[-1])
        slice_height = y_max_local - y_min_local + 1
        
        # Center y in global coordinates
        cy_local = (y_min_local + y_max_local) // 2
        cy_global = y + cy_local
        
        # Center x: use middle of slice (like legacy does approximately)
        cx_global = gx0 + (lx1 - lx0) // 2
        
        centers_xy.append((cx_global, cy_global))
        thicknesses.append(slice_height)
    
    all_cy = [cy for (_, cy) in centers_xy]
    max_ydelta = float(max(all_cy) - min(all_cy))
    mean_th = float(np.mean(thicknesses))
    mean_cy = float(np.mean(all_cy))
    
    if max_ydelta > mean_th:
        target_y = int(round(mean_cy))
        input_pts = [[cy, cx] for (cx, cy) in centers_xy]
        output_pts = [[target_y, cx] for (cx, cy) in centers_xy]
        return True, input_pts, output_pts, max_ydelta
    
    return False, None, None, 0.0


# -----------------------------
# TPS: legacy mask-writing detection (fallback)
# -----------------------------
def _get_global_center_from_slice(slice_image_2d: npt.NDArray[np.uint8], start_x: int, bbox_y: int) -> Tuple[int, int, int]:
    contours, _ = cv2.findContours(slice_image_2d, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        cx = slice_image_2d.shape[1] // 2
        cy = slice_image_2d.shape[0] // 2
        bh = slice_image_2d.shape[0]
        return start_x + cx, bbox_y + cy, bh

    areas = [cv2.contourArea(c) for c in contours]
    biggest = contours[int(np.argmax(areas))]
    _, _, _, bh = cv2.boundingRect(biggest)
    (cx_f, cy_f), _, _ = cv2.minAreaRect(biggest)
    return start_x + int(cx_f), bbox_y + int(cy_f), int(bh)


def _check_line_tps_legacy(
    contour: npt.NDArray[np.int32],
    h: int,
    w: int,
    *,
    slice_width: int,
) -> Tuple[bool, Optional[List[List[int]]], Optional[List[List[int]]], float]:
    mask = np.zeros((h, w), dtype=np.uint8)
    x, y, bw, bh = cv2.boundingRect(contour)
    cv2.drawContours(mask, [contour], contourIdx=0, color=255, thickness=-1)

    def clamp(a: int, lo: int, hi: int) -> int:
        return max(lo, min(hi, a))

    # 5 slices
    s1x0, s1x1 = x, x + slice_width
    s2x0, s2x1 = x + bw // 4 - slice_width, x + bw // 4
    s3x0, s3x1 = x + bw // 2, x + bw // 2 + slice_width
    s4x0, s4x1 = x + bw // 2 + bw // 4, x + bw // 2 + (bw // 4) + slice_width
    s5x0, s5x1 = x + bw - slice_width, x + bw

    s1x0, s1x1 = clamp(s1x0, 0, w), clamp(s1x1, 0, w)
    s2x0, s2x1 = clamp(s2x0, 0, w), clamp(s2x1, 0, w)
    s3x0, s3x1 = clamp(s3x0, 0, w), clamp(s3x1, 0, w)
    s4x0, s4x1 = clamp(s4x0, 0, w), clamp(s4x1, 0, w)
    s5x0, s5x1 = clamp(s5x0, 0, w), clamp(s5x1, 0, w)

    sl1 = mask[y:y + bh, s1x0:s1x1]
    sl2 = mask[y:y + bh, s2x0:s2x1]
    sl3 = mask[y:y + bh, s3x0:s3x1]
    sl4 = mask[y:y + bh, s4x0:s4x1]
    sl5 = mask[y:y + bh, s5x0:s5x1]

    p1x, p1y, b1h = _get_global_center_from_slice(sl1, s1x0, y)
    p2x, p2y, b2h = _get_global_center_from_slice(sl2, s2x0, y)
    p3x, p3y, b3h = _get_global_center_from_slice(sl3, s3x0, y)
    p4x, p4y, b4h = _get_global_center_from_slice(sl4, s4x0, y)
    p5x, p5y, b5h = _get_global_center_from_slice(sl5, s5x0, y)

    all_bh = [b1h, b2h, b3h, b4h, b5h]
    all_cy = [p1y, p2y, p3y, p4y, p5y]

    max_ydelta = float(max(all_cy) - min(all_cy))
    mean_bh = float(np.mean(all_bh))
    mean_cy = float(np.mean(all_cy))

    if max_ydelta > mean_bh:
        target_y = int(round(mean_cy))
        input_pts = [[p1y, p1x], [p2y, p2x], [p3y, p3x], [p4y, p4x], [p5y, p5x]]      # [y,x]
        output_pts = [[target_y, p1x], [target_y, p2x], [target_y, p3x], [target_y, p4x], [target_y, p5x]]
        return True, input_pts, output_pts, max_ydelta

    return False, None, None, 0.0


def _get_global_tps_line_idx(line_data: List[Dict[str, Any]]) -> int:
    all_y_deltas = [float(ld["max_yd"]) if ld["tps"] else 0.0 for ld in line_data]
    mean_delta = float(np.mean(all_y_deltas))

    best_diff = float(max(all_y_deltas))
    best_y = None
    for yd in all_y_deltas:
        if yd > 0:
            diff = abs(mean_delta - yd)
            if diff < best_diff:
                best_diff = diff
                best_y = yd

    if best_y is None:
        return -1
    return int(all_y_deltas.index(best_y))


# -----------------------------
# Public API 4: TPS points (robust default, legacy optional)
# -----------------------------
def get_tps_points(
    contours: Sequence[npt.NDArray[np.int32]],
    h: int,
    w: int,
    *,
    legacy_tps_detect: bool = False,  # uses full-mask approach (slower but original behavior)
    slice_width: int = TPS_SLICE_WIDTH_DEFAULT,
    alpha: float = TPS_ALPHA_DEFAULT,
    add_corners: bool = True,
) -> Optional[Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]]:
    """
    Returns TPS control points (input_pts, output_pts) in float64 or None if no TPS correction needed.

    IMPORTANT:
      - Points are in [y, x] order
      - If add_corners=True (default), image corners are INCLUDED
        in input_pts / output_pts
    """
    if not contours:
        return None

    line_data: List[Dict[str, Any]] = []

    for cnt in contours:
        if legacy_tps_detect:
            tps_status, input_pts, output_pts, max_yd = _check_line_tps_legacy(
                cnt, h, w, slice_width=slice_width
            )
        else:
            tps_status, input_pts, output_pts, max_yd = _check_line_tps_optimized(
                cnt, h, w,
                slice_width=slice_width,
            )

        line_data.append(
            {
                "tps": tps_status,
                "input_pts": input_pts,
                "output_pts": output_pts,
                "max_yd": max_yd,
            }
        )

    if not any(ld["tps"] for ld in line_data):
        return None

    best_idx = _get_global_tps_line_idx(line_data)
    if best_idx < 0:
        return None

    input_pts = line_data[best_idx]["input_pts"]
    output_pts = line_data[best_idx]["output_pts"]
    if input_pts is None or output_pts is None:
        return None

    input_pts_np = np.asarray(input_pts, dtype=np.float64)
    output_pts_np = np.asarray(output_pts, dtype=np.float64)

    if add_corners:
        corners = np.array(
            [
                [0.0, 0.0],
                [0.0, float(w - 1)],
                [float(h - 1), 0.0],
                [float(h - 1), float(w - 1)],
            ],
            dtype=np.float64,
        )
        input_pts_np = np.ascontiguousarray(np.concatenate([input_pts_np, corners], axis=0))
        output_pts_np = np.ascontiguousarray(np.concatenate([output_pts_np, corners], axis=0))

    # points are [y,x], corners already included
    return (input_pts_np, output_pts_np)