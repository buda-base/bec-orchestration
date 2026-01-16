"""
TileBatcher: Prepares batches of tiled frames for GPU inference.

This component sits between Decoder/PostProcessor and LDInferenceRunner,
handling binarization, tiling, and batch assembly. Similar to PyTorch
DataLoader's worker + collate_fn role.
"""

import asyncio
import logging
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from .types_common import DecodedFrame, PipelineError, EndOfStream, TiledBatch
from .img_helpers import adaptive_binarize

logger = logging.getLogger(__name__)
# Separate logger for timing/performance logs (ERROR by default)
timings_logger = logging.getLogger("bec_timings")


# -----------------------------------------------------------------------------
# Tiling functions (from utils_alt.py)
# -----------------------------------------------------------------------------

def pad_to_multiple(img: torch.Tensor, patch_size: int = 512, value: float = 255.0) -> Tuple[torch.Tensor, int, int]:
    """
    Pad image to make dimensions divisible by patch_size.
    
    Args:
        img: [C, H, W] tensor
        patch_size: tile size
        value: padding value
    
    Returns:
        (padded_img, pad_w, pad_h)
    """
    _, H, W = img.shape
    pad_h = (patch_size - H % patch_size) % patch_size
    pad_w = (patch_size - W % patch_size) % patch_size
    # F.pad order: (left, right, top, bottom)
    img = F.pad(img, (0, pad_w, 0, pad_h), value=value)
    return img, pad_w, pad_h


def tile_image(img: torch.Tensor, patch_size: int = 512) -> Tuple[torch.Tensor, int, int]:
    """
    Tile image using unfold (no overlap).
    
    Args:
        img: [C, H, W] tensor (H, W must be divisible by patch_size)
    
    Returns:
        (tiles, x_steps, y_steps) where tiles is [N, C, patch_size, patch_size]
    """
    C, H, W = img.shape
    y_steps = H // patch_size
    x_steps = W // patch_size
    
    tiles = img.unfold(1, patch_size, patch_size).unfold(2, patch_size, patch_size)
    # tiles shape: [C, y_steps, x_steps, patch_size, patch_size]
    tiles = tiles.permute(1, 2, 0, 3, 4).contiguous()
    # tiles shape: [y_steps, x_steps, C, patch_size, patch_size]
    tiles = tiles.view(-1, C, patch_size, patch_size)
    # tiles shape: [N, C, patch_size, patch_size] where N = y_steps * x_steps
    
    return tiles, x_steps, y_steps


# -----------------------------------------------------------------------------
# Precision helpers
# -----------------------------------------------------------------------------

def get_tile_dtype(precision: str) -> torch.dtype:
    """
    Get torch dtype from precision string.
    
    Args:
        precision: "fp32", "fp16", "bf16", or "auto"
    
    Returns:
        torch.dtype
    """
    if precision == "fp16":
        return torch.float16
    elif precision == "bf16":
        return torch.bfloat16
    elif precision == "auto":
        # Auto: use fp16 if CUDA available, else fp32
        if torch.cuda.is_available():
            return torch.float16
        return torch.float32
    else:  # fp32 or default
        return torch.float32


# -----------------------------------------------------------------------------
# Synchronous tiling function (can run in thread pool)
# -----------------------------------------------------------------------------

def tile_frame_sync(
    gray: np.ndarray,
    patch_size: int,
    dtype: torch.dtype = torch.float32,
    pin_memory: bool = True,
) -> Tuple[torch.Tensor, int, int, int, int]:
    """
    Tile a grayscale frame.
    
    Args:
        gray: [H, W] uint8 numpy array
        patch_size: tile size
        dtype: torch dtype for output tiles (fp32, fp16, bf16)
        pin_memory: if True and CUDA available, pin the output tensor for faster H2D transfer
    
    Returns:
        (tiles, x_steps, y_steps, pad_x, pad_y)
        - tiles: [N, 3, patch_size, patch_size] tensor in specified dtype (on CPU, optionally pinned)
    """
    # Convert to torch [1, H, W] and normalize in one step
    # Using the target dtype directly if possible to avoid extra conversion
    img = torch.from_numpy(gray).unsqueeze(0).to(dtype).div_(255.0)
    
    # Pad to multiple of patch_size (white background = 1.0 after normalization)
    img, pad_x, pad_y = pad_to_multiple(img, patch_size, value=1.0)
    
    # Tile the image using unfold (creates views, very fast)
    tiles, x_steps, y_steps = tile_image(img, patch_size)
    # tiles shape: [N, 1, patch_size, patch_size]
    
    # Expand grayscale to 3 channels using repeat (single operation)
    # repeat() allocates new memory and copies, but is faster than expand+contiguous
    tiles = tiles.repeat(1, 3, 1, 1)
    
    # Pin memory for faster async GPU transfer (requires CUDA)
    # pin_memory() returns a new tensor backed by page-locked memory
    if pin_memory and torch.cuda.is_available():
        tiles = tiles.pin_memory()
    
    return tiles, x_steps, y_steps, pad_x, pad_y


# -----------------------------------------------------------------------------
# Main class
# -----------------------------------------------------------------------------

class TileBatcher:
    """
    Tiles frames and batches them for GPU inference.
    
    Inputs (load-balanced, pass-2 has priority):
      - q_from_decoder: DecodedFrame (pass-1)
      - q_from_postprocessor: DecodedFrame (pass-2)
    
    Output:
      - q_to_inference: TiledBatch (short queue for GPU)
    """

    def __init__(
        self,
        cfg,
        q_from_decoder: asyncio.Queue,
        q_from_postprocessor: asyncio.Queue,
        q_to_inference: asyncio.Queue,
    ):
        self.cfg = cfg
        self.q_from_decoder = q_from_decoder
        self.q_from_postprocessor = q_from_postprocessor
        self.q_to_inference = q_to_inference

        # Configuration
        self.batch_size: int = getattr(cfg, "batch_size", 8)
        self.batch_timeout_s: float = cfg.batch_timeout_ms / 1000.0
        self.patch_size: int = getattr(cfg, "patch_size", 512)
        self.tile_workers: int = getattr(cfg, "tile_workers", 4)
        
        # Tile precision (fp16/bf16 saves ~50% memory)
        precision = getattr(cfg, "precision", "fp32")
        self.tile_dtype: torch.dtype = get_tile_dtype(precision)
        
        # Pin memory for faster H2D transfer
        self.pin_tile_memory: bool = getattr(cfg, "pin_tile_memory", True)

        # State tracking
        self._decoder_done = False
        self._postprocessor_done = False
        self._pass1_eos_sent = False  # Track if we've sent pass-1 EOS

        # Thread pool for parallel tiling (optional, can be disabled)
        self._use_parallel_tiling = getattr(cfg, "parallel_tiling", True)
        if self._use_parallel_tiling:
            self._tile_executor = ThreadPoolExecutor(max_workers=self.tile_workers)
            logger.info(
                f"[TileBatcher] Initialized with parallel tiling ({self.tile_workers} workers), "
                f"batch_size={self.batch_size}, patch_size={self.patch_size}, dtype={self.tile_dtype}, "
                f"max_buffer={self._max_buffer_size}"
            )
        else:
            self._tile_executor = None
            logger.info(
                f"[TileBatcher] Initialized with sequential tiling, "
                f"batch_size={self.batch_size}, patch_size={self.patch_size}, dtype={self.tile_dtype}, "
                f"max_buffer={self._max_buffer_size}"
            )

        # Buffer for accumulating tiled frames before emitting a batch
        # Each entry: dict with tiles, metadata, etc.
        self._buffer: List[Dict[str, Any]] = []
        
        # Pending tiling tasks (for parallel tiling)
        self._pending_tiles: List[asyncio.Task] = []
        
        # Buffer size limit to prevent memory accumulation
        # Each buffered frame holds ~25MB (tiles tensor) + metadata
        # Set to 0 to disable limit
        self._max_buffer_size: int = getattr(cfg, "max_tilebatcher_buffer", 64)

    # -------------------------------------------------------------------------
    # Error handling
    # -------------------------------------------------------------------------

    async def _emit_pipeline_error(
        self,
        *,
        internal_stage: str,
        exc: BaseException,
        lane_second_pass: bool,
        task: Any,
        source_etag: Optional[str],
        retryable: bool = False,
        attempt: int = 1,
    ) -> None:
        import logging
        logger = logging.getLogger(__name__)
        
        tb = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
        err = PipelineError(
            stage="TileBatcher",
            task=task,
            source_etag=source_etag,
            error_type=type(exc).__name__,
            message=f"[{internal_stage}] {exc}",
            traceback=tb,
            retryable=bool(retryable),
            attempt=int(attempt),
        )
        
        # Errors flow through to inference runner, which will route them
        try:
            await asyncio.wait_for(self.q_to_inference.put(err), timeout=5.0)
        except asyncio.TimeoutError:
            logger.critical(
                f"Failed to emit error: queue full. "
                f"Dropping error for {task.img_filename if task else 'unknown'}"
            )

    # -------------------------------------------------------------------------
    # Frame preprocessing
    # -------------------------------------------------------------------------

    def _preprocess_frame_sync(self, dec_frame: DecodedFrame) -> np.ndarray:
        """
        Preprocess a decoded frame: validate and optionally binarize.
        
        WARNING: This runs adaptive_binarize which is CPU-intensive!
        Should be called from thread pool, not main async loop.
        
        Returns grayscale uint8 numpy array [H, W].
        """
        gray = dec_frame.frame
        if not isinstance(gray, np.ndarray) or gray.ndim != 2 or gray.dtype != np.uint8:
            raise ValueError("DecodedFrame.frame must be a 2D numpy uint8 array (H, W)")

        # Optional binarization (CPU-intensive!)
        is_binary = bool(dec_frame.is_binary)
        if not is_binary:
            gray = adaptive_binarize(
                gray,
                block_size=self.cfg.binarize_block_size,
                c=self.cfg.binarize_c,
            )
        
        return gray

    async def _tile_frame_async(
        self,
        dec_frame: DecodedFrame,
        is_pass2: bool,
    ) -> Dict[str, Any]:
        """
        Binarize and tile a frame, running CPU-intensive work in thread pool.
        
        This combines preprocessing (binarization) and tiling into a single
        thread pool operation to avoid blocking the async event loop.
        """
        t0 = time.perf_counter()
        
        if self._use_parallel_tiling and self._tile_executor is not None:
            loop = asyncio.get_event_loop()
            # Run BOTH binarization and tiling in thread pool
            gray, tiles, x_steps, y_steps, pad_x, pad_y = await loop.run_in_executor(
                self._tile_executor,
                self._preprocess_and_tile_sync,
                dec_frame,
            )
        else:
            # Fallback: run synchronously (will block)
            gray = self._preprocess_frame_sync(dec_frame)
            tiles, x_steps, y_steps, pad_x, pad_y = tile_frame_sync(
                gray, self.patch_size, self.tile_dtype, self.pin_tile_memory
            )
        
        tile_time = time.perf_counter() - t0
        
        # Note: 'gray' (binarized image) is NOT stored here - it was only needed for tiling.
        # Downstream uses dec_frame.frame (original grayscale) for pass-2 transforms.
        return {
            "dec_frame": dec_frame,
            "second_pass": is_pass2,
            "tiles": tiles,
            "x_steps": x_steps,
            "y_steps": y_steps,
            "pad_x": pad_x,
            "pad_y": pad_y,
            "tile_time": tile_time,
        }
    
    def _preprocess_and_tile_sync(
        self,
        dec_frame: DecodedFrame,
    ) -> Tuple[np.ndarray, torch.Tensor, int, int, int, int]:
        """
        Combined binarization + tiling, designed to run in thread pool.
        
        Returns (gray, tiles, x_steps, y_steps, pad_x, pad_y).
        """
        # Step 1: Binarize (CPU-intensive)
        gray = self._preprocess_frame_sync(dec_frame)
        
        # Step 2: Tile (also CPU work + memory ops)
        tiles, x_steps, y_steps, pad_x, pad_y = tile_frame_sync(
            gray, self.patch_size, self.tile_dtype, self.pin_tile_memory
        )
        
        return gray, tiles, x_steps, y_steps, pad_x, pad_y

    def _collect_completed_tiles(self) -> Tuple[int, float]:
        """
        Check pending tile tasks and move completed ones to buffer.
        
        Returns:
            (n_completed, total_tile_time)
        """
        completed = 0
        total_time = 0.0
        still_pending = []
        
        for task in self._pending_tiles:
            if task.done():
                try:
                    entry = task.result()
                    self._buffer.append(entry)
                    completed += 1
                    total_time += entry.get("tile_time", 0.0)
                except Exception as e:
                    # Task failed - log error but continue
                    logger.error(f"[TileBatcher] Tile task failed: {e}")
            else:
                still_pending.append(task)
        
        self._pending_tiles = still_pending
        return completed, total_time

    # -------------------------------------------------------------------------
    # Batch preparation (multi_image_collate_fn logic)
    # -------------------------------------------------------------------------

    def _prepare_batch(self) -> TiledBatch:
        """
        Combine buffered tiled frames into a single TiledBatch.
        
        This is the collate_fn equivalent from the reference code.
        Respects batch_size (frames) and max_tiles_per_batch limits.
        """
        all_tiles = []
        tile_ranges = []
        metas = []
        offset = 0
        
        max_tiles = getattr(self.cfg, "max_tiles_per_batch", 80)
        frames_to_take = 0
        total_tiles = 0

        # First pass: count how many frames we can take
        for entry in self._buffer:
            n_tiles = entry["tiles"].shape[0]
            
            # Stop if adding this frame would exceed limits
            if frames_to_take >= self.batch_size:
                break
            if total_tiles + n_tiles > max_tiles and total_tiles > 0:
                break  # Don't exceed max_tiles (but always take at least 1)
            
            frames_to_take += 1
            total_tiles += n_tiles
        
        # Track batch composition (pass 1 vs pass 2)
        p1_count = 0
        p2_count = 0
        
        # Second pass: collect the frames
        for entry in self._buffer[:frames_to_take]:
            tiles = entry["tiles"]
            n_tiles = tiles.shape[0]
            
            tile_ranges.append((offset, offset + n_tiles))
            all_tiles.append(tiles)
            
            # Track pass composition
            if entry["second_pass"]:
                p2_count += 1
            else:
                p1_count += 1
            
            # Note: 'gray' is not stored in entry - downstream uses dec_frame.frame
            metas.append({
                "dec_frame": entry["dec_frame"],
                "second_pass": entry["second_pass"],
                "x_steps": entry["x_steps"],
                "y_steps": entry["y_steps"],
                "pad_x": entry["pad_x"],
                "pad_y": entry["pad_y"],
            })
            
            offset += n_tiles

        # Stack all tiles into single tensor
        all_tiles_tensor = torch.cat(all_tiles, dim=0)
        
        # Explicitly delete individual tiles tensors to help GC
        # (they've been concatenated into all_tiles_tensor)
        del all_tiles
        for entry in self._buffer[:frames_to_take]:
            if "tiles" in entry:
                del entry["tiles"]
        
        # Pin the concatenated tensor for faster H2D transfer
        # (torch.cat creates a new tensor, so we need to re-pin)
        if self.pin_tile_memory and torch.cuda.is_available() and not all_tiles_tensor.is_pinned():
            all_tiles_tensor = all_tiles_tensor.pin_memory()
        
        # Remove taken frames from buffer
        self._buffer = self._buffer[frames_to_take:]
        
        # Store composition for logging (will be used by caller)
        batch = TiledBatch(
            all_tiles=all_tiles_tensor,
            tile_ranges=tile_ranges,
            metas=metas,
        )
        # Attach composition info as attributes for logging
        batch._p1_count = p1_count
        batch._p2_count = p2_count
        
        return batch

    # -------------------------------------------------------------------------
    # Queue helpers
    # -------------------------------------------------------------------------

    def _try_get_nowait(self, q: asyncio.Queue):
        """Try to get an item from queue without waiting. Returns None if empty."""
        try:
            return q.get_nowait()
        except asyncio.QueueEmpty:
            return None
    
    async def _pop_one(self, q: asyncio.Queue, timeout_s: float):
        """Pop one item from queue with timeout. Returns None on timeout."""
        try:
            return await asyncio.wait_for(q.get(), timeout=timeout_s)
        except asyncio.TimeoutError:
            return None

    async def _maybe_emit_pass1_eos(self) -> None:
        """
        Emit pass-1 EOS when decoder lane is fully drained.
        
        This breaks the EOS cycle between TileBatcher and PostProcessor:
        - TileBatcher sends tiled_pass_1 EOS
        - LDInferenceRunner forwards as gpu_pass_1 EOS
        - PostProcessor receives gpu_pass_1 EOS, can finish pass-1 processing
        - PostProcessor sends transformed_pass_1 EOS back to TileBatcher
        - TileBatcher can now finish
        """
        if self._pass1_eos_sent:
            return
        if not self._decoder_done:
            return
        
        # Check if any frames in buffer are from pass-1
        has_pass1_frames = any(not entry["second_pass"] for entry in self._buffer)
        if has_pass1_frames:
            return

        await self.q_to_inference.put(
            EndOfStream(stream="tiled_pass_1", producer="TileBatcher")
        )
        self._pass1_eos_sent = True

    # -------------------------------------------------------------------------
    # Main run loop
    # -------------------------------------------------------------------------

    async def run(self) -> None:
        """
        Main loop: receive frames, tile them in parallel, batch, and emit.
        
        Key optimization: Fire off multiple tiling tasks in parallel instead
        of awaiting each one sequentially. This keeps the thread pool busy
        while we fetch more frames.
        
        Load balancing: prefer pass-2 (postprocessor) over pass-1 (decoder).
        """
        # Timing stats
        loop_count = 0
        total_wait_time = 0.0
        total_tile_time = 0.0
        total_batch_emit_time = 0.0
        frames_submitted = 0
        batches_emitted = 0
        
        # Pass composition stats (for diagnosing pass-2 overhead)
        total_p1_frames = 0
        total_p2_frames = 0
        p2_only_batches = 0  # Batches with only pass-2 frames (inefficient)
        
        # Max frames to have in-flight (tiling) at once
        max_inflight = self.tile_workers * 2  # 2x workers for good overlap
        
        # Warmup: wait for buffer to fill before emitting first batch
        # This ensures consistent batch sizes when upstream (S3) is bursty
        warmup_frames = getattr(self.cfg, "inference_warmup_frames", 0)
        warmup_complete = (warmup_frames == 0)  # Skip warmup if disabled
        
        try:
            while True:
                loop_start = time.perf_counter()
                loop_count += 1
                
                # --- Collect completed tiling tasks ---
                n_completed, tile_time = self._collect_completed_tiles()
                total_tile_time += tile_time
                
                # --- Check termination ---
                # Only terminate when both inputs are done AND no pending work
                all_done = self._decoder_done and self._postprocessor_done
                no_pending = len(self._pending_tiles) == 0
                
                if all_done and no_pending:
                    timings_logger.info(
                        f"[TileBatcher] DONE - loops={loop_count}, frames_submitted={frames_submitted}, "
                        f"batches={batches_emitted}, total_wait={total_wait_time:.2f}s, "
                        f"total_tile={total_tile_time:.2f}s, total_emit={total_batch_emit_time:.2f}s"
                    )
                    # Log pass composition summary if any pass-2 frames were processed
                    if total_p2_frames > 0:
                        timings_logger.info(
                            f"[TileBatcher] COMPOSITION: p1_frames={total_p1_frames}, p2_frames={total_p2_frames}, "
                            f"p2_only_batches={p2_only_batches} (potential inefficiency if >0)"
                        )
                    # Flush remaining buffer
                    if self._buffer:
                        batch = self._prepare_batch()
                        batches_emitted += 1
                        await self.q_to_inference.put(batch)
                        logger.info(
                            f"[TileBatcher] Emitted final batch #{batches_emitted}: "
                            f"{len(batch.metas)} frames, {batch.all_tiles.shape[0]} tiles"
                        )
                    
                    # Send pass-1 EOS if not already sent
                    if not self._pass1_eos_sent:
                        await self.q_to_inference.put(
                            EndOfStream(stream="tiled_pass_1", producer="TileBatcher")
                        )
                        self._pass1_eos_sent = True
                    
                    # Send final pass-2 EOS
                    await self.q_to_inference.put(
                        EndOfStream(stream="tiled_pass_2", producer="TileBatcher")
                    )
                    break

                # --- Warmup phase: wait for buffer to fill ---
                if not warmup_complete:
                    total_ready = len(self._buffer) + len(self._pending_tiles)
                    if total_ready >= warmup_frames or self._decoder_done:
                        warmup_complete = True
                        timings_logger.debug(
                            f"[TileBatcher] Warmup complete: {len(self._buffer)} frames ready, "
                            f"{len(self._pending_tiles)} pending"
                        )
                    else:
                        # Still warming up - don't emit, just collect more frames
                        # Continue to the fetch section below
                        pass
                
                # --- Calculate current tiles in buffer ---
                buffer_tiles = sum(entry["tiles"].shape[0] for entry in self._buffer) if self._buffer else 0
                max_tiles = getattr(self.cfg, "max_tiles_per_batch", 80)
                
                # --- Emit batch when full (only after warmup) ---
                # Emit when: batch_size frames reached OR max_tiles exceeded
                if warmup_complete and (len(self._buffer) >= self.batch_size or buffer_tiles >= max_tiles):
                    emit_start = time.perf_counter()
                    batch = self._prepare_batch()
                    n_frames = len(batch.metas)  # Actual frames in batch (after max_tiles limit)
                    n_tiles = batch.all_tiles.shape[0]
                    # Get batch composition (p1/p2)
                    p1_count = getattr(batch, '_p1_count', 0)
                    p2_count = getattr(batch, '_p2_count', 0)
                    
                    put_start = time.perf_counter()
                    await self.q_to_inference.put(batch)
                    put_time = time.perf_counter() - put_start
                    
                    emit_time = time.perf_counter() - emit_start
                    total_batch_emit_time += emit_time
                    batches_emitted += 1
                    
                    # Track composition totals
                    total_p1_frames += p1_count
                    total_p2_frames += p2_count
                    if p1_count == 0 and p2_count > 0:
                        p2_only_batches += 1
                    
                    # Only show composition if there are pass-2 frames
                    composition = f", p1={p1_count}/p2={p2_count}" if p2_count > 0 else ""
                    timings_logger.debug(
                        f"[TileBatcher] Emitted batch #{batches_emitted}: {n_frames} frames, {n_tiles} tiles{composition}, "
                        f"reason=full, prepare={emit_time-put_time:.3f}s, put_wait={put_time:.3f}s, "
                        f"pending={len(self._pending_tiles)}, buffer={len(self._buffer)}"
                    )

                # --- Fetch more frames if we have capacity ---
                # Keep fetching while we have room for more in-flight tasks
                # Also respect buffer size limit to prevent memory accumulation
                frames_fetched_this_loop = 0
                max_fetch_per_loop = max_inflight  # Don't spin forever
                
                # Check buffer capacity (buffer + pending tiles)
                total_in_flight = len(self._buffer) + len(self._pending_tiles)
                buffer_has_space = (self._max_buffer_size == 0 or total_in_flight < self._max_buffer_size)
                
                while (len(self._pending_tiles) < max_inflight and 
                       frames_fetched_this_loop < max_fetch_per_loop and
                       buffer_has_space):
                    
                    msg = None
                    is_pass2 = False
                    
                    # --- First try NON-BLOCKING gets from both queues ---
                    # This avoids wasting 25ms on empty queues
                    
                    # Prefer pass-2 (from PostProcessor)
                    if not self._postprocessor_done:
                        msg = self._try_get_nowait(self.q_from_postprocessor)
                        if msg is not None:
                            is_pass2 = True
                            if isinstance(msg, EndOfStream) and msg.stream == "transformed_pass_1":
                                self._postprocessor_done = True
                                logger.debug("[TileBatcher] Received transformed_pass_1 EOS")
                                msg = None
                            elif isinstance(msg, PipelineError):
                                await self.q_to_inference.put(msg)
                                msg = None

                    # Then pass-1 (from Decoder)
                    if msg is None and not self._decoder_done:
                        msg = self._try_get_nowait(self.q_from_decoder)
                        if msg is not None:
                            is_pass2 = False
                            if isinstance(msg, EndOfStream) and msg.stream == "decoded":
                                self._decoder_done = True
                                logger.debug("[TileBatcher] Received decoded EOS")
                                msg = None
                            elif isinstance(msg, PipelineError):
                                await self.q_to_inference.put(msg)
                                msg = None
                    
                    # --- If both queues empty, wait briefly on decoder queue ---
                    if msg is None and not self._decoder_done:
                        wait_start = time.perf_counter()
                        msg = await self._pop_one(self.q_from_decoder, self.batch_timeout_s)
                        wait_time = time.perf_counter() - wait_start
                        total_wait_time += wait_time
                        
                        if msg is not None:
                            is_pass2 = False
                            if isinstance(msg, EndOfStream) and msg.stream == "decoded":
                                self._decoder_done = True
                                logger.debug("[TileBatcher] Received decoded EOS")
                                msg = None
                            elif isinstance(msg, PipelineError):
                                await self.q_to_inference.put(msg)
                                msg = None
                    
                    if msg is None:
                        # No frame available, break out of fetch loop
                        break
                    
                    frames_fetched_this_loop += 1
                    
                    # --- Start tiling task (don't await!) ---
                    if isinstance(msg, DecodedFrame):
                        # Fire off async task that does BOTH binarization and tiling
                        # in thread pool (avoids blocking the event loop)
                        task = asyncio.create_task(
                            self._tile_frame_async(msg, is_pass2)
                        )
                        self._pending_tiles.append(task)
                        frames_submitted += 1
                        
                        # Update buffer capacity check for next iteration
                        total_in_flight = len(self._buffer) + len(self._pending_tiles)
                        buffer_has_space = (self._max_buffer_size == 0 or total_in_flight < self._max_buffer_size)

                # --- Timeout flush: emit partial batch if nothing is coming ---
                # Only flush if:
                # 1. Warmup is complete
                # 2. We have frames AND couldn't fetch any new ones AND no pending tiles
                # 3. AND either: postprocessor is done OR batch is at least 75% full
                # This prevents tiny batches when S3 fetches are slow/bursty
                # IMPORTANT: Don't force-flush just because decoder is done - pass 2 frames
                # may still be coming from PostProcessor transforms. Wait until postprocessor_done.
                if (warmup_complete and
                    frames_fetched_this_loop == 0 and 
                    self._buffer and 
                    len(self._pending_tiles) == 0):
                    
                    # Determine if we should flush or wait for more
                    frame_fullness = len(self._buffer) / self.batch_size
                    tile_fullness = buffer_tiles / max_tiles
                    min_flush_ratio = 0.75  # Only flush if at least 75% full
                    
                    # Check if buffer contains any pass 2 frames
                    has_pass2_in_buffer = any(entry["second_pass"] for entry in self._buffer)
                    
                    # After decoder done but before postprocessor done, be patient with pass 2 frames
                    # They're still trickling in from transforms - wait for a reasonable batch
                    if self._decoder_done and not self._postprocessor_done and has_pass2_in_buffer:
                        # Only flush if we have enough pass 2 frames OR waited too long
                        min_pass2_batch = max(4, self.batch_size // 4)  # At least 4 frames or 25% of batch
                        should_flush = (
                            len(self._buffer) >= min_pass2_batch or
                            frame_fullness >= min_flush_ratio or
                            tile_fullness >= min_flush_ratio
                        )
                    else:
                        should_flush = (
                            self._postprocessor_done or  # All pass 2 frames submitted, flush remaining
                            self._decoder_done or  # Decoder done and no pass 2 in buffer
                            frame_fullness >= min_flush_ratio or  # Nearly full by frames
                            tile_fullness >= min_flush_ratio  # Nearly full by tiles
                        )
                    
                    if should_flush:
                        emit_start = time.perf_counter()
                        batch = self._prepare_batch()
                        n_frames = len(batch.metas)  # Actual frames in batch
                        n_tiles = batch.all_tiles.shape[0]
                        # Get batch composition (p1/p2)
                        p1_count = getattr(batch, '_p1_count', 0)
                        p2_count = getattr(batch, '_p2_count', 0)
                        
                        put_start = time.perf_counter()
                        await self.q_to_inference.put(batch)
                        put_time = time.perf_counter() - put_start
                        
                        emit_time = time.perf_counter() - emit_start
                        total_batch_emit_time += emit_time
                        batches_emitted += 1
                        
                        # Track composition totals
                        total_p1_frames += p1_count
                        total_p2_frames += p2_count
                        if p1_count == 0 and p2_count > 0:
                            p2_only_batches += 1
                        
                        # Determine flush reason for logging
                        if self._postprocessor_done:
                            reason = "postprocessor_done"
                        elif self._decoder_done and has_pass2_in_buffer:
                            reason = "pass2_batch_ready"
                        elif self._decoder_done:
                            reason = "decoder_done"
                        else:
                            reason = "nearly_full"
                        composition = f", p1={p1_count}/p2={p2_count}" if p2_count > 0 else ""
                        timings_logger.debug(
                            f"[TileBatcher] Emitted batch #{batches_emitted}: {n_frames} frames, {n_tiles} tiles{composition}, "
                            f"reason={reason}, prepare={emit_time-put_time:.3f}s, put_wait={put_time:.3f}s"
                        )
                    else:
                        # Wait for more frames with blocking timeout (avoid busy-spin)
                        # This gives decoder/prefetcher time to provide more data
                        # But only if we have buffer capacity
                        total_in_flight = len(self._buffer) + len(self._pending_tiles)
                        if self._max_buffer_size > 0 and total_in_flight >= self._max_buffer_size:
                            # Buffer is full, don't fetch more - just yield
                            await asyncio.sleep(0.01)
                        else:
                            wait_start = time.perf_counter()
                            msg = await self._pop_one(self.q_from_decoder, self.batch_timeout_s * 4)  # 100ms wait
                            total_wait_time += time.perf_counter() - wait_start
                            
                            if msg is not None:
                                if isinstance(msg, EndOfStream) and msg.stream == "prefetched":
                                    self._decoder_done = True
                                elif isinstance(msg, PipelineError):
                                    await self.q_to_inference.put(msg)
                                elif isinstance(msg, DecodedFrame):
                                    # Fire off async task that does BOTH binarization and tiling
                                    task = asyncio.create_task(
                                        self._tile_frame_async(msg, False)
                                    )
                                    self._pending_tiles.append(task)
                                    frames_submitted += 1

                # --- Check if we should send pass-1 EOS ---
                await self._maybe_emit_pass1_eos()

                # Log progress periodically
                loop_time = time.perf_counter() - loop_start
                if loop_count % 100 == 0:
                    logger.debug(
                        f"[TileBatcher] Loop #{loop_count}: {loop_time*1000:.1f}ms, "
                        f"buffer={len(self._buffer)}, pending={len(self._pending_tiles)}, "
                        f"submitted={frames_submitted}"
                    )

                # Yield to event loop (allow pending tasks to progress)
                await asyncio.sleep(0)

        except asyncio.CancelledError:
            logger.info("TileBatcher cancelled, cleaning up...")
            # Cancel pending tasks
            for task in self._pending_tiles:
                task.cancel()
            raise
        except KeyboardInterrupt:
            logger.info("TileBatcher interrupted by user")
            raise
        finally:
            # Shutdown thread pool
            if self._tile_executor is not None:
                self._tile_executor.shutdown(wait=False)

