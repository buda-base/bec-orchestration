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

        # SEPARATE buffers for pass-1 and pass-2 frames to prevent deadlock
        # When pass-2 output queue is full, pass-1 can still flow through independently
        # Each entry: dict with tiles, metadata, etc.
        self._buffer_pass1: List[Dict[str, Any]] = []
        self._buffer_pass2: List[Dict[str, Any]] = []
        
        # Pending tiling tasks (for parallel tiling)
        # Tasks are tracked with their pass type for routing to correct buffer
        self._pending_tiles: List[asyncio.Task] = []
        
        # Buffer size limit to prevent memory accumulation
        # Each buffered frame holds ~25MB (tiles tensor) + metadata
        # Set to 0 to disable limit (limit applies to EACH buffer separately)
        self._max_buffer_size: int = getattr(cfg, "max_tilebatcher_buffer", 64)

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

    def _collect_completed_tiles(self) -> Tuple[int, int, float]:
        """
        Check pending tile tasks and move completed ones to appropriate buffer.
        
        Routes completed tiles to _buffer_pass1 or _buffer_pass2 based on second_pass flag.
        This separation prevents deadlock by allowing pass-1 and pass-2 to flow independently.
        
        Returns:
            (n_completed_pass1, n_completed_pass2, total_tile_time)
        """
        completed_p1 = 0
        completed_p2 = 0
        total_time = 0.0
        still_pending = []
        
        for task in self._pending_tiles:
            if task.done():
                try:
                    entry = task.result()
                    # Route to appropriate buffer based on pass type
                    if entry["second_pass"]:
                        self._buffer_pass2.append(entry)
                        completed_p2 += 1
                    else:
                        self._buffer_pass1.append(entry)
                        completed_p1 += 1
                    total_time += entry.get("tile_time", 0.0)
                except Exception as e:
                    # Task failed - log error but continue
                    logger.error(f"[TileBatcher] Tile task failed: {e}")
            else:
                still_pending.append(task)
        
        self._pending_tiles = still_pending
        return completed_p1, completed_p2, total_time

    # -------------------------------------------------------------------------
    # Batch preparation (multi_image_collate_fn logic)
    # -------------------------------------------------------------------------

    def _prepare_batch_from_buffer(self, buffer: List[Dict[str, Any]], is_pass2: bool) -> Tuple[TiledBatch, int]:
        """
        Combine buffered tiled frames into a single TiledBatch.
        
        This is the collate_fn equivalent from the reference code.
        Respects batch_size (frames) and max_tiles_per_batch limits.
        
        IMPORTANT: Takes from a SINGLE buffer (pass-1 OR pass-2), never mixed.
        This prevents deadlock by ensuring pass-1 and pass-2 pipelines flow independently.
        
        Args:
            buffer: The buffer to take frames from (_buffer_pass1 or _buffer_pass2)
            is_pass2: Whether this is the pass-2 buffer (for logging)
        
        Returns:
            Tuple of (TiledBatch, frames_taken)
        """
        all_tiles = []
        tile_ranges = []
        metas = []
        offset = 0
        
        max_tiles = getattr(self.cfg, "max_tiles_per_batch", 80)
        frames_to_take = 0
        total_tiles = 0

        # First pass: count how many frames we can take
        for entry in buffer:
            n_tiles = entry["tiles"].shape[0]
            
            # Stop if adding this frame would exceed limits
            if frames_to_take >= self.batch_size:
                break
            if total_tiles + n_tiles > max_tiles and total_tiles > 0:
                break  # Don't exceed max_tiles (but always take at least 1)
            
            frames_to_take += 1
            total_tiles += n_tiles
        
        # Collect the frames
        for entry in buffer[:frames_to_take]:
            tiles = entry["tiles"]
            n_tiles = tiles.shape[0]
            
            tile_ranges.append((offset, offset + n_tiles))
            all_tiles.append(tiles)
            
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
        for entry in buffer[:frames_to_take]:
            if "tiles" in entry:
                del entry["tiles"]
        
        # Pin the concatenated tensor for faster H2D transfer
        # (torch.cat creates a new tensor, so we need to re-pin)
        if self.pin_tile_memory and torch.cuda.is_available() and not all_tiles_tensor.is_pinned():
            all_tiles_tensor = all_tiles_tensor.pin_memory()
        
        # Store composition for logging (will be used by caller)
        batch = TiledBatch(
            all_tiles=all_tiles_tensor,
            tile_ranges=tile_ranges,
            metas=metas,
        )
        # Attach composition info as attributes for logging
        # With separate buffers, batches are now homogeneous (all pass-1 or all pass-2)
        batch._p1_count = 0 if is_pass2 else frames_to_take
        batch._p2_count = frames_to_take if is_pass2 else 0
        
        return batch, frames_to_take
    
    def _remove_from_buffer(self, buffer: List[Dict[str, Any]], count: int, is_pass2: bool) -> None:
        """Remove taken frames from the appropriate buffer."""
        if is_pass2:
            del self._buffer_pass2[:count]
        else:
            del self._buffer_pass1[:count]

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

    def _has_pending_pass1_tiles(self) -> bool:
        """Check if there are any pending pass-1 tiles in the tiling queue."""
        for task in self._pending_tiles:
            if not task.done():
                # Can't know the pass type until done, be conservative
                # Actually we can check the task's coroutine args, but simpler to be safe
                continue
            try:
                entry = task.result()
                if not entry["second_pass"]:
                    return True
            except Exception:
                pass
        return False
    
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
        
        # Check if any frames in pass-1 buffer (now separate from pass-2)
        if self._buffer_pass1:
            return

        await self.q_to_inference.put(
            EndOfStream(stream="tiled_pass_1", producer="TileBatcher")
        )
        self._pass1_eos_sent = True

    # -------------------------------------------------------------------------
    # Main run loop
    # -------------------------------------------------------------------------

    def _get_buffer_tiles(self, buffer: List[Dict[str, Any]]) -> int:
        """Calculate total tiles in a buffer."""
        return sum(entry["tiles"].shape[0] for entry in buffer) if buffer else 0
    
    def _should_emit_from_buffer(
        self, 
        buffer: List[Dict[str, Any]], 
        max_tiles: int,
        min_flush_ratio: float = 0.75,
        force_flush: bool = False,
    ) -> bool:
        """
        Determine if we should emit a batch from the given buffer.
        
        Args:
            buffer: The buffer to check
            max_tiles: Maximum tiles per batch
            min_flush_ratio: Minimum fullness ratio for partial flush
            force_flush: If True, emit even partial batches
        
        Returns:
            True if we should emit a batch from this buffer
        """
        if not buffer:
            return False
        
        if force_flush:
            return True
        
        buffer_tiles = self._get_buffer_tiles(buffer)
        frame_fullness = len(buffer) / self.batch_size
        tile_fullness = buffer_tiles / max_tiles
        
        # Emit if buffer is at least batch_size or max_tiles
        if len(buffer) >= self.batch_size or buffer_tiles >= max_tiles:
            return True
        
        # Emit partial if at least 75% full
        if frame_fullness >= min_flush_ratio or tile_fullness >= min_flush_ratio:
            return True
        
        return False

    async def run(self) -> None:
        """
        Main loop: receive frames, tile them in parallel, batch, and emit.
        
        Key optimizations:
        1. Fire off multiple tiling tasks in parallel instead of awaiting each one sequentially
        2. SEPARATE BUFFERS for pass-1 and pass-2 to prevent deadlock
        3. Pass-2 has priority but pass-1 can always flow independently
        
        The two-buffer design breaks the circular dependency that causes deadlock:
        - When pass-2 output queue is full, pass-1 batches can still be emitted
        - Pass-1 and pass-2 flow through the GPU independently
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
        p1_batches = 0
        p2_batches = 0
        
        # Max frames to have in-flight (tiling) at once
        max_inflight = self.tile_workers * 2  # 2x workers for good overlap
        
        # Warmup: wait for buffer to fill before emitting first batch
        # This ensures consistent batch sizes when upstream (S3) is bursty
        warmup_frames = getattr(self.cfg, "inference_warmup_frames", 0)
        warmup_complete = (warmup_frames == 0)  # Skip warmup if disabled
        
        max_tiles = getattr(self.cfg, "max_tiles_per_batch", 80)
        
        try:
            while True:
                loop_start = time.perf_counter()
                loop_count += 1
                
                # --- Collect completed tiling tasks (routes to correct buffer) ---
                n_p1_completed, n_p2_completed, tile_time = self._collect_completed_tiles()
                total_tile_time += tile_time
                
                # --- Check termination ---
                # Only terminate when both inputs are done AND no pending work AND buffers empty
                all_done = self._decoder_done and self._postprocessor_done
                no_pending = len(self._pending_tiles) == 0
                buffers_empty = not self._buffer_pass1 and not self._buffer_pass2
                
                if all_done and no_pending and buffers_empty:
                    timings_logger.info(
                        f"[TileBatcher] DONE - loops={loop_count}, frames_submitted={frames_submitted}, "
                        f"batches={batches_emitted}, total_wait={total_wait_time:.2f}s, "
                        f"total_tile={total_tile_time:.2f}s, total_emit={total_batch_emit_time:.2f}s"
                    )
                    # Log pass composition summary
                    timings_logger.info(
                        f"[TileBatcher] COMPOSITION: p1_frames={total_p1_frames} ({p1_batches} batches), "
                        f"p2_frames={total_p2_frames} ({p2_batches} batches)"
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
                
                # --- Flush remaining buffers at termination ---
                if all_done and no_pending:
                    # Flush pass-1 buffer first (pass-2 might still be trickling)
                    if self._buffer_pass1:
                        emit_start = time.perf_counter()
                        batch, frames_taken = self._prepare_batch_from_buffer(self._buffer_pass1, is_pass2=False)
                        self._remove_from_buffer(self._buffer_pass1, frames_taken, is_pass2=False)
                        
                        await self.q_to_inference.put(batch)
                        batches_emitted += 1
                        p1_batches += 1
                        total_p1_frames += frames_taken
                        total_batch_emit_time += time.perf_counter() - emit_start
                        logger.info(
                            f"[TileBatcher] Emitted final pass-1 batch #{batches_emitted}: "
                            f"{len(batch.metas)} frames, {batch.all_tiles.shape[0]} tiles"
                        )
                    
                    # Then flush pass-2 buffer
                    if self._buffer_pass2:
                        emit_start = time.perf_counter()
                        batch, frames_taken = self._prepare_batch_from_buffer(self._buffer_pass2, is_pass2=True)
                        self._remove_from_buffer(self._buffer_pass2, frames_taken, is_pass2=True)
                        
                        await self.q_to_inference.put(batch)
                        batches_emitted += 1
                        p2_batches += 1
                        total_p2_frames += frames_taken
                        total_batch_emit_time += time.perf_counter() - emit_start
                        logger.info(
                            f"[TileBatcher] Emitted final pass-2 batch #{batches_emitted}: "
                            f"{len(batch.metas)} frames, {batch.all_tiles.shape[0]} tiles"
                        )
                    continue

                # --- Warmup phase: wait for pass-1 buffer to fill ---
                if not warmup_complete:
                    total_ready = len(self._buffer_pass1) + len(self._pending_tiles)
                    if total_ready >= warmup_frames or self._decoder_done:
                        warmup_complete = True
                        timings_logger.debug(
                            f"[TileBatcher] Warmup complete: {len(self._buffer_pass1)} pass-1 frames ready, "
                            f"{len(self._buffer_pass2)} pass-2 frames, {len(self._pending_tiles)} pending"
                        )
                
                # --- Emit batches with PRIORITY: pass-2 first, then pass-1 ---
                # This ensures pass-2 frames (which complete the image) get processed quickly
                # But pass-1 can still flow even when pass-2 output is backed up
                emitted_this_loop = False
                
                if warmup_complete:
                    # Try pass-2 first (higher priority, smaller volume)
                    if self._should_emit_from_buffer(self._buffer_pass2, max_tiles):
                        emit_start = time.perf_counter()
                        batch, frames_taken = self._prepare_batch_from_buffer(self._buffer_pass2, is_pass2=True)
                        self._remove_from_buffer(self._buffer_pass2, frames_taken, is_pass2=True)
                        
                        put_start = time.perf_counter()
                        await self.q_to_inference.put(batch)
                        put_time = time.perf_counter() - put_start
                        
                        emit_time = time.perf_counter() - emit_start
                        total_batch_emit_time += emit_time
                        batches_emitted += 1
                        p2_batches += 1
                        total_p2_frames += frames_taken
                        emitted_this_loop = True
                        
                        timings_logger.debug(
                            f"[TileBatcher] Emitted pass-2 batch #{batches_emitted}: {frames_taken} frames, "
                            f"{batch.all_tiles.shape[0]} tiles, prepare={emit_time-put_time:.3f}s, put={put_time:.3f}s"
                        )
                    
                    # Then try pass-1 (can flow independently of pass-2)
                    # This is the KEY deadlock fix: pass-1 doesn't wait for pass-2
                    elif self._should_emit_from_buffer(self._buffer_pass1, max_tiles):
                        emit_start = time.perf_counter()
                        batch, frames_taken = self._prepare_batch_from_buffer(self._buffer_pass1, is_pass2=False)
                        self._remove_from_buffer(self._buffer_pass1, frames_taken, is_pass2=False)
                        
                        put_start = time.perf_counter()
                        await self.q_to_inference.put(batch)
                        put_time = time.perf_counter() - put_start
                        
                        emit_time = time.perf_counter() - emit_start
                        total_batch_emit_time += emit_time
                        batches_emitted += 1
                        p1_batches += 1
                        total_p1_frames += frames_taken
                        emitted_this_loop = True
                        
                        timings_logger.debug(
                            f"[TileBatcher] Emitted pass-1 batch #{batches_emitted}: {frames_taken} frames, "
                            f"{batch.all_tiles.shape[0]} tiles, prepare={emit_time-put_time:.3f}s, put={put_time:.3f}s"
                        )

                # --- Fetch more frames if we have capacity ---
                # Keep fetching while we have room for more in-flight tasks
                # Buffer limit applies to EACH buffer independently
                frames_fetched_this_loop = 0
                max_fetch_per_loop = max_inflight  # Don't spin forever
                
                # Check buffer capacity (both buffers + pending tiles)
                total_p1 = len(self._buffer_pass1)
                total_p2 = len(self._buffer_pass2)
                total_pending = len(self._pending_tiles)
                
                # Each buffer has its own limit
                p1_has_space = (self._max_buffer_size == 0 or total_p1 < self._max_buffer_size)
                p2_has_space = (self._max_buffer_size == 0 or total_p2 < self._max_buffer_size)
                
                while (len(self._pending_tiles) < max_inflight and 
                       frames_fetched_this_loop < max_fetch_per_loop):
                    
                    msg = None
                    is_pass2 = False
                    
                    # --- First try NON-BLOCKING gets from both queues ---
                    # This avoids wasting 25ms on empty queues
                    
                    # Prefer pass-2 (from PostProcessor) if buffer has space
                    if not self._postprocessor_done and p2_has_space:
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

                    # Then pass-1 (from Decoder) if buffer has space
                    if msg is None and not self._decoder_done and p1_has_space:
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
                    if msg is None and not self._decoder_done and p1_has_space:
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
                        # (pending tiles could be either pass, be conservative)
                        total_pending = len(self._pending_tiles)

                # --- Timeout flush: emit partial batches if nothing is coming ---
                # Only flush if we couldn't fetch and didn't emit
                if (warmup_complete and
                    frames_fetched_this_loop == 0 and 
                    not emitted_this_loop and
                    len(self._pending_tiles) == 0):
                    
                    min_flush_ratio = 0.5  # Lower threshold for timeout flush
                    min_batch = max(4, self.batch_size // 4)  # At least 4 frames
                    
                    # Flush pass-2 if we're past decoder done and have enough frames
                    if (self._decoder_done and self._buffer_pass2 and 
                        (len(self._buffer_pass2) >= min_batch or self._postprocessor_done)):
                        emit_start = time.perf_counter()
                        batch, frames_taken = self._prepare_batch_from_buffer(self._buffer_pass2, is_pass2=True)
                        self._remove_from_buffer(self._buffer_pass2, frames_taken, is_pass2=True)
                        
                        await self.q_to_inference.put(batch)
                        batches_emitted += 1
                        p2_batches += 1
                        total_p2_frames += frames_taken
                        total_batch_emit_time += time.perf_counter() - emit_start
                        
                        timings_logger.debug(
                            f"[TileBatcher] Timeout flush pass-2 batch #{batches_emitted}: {frames_taken} frames"
                        )
                    
                    # Flush pass-1 if decoder is done or buffer is reasonably full
                    elif self._buffer_pass1 and (self._decoder_done or len(self._buffer_pass1) >= min_batch):
                        emit_start = time.perf_counter()
                        batch, frames_taken = self._prepare_batch_from_buffer(self._buffer_pass1, is_pass2=False)
                        self._remove_from_buffer(self._buffer_pass1, frames_taken, is_pass2=False)
                        
                        await self.q_to_inference.put(batch)
                        batches_emitted += 1
                        p1_batches += 1
                        total_p1_frames += frames_taken
                        total_batch_emit_time += time.perf_counter() - emit_start
                        
                        timings_logger.debug(
                            f"[TileBatcher] Timeout flush pass-1 batch #{batches_emitted}: {frames_taken} frames"
                        )
                    else:
                        # Nothing to flush, wait a bit to avoid busy spin
                        await asyncio.sleep(0.01)

                # --- Check if we should send pass-1 EOS ---
                await self._maybe_emit_pass1_eos()

                # Log progress periodically
                loop_time = time.perf_counter() - loop_start
                if loop_count % 100 == 0:
                    logger.debug(
                        f"[TileBatcher] Loop #{loop_count}: {loop_time*1000:.1f}ms, "
                        f"p1_buf={len(self._buffer_pass1)}, p2_buf={len(self._buffer_pass2)}, "
                        f"pending={len(self._pending_tiles)}, submitted={frames_submitted}"
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

