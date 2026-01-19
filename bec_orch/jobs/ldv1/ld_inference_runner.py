"""
LDInferenceRunner: Runs GPU inference on pre-tiled batches.

This is a simplified GPU component that receives TiledBatch from TileBatcher,
runs the model, stitches results, and emits InferredFrames.

The heavy lifting (tiling, batching) is done by TileBatcher, so this
component is straightforward: receive batch → GPU forward → stitch → emit.
"""

import asyncio
import logging
import time
import traceback
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from .types_common import (
    DecodedFrame,
    PipelineError,
    EndOfStream,
    InferredFrame,
    TiledBatch,
    trace_frame,
    trace_frame_error,
)

logger = logging.getLogger(__name__)
# Separate logger for timing/performance warnings (ERROR by default)
timings_logger = logging.getLogger("bec_timings")


# -----------------------------------------------------------------------------
# Stitching functions (from utils_alt.py)
# -----------------------------------------------------------------------------

def stitch_tiles(preds: torch.Tensor, x_steps: int, y_steps: int, patch_size: int = 512) -> torch.Tensor:
    """
    Reconstruct full image from tiles.
    
    Args:
        preds: [N, C, H, W] tensor of tile predictions
        x_steps: number of horizontal tiles
        y_steps: number of vertical tiles
        patch_size: tile size
    
    Returns:
        [C, H_full, W_full] tensor
    """
    N, C, H, W = preds.shape
    assert H == patch_size and W == patch_size
    assert N == x_steps * y_steps
    
    # [N, C, H, W] → [y_steps, x_steps, C, H, W]
    tiles = preds.view(y_steps, x_steps, C, H, W)
    
    # Stitch width (concat along W dimension)
    rows = []
    for y in range(y_steps):
        rows.append(torch.cat(list(tiles[y]), dim=-1))  # [C, H, W*x_steps]
    
    # Stitch height (concat along H dimension)
    full = torch.cat(rows, dim=-2)  # [C, H*y_steps, W*x_steps]
    
    return full


def crop_padding(mask: torch.Tensor, pad_x: int, pad_y: int) -> torch.Tensor:
    """
    Remove padding from mask.
    
    Args:
        mask: [C, H, W] tensor
        pad_x: horizontal padding to remove
        pad_y: vertical padding to remove
    
    Returns:
        Cropped [C, H-pad_y, W-pad_x] tensor
    """
    if pad_y > 0:
        mask = mask[:, :-pad_y, :]
    if pad_x > 0:
        mask = mask[:, :, :-pad_x]
    return mask


# -----------------------------------------------------------------------------
# Inference function (from utils_alt.py infer_batch)
# -----------------------------------------------------------------------------

@dataclass
class InferTiming:
    """Detailed timing breakdown for a single batch inference."""
    n_tiles: int
    n_frames: int
    h2d_ms: float      # Host-to-device transfer
    forward_ms: float  # Model forward pass
    stitch_ms: float   # Stitching on GPU
    d2h_ms: float      # Device-to-host transfer
    
    @property
    def total_ms(self) -> float:
        return self.h2d_ms + self.forward_ms + self.stitch_ms + self.d2h_ms
    
    @property
    def gpu_ms(self) -> float:
        """Time spent on GPU (forward + stitch)."""
        return self.forward_ms + self.stitch_ms
    
    @property
    def transfer_ms(self) -> float:
        """Time spent on PCIe transfers (H2D + D2H)."""
        return self.h2d_ms + self.d2h_ms


def infer_batch(
    model: torch.nn.Module,
    batch: TiledBatch,
    class_threshold: float,
    device: str,
    patch_size: int = 512,
    staged_tiles: Optional[torch.Tensor] = None,
    compute_stream: Optional[torch.cuda.Stream] = None,
) -> Tuple[List[Tuple[Dict[str, Any], np.ndarray]], InferTiming]:
    """
    Run inference on a TiledBatch and stitch results.
    
    This is the core inference function, nearly identical to utils_alt.py infer_batch.
    
    Args:
        model: segmentation model
        batch: TiledBatch with pre-tiled frames
        class_threshold: threshold for binary mask
        device: "cuda" or "cpu"
        patch_size: tile size
        staged_tiles: Optional pre-staged GPU tensor (if using CUDA streams)
        compute_stream: Optional CUDA stream for compute operations
    
    Returns:
        Tuple of:
        - List of (meta, mask_np) tuples where mask_np is uint8 [H, W] with values {0, 255}
        - InferTiming with detailed timing breakdown
    """
    n_tiles = batch.all_tiles.shape[0]
    n_frames = len(batch.metas)
    
    # -------------------------------------------------------------------------
    # Phase 1: H2D Transfer (skip if tiles already staged)
    # -------------------------------------------------------------------------
    t0 = time.perf_counter()
    if staged_tiles is not None:
        all_tiles = staged_tiles
        h2d_time = 0.0  # Already transferred
    else:
        all_tiles = batch.all_tiles.to(device, non_blocking=True)
        if device == "cuda":
            torch.cuda.synchronize()
        h2d_time = (time.perf_counter() - t0) * 1000
    t1 = time.perf_counter()
    
    # -------------------------------------------------------------------------
    # Phase 2: Forward pass (on compute stream if provided)
    # -------------------------------------------------------------------------
    stream_ctx = torch.cuda.stream(compute_stream) if compute_stream else nullcontext()
    with stream_ctx:
        with torch.inference_mode():
            preds = model(all_tiles)
            soft = torch.sigmoid(preds)

    # Synchronize compute stream
    if compute_stream:
        compute_stream.synchronize()
    elif device == "cuda":
        torch.cuda.synchronize()
    t2 = time.perf_counter()

    # -------------------------------------------------------------------------
    # Phase 3: Stitching (on GPU, same stream as compute)
    # -------------------------------------------------------------------------
    with stream_ctx:
        stitched_tensors = []  # Keep on GPU, batch the D2H
        for (start, end), meta in zip(batch.tile_ranges, batch.metas):
            preds_img = soft[start:end]
            
            stitched = stitch_tiles(preds_img, meta["x_steps"], meta["y_steps"], patch_size)
            stitched = crop_padding(stitched, meta["pad_x"], meta["pad_y"])
            
            # Threshold on GPU
            binary = (stitched > class_threshold).to(torch.uint8) * 255
            stitched_tensors.append((meta, binary.squeeze(0)))
    
    # Synchronize stitch
    if compute_stream:
        compute_stream.synchronize()
    elif device == "cuda":
        torch.cuda.synchronize()
    t3 = time.perf_counter()
    
    # -------------------------------------------------------------------------
    # Phase 4: D2H Transfer (batch all at once)
    # -------------------------------------------------------------------------
    results = []
    for meta, binary_gpu in stitched_tensors:
        mask_np = binary_gpu.cpu().numpy()
        results.append((meta, mask_np))
    
    # Synchronize D2H
    if device == "cuda":
        torch.cuda.synchronize()
    t4 = time.perf_counter()
    
    # -------------------------------------------------------------------------
    # Cleanup
    # -------------------------------------------------------------------------
    del all_tiles, preds, soft, stitched_tensors
    
    timing = InferTiming(
        n_tiles=n_tiles,
        n_frames=n_frames,
        h2d_ms=h2d_time if staged_tiles is not None else (t1 - t0) * 1000,
        forward_ms=(t2 - t1) * 1000,
        stitch_ms=(t3 - t2) * 1000,
        d2h_ms=(t4 - t3) * 1000,
    )
    
    return results, timing


# Context manager helper for optional stream
from contextlib import nullcontext


# -----------------------------------------------------------------------------
# Main class
# -----------------------------------------------------------------------------

class LDInferenceRunner:
    """
    Runs GPU inference on pre-tiled batches.
    
    Input:
      - q_from_tilebatcher: TiledBatch (short queue, e.g. size 8)
    
    Outputs:
      - q_gpu_pass_1_to_post_processor: InferredFrame (pass-1 results)
      - q_gpu_pass_2_to_post_processor: InferredFrame (pass-2 results)
    """

    def __init__(
        self,
        cfg,
        q_from_tilebatcher: asyncio.Queue,
        q_gpu_pass_1_to_post_processor: asyncio.Queue,
        q_gpu_pass_2_to_post_processor: asyncio.Queue,
    ):
        self.cfg = cfg
        self.q_in = q_from_tilebatcher
        self.q_gpu_pass_1_to_post_processor = q_gpu_pass_1_to_post_processor
        self.q_gpu_pass_2_to_post_processor = q_gpu_pass_2_to_post_processor

        # Configuration
        self.patch_size: int = getattr(cfg, "patch_size", 512)
        self.class_threshold: float = getattr(cfg, "class_threshold", 0.5)

        # Device/model configuration
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = getattr(cfg, "model", None)
        if self.model is None:
            raise ValueError("cfg.model must be set (torch.nn.Module)")
        if not isinstance(self.model, torch.nn.Module):
            raise TypeError(f"cfg.model must be torch.nn.Module, got {type(self.model)}")
        self.model.to(self.device)
        self.model.eval()

        # State tracking
        self._done = False
        self._p1_eos_sent = False
        self._p2_eos_sent = False
        
        # Timeout for queue put operations
        self._queue_put_timeout_s = getattr(cfg, "queue_put_timeout_s", 30.0)
        
        # CUDA streams for overlapping H2D with compute
        self._use_streams = self.device == "cuda" and getattr(cfg, "cuda_streams", 2) >= 2
        self._h2d_stream: Optional[torch.cuda.Stream] = None
        self._compute_stream: Optional[torch.cuda.Stream] = None
        self._staged_tiles: Optional[torch.Tensor] = None  # Pre-staged batch on GPU
        self._staged_batch: Optional[TiledBatch] = None    # Metadata for staged batch
        
        if self._use_streams:
            self._h2d_stream = torch.cuda.Stream()
            self._compute_stream = torch.cuda.Stream()
            logger.info("[InferenceRunner] CUDA streams enabled for H2D/compute overlap")
        
        # Warmup the model on first init (eliminates 5s JIT compilation on first batch)
        self._warmup_model()

    # -------------------------------------------------------------------------
    # Model warmup
    # -------------------------------------------------------------------------

    def _warmup_model(self) -> None:
        """
        Run dummy forward passes to trigger CUDA JIT compilation and memory allocation.
        
        This eliminates the warmup penalty by:
        1. Running inference with multiple batch sizes (forces CUDA to pre-allocate memory pools)
        2. Running multiple passes to ensure all CUDA kernels are compiled
        3. Testing both large and small batches to warm up different code paths
        """
        logger.info(f"[InferenceRunner] Warming up model on {self.device}...")
        warmup_start = time.perf_counter()
        
        try:
            # Use the configured tile dtype
            precision = getattr(self.cfg, "precision", "fp32")
            if precision == "fp16":
                dtype = torch.float16
            elif precision == "bf16":
                dtype = torch.bfloat16
            else:
                dtype = torch.float32
            
            # Calculate batch sizes to warm up
            batch_size = getattr(self.cfg, "batch_size", 8)
            tiles_per_frame = 5  # Conservative estimate
            max_tiles = batch_size * tiles_per_frame  # e.g., 8 * 5 = 40 tiles
            
            # Warm up with different batch sizes to pre-allocate CUDA memory pools
            # This prevents slowdowns when batch size varies (e.g., last batch, pass-2 batches)
            warmup_sizes = [max_tiles, max_tiles // 2, max_tiles // 4, 1]
            
            with torch.inference_mode():
                for n_tiles in warmup_sizes:
                    if n_tiles < 1:
                        continue
                    dummy_input = torch.zeros(
                        n_tiles, 3, self.patch_size, self.patch_size,
                        dtype=dtype, device=self.device
                    )
                    # Run twice per size
                    for _ in range(2):
                        _ = self.model(dummy_input)
                        if self.device == "cuda":
                            torch.cuda.synchronize()
                    del dummy_input
            
            warmup_time = time.perf_counter() - warmup_start
            logger.info(
                f"[InferenceRunner] Model warmup complete in {warmup_time:.2f}s "
                f"(sizes={warmup_sizes}, dtype={dtype})"
            )
            
            # Clean up
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
        except Exception as e:
            logger.warning(f"[InferenceRunner] Model warmup failed: {e}")

    # -------------------------------------------------------------------------
    # Error handling
    # -------------------------------------------------------------------------

    @staticmethod
    def _is_cuda_oom(exc: BaseException) -> bool:
        msg = str(exc).lower()
        return "out of memory" in msg or "cuda out of memory" in msg

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
        timeout_s: float = 5.0,
    ) -> None:
        import logging
        logger = logging.getLogger(__name__)
        
        tb = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
        err = PipelineError(
            stage="LDInferenceRunner",
            task=task,
            source_etag=source_etag,
            error_type=type(exc).__name__,
            message=f"[{internal_stage}] {exc}",
            traceback=tb,
            retryable=bool(retryable),
            attempt=int(attempt),
        )
        target_q = self.q_gpu_pass_2_to_post_processor if lane_second_pass else self.q_gpu_pass_1_to_post_processor
        
        try:
            await asyncio.wait_for(target_q.put(err), timeout=timeout_s)
        except asyncio.TimeoutError:
            logger.critical(
                f"Failed to emit error after {timeout_s}s: queue full. "
                f"Dropping error for {task.img_filename if task else 'unknown'}"
            )

    # -------------------------------------------------------------------------
    # Result emission
    # -------------------------------------------------------------------------

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
        try:
            await asyncio.wait_for(queue.put(item), timeout=self._queue_put_timeout_s)
            return True
        except asyncio.TimeoutError:
            logger.error(
                f"[InferenceRunner] Timeout ({self._queue_put_timeout_s}s) putting {item_desc} to {queue_name} "
                f"(qsize={queue.qsize()}/{queue.maxsize}). Pipeline may be deadlocked."
            )
            return False

    async def _emit_result(self, meta: Dict[str, Any], mask_np: np.ndarray) -> bool:
        """
        Emit InferredFrame to appropriate output queue.
        
        Returns True if successful, False if the put timed out (frame will be lost).
        """
        dec_frame: DecodedFrame = meta["dec_frame"]
        second_pass: bool = meta["second_pass"]

        # Use original grayscale frame from decoder, not the binarized version
        # (binarization in TileBatcher is only for tiling, not for downstream)
        out = InferredFrame(
            task=dec_frame.task,
            source_etag=dec_frame.source_etag,
            frame=dec_frame.frame,  # Original grayscale from decoder
            orig_h=dec_frame.orig_h,
            orig_w=dec_frame.orig_w,
            is_binary=dec_frame.is_binary,  # Preserve original binary flag
            first_pass=dec_frame.first_pass,
            rotation_angle=dec_frame.rotation_angle,
            tps_data=dec_frame.tps_data,
            line_mask=mask_np,  # grayscale uint8, {0, 255}
        )

        img_filename = dec_frame.task.img_filename if dec_frame.task else "unknown"
        pass_str = "pass2" if second_pass else "pass1"
        
        if second_pass:
            success = await self._put_with_timeout(
                self.q_gpu_pass_2_to_post_processor,
                out,
                "q_gpu_pass_2_to_post_processor",
                f"InferredFrame({img_filename}, pass2)",
            )
        else:
            success = await self._put_with_timeout(
                self.q_gpu_pass_1_to_post_processor,
                out,
                "q_gpu_pass_1_to_post_processor",
                f"InferredFrame({img_filename}, pass1)",
            )
        
        # If put failed due to timeout, emit error so frame is tracked
        if not success:
            trace_frame_error("InferenceRunner", img_filename, f"queue_timeout_{pass_str}")
            await self._emit_pipeline_error(
                internal_stage="_emit_result.queue_timeout",
                exc=TimeoutError(f"Could not emit result for {img_filename} after {self._queue_put_timeout_s}s"),
                lane_second_pass=second_pass,
                task=dec_frame.task,
                source_etag=dec_frame.source_etag,
                retryable=False,
                attempt=1,
                timeout_s=2.0,  # Short timeout for error emission to avoid cascading
            )
        else:
            trace_frame("InferenceRunner", f"emitted_{pass_str}", img_filename)
        
        return success

    # -------------------------------------------------------------------------
    # Batch processing with CUDA streams
    # -------------------------------------------------------------------------

    def _stage_batch_h2d(self, batch: TiledBatch) -> None:
        """
        Stage a batch's tiles on GPU using H2D stream (non-blocking).
        Call this BEFORE processing the current batch to overlap H2D with compute.
        """
        if not self._use_streams or self._h2d_stream is None:
            return
        
        with torch.cuda.stream(self._h2d_stream):
            self._staged_tiles = batch.all_tiles.to(self.device, non_blocking=True)
            self._staged_batch = batch

    def _wait_for_staged(self) -> Tuple[Optional[torch.Tensor], Optional[TiledBatch]]:
        """
        Wait for staged H2D transfer to complete and return the staged data.
        """
        if not self._use_streams or self._staged_tiles is None:
            return None, None
        
        # Sync H2D stream to ensure transfer is complete
        if self._h2d_stream is not None:
            self._h2d_stream.synchronize()
        
        tiles = self._staged_tiles
        batch = self._staged_batch
        self._staged_tiles = None
        self._staged_batch = None
        return tiles, batch

    async def _process_batch(
        self,
        batch: TiledBatch,
        staged_tiles: Optional[torch.Tensor] = None,
    ) -> Tuple[Optional[InferTiming], float]:
        """
        Process a TiledBatch: run GPU inference, stitch, emit results.
        
        Args:
            batch: TiledBatch to process
            staged_tiles: Optional pre-staged GPU tensor (from H2D stream)
        
        Returns:
            Tuple of (InferTiming if successful else None, emit_time_seconds).
        """
        emit_time = 0.0
        try:
            results, timing = infer_batch(
                self.model,
                batch,
                self.class_threshold,
                self.device,
                self.patch_size,
                staged_tiles=staged_tiles,
                compute_stream=self._compute_stream,
            )

            emit_start = time.perf_counter()
            for meta, mask_np in results:
                await self._emit_result(meta, mask_np)
            emit_time = time.perf_counter() - emit_start
            
            return timing, emit_time

        except Exception as e:
            # On error, emit PipelineError for each frame in the batch
            retryable = self._is_cuda_oom(e)
            for meta in batch.metas:
                await self._emit_pipeline_error(
                    internal_stage="inference",
                    exc=e,
                    lane_second_pass=meta["second_pass"],
                    task=meta["dec_frame"].task,
                    source_etag=meta["dec_frame"].source_etag,
                    retryable=retryable,
                    attempt=1,
                )
            return None, 0.0

    # -------------------------------------------------------------------------
    # Main run loop
    # -------------------------------------------------------------------------

    async def _fetch_next_batch(self) -> Any:
        """Fetch next message from queue (runs as background task)."""
        return await self.q_in.get()

    async def run(self) -> None:
        """
        Main loop with batch pre-fetching for pipeline overlap.
        
        Key optimization: Start fetching batch N+1 while GPU processes batch N.
        This overlaps queue wait time with GPU compute time, reducing idle gaps.
        """
        # Timing stats
        batches_processed = 0
        total_wait_time = 0.0
        total_emit_time = 0.0  # Time emitting results to queues
        total_tiles = 0
        total_frames = 0
        
        # Detailed timing accumulators
        total_h2d_ms = 0.0
        total_forward_ms = 0.0
        total_stitch_ms = 0.0
        total_d2h_ms = 0.0
        
        detailed_timing = getattr(self.cfg, "detailed_inference_timing", False)
        
        # Track time from GPU perspective
        run_start_time = time.perf_counter()
        
        # Pre-fetch task for batch pipelining
        prefetch_task: Optional[asyncio.Task] = None
        
        try:
            # Start pre-fetching the first batch
            prefetch_task = asyncio.create_task(self._fetch_next_batch())
            
            while True:
                # Wait for the pre-fetched batch (measure actual blocking time)
                wait_start = time.perf_counter()
                if prefetch_task is not None:
                    msg = await prefetch_task
                    prefetch_task = None
                else:
                    msg = await self.q_in.get()
                wait_time = time.perf_counter() - wait_start
                total_wait_time += wait_time
                
                if wait_time > 0.5:
                    timings_logger.warning(
                        f"[InferenceRunner] Long wait for batch: {wait_time:.2f}s, "
                        f"queue_size={self.q_in.qsize()}"
                    )

                # Handle different message types
                if isinstance(msg, EndOfStream):
                    if msg.stream == "tiled_pass_1":
                        logger.info(f"[InferenceRunner] Received tiled_pass_1 EOS, forwarding gpu_pass_1")
                        if not self._p1_eos_sent:
                            await self.q_gpu_pass_1_to_post_processor.put(
                                EndOfStream(stream="gpu_pass_1", producer="LDInferenceRunner")
                            )
                            self._p1_eos_sent = True
                        # Continue processing - there may be more pass-2 batches
                        prefetch_task = asyncio.create_task(self._fetch_next_batch())
                        continue
                    
                    elif msg.stream == "tiled_pass_2":
                        # Log detailed summary
                        total_infer_ms = total_h2d_ms + total_forward_ms + total_stitch_ms + total_d2h_ms
                        total_transfer_ms = total_h2d_ms + total_d2h_ms
                        total_gpu_ms = total_forward_ms + total_stitch_ms
                        
                        # Calculate actual GPU utilization
                        run_total_time = time.perf_counter() - run_start_time
                        gpu_utilization = (total_infer_ms / 1000) / run_total_time * 100 if run_total_time > 0 else 0
                        
                        timings_logger.info(
                            f"[InferenceRunner] DONE - batches={batches_processed}, frames={total_frames}, tiles={total_tiles}"
                        )
                        if total_infer_ms > 0:
                            timings_logger.info(
                                f"[InferenceRunner] TIMING SUMMARY: "
                                f"total={total_infer_ms:.0f}ms, "
                                f"h2d={total_h2d_ms:.0f}ms ({100*total_h2d_ms/total_infer_ms:.1f}%), "
                                f"forward={total_forward_ms:.0f}ms ({100*total_forward_ms/total_infer_ms:.1f}%), "
                                f"stitch={total_stitch_ms:.0f}ms ({100*total_stitch_ms/total_infer_ms:.1f}%), "
                                f"d2h={total_d2h_ms:.0f}ms ({100*total_d2h_ms/total_infer_ms:.1f}%)"
                            )
                        
                        # Key insight: GPU utilization shows if GPU is the bottleneck
                        # - >80%: GPU is bottleneck (good!)
                        # - <50%: Pipeline is bottleneck (GPU starved)
                        timings_logger.info(
                            f"[InferenceRunner] GPU UTILIZATION: {gpu_utilization:.1f}% "
                            f"(run={run_total_time:.1f}s, gpu_work={total_infer_ms/1000:.1f}s, "
                            f"queue_wait={total_wait_time:.1f}s, emit={total_emit_time:.1f}s)"
                        )
                        
                        if total_infer_ms > 0:
                            timings_logger.info(
                                f"[InferenceRunner] Throughput: {total_tiles/(total_infer_ms/1000):.1f} tiles/s, "
                                f"{total_frames/(total_infer_ms/1000):.1f} frames/s"
                            )
                        
                        self._done = True
                        if not self._p1_eos_sent:
                            await self.q_gpu_pass_1_to_post_processor.put(
                                EndOfStream(stream="gpu_pass_1", producer="LDInferenceRunner")
                            )
                            self._p1_eos_sent = True
                        await self.q_gpu_pass_2_to_post_processor.put(
                            EndOfStream(stream="gpu_pass_2", producer="LDInferenceRunner")
                        )
                        break
                
                elif isinstance(msg, PipelineError):
                    await self.q_gpu_pass_1_to_post_processor.put(msg)
                    # Continue fetching
                    prefetch_task = asyncio.create_task(self._fetch_next_batch())
                
                elif isinstance(msg, TiledBatch):
                    n_tiles = msg.all_tiles.shape[0]
                    n_frames = len(msg.metas)
                    
                    # Trace each frame in the batch
                    for meta in msg.metas:
                        dec_frame = meta.get("dec_frame")
                        if dec_frame:
                            pass_str = "pass2" if meta.get("second_pass", False) else "pass1"
                            trace_frame("InferenceRunner", f"received_{pass_str}", dec_frame.task.img_filename)
                    
                    # DOUBLE BUFFERING WITH CUDA STREAMS:
                    # 1. Stage current batch to GPU (H2D stream, non-blocking)
                    # 2. Start prefetching next batch from queue
                    # 3. Wait for H2D to complete
                    # 4. Process current batch (compute stream) while next batch prefetches
                    
                    h2d_start = time.perf_counter()
                    staged_tiles: Optional[torch.Tensor] = None
                    
                    if self._use_streams:
                        # Stage current batch on H2D stream (async)
                        self._stage_batch_h2d(msg)
                        
                        # Start prefetching next batch while H2D runs
                        prefetch_task = asyncio.create_task(self._fetch_next_batch())
                        
                        # Yield to let prefetch start
                        await asyncio.sleep(0)
                        
                        # Wait for H2D to complete and get staged tiles
                        staged_tiles, _ = self._wait_for_staged()
                    else:
                        # No streams - just prefetch next batch
                        prefetch_task = asyncio.create_task(self._fetch_next_batch())
                    
                    h2d_overlap_time = time.perf_counter() - h2d_start
                    
                    # Process current batch on GPU (using staged tiles if available)
                    timing, emit_time = await self._process_batch(msg, staged_tiles=staged_tiles)
                    total_emit_time += emit_time
                    
                    if timing:
                        total_tiles += n_tiles
                        total_frames += n_frames
                        batches_processed += 1
                        
                        # Accumulate detailed timing
                        total_h2d_ms += timing.h2d_ms
                        total_forward_ms += timing.forward_ms
                        total_stitch_ms += timing.stitch_ms
                        total_d2h_ms += timing.d2h_ms
                        
                        tiles_per_sec = n_tiles / (timing.total_ms / 1000) if timing.total_ms > 0 else 0
                        
                        if detailed_timing:
                            logger.info(
                                f"[InferenceRunner] Batch #{batches_processed}: {n_frames} frames, {n_tiles} tiles, "
                                f"total={timing.total_ms:.1f}ms ({tiles_per_sec:.0f} t/s) | "
                                f"h2d={timing.h2d_ms:.1f}ms, fwd={timing.forward_ms:.1f}ms, "
                                f"stitch={timing.stitch_ms:.1f}ms, d2h={timing.d2h_ms:.1f}ms | "
                                f"wait={wait_time*1000:.0f}ms, emit={emit_time*1000:.0f}ms"
                            )
                        else:
                            logger.info(
                                f"[InferenceRunner] Batch #{batches_processed}: {n_frames} frames, {n_tiles} tiles, "
                                f"infer={timing.total_ms:.0f}ms ({tiles_per_sec:.0f} t/s), wait={wait_time*1000:.0f}ms"
                            )
                    
                    # Periodic memory cleanup to prevent fragmentation
                    if batches_processed % 10 == 0:
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                
                else:
                    logger.warning(f"[InferenceRunner] Unexpected message type: {type(msg)}")
                    # Continue fetching
                    prefetch_task = asyncio.create_task(self._fetch_next_batch())

                # Yield to event loop (allows prefetch task to progress)
                await asyncio.sleep(0)

        except asyncio.CancelledError:
            logger.info("[InferenceRunner] Cancelled, cleaning up...")
            if prefetch_task and not prefetch_task.done():
                prefetch_task.cancel()
            raise
        except KeyboardInterrupt:
            logger.info("[InferenceRunner] Interrupted by user")
            if prefetch_task and not prefetch_task.done():
                prefetch_task.cancel()
            raise
        finally:
            if prefetch_task and not prefetch_task.done():
                prefetch_task.cancel()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

