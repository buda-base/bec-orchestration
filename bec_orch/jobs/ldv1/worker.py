import asyncio
import os
import contextlib
from typing import Optional, Dict, Any
from .config import PipelineConfig
from .types_common import *
from .prefetch import BasePrefetcher, LocalPrefetcher, S3Prefetcher
from .decoder import Decoder
from .ld_postprocessor import LDPostProcessor
from .tile_batcher import TileBatcher
from .ld_inference_runner import LDInferenceRunner
from .parquet_writer import ParquetWriter
from .s3ctx import S3Context

# TODO: adjust from async worker to sync worker, adjust signature so it inherits from JobWorker (jobs/base.py)

class LDVolumeWorker:
    """Owns a single volume and runs all stages concurrently.

    Wires the queues, starts:
      - Prefetcher → Decoder → GpuBatcher → LDPostProcessor → S3ParquetWriter
    All queues are bounded to enforce backpressure.
    """
    def __init__(self, cfg: PipelineConfig, volume_task: VolumeTask, progress: Optional[ProgressHook] = None, s3ctx: Optional[S3Context]=None):
        self.cfg: PipelineConfig = cfg
        self.volume_task: VolumeTask = volume_task
        self.s3ctx: Optional[S3Context] = s3ctx

        # Queues
        self.q_prefetcher_to_decoder: asyncio.Queue[FetchedBytesMsg] = asyncio.Queue(maxsize=cfg.max_q_prefetcher_to_decoder)
        self.q_decoder_to_tilebatcher: asyncio.Queue[DecodedFrameMsg] = asyncio.Queue(maxsize=cfg.max_q_decoder_to_tilebatcher)
        self.q_postprocessor_to_tilebatcher: asyncio.Queue[DecodedFrameMsg] = asyncio.Queue(maxsize=cfg.max_q_post_processor_to_tilebatcher)
        self.q_tilebatcher_to_inference: asyncio.Queue[TiledBatchMsg] = asyncio.Queue(maxsize=cfg.max_q_tilebatcher_to_inference)
        self.q_gpu_pass_1_to_post_processor: asyncio.Queue[InferredFrameMsg] = asyncio.Queue(maxsize=cfg.max_q_gpu_pass_1_to_post_processor)
        self.q_gpu_pass_2_to_post_processor: asyncio.Queue[InferredFrameMsg] = asyncio.Queue(maxsize=cfg.max_q_gpu_pass_2_to_post_processor)
        self.q_post_processor_to_writer: asyncio.Queue[RecordMsg] = asyncio.Queue(maxsize=cfg.max_q_post_processor_to_writer)

        # Components
        if volume_task.io_mode == "local":
            self.prefetcher: BasePrefetcher = LocalPrefetcher(cfg, volume_task, self.q_prefetcher_to_decoder)
        else:
            self.prefetcher: BasePrefetcher = S3Prefetcher(cfg, self.s3ctx, volume_task, self.q_prefetcher_to_decoder)
        self.decoder = Decoder(cfg, self.q_prefetcher_to_decoder, self.q_decoder_to_tilebatcher)
        self.tilebatcher = TileBatcher(
            cfg,
            q_from_decoder=self.q_decoder_to_tilebatcher,
            q_from_postprocessor=self.q_postprocessor_to_tilebatcher,
            q_to_inference=self.q_tilebatcher_to_inference,
        )
        self.inference_runner = LDInferenceRunner(
            cfg,
            q_from_tilebatcher=self.q_tilebatcher_to_inference,
            q_gpu_pass_1_to_post_processor=self.q_gpu_pass_1_to_post_processor,
            q_gpu_pass_2_to_post_processor=self.q_gpu_pass_2_to_post_processor,
        )
        self.postprocessor = LDPostProcessor(
            cfg,
            self.q_gpu_pass_1_to_post_processor,
            self.q_gpu_pass_2_to_post_processor,
            self.q_postprocessor_to_tilebatcher,  # Pass-2 frames go back to TileBatcher
            self.q_post_processor_to_writer,
        )
        self.writer = ParquetWriter(cfg, self.q_post_processor_to_writer, volume_task.output_parquet_uri, volume_task.output_jsonl_uri, progress=progress)
        
        # Health check state
        self._is_running = False
        self._is_healthy = True
        self._last_error_time: Optional[float] = None
        # Task tracking for cancellation/cleanup
        self._tasks: list[asyncio.Task[Any]] = []

    async def __aenter__(self) -> "LDVolumeWorker":
        # No heavy initialization needed; stages are created in __init__.
        return self

    async def __aexit__(self, exc_type, exc, tb) -> bool:
        # Best-effort cleanup if caller exits early due to exceptions/cancellation.
        await self.aclose()
        # Don't suppress exceptions.
        return False

    async def aclose(self) -> None:
        """Best-effort cancellation of any running stage tasks."""
        if not self._tasks:
            return
        for t in self._tasks:
            if not t.done():
                t.cancel()
        # Drain cancellations; never raise from cleanup.
        with contextlib.suppress(Exception):
            await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks = []


    def health_check(self) -> Dict[str, Any]:
        """
        Return health status for readiness probes.
        
        Returns:
            Dict with 'healthy' (bool), 'stage' (str), and optional 'error' (str)
        """
        if not self._is_running:
            return {"healthy": False, "stage": "not_started", "error": "Worker not started"}
        
        # Check if queues are critically full (backpressure indicator)
        queue_status = {
            "prefetcher_to_decoder": self.q_prefetcher_to_decoder.qsize(),
            "decoder_to_tilebatcher": self.q_decoder_to_tilebatcher.qsize(),
            "postprocessor_to_tilebatcher": self.q_postprocessor_to_tilebatcher.qsize(),
            "tilebatcher_to_inference": self.q_tilebatcher_to_inference.qsize(),
            "gpu_pass_1_to_post_processor": self.q_gpu_pass_1_to_post_processor.qsize(),
            "gpu_pass_2_to_post_processor": self.q_gpu_pass_2_to_post_processor.qsize(),
            "post_processor_to_writer": self.q_post_processor_to_writer.qsize(),
        }
        
        # Check for critical backpressure (queue > 90% full)
        max_sizes = {
            "prefetcher_to_decoder": self.cfg.max_q_prefetcher_to_decoder,
            "decoder_to_tilebatcher": self.cfg.max_q_decoder_to_tilebatcher,
            "postprocessor_to_tilebatcher": self.cfg.max_q_post_processor_to_tilebatcher,
            "tilebatcher_to_inference": self.cfg.max_q_tilebatcher_to_inference,
            "gpu_pass_1_to_post_processor": self.cfg.max_q_gpu_pass_1_to_post_processor,
            "gpu_pass_2_to_post_processor": self.cfg.max_q_gpu_pass_2_to_post_processor,
            "post_processor_to_writer": self.cfg.max_q_post_processor_to_writer,
        }
        
        critical_queues = []
        for name, size in queue_status.items():
            max_size = max_sizes[name]
            if max_size > 0 and size > 0.9 * max_size:
                critical_queues.append(name)
        
        healthy = self._is_healthy and len(critical_queues) == 0
        
        result: Dict[str, Any] = {
            "healthy": healthy,
            "stage": "running" if self._is_running else "stopped",
            "queue_status": queue_status,
        }
        
        if critical_queues:
            result["warning"] = f"Queues near capacity: {', '.join(critical_queues)}"
        
        if self._last_error_time:
            result["last_error_time"] = self._last_error_time
        
        return result

    async def run(self) -> None:
        """Run all pipeline stages concurrently with proper exception handling."""
        import logging
        import time
        logger = logging.getLogger(__name__)
        
        self._is_running = True
        self._is_healthy = True
        
        try:
            tasks: list[asyncio.Task[Any]] = [
                asyncio.create_task(self.prefetcher.run(), name="prefetcher"),
                asyncio.create_task(self.decoder.run(), name="decoder"),
                asyncio.create_task(self.tilebatcher.run(), name="tilebatcher"),
                asyncio.create_task(self.inference_runner.run(), name="inference_runner"),
                asyncio.create_task(self.postprocessor.run(), name="postprocessor"),
                asyncio.create_task(self.writer.run(), name="writer"),
            ]
            self._tasks = tasks
            
            # Wait for all tasks, but handle exceptions per-task to avoid cascading failures
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Log any exceptions
            stage_names = ["prefetcher", "decoder", "tilebatcher", "inference_runner", "postprocessor", "writer"]
            for name, result in zip(stage_names, results):
                if isinstance(result, Exception):
                    logger.error(f"Stage {name} failed: {result}", exc_info=result)
                    self._is_healthy = False
                    self._last_error_time = time.time()
            
            # Re-raise writer failure as it's critical (data loss risk)
            if isinstance(results[5], Exception):
                raise RuntimeError(f"Writer stage failed: {results[5]}") from results[5]
        finally:
            # If we're being cancelled or a stage failed unexpectedly, ensure no background tasks linger.
            await self.aclose()
            self._is_running = False
