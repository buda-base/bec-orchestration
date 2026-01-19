import asyncio
import logging
import os
import contextlib
import time
from pathlib import Path
from typing import Optional, Dict, Any

from .config import PipelineConfig
from .types_common import *
from ..shared.prefetch import BasePrefetcher, LocalPrefetcher, S3Prefetcher
from ..shared.decoder import Decoder
from ..shared.memory_monitor import MemoryMonitor, log_memory_snapshot
from .ld_postprocessor import LDPostProcessor
from .tile_batcher import TileBatcher
from .ld_inference_runner import LDInferenceRunner
from .parquet_writer import ParquetWriter
from ..shared.s3ctx import S3Context

logger = logging.getLogger(__name__)


class VolumeTimeoutError(Exception):
    """Raised when volume processing exceeds the configured timeout."""
    
    def __init__(self, message: str, diagnostic_state: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.diagnostic_state = diagnostic_state


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

        # Create frame tracker for this volume and set it globally + on config
        self._frame_tracker = FrameTracker()
        set_tracker(self._frame_tracker)
        cfg.frame_tracker = self._frame_tracker

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
        # Build list of expected filenames for missing records tracking
        expected_filenames = [t.img_filename for t in volume_task.image_tasks]
        self.writer = ParquetWriter(
            cfg,
            self.q_post_processor_to_writer,
            volume_task.output_parquet_uri,
            volume_task.output_jsonl_uri,
            progress=progress,
            expected_filenames=expected_filenames,
        )
        
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

    async def aclose(
        self,
        graceful_writer_flush: bool = False,
        timeout_diagnostic_state: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Best-effort cancellation of any running stage tasks.
        
        Args:
            graceful_writer_flush: If True, try to send EndOfStream to writer
                                   and wait briefly for it to flush before cancelling.
            timeout_diagnostic_state: If provided (on timeout), send a PipelineError
                                      to the writer with this diagnostic info.
        """
        if not self._tasks:
            return
        
        if graceful_writer_flush:
            # Try to let the writer flush its buffer before cancellation
            await self._try_graceful_writer_flush(timeout_diagnostic_state)
        
        for t in self._tasks:
            if not t.done():
                t.cancel()
        # Drain cancellations; never raise from cleanup.
        with contextlib.suppress(Exception):
            await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks = []
        
        # Clear global frame tracker to prevent memory leaks between volumes
        set_tracker(None)
    
    async def _try_graceful_writer_flush(
        self,
        diagnostic_state: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Try to flush the ParquetWriter's buffer gracefully on timeout.
        
        Sends a timeout error (if diagnostic_state provided) and EndOfStream to the
        writer queue, then waits briefly for it to process and flush.
        This preserves any errors/records already collected before the timeout.
        """
        import json
        import logging
        logger = logging.getLogger(__name__)
        
        try:
            # First, cancel all stages EXCEPT the writer
            writer_task = None
            for t in self._tasks:
                if t.get_name() == "writer":
                    writer_task = t
                elif not t.done():
                    t.cancel()
            
            # Drain the cancelled tasks
            non_writer_tasks = [t for t in self._tasks if t.get_name() != "writer"]
            with contextlib.suppress(Exception):
                await asyncio.gather(*non_writer_tasks, return_exceptions=True)
            
            if writer_task is None or writer_task.done():
                return
            
            # If we have diagnostic state, send a timeout error to the writer
            if diagnostic_state is not None:
                try:
                    # Create a volume-level timeout error
                    # Use a dummy ImageTask with None values for volume-level error
                    timeout_error = PipelineError(
                        stage="Pipeline",
                        task=ImageTask(source_uri="", img_filename=""),  # Dummy task for volume-level
                        source_etag=None,
                        error_type="VolumeTimeout",
                        message=f"Volume processing exceeded timeout. Diagnostic: {json.dumps(diagnostic_state)}",
                        traceback=None,
                        retryable=True,
                    )
                    await asyncio.wait_for(
                        self.q_post_processor_to_writer.put(timeout_error),
                        timeout=1.0
                    )
                    logger.debug("[aclose] Sent timeout error to writer")
                except asyncio.TimeoutError:
                    logger.debug("[aclose] Could not send timeout error to writer (queue full)")
            
            # Try to send EndOfStream to writer with a short timeout
            try:
                await asyncio.wait_for(
                    self.q_post_processor_to_writer.put(EndOfStream(source="timeout_shutdown")),
                    timeout=1.0
                )
                logger.debug("[aclose] Sent EndOfStream to writer for graceful shutdown")
            except asyncio.TimeoutError:
                logger.debug("[aclose] Could not send EndOfStream to writer (queue full)")
                return
            
            # Wait briefly for writer to flush
            try:
                await asyncio.wait_for(writer_task, timeout=5.0)
                logger.info("[aclose] Writer flushed successfully before cancellation")
            except asyncio.TimeoutError:
                logger.debug("[aclose] Writer did not complete in time, will cancel")
        except Exception as e:
            logger.debug(f"[aclose] Graceful writer flush failed: {e}")


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

    def get_diagnostic_state(self) -> Dict[str, Any]:
        """
        Return detailed diagnostic state for debugging hangs.
        
        Called when volume timeout occurs to help identify where the pipeline is stuck.
        """
        state: Dict[str, Any] = {
            "queues": {
                "prefetcher_to_decoder": f"{self.q_prefetcher_to_decoder.qsize()}/{self.cfg.max_q_prefetcher_to_decoder}",
                "decoder_to_tilebatcher": f"{self.q_decoder_to_tilebatcher.qsize()}/{self.cfg.max_q_decoder_to_tilebatcher}",
                "tilebatcher_to_inference": f"{self.q_tilebatcher_to_inference.qsize()}/{self.cfg.max_q_tilebatcher_to_inference}",
                "gpu_pass_1": f"{self.q_gpu_pass_1_to_post_processor.qsize()}/{self.cfg.max_q_gpu_pass_1_to_post_processor}",
                "gpu_pass_2": f"{self.q_gpu_pass_2_to_post_processor.qsize()}/{self.cfg.max_q_gpu_pass_2_to_post_processor}",
                "postprocessor_to_tilebatcher": f"{self.q_postprocessor_to_tilebatcher.qsize()}/{self.cfg.max_q_post_processor_to_tilebatcher}",
                "postprocessor_to_writer": f"{self.q_post_processor_to_writer.qsize()}/{self.cfg.max_q_post_processor_to_writer}",
            },
        }
        
        # PostProcessor state
        try:
            state["postprocessor"] = {
                "pending_transforms": len(self.postprocessor._pending_transforms),
                "p1_done": self.postprocessor._p1_done,
                "p2_done": self.postprocessor._p2_done,
            }
        except Exception:
            state["postprocessor"] = {"error": "could not read state"}
        
        # Writer state
        try:
            state["writer"] = {
                "success_count": self.writer._success_count,
                "error_count": self.writer._error_count,
                "buffer_size": len(self.writer._buffer),
                "expected_count": len(self.writer._expected_filenames) if hasattr(self.writer, '_expected_filenames') else 0,
                "received_count": len(self.writer._received_filenames) if hasattr(self.writer, '_received_filenames') else 0,
            }
        except Exception:
            state["writer"] = {"error": "could not read state"}
        
        # TileBatcher state
        try:
            state["tilebatcher"] = {
                "decoder_done": self.tilebatcher._decoder_done,
                "postprocessor_done": self.tilebatcher._postprocessor_done,
                "buffer_pass1": len(self.tilebatcher._buffer_pass1),
                "buffer_pass2": len(self.tilebatcher._buffer_pass2),
                "pending_tiles": len(self.tilebatcher._pending_tiles),
            }
        except Exception:
            state["tilebatcher"] = {"error": "could not read state"}
        
        return state

    def get_error_stats(self) -> Dict[str, Any]:
        """
        Get error statistics from the writer after pipeline completion.
        
        Returns:
            Dict with:
            - success_count: Number of successfully processed records
            - error_count: Total number of errors
            - dropped_count: Number of records dropped due to pipeline issues
            - errors_by_stage: Dict mapping stage name to error count
        """
        try:
            dropped = len(self.writer._expected_filenames - self.writer._received_filenames) \
                if hasattr(self.writer, '_expected_filenames') and hasattr(self.writer, '_received_filenames') \
                else 0
            return {
                "success_count": self.writer._success_count,
                "error_count": self.writer._error_count,
                "dropped_count": dropped,
                "errors_by_stage": dict(self.writer._error_by_stage),
            }
        except Exception:
            return {
                "success_count": 0,
                "error_count": 0,
                "dropped_count": 0,
                "errors_by_stage": {},
            }

    async def run(self) -> None:
        """Run all pipeline stages with timeout protection."""
        import json
        import logging
        logger = logging.getLogger(__name__)
        
        timeout_s = getattr(self.cfg, 'volume_timeout_s', 300.0)
        
        try:
            await asyncio.wait_for(self._run_pipeline(), timeout=timeout_s)
        except asyncio.TimeoutError:
            # Dump diagnostic state before raising
            state = self.get_diagnostic_state()
            logger.error(
                f"[LDVolumeWorker] Volume timeout after {timeout_s}s. "
                f"Diagnostic state: {json.dumps(state, indent=2)}"
            )
            # Try to flush writer gracefully (with timeout error), then cancel all tasks
            await self.aclose(graceful_writer_flush=True, timeout_diagnostic_state=state)
            self._is_running = False
            # Re-raise with diagnostic info
            raise VolumeTimeoutError(
                f"Volume processing exceeded {timeout_s}s timeout",
                diagnostic_state=state
            )

    async def _run_pipeline(self) -> None:
        """Run all pipeline stages concurrently with proper exception handling."""
        import logging
        import time
        logger = logging.getLogger(__name__)
        
        self._is_running = True
        self._is_healthy = True
        
        # Create memory monitor with all pipeline queues
        memory_monitor = MemoryMonitor(
            interval_s=5.0,
            queues={
                "prefetch": self.q_prefetcher_to_decoder,
                "decode": self.q_decoder_to_tilebatcher,
                "tiles": self.q_tilebatcher_to_inference,
                "gpu_p1": self.q_gpu_pass_1_to_post_processor,
                "gpu_p2": self.q_gpu_pass_2_to_post_processor,
                "writer": self.q_post_processor_to_writer,
            },
        )
        monitor_task: Optional[asyncio.Task[Any]] = None
        
        try:
            # Start memory monitor as background task
            monitor_task = asyncio.create_task(memory_monitor.run(), name="memory_monitor")
            
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
            
            # Collect all stage failures
            stage_names = ["prefetcher", "decoder", "tilebatcher", "inference_runner", "postprocessor", "writer"]
            failed_stages = []
            
            for name, result in zip(stage_names, results):
                if isinstance(result, Exception):
                    logger.error(f"Pipeline stage '{name}' failed with {type(result).__name__}: {result}", exc_info=result)
                    self._is_healthy = False
                    self._last_error_time = time.time()
                    failed_stages.append((name, result))
                else:
                    logger.debug(f"Pipeline stage '{name}' completed successfully")
            
            # If any stage failed, raise a consolidated error with all failures
            if failed_stages:
                if len(failed_stages) == 1:
                    name, exc = failed_stages[0]
                    raise RuntimeError(f"Pipeline stage '{name}' failed: {exc}") from exc
                else:
                    # Multiple stages failed - report all
                    failure_summary = ", ".join([f"{name}({type(exc).__name__})" for name, exc in failed_stages])
                    logger.error(f"Multiple pipeline stages failed: {failure_summary}")
                    # Raise from the first failure
                    first_name, first_exc = failed_stages[0]
                    raise RuntimeError(
                        f"Multiple pipeline stages failed: {failure_summary}. "
                        f"First failure in '{first_name}': {first_exc}"
                    ) from first_exc
        finally:
            # Stop memory monitor
            if monitor_task is not None and not monitor_task.done():
                monitor_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await monitor_task
            
            # If we're being cancelled or a stage failed unexpectedly, ensure no background tasks linger.
            await self.aclose()
            self._is_running = False


# ============================================================================
# JobWorker Adapter for BEC Orchestration
# ============================================================================

class LDV1JobWorker:
    """
    Synchronous adapter for LDVolumeWorker to integrate with BEC orchestration.
    
    Implements the JobWorker protocol expected by BECWorkerRuntime.
    
    The model is loaded once in __init__ and reused across all volumes to avoid
    reloading overhead. The model stays on GPU between volumes.
    """
    
    def __init__(self):
        """Initialize the job worker and load the model."""
        # Get model path from environment (required for ldv1)
        model_path = os.environ.get('BEC_LD_MODEL_PATH')
        if not model_path:
            raise ValueError(
                "BEC_LD_MODEL_PATH environment variable not set. "
                "Set it to the path of the segmentation model checkpoint (.pth file)."
            )
        
        # Strip quotes if present (common in .env files)
        model_path = model_path.strip('"\'')
        
        # Validate model file exists early
        model_path_obj = Path(model_path)
        if not model_path_obj.exists():
            raise FileNotFoundError(
                f"Model checkpoint not found: {model_path}\n"
                f"Please check that BEC_LD_MODEL_PATH is set correctly and the file exists."
            )
        
        if not model_path_obj.is_file():
            raise ValueError(
                f"Model path is not a file: {model_path}\n"
                f"BEC_LD_MODEL_PATH must point to a .pth checkpoint file."
            )
        
        logger.info(f"Loading model from: {model_path}")
        
        # Determine device for model loading
        import torch
        device = None
        if torch.cuda.is_available():
            device = "cuda"
            logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
        else:
            logger.warning("CUDA not available, using CPU (this will be slow)")
        
        # Load model with proper error handling
        try:
            from .model_utils import load_model
            
            # Use default precision and settings (can be overridden per-volume if needed)
            self.model = load_model(
                model_path,
                classes=1,
                device=device,
                precision="fp16" if device == "cuda" else "fp32",
                compile_model=False,  # Can be enabled via config if needed
            )
            logger.info(f"Model loaded successfully on {device}")
            
        except FileNotFoundError as e:
            raise FileNotFoundError(
                f"Model checkpoint file not found: {model_path}\n"
                f"Original error: {e}"
            ) from e
        except ValueError as e:
            if "state_dict" in str(e).lower():
                raise ValueError(
                    f"Invalid model checkpoint format: {model_path}\n"
                    f"Expected a checkpoint dict with a 'state_dict' key.\n"
                    f"Original error: {e}"
                ) from e
            raise
        except Exception as e:
            raise RuntimeError(
                f"Failed to load model from {model_path}.\n"
                f"Error type: {type(e).__name__}\n"
                f"Error message: {e}\n"
                f"Please verify the checkpoint file is valid and dependencies are installed."
            ) from e
    
    def run(self, ctx: 'JobContext') -> 'TaskResult':
        """
        Run line detection on a volume.
        
        Args:
            ctx: Job context with volume info, config, and artifact location
            
        Returns:
            TaskResult with metrics
        """
        from bec_orch.jobs.base import JobContext
        from bec_orch.core.models import TaskResult
        
        logger.info(f"Starting LDV1 job for volume {ctx.volume.w_id}/{ctx.volume.i_id}")
        start_time = time.time()
        
        # Log memory at start of volume processing
        log_memory_snapshot(f"[LDV1] Start volume {ctx.volume.w_id}/{ctx.volume.i_id}")
        
        # Build PipelineConfig from job config with defaults
        pipeline_cfg = self._build_pipeline_config(ctx)
        
        # Build VolumeTask from context
        volume_task = self._build_volume_task(ctx)
        
        # Track metrics
        metrics = {
            'total_images': len(ctx.volume_manifest.manifest),
            'nb_errors': 0,
            'image_durations': [],
        }
        
        # Progress hook to collect metrics
        def progress_hook(event: Dict[str, Any]) -> None:
            if event.get('type') == 'image_complete':
                duration_ms = event.get('duration_ms', 0)
                metrics['image_durations'].append(duration_ms)
            elif event.get('type') == 'error':
                metrics['nb_errors'] += 1
        
        # Run async pipeline and collect error stats
        error_stats: Dict[str, Any] = {}
        
        try:
            async def run_pipeline():
                nonlocal error_stats
                # Create S3 context within async context
                global_sem = asyncio.Semaphore(pipeline_cfg.s3_max_inflight_global)
                s3ctx = S3Context(pipeline_cfg, global_sem)
                
                async with LDVolumeWorker(pipeline_cfg, volume_task, progress=progress_hook, s3ctx=s3ctx) as worker:
                    await worker.run()
                    # Collect error stats after successful pipeline completion
                    error_stats.update(worker.get_error_stats())
            
            asyncio.run(run_pipeline())
            
        except VolumeTimeoutError as e:
            # Volume timeout is always retryable
            log_memory_snapshot(f"[LDV1] TIMEOUT volume {ctx.volume.w_id}/{ctx.volume.i_id}", level=logging.ERROR)
            logger.error(
                f"LDV1 pipeline timeout for volume {ctx.volume.w_id}/{ctx.volume.i_id}: {e}",
                exc_info=True
            )
            from bec_orch.errors import RetryableTaskError
            raise RetryableTaskError(f"Volume timeout: {e}") from e
            
        except Exception as e:
            # Log memory state at failure (helps diagnose OOM)
            log_memory_snapshot(f"[LDV1] FAILED volume {ctx.volume.w_id}/{ctx.volume.i_id}", level=logging.ERROR)
            
            # Log detailed error information
            error_type = type(e).__name__
            logger.error(
                f"LDV1 pipeline failed for volume {ctx.volume.w_id}/{ctx.volume.i_id}: "
                f"{error_type}: {e}",
                exc_info=True
            )
            
            # Re-classify exceptions for orchestration layer
            from bec_orch.errors import RetryableTaskError, TerminalTaskError
            
            # Classify based on error type
            if 'CUDA out of memory' in str(e) or 'OOM' in str(e):
                logger.error("GPU out of memory error detected - task will be retried")
                raise RetryableTaskError(f"GPU OOM error: {e}") from e
            elif 'NotFound' in error_type or '404' in str(e):
                logger.error("Resource not found - task is terminal")
                raise TerminalTaskError(f"Resource not found: {e}") from e
            elif 'FileNotFoundError' == error_type:
                logger.error("File not found - task is terminal")
                raise TerminalTaskError(f"File not found: {e}") from e
            else:
                # Default: retryable
                logger.warning(f"Unclassified error ({error_type}) - task will be retried")
                raise RetryableTaskError(f"Pipeline error ({error_type}): {e}") from e
        
        # Calculate metrics
        elapsed_ms = (time.time() - start_time) * 1000
        total_images = metrics['total_images']
        
        # Get error counts from writer stats (more accurate than progress hook)
        nb_errors = error_stats.get('error_count', metrics['nb_errors'])
        nb_dropped = error_stats.get('dropped_count', 0)
        errors_by_stage = error_stats.get('errors_by_stage', {})
        
        # Average duration per image (from collected durations or fallback to elapsed/count)
        if metrics['image_durations']:
            avg_duration_per_page_ms = sum(metrics['image_durations']) / len(metrics['image_durations'])
            total_duration_ms = sum(metrics['image_durations'])
        else:
            # Fallback: use wall clock time
            total_duration_ms = elapsed_ms
            avg_duration_per_page_ms = elapsed_ms / max(1, total_images)
        
        # Log memory at end of volume processing
        log_memory_snapshot(f"[LDV1] End volume {ctx.volume.w_id}/{ctx.volume.i_id}")
        
        logger.info(
            f"LDV1 job completed: {total_images} images, {nb_errors} errors "
            f"({nb_dropped} dropped), avg {avg_duration_per_page_ms:.2f}ms/page"
        )
        
        # Check for errors and raise appropriate exceptions
        # (after logging so we have the metrics even on failure)
        from bec_orch.errors import RetryableTaskError, TerminalTaskError
        
        if nb_dropped > 0:
            # Dropped records indicate pipeline issues (timeout, backpressure)
            # This is retryable - the issue might be transient
            raise RetryableTaskError(
                f"Volume had {nb_dropped} dropped records (pipeline timeout/backpressure). "
                f"Total errors: {nb_errors}, errors by stage: {errors_by_stage}"
            )
        
        if nb_errors > 0:
            # Other processing errors are not retryable - the images have issues
            # that won't be fixed by retrying
            raise TerminalTaskError(
                f"Volume had {nb_errors} processing errors. "
                f"Errors by stage: {errors_by_stage}"
            )
        
        return TaskResult(
            total_images=total_images,
            nb_errors=nb_errors,
            total_duration_ms=total_duration_ms,
            avg_duration_per_page_ms=avg_duration_per_page_ms,
            nb_dropped_records=nb_dropped,
            errors_by_stage=errors_by_stage if errors_by_stage else None,
        )
    
    def _build_pipeline_config(self, ctx: 'JobContext') -> PipelineConfig:
        """
        Build PipelineConfig from job context.
        
        Starts with defaults and overrides with job config from DB.
        The model is already loaded in __init__ and will be attached to the config.
        """
        # Get model path from environment (for reference, but model is already loaded)
        model_path = os.environ.get('BEC_LD_MODEL_PATH', '').strip('"\'')
        
        # Start with default config
        defaults = {
            's3_bucket': ctx.artifacts_location.bucket,
            's3_region': os.environ.get('BEC_REGION', 'us-east-1'),
            'aws_profile': 'default',
            'model_path': model_path,  # Keep for reference/debugging
            'use_gpu': True,
            'precision': 'fp16',
            'batch_size': 16,
            'debug_mode': False,
        }
        
        # Merge with job config from DB
        config_dict = {**defaults, **ctx.job_config}
        
        # Create PipelineConfig instance
        cfg = PipelineConfig(**config_dict)
        
        # Attach the pre-loaded model to the config
        # This model stays on GPU and is reused across all volumes
        cfg.model = self.model
        
        return cfg
    
    def _build_volume_task(self, ctx: 'JobContext') -> VolumeTask:
        """
        Build VolumeTask from job context.
        """
        # Build list of ImageTask from manifest
        image_tasks = []
        
        # Get S3 folder prefix for source images
        from bec_orch.core.worker_runtime import get_s3_folder_prefix
        vol_prefix = get_s3_folder_prefix(ctx.volume.w_id, ctx.volume.i_id)
        
        for item in ctx.volume_manifest.manifest:
            filename = item.get('filename')
            if not filename:
                continue
            
            # Build S3 URI for source image
            source_uri = f"s3://archive.tbrc.org/{vol_prefix}{filename}"
            
            image_tasks.append(ImageTask(
                source_uri=source_uri,
                img_filename=filename
            ))
        
        # Build output URIs
        prefix = ctx.artifacts_location.prefix.rstrip('/')
        basename = ctx.artifacts_location.basename
        
        output_parquet_uri = f"s3://{ctx.artifacts_location.bucket}/{prefix}/{basename}.parquet"
        output_jsonl_uri = f"s3://{ctx.artifacts_location.bucket}/{prefix}/{basename}-errors.jsonl"
        
        # Debug folder (local only, not used in production)
        debug_folder_path = os.environ.get('BEC_DEBUG_FOLDER', '/tmp/bec_debug')
        
        return VolumeTask(
            io_mode='s3',
            debug_folder_path=debug_folder_path,
            output_parquet_uri=output_parquet_uri,
            output_jsonl_uri=output_jsonl_uri,
            image_tasks=image_tasks,
        )
