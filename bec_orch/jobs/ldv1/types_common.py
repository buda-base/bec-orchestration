from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Optional, Union, Tuple, List, Callable, Dict

# --- Sentinel ---------------------------------------------------------------

@dataclass(frozen=True)
class EndOfStream:
    """
    Explicit end-of-stream marker for multi-lane pipelines.
    """
    stream: Literal["prefetched", "decoded", "tiled", "gpu_pass_1", "transformed_pass_1", "gpu_pass_2", "transformed_pass_2", "record"]
    producer: Optional[str] = None


# --- Core tasks / payloads --------------------------------------------------

@dataclass(frozen=True)
class ImageTask:
    """
    A unit of work.

    - source_uri: canonical location for the input bytes (s3://bucket/key or file:///abs/path)
    - img_filename: your existing basename used elsewhere (e.g. output naming / debug)
    """
    source_uri: str
    img_filename: str

@dataclass(frozen=True)
class VolumeTask:
    """Input descriptor for the prefetcher.
    """
    io_mode: str # "local" or "s3"
    debug_folder_path: str # local folder for debugging output, never on s3 (for now)
    output_parquet_uri: str
    output_jsonl_uri: str
    image_tasks: List[ImageTask]

@dataclass(frozen=True)
class FetchedBytes:
    """Input descriptor for the decoder.
    """
    task: ImageTask
    source_etag: Optional[str] # null for local pipeline
    file_bytes: bytes

@dataclass(frozen=True)
class DecodedFrame:
    """Output of the decode stage and transform stage
    """
    task: ImageTask
    source_etag: Optional[str]
    frame: Any # grayscale H, W, uint8
    orig_h: int
    orig_w: int
    is_binary: bool # if the value space is {0, 255}
    first_pass: bool # if it's a first pass for an image
    rotation_angle: Optional[float] # if in the second pass, value of the rotation angle in degrees that was applied after first pass, null if first pass
    tps_data: Optional[Any] # (input_pts, output_pts, alpha), if in the second pass, tps data that was applied after first pass, null if first pass

@dataclass(frozen=True)
class InferredFrame:
    """Output of the inference stage.
    """
    task: ImageTask
    source_etag: Optional[str]
    frame: Any
    orig_h: int
    orig_w: int
    is_binary: bool
    first_pass: bool
    rotation_angle: Optional[float]
    tps_data: Optional[Any]
    line_mask: Any # result of inference, H, W, uint8, binary {0, 255}, same H, W as frame

@dataclass(frozen=True)
class Record:
    """input for the Parquet writer, output of the transform stage
    """
    task: ImageTask
    source_etag: Optional[str]
    rotation_angle: Optional[float]
    tps_data: Optional[Any] # should be scaled to original image dimension
    contours: Optional[Any] # NDArray of (x,y) points, contours of line segments (not final merged lines), scaled to original image dimensions
    nb_contours: int
    contours_bboxes: Optional[Any] # bboxes (x, y, w, h) of the contours, scaled to original image dimensions

@dataclass
class TiledBatch:
    """
    Output of TileBatcher, input to LDInferenceRunner.
    
    Contains pre-tiled frames ready for GPU inference.
    Not frozen because it contains mutable tensors.
    """
    all_tiles: Any  # torch.Tensor [total_tiles, 3, patch_size, patch_size] on CPU
    tile_ranges: List[Tuple[int, int]]  # (start, end) indices for each frame
    metas: List[Dict[str, Any]]  # Metadata per frame (includes original DecodedFrame, gray, second_pass, etc.)

# --- Error envelope ---------------------------------------------------------

@dataclass(frozen=True)
class PipelineError:
    """Error message that can flow through queues."""
    stage: Literal["Prefetcher", "Decoder", "TileBatcher", "LDInferenceRunner", "LDGpuBatcher", "LDPostProcessor", "ParquetWriter", "Pipeline"]
    task: ImageTask
    source_etag: Optional[str]
    error_type: str
    message: str
    traceback: Optional[str] = None
    retryable: bool = False
    attempt: int = 1


# --- Queue message unions ---------------------------------------------------

FetchedBytesMsg = Union[FetchedBytes, PipelineError, EndOfStream]
DecodedFrameMsg = Union[DecodedFrame, PipelineError, EndOfStream]
TiledBatchMsg = Union[TiledBatch, PipelineError, EndOfStream]
InferredFrameMsg = Union[InferredFrame, PipelineError, EndOfStream]
RecordMsg = Union[Record, PipelineError, EndOfStream]

# --- For UI

ProgressEvent = Dict[str, Any]
ProgressHook = Callable[[ProgressEvent], None]


# --- Frame Tracking ---------------------------------------------------------

import os
import logging
import threading
import time
from dataclasses import dataclass as dc_dataclass, field as dc_field

_trace_logger = logging.getLogger("bec.frame_trace")
_FRAME_TRACE_ENABLED = os.environ.get("BEC_FRAME_TRACE", "0") == "1"


@dc_dataclass
class FrameState:
    """State of a single frame in the pipeline."""
    last_step: str
    error: Optional[str] = None


class FrameTracker:
    """
    Thread-safe tracker for frame progress through the pipeline.
    
    Each stage calls tracker.touch(filename, step) when it processes a frame.
    At the end, missing frames can be reported with their last known step.
    
    Usage:
        tracker = FrameTracker()
        tracker.touch("image001.jpg", "Decoder.received")
        tracker.touch("image001.jpg", "Decoder.decoded")
        tracker.touch("image001.jpg", "TileBatcher.buffered_pass1")
        ...
        # At end, get report for missing frames
        report = tracker.get_missing_report(expected_filenames, received_filenames)
    """
    
    def __init__(self):
        self._states: Dict[str, FrameState] = {}
        self._lock = threading.Lock()
    
    def touch(self, filename: str, step: str) -> None:
        """Update the last step for a frame. Thread-safe."""
        with self._lock:
            if filename not in self._states:
                self._states[filename] = FrameState(step)
            else:
                self._states[filename].last_step = step
        
        # Also log if tracing is enabled
        if _FRAME_TRACE_ENABLED:
            _trace_logger.warning(f"[TRACE] {time.time():.3f} | {step:40s} | {filename}")
    
    def error(self, filename: str, step: str, error_msg: str) -> None:
        """Record an error for a frame. Thread-safe."""
        with self._lock:
            self._states[filename] = FrameState(step, error_msg)
        
        # Also log if tracing is enabled
        if _FRAME_TRACE_ENABLED:
            _trace_logger.error(f"[TRACE] {time.time():.3f} | {step:40s} | {filename} | ERROR: {error_msg}")
    
    def get_state(self, filename: str) -> Optional[FrameState]:
        """Get the current state of a frame. Thread-safe."""
        with self._lock:
            return self._states.get(filename)
    
    def get_last_step(self, filename: str) -> str:
        """Get just the last step for a frame, or 'never_seen'."""
        with self._lock:
            state = self._states.get(filename)
            return state.last_step if state else "never_seen"
    
    def get_missing_info(self, missing_filenames: Iterable[str]) -> Dict[str, FrameState]:
        """
        Get state info for a set of missing filenames.
        
        Returns dict mapping filename -> FrameState (or None if never seen).
        """
        with self._lock:
            return {f: self._states.get(f) for f in missing_filenames}
    
    def get_missing_report(self, expected: Iterable[str], received: Iterable[str], max_items: int = 10) -> str:
        """
        Generate a human-readable report of missing frames with their last known steps.
        """
        missing = set(expected) - set(received)
        if not missing:
            return "No missing frames"
        
        with self._lock:
            lines = []
            for f in sorted(missing)[:max_items]:
                state = self._states.get(f)
                if state:
                    if state.error:
                        lines.append(f"  {f}: last_step={state.last_step}, error={state.error}")
                    else:
                        lines.append(f"  {f}: last_step={state.last_step}")
                else:
                    lines.append(f"  {f}: never_seen (not in manifest or fetch failed silently)")
            
            if len(missing) > max_items:
                lines.append(f"  ... and {len(missing) - max_items} more")
            
            return "\n".join(lines)


# Global tracker instance - will be replaced per-volume by LDVolumeWorker
_global_tracker: Optional[FrameTracker] = None


def get_tracker() -> Optional[FrameTracker]:
    """Get the current frame tracker (if set)."""
    return _global_tracker


def set_tracker(tracker: Optional[FrameTracker]) -> None:
    """Set the current frame tracker."""
    global _global_tracker
    _global_tracker = tracker


# Convenience functions that use the global tracker
def trace_frame(stage: str, action: str, filename: str, extra: str = "") -> None:
    """
    Update frame tracker and optionally log (if BEC_FRAME_TRACE=1).
    
    This is the main function stages should call to track frame progress.
    """
    step = f"{stage}.{action}"
    tracker = _global_tracker
    if tracker:
        tracker.touch(filename, step)
    elif _FRAME_TRACE_ENABLED:
        # No tracker but tracing enabled - just log
        msg = f"[TRACE] {time.time():.3f} | {step:40s} | {filename}"
        if extra:
            msg += f" | {extra}"
        _trace_logger.warning(msg)


def trace_frame_error(stage: str, filename: str, error: str) -> None:
    """Record an error for a frame in the tracker."""
    step = f"{stage}.error"
    tracker = _global_tracker
    if tracker:
        tracker.error(filename, step, error)
    elif _FRAME_TRACE_ENABLED:
        _trace_logger.error(f"[TRACE] {time.time():.3f} | {step:40s} | {filename} | ERROR: {error}")