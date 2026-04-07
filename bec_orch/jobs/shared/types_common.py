from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Optional, Callable, Dict, Union
import os
import logging
import time

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
class FetchedBytes:
    """Input descriptor for the decoder.
    """
    task: ImageTask
    source_etag: Optional[str]  # null for local pipeline
    file_bytes: bytes


@dataclass(frozen=True)
class DecodedFrame:
    """Output of the decode stage and transform stage.
    
    Note: Some fields (first_pass, rotation_angle, tps_data) are specific to
    LDV1's two-pass processing pipeline but are included here for compatibility.
    """
    task: ImageTask
    source_etag: Optional[str]
    frame: Any  # grayscale H, W, uint8
    orig_h: int
    orig_w: int
    is_binary: bool  # if the value space is {0, 255}
    first_pass: bool = True  # if it's a first pass for an image
    rotation_angle: Optional[float] = None  # if in the second pass, value of the rotation angle in degrees that was applied after first pass, null if first pass
    tps_data: Optional[Any] = None  # (input_pts, output_pts, alpha), if in the second pass, tps data that was applied after first pass, null if first pass


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


# --- For UI ----------------------------------------------------------------

ProgressEvent = Dict[str, Any]
ProgressHook = Callable[[ProgressEvent], None]


# --- Frame Tracking Helpers -------------------------------------------------

# These functions work with an optional tracker that can be set by job-specific code.
# The tracker interface is: touch(filename: str, step: str) and error(filename: str, step: str, error_msg: str)

_trace_logger = logging.getLogger("bec.frame_trace")
_FRAME_TRACE_ENABLED = os.environ.get("BEC_FRAME_TRACE", "0") == "1"

# Global tracker instance - can be set by job-specific code (e.g., LDV1's FrameTracker)
_global_tracker: Optional[Any] = None


def set_tracker(tracker: Optional[Any]) -> None:
    """Set the current frame tracker (can be job-specific tracker instance)."""
    global _global_tracker
    _global_tracker = tracker


def get_tracker() -> Optional[Any]:
    """Get the current frame tracker (if set)."""
    return _global_tracker


def trace_frame(stage: str, action: str, filename: str, extra: str = "") -> None:
    """
    Update frame tracker and optionally log (if BEC_FRAME_TRACE=1).
    
    This is the main function stages should call to track frame progress.
    Works with any tracker that has a touch(filename, step) method.
    """
    step = f"{stage}.{action}"
    tracker = _global_tracker
    if tracker and hasattr(tracker, "touch"):
        tracker.touch(filename, step)
    elif _FRAME_TRACE_ENABLED:
        # No tracker but tracing enabled - just log
        msg = f"[TRACE] {time.time():.3f} | {step:40s} | {filename}"
        if extra:
            msg += f" | {extra}"
        _trace_logger.warning(msg)


def trace_frame_error(stage: str, filename: str, error: str) -> None:
    """Record an error for a frame in the tracker.
    
    Works with any tracker that has an error(filename, step, error_msg) method.
    """
    step = f"{stage}.error"
    tracker = _global_tracker
    if tracker and hasattr(tracker, "error"):
        tracker.error(filename, step, error)
    elif _FRAME_TRACE_ENABLED:
        _trace_logger.error(f"[TRACE] {time.time():.3f} | {step:40s} | {filename} | ERROR: {error}")
