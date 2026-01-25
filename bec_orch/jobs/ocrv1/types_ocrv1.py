"""
Type system for OCRv1 pipeline.

Following ldv1 pattern: clean dataclasses with frozen=True for pipeline messages.
Each stage has well-defined input/output types.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Optional, Union

import numpy.typing as npt


# ============================================================================
# Sentinel
# ============================================================================

@dataclass(frozen=True)
class EndOfStream:
    """Explicit end-of-stream marker for pipeline stages."""
    stream: Literal["prefetched", "processed", "inferred", "decoded", "written"]
    producer: Optional[str] = None


# ============================================================================
# Core Data Types
# ============================================================================

@dataclass(frozen=True)
class LDResult:
    """
    Structured line detection results (replaces untyped dict).
    
    Contains transformation parameters and detected line contours.
    """
    rotation_angle: float
    tps_points: Optional[tuple]  # (input_pts, output_pts) if TPS needed
    tps_alpha: float
    contours: list  # List of line contour points


@dataclass(frozen=True)
class FetchedBytes:
    """
    Output of Prefetcher, input to LineProcessor.
    
    Contains raw image bytes from S3 and line detection metadata.
    """
    page_idx: int
    filename: str
    source_etag: str
    file_bytes: bytes
    ld_data: LDResult


@dataclass(frozen=True)
class LineTensor:
    """
    A single preprocessed line ready for model inference.
    
    Tensor is already resized, binarized, and normalized to [-1, 1].
    """
    tensor: npt.NDArray  # (1, H, W) float32, normalized to [-1, 1]
    original_width: int  # Original line width before model resize (for CTC time-axis cropping)


@dataclass(frozen=True)
class ProcessedPage:
    """
    Output of LineProcessor, input to GPUBatcher.
    
    Contains list of preprocessed line tensors ready for GPU inference.
    """
    page_idx: int
    filename: str
    source_etag: str
    lines: list[LineTensor]
    error: Optional[str] = None


@dataclass(frozen=True)
class LineLogits:
    """
    Inference results for a single line.
    
    Logits are already cropped to original_width to match the line's actual content.
    """
    logits: npt.NDArray  # (time, vocab) or (vocab, time) - already cropped
    original_width: int  # Used for CTC decoding
    keep_indices: Optional[npt.NDArray]  # Vocabulary pruning indices (if pruning enabled)


@dataclass(frozen=True)
class InferredPage:
    """
    Output of GPUBatcher, input to CTCDecoder.
    
    Contains logits for all lines in the page.
    """
    page_idx: int
    filename: str
    source_etag: str
    logits: list[LineLogits]
    error: Optional[str] = None


@dataclass(frozen=True)
class PageResult:
    """
    Output of CTCDecoder, input to ParquetWriter.
    
    Contains decoded text for all lines in the page.
    """
    page_idx: int
    filename: str
    source_etag: str
    texts: list[str]
    error: Optional[str] = None


# ============================================================================
# Error Handling
# ============================================================================

@dataclass(frozen=True)
class PipelineError:
    """
    Error that can flow through queues.
    
    Allows errors to be tracked and written to output alongside successful results.
    """
    stage: Literal["Prefetcher", "LineProcessor", "GPUBatcher", "CTCDecoder", "ParquetWriter"]
    page_idx: int
    filename: str
    source_etag: str
    error_type: str
    message: str
    traceback: Optional[str] = None
    retryable: bool = False


# ============================================================================
# Queue Message Unions
# ============================================================================

FetchedBytesMsg = Union[FetchedBytes, PipelineError, EndOfStream]
ProcessedPageMsg = Union[ProcessedPage, PipelineError, EndOfStream]
InferredPageMsg = Union[InferredPage, PipelineError, EndOfStream]
PageResultMsg = Union[PageResult, PipelineError, EndOfStream]
