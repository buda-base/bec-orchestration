"""Shared data structures for OCR pipeline.

This module defines:
- Pipeline message types (EndOfStream, PipelineError)
- Intermediate data structures (LineLogits, InferredPage, etc.)
- Output data structures (PageOCRResult, LineResult, etc.)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from .line_decoder import PrefetchedBytes, ProcessedPage

if TYPE_CHECKING:
    import numpy.typing as npt

    from .ctc_decoder import SyllableSegment
    from .gpu_inference import InferredPage
    from .line import BBox
    from .output_writer import PageOCRResult

# =============================================================================
# Task Identity
# =============================================================================


@dataclass(frozen=True)
class ImageTask:
    """
    Identity of a single image/page to process.

    Frozen dataclass that uniquely identifies a page in the pipeline.
    Replaces tuple-based (page_idx, filename) passing.
    """

    page_idx: int
    filename: str


# =============================================================================
# Pipeline Control Messages
# =============================================================================


@dataclass(frozen=True)
class EndOfStream:
    """
    Explicit end-of-stream marker for pipeline stages.

    Each stage sends this when it has finished producing messages.
    The stream field identifies which queue this EOS is for.
    """

    stream: Literal["fetched", "processed", "inferred", "results"]
    producer: str | None = None


@dataclass(frozen=True)
class PipelineError:
    """
    Error message that can flow through queues.

    When a stage encounters an error processing a page, it creates a PipelineError
    and sends it downstream instead of the normal data. This allows:
    - Errors to be tracked and logged by the writer
    - Other pages to continue processing
    - Consistent error handling across all stages
    """

    stage: Literal[
        "Prefetcher",
        "ImageProcessor",
        "GPUInference",
        "CTCDecoder",
        "OutputWriter",
        "Pipeline",
    ]
    task: ImageTask
    source_etag: str | None
    error_type: str
    message: str

    # Convenience properties for backward compatibility
    @property
    def page_idx(self) -> int:
        return self.task.page_idx

    @property
    def filename(self) -> str:
        return self.task.filename


# =============================================================================
# Internal Pipeline Data Structures
# =============================================================================


@dataclass
class LineLogits:
    """Logits data for a single line from GPU inference.

    This structure encapsulates all the data needed for CTC decoding
    of a single line, including the logits tensor and metadata.
    """

    logits: npt.NDArray  # Logits tensor, already cropped and in (time, vocab) shape
    content_width: int  # Width of content region in original tensor (before padding)
    left_pad_width: int  # Width of left padding in original tensor
    keep_indices: npt.NDArray | None  # Indices of kept vocabulary tokens after pruning

    @property
    def vocab_size(self) -> int:
        """Get the effective vocabulary size for these logits."""
        return self.keep_indices.shape[0] if self.keep_indices is not None else self.logits.shape[1]


@dataclass
class PendingTensor:
    """Data for a tensor waiting to be processed by GPU inference.

    This structure encapsulates the tensor and its metadata needed
    for batching and later result distribution.
    """

    page_idx: int  # Page index this line belongs to
    line_idx: int  # Line index within the page
    tensor: npt.NDArray  # The tensor to process
    content_width: int  # Width of content region (actual text)
    left_pad_width: int  # Width of left padding


@dataclass
class PageInFlight:
    """Data for a page being processed by GPU inference.

    This structure tracks a page through the inference process,
    including the processed page data and collected line results.
    """

    processed_page: ProcessedPage  # The processed page with line tensors
    expected_lines: int  # Number of lines expected for this page
    line_logits: dict[int, LineLogits]  # Collected LineLogits objects by line index


@dataclass
class InferredPage:
    """
    Output of GPU inference stage, input to CTC decoder.

    Contains the logits for all lines in a page, plus the original
    ProcessedPage data needed for building the final result.
    """

    task: ImageTask
    source_etag: str
    logits_list: list[LineLogits]
    processed_page: ProcessedPage
    error: str | None = None

    # Convenience properties
    @property
    def page_idx(self) -> int:
        return self.task.page_idx

    @property
    def filename(self) -> str:
        return self.task.filename


# =============================================================================
# Output Data Structures
# =============================================================================


@dataclass
class LineResult:
    """Result for a single OCR line with text, bounding box, and syllable details."""

    line_idx: int
    bbox: BBox
    text: str
    confidence: float
    syllables: list[SyllableSegment]  # from CTC decoder


@dataclass
class PageOCRResult:
    """Complete OCR result for a page with structured line/segment/syllable data."""

    img_file_name: str
    source_etag: str
    rotation_angle: float
    tps_points: tuple | None
    lines: list[LineResult]
    error: str | None = None

    @property
    def page_text(self) -> str:
        """Get the full page text by joining all lines."""
        return "\n".join(line.text for line in self.lines)

    @property
    def page_confidence(self) -> float:
        """Get the overall page confidence as weighted average by character count."""
        if not self.lines:
            return 0.0

        total_weighted_confidence = sum(line.confidence * len(line.text) for line in self.lines)
        total_chars = sum(len(line.text) for line in self.lines)

        return total_weighted_confidence / total_chars if total_chars > 0 else 0.0


@dataclass
class PageResult:
    """Legacy page result for backward compatibility."""

    task: ImageTask
    source_etag: str
    texts: list[str]
    error: str | None = None

    # Convenience properties
    @property
    def page_idx(self) -> int:
        return self.task.page_idx

    @property
    def filename(self) -> str:
        return self.task.filename


# =============================================================================
# Queue Message Type Unions
# =============================================================================

# Each queue can receive one of these message types:
# - The normal data payload
# - A PipelineError (if processing failed for this page)
# - EndOfStream (when the producer is done)

# q_fetched: Prefetcher -> ImageProcessor (PrefetchedBytes, no LD metadata yet)
PrefetchedBytesMsg = PrefetchedBytes | PipelineError | EndOfStream

# q_processed: ImageProcessor -> GPUInference
ProcessedPageMsg = ProcessedPage | PipelineError | EndOfStream

# q_inferred: GPUInference -> CTCDecoder
InferredPageMsg = InferredPage | PipelineError | EndOfStream

# q_results: CTCDecoder -> OutputWriter
PageOCRResultMsg = PageOCRResult | PipelineError | EndOfStream
