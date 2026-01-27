"""Shared data structures for OCR pipeline."""

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy.typing as npt

from .line import BBox

if TYPE_CHECKING:
    from .ctc_decoder import SyllableSegment
    from .line_decoder import ProcessedPage


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

    processed_page: "ProcessedPage"  # The processed page with line tensors
    expected_lines: int  # Number of lines expected for this page
    line_logits: dict[int, LineLogits]  # Collected LineLogits objects by line index


# =============================================================================
# New data structures for dual output format
# =============================================================================


@dataclass
class LineResult:
    """Result for a single OCR line with text, bounding box, and syllable details."""

    line_idx: int
    bbox: BBox
    text: str
    confidence: float
    syllables: list["SyllableSegment"]  # from CTC decoder, always populated


@dataclass
class PageOCRResult:
    """Complete OCR result for a page with structured line/syllable data."""

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

    page_idx: int
    filename: str
    source_etag: str
    texts: list[str]
    error: str | None = None
