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