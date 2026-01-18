
from dataclasses import dataclass, field
from typing import Dict, Literal, Optional, Any

Precision = Literal["fp32", "fp16", "bf16", "auto"]

@dataclass
class PipelineConfig:
    """Configuration knobs for the pipeline.

    Key fields:
      - S3 concurrency caps (global/per-worker)
      - Bounded queue sizes per stage (backpressure)
      - GPU micro-batching (batch size/timeout)
      - Output prefixes (staging and final) and Parquet options
    """
    # S3
    s3_bucket: str
    s3_region: str = "us-east-1"
    aws_profile: str = "default"
    s3_max_inflight_global: int = 128  # GLOBAL cap - match bulk_prefetch_concurrency for max throughput
    inflight_per_worker: int = 24       # per-worker S3 GET concurrency or local concurrency
    s3_get_timeout_s: int = 60

    # Queues (bounded)
    max_q_prefetcher_to_decoder: int = 500  # Large enough to hold entire volume (~200MB raw bytes)
    max_q_decoder_to_tilebatcher: int = 32
    max_q_tilebatcher_to_inference: int = 8  # Allow more batches queued (reduces GPU starvation)
    max_q_gpu_pass_1_to_post_processor: int = 32
    max_q_post_processor_to_tilebatcher: int = 16
    max_q_gpu_pass_2_to_post_processor: int = 16
    max_q_post_processor_to_writer: int = 64


    # CPU decode threads
    decode_threads: int = 8
    tile_workers: int = 24  # ThreadPoolExecutor workers for parallel tiling (must outpace GPU)
    parallel_tiling: bool = True  # Enable parallel tiling in TileBatcher

    # GPU batching
    use_gpu: bool = True
    model_path: Optional[str] = None  # Path to model .pth file (required for inference)
    model: Optional[Any] = None  # Loaded torch.nn.Module (set by worker, reused across volumes)
    compile_model: bool = False
    precision: Precision = "fp16" # bf16, fp16, fp32 or auto
    pin_tile_memory: bool = True  # Pin tiled tensors for faster async GPU transfer
    # Optional cap for the internal tile pool to limit peak memory under backpressure.
    # 0 disables throttling.
    batch_size: int = 16  # Number of images per batch (16 images × ~5 tiles = ~80 tiles per batch)
    batch_timeout_ms: int = 25
    cuda_streams: int = 2
    binarize_block_size: int = 31
    binarize_c: int = 15
    batch_type: str = "tiles" # "tiles" for batching tiles (images batch mode removed)
    class_threshold: float = 0.85
    gpu_reinit_on_error = False
    gpu_reinit_on_oom = True
    reprocess_budget: int = 3  # Priority weight for reprocess lane (higher = more priority)
    controller_poll_ms: int = 5  # Polling timeout for post-processor

    # Output
    parquet_compression: str = "zstd"
    parquet_data_page_size: int = 65536  # 64KB; tune
    parquet_dictionary_enabled: bool = True
    schema_version: str = "v1"

    # Decoder
    frame_max_width = 4096
    frame_max_height = 2048
    patch_size = 512
    patch_vertical_overlap_px = 0 # about 15%
    patch_horizontal_overlap_px = 0
    snap_extra_patch_row_threshold_px = 78 # if a patch row would only have h=78px, downscale to remove the patch row
    max_patch_rows = 2 # resize so it fits into two patch rows (considering vertical overlap)
    linearize = True # remove gamma from jpeg encoding, should improve binarization
    normalize_background = False

    # LDTransform
    max_angle_deg: float = 5.0
    min_angle_deg: float = 0.3
    skip_pass2_rotation_threshold: float = 3.0  # Skip GPU pass-2 for rotations below this (just rotate contours)
    max_concurrent_transforms: int = 8  # Max concurrent rotation/TPS transforms (fire-and-forget)
    
    # Prefetch mode: bulk fetches ALL images into memory before processing
    # Best for S3 sources where images fit in RAM (~2GB max volume)
    bulk_prefetch: bool = True
    bulk_prefetch_concurrency: int = 128  # High concurrency for bulk mode (mimics aws s3 sync)
    
    # Warmup: wait for buffer to fill before starting GPU inference
    # Short warmup helps batch consistency without adding much latency
    inference_warmup_frames: int = 40  # ~5 batches worth, builds larger buffer before GPU starts
    
    # Max tiles per batch to prevent CUDA OOM (images have variable tile counts)
    max_tiles_per_batch: int = 80  # ~80 tiles = ~400MB GPU memory for forward pass
    
    # TileBatcher buffer limit to prevent memory accumulation
    # Each buffered frame holds ~41MB (8MB gray + 8MB orig frame + 25MB pinned tiles)
    # Set to 0 to disable limit (not recommended for large volumes)
    max_tilebatcher_buffer: int = 64  # ~2.6GB max in TileBatcher buffer
    tps_add_corners: bool = True
    tps_alpha: float = 0.5
    legacy_tps_detect: bool = False
    add_corners: bool = True

    # Artefact writer
    max_error_message_len = 128
    flush_every = 4096  # Flush Parquet buffer every N records
    
    # Debug
    debug_mode: bool = False
    debug_folder: Optional[str] = None  # Local folder path for debug output
    debug_images: Optional[set[str]] = None  # Set of image filenames to debug (None = all images)
    
    # GPU profiling
    enable_pytorch_profiler: bool = False  # Enable PyTorch profiler for GPU timing analysis
    profiler_trace_output: Optional[str] = None  # Output path for profiler trace (default: pytorch_trace.json)
    detailed_inference_timing: bool = True  # Log H2D/forward/stitch/D2H breakdown for each batch
    
    # Timeout settings (deadlock prevention)
    volume_timeout_s: float = 300.0  # Overall volume timeout (5 minutes)
    writer_eos_timeout_s: float = 10.0  # EOS wait timeout after all records received
    queue_put_timeout_s: float = 60.0  # Timeout for queue.put() operations
    transform_gather_timeout_s: float = 60.0  # Timeout waiting for pending transforms at EOS
    
    def __post_init__(self):
        """Validate configuration values."""

        # GPU settings
        if self.use_gpu:
            try:
                import torch
                if not torch.cuda.is_available():
                    raise ValueError("GPU requested but CUDA is not available")
            except ImportError:
                pass  # torch may not be available at config time
        
        # Overlap validations
        if not (0 <= self.patch_vertical_overlap_px < self.patch_size):
            raise ValueError(
                f"patch_vertical_overlap_px must be in [0, patch_size), "
                f"got {self.patch_vertical_overlap_px} (patch_size={self.patch_size})"
            )
        if not (0 <= self.patch_horizontal_overlap_px < self.patch_size):
            raise ValueError(
                f"patch_horizontal_overlap_px must be in [0, patch_size), "
                f"got {self.patch_horizontal_overlap_px} (patch_size={self.patch_size})"
            )
        
        # Timeout validations
        if self.batch_timeout_ms <= 0:
            raise ValueError(f"batch_timeout_ms must be > 0, got {self.batch_timeout_ms}")
        if self.s3_get_timeout_s <= 0:
            raise ValueError(f"s3_get_timeout_s must be > 0, got {self.s3_get_timeout_s}")
        
        # Batch type validation
        if self.batch_type != "tiles":
            raise ValueError(f"batch_type must be 'tiles', got '{self.batch_type}'")