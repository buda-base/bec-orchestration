"""
Configuration for OCRV1 pipeline.

All configuration options in one place for easy tuning and reference.
"""

from dataclasses import dataclass, field

# Tibetan word/syllable delimiters for CTC decoding
# These characters trigger word boundaries for language model scoring and frame tracking.
TIBETAN_WORD_DELIMITERS: frozenset[str] = frozenset(
    {
        "་",  # tsheg (most common syllable separator)
        "༌",  # tsheg-like / non-breaking tsheg
        " ",  # space
        "།",  # shad (sentence/phrase delimiter)
        "༴",  # sbrul-shad (repetition mark)
        "ཿ",  # visarga / rnam-bcad
        "࿒",  # head mark
        "༼",  # opening bracket
        "༽",  # closing bracket
        "࿙",  # leading ornament
        "࿚",  # trailing ornament
        "༔",  # ter-tsheg / gter-tsheg
    }
)

# Space-only delimiters - original behavior for backward compatibility
SPACE_ONLY_DELIMITERS: frozenset[str] = frozenset({" "})

# Default word delimiters
DEFAULT_WORD_DELIMITERS: frozenset[str] = TIBETAN_WORD_DELIMITERS


@dataclass
class OCRV1Config:
    """Configuration for OCRV1 pipeline.

    Groups all configuration options for easy passing between components.
    """

    # -------------------------------------------------------------------------
    # Required fields (no defaults - must come first)
    # -------------------------------------------------------------------------
    model: str  # Model name/subdirectory: "Woodblock", "Ume_Druma", etc. (REQUIRED)

    # -------------------------------------------------------------------------
    # Model configuration (set from model_config.json)
    # -------------------------------------------------------------------------
    apply_log_softmax: bool = False  # Set to false for the new version of the model

    # -------------------------------------------------------------------------
    # Model Type Configuration (for different models like ume)
    # -------------------------------------------------------------------------
    model_variant: str | None = None  # Model variant for fine-tuning
    confidence_threshold: float = 0.5  # Confidence threshold for filtering results
    max_line_length: int = 1000  # Maximum line length to process

    # -------------------------------------------------------------------------
    # Advanced Processing Options
    # -------------------------------------------------------------------------
    enable_line_merge: bool = True  # Enable line segment merging
    line_merge_threshold: float | None = None  # Threshold for line merging (None = auto)
    enable_rotation_correction: bool = True  # Enable rotation correction
    rotation_confidence_threshold: float = 0.7  # Confidence threshold for rotation

    # -------------------------------------------------------------------------
    # Performance Tuning
    # -------------------------------------------------------------------------
    enable_model_caching: bool = True  # Cache model in memory
    cache_size_mb: int = 1024  # Model cache size in MB

    # -------------------------------------------------------------------------
    # Image processing
    # -------------------------------------------------------------------------
    max_image_width: int = 6000  # Maximum image width before scaling
    max_image_height: int = 3000  # Maximum image height before scaling
    use_line_prepadding: bool = True  # Add h*h padding on left/right before resize
    merge_line_segments: bool = True  # Merge contours that belong to same line

    # -------------------------------------------------------------------------
    # Pipeline parallelism
    # -------------------------------------------------------------------------
    prefetch_concurrency: int = 64  # Max concurrent S3 fetches
    image_processor_workers: int = 16  # Thread pool size for image processing
    ctc_workers: int = 8  # Process pool size for CTC decoding
    gpu_batch_size: int = 16  # Batch size for GPU inference
    use_sequential_pipeline: bool = False  # Run GPU first, then CTC (reduces contention)
    volume_timeout_s: float = 600.0  # Timeout for volume processing (10 minutes)

    # -------------------------------------------------------------------------
    # CTC Decoding
    # -------------------------------------------------------------------------
    beam_width: int = 64  # Beam width for beam search
    token_min_logp: float = -3.0  # Skip tokens with log prob below this
    vocab_prune_threshold: float | None = -10.0  # Vocabulary pruning threshold (None = no pruning)
    vocab_prune_mode: str | None = "line"  # Vocabulary pruning mode ('line', 'batch', etc.)
    use_greedy_decode: bool = False  # Use fast greedy decode (less accurate)
    use_hybrid_decode: bool = True  # Greedy + beam search fallback
    greedy_confidence_threshold: float = -0.5  # Threshold for hybrid decode
    kenlm_path: str | None = None  # Path to KenLM language model
    word_delimiters: frozenset[str] = field(default_factory=lambda: DEFAULT_WORD_DELIMITERS)

    # -------------------------------------------------------------------------
    # Output Configuration
    # -------------------------------------------------------------------------
    enable_jsonl_output: bool = False  # Write JSONL.gz output alongside Parquet

    # -------------------------------------------------------------------------
    # Debug
    # -------------------------------------------------------------------------
    debug_output_dir: str | None = None  # Directory to save preprocessed line images
