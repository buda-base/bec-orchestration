"""
Configuration dataclass for OCRV1 worker.

Provides a clean, typed interface for all OCR pipeline parameters.

Usage Examples:
    
    # 1. Basic usage with defaults
    from bec_orch.jobs.ocrv1.worker_async import OCRV1JobWorkerAsync
    from bec_orch.jobs.ocrv1.config import OCRV1Config
    
    worker = OCRV1JobWorkerAsync()  # Uses OCRV1Config() defaults
    # or explicit
    worker = OCRV1JobWorkerAsync(OCRV1Config())
    
    # 2. High accuracy mode (slower, best quality)
    config = OCRV1Config.high_accuracy()
    worker = OCRV1JobWorkerAsync(config)
    
    # 3. High speed mode (faster, lower quality)
    config = OCRV1Config.high_speed()
    worker = OCRV1JobWorkerAsync(config)
    
    # 4. Debug mode with image output
    config = OCRV1Config.debug(
        output_dir="debug_output",
        reference_file="test/reference_output_lines.txt"
    )
    worker = OCRV1JobWorkerAsync(config)
    
    # 5. Custom configuration
    config = OCRV1Config(
        prefetch_concurrency=128,
        image_processor_workers=32,
        gpu_batch_size=32,
        beam_width=50,
        min_line_padding=200,  # Ensure 200px total padding for edge-sensitive models
    )
    worker = OCRV1JobWorkerAsync(config)
    
    # 6. Modify preset config
    config = OCRV1Config.high_speed()
    config.min_line_padding = 100
    config.debug_output_dir = "debug_output"
    worker = OCRV1JobWorkerAsync(config)
"""

from dataclasses import dataclass, field
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .ctc_decoder import DEFAULT_WORD_DELIMITERS


@dataclass
class OCRV1Config:
    """
    Configuration for OCRV1 OCR pipeline.
    
    This class encapsulates all tunable parameters for the OCR worker,
    including concurrency settings, decoding parameters, and debug options.
    """
    
    # ============================================================================
    # Word Delimiters
    # ============================================================================
    
    word_delimiters: Optional[frozenset[str]] = None
    """
    Characters that trigger word boundaries for CTC decoding.
    - None (default): Use DEFAULT_WORD_DELIMITERS (Tibetan syllable-level)
    - Custom frozenset: Specify your own delimiters
    
    Import options from bec_orch.jobs.ocrv1.ctc_decoder:
    - DEFAULT_WORD_DELIMITERS: Tibetan syllable-level delimiters
    - TIBETAN_WORD_DELIMITERS: Same as DEFAULT
    - SPACE_ONLY_DELIMITERS: Original space-only behavior
    """
    
    # ============================================================================
    # Pipeline Concurrency & Workers
    # ============================================================================
    
    prefetch_concurrency: int = 64
    """Number of concurrent S3 fetches for line detection data and images."""
    
    image_processor_workers: int = 16
    """Number of thread pool workers for CPU-bound image processing (line extraction, preprocessing)."""
    
    ctc_workers: int = 8
    """Number of thread pool workers (or processes for NeMo) for CTC decoding."""
    
    gpu_batch_size: int = 16
    """Batch size for GPU inference. Larger = more GPU memory, potentially faster."""
    
    # ============================================================================
    # CTC Decoding Parameters
    # ============================================================================
    
    beam_width: Optional[int] = None
    """Beam width for CTC beam search. None = use decoder's default (typically 100)."""
    
    token_min_logp: Optional[float] = None
    """Minimum log probability for tokens during beam search. None = use decoder's default."""
    
    vocab_prune_threshold: Optional[float] = None
    """Vocabulary pruning threshold for beam search optimization. None = no pruning."""
    
    vocab_prune_mode: Optional[str] = None
    """Vocabulary pruning mode ('logp' or 'rank'). None = no pruning."""
    
    use_greedy_decode: bool = False
    """Use fast greedy decoding instead of beam search (ignores beam_width)."""
    
    use_hybrid_decode: bool = True
    """Use hybrid decode: greedy first, beam search fallback for low-confidence lines."""
    
    greedy_confidence_threshold: Optional[float] = None
    """Confidence threshold for hybrid decode. None = use decoder's default (-0.5)."""
    
    use_nemo_decoder: bool = False
    """Use NeMo GPU-accelerated CTC decoder instead of pyctcdecode (CPU)."""
    
    kenlm_path: Optional[str] = None
    """Path to KenLM language model file (.arpa or .bin) for NeMo decoder with LM support."""
    
    # ============================================================================
    # Pipeline Execution Mode
    # ============================================================================
    
    use_sequential_pipeline: bool = False
    """
    Run GPU inference sequentially before CTC decoding starts (instead of parallel).
    Useful for memory-constrained environments or debugging.
    """
    
    # ============================================================================
    # Debug & Development
    # ============================================================================
    
    debug_output_dir: Optional[str] = None
    """
    Directory to save debug line images (original + preprocessed).
    None = debug mode disabled. Example: "debug_output"
    """
    
    debug_reference_lines: Optional[list[str]] = None
    """
    Reference text lines for diff comparison in debug mode.
    Typically loaded from a reference output file.
    """
    
    def __post_init__(self):
        """Validate configuration parameters."""
        if self.prefetch_concurrency < 1:
            raise ValueError(f"prefetch_concurrency must be >= 1, got {self.prefetch_concurrency}")
        
        if self.image_processor_workers < 1:
            raise ValueError(f"image_processor_workers must be >= 1, got {self.image_processor_workers}")
        
        if self.ctc_workers < 1:
            raise ValueError(f"ctc_workers must be >= 1, got {self.ctc_workers}")
        
        if self.gpu_batch_size < 1:
            raise ValueError(f"gpu_batch_size must be >= 1, got {self.gpu_batch_size}")
        
        if self.beam_width is not None and self.beam_width < 1:
            raise ValueError(f"beam_width must be >= 1 or None, got {self.beam_width}")
        
        if self.use_nemo_decoder and self.use_greedy_decode:
            raise ValueError("use_nemo_decoder and use_greedy_decode are mutually exclusive")
        
        if self.vocab_prune_mode is not None and self.vocab_prune_mode not in ("logp", "rank"):
            raise ValueError(f"vocab_prune_mode must be 'logp', 'rank', or None, got {self.vocab_prune_mode}")
    
    @classmethod
    def default(cls) -> "OCRV1Config":
        """Create a config with default values (same as dataclass defaults)."""
        return cls()
    
    @classmethod
    def high_accuracy(cls) -> "OCRV1Config":
        """
        Configuration optimized for maximum accuracy (slower).
        
        - Large beam width (100)
        - No greedy decode
        - No vocab pruning
        """
        return cls(
            beam_width=100,
            use_greedy_decode=False,
            use_hybrid_decode=False,
            vocab_prune_threshold=None,
            vocab_prune_mode=None,
        )
    
    @classmethod
    def high_speed(cls) -> "OCRV1Config":
        """
        Configuration optimized for speed (lower accuracy).
        
        - Greedy decode only
        - Reduced workers
        - Larger GPU batch
        """
        return cls(
            use_greedy_decode=True,
            use_hybrid_decode=False,
            image_processor_workers=8,
            ctc_workers=4,
            gpu_batch_size=32,
        )
    
    @classmethod
    def balanced(cls) -> "OCRV1Config":
        """
        Balanced configuration (default settings).
        
        - Hybrid decode (greedy + beam fallback)
        - Moderate concurrency
        """
        return cls()
    
    @classmethod
    def debug(cls, output_dir: str = "debug_output", reference_file: Optional[str] = None) -> "OCRV1Config":
        """
        Configuration for debugging with image output.
        
        Args:
            output_dir: Directory to save debug images
            reference_file: Optional path to reference text file for diff comparison
        
        Returns:
            Config with debug mode enabled
        """
        debug_reference_lines = None
        if reference_file:
            with open(reference_file, "r", encoding="utf-8") as f:
                debug_reference_lines = [line.rstrip("\n") for line in f]
        
        return cls(
            debug_output_dir=output_dir,
            debug_reference_lines=debug_reference_lines,
            # Use sequential pipeline for easier debugging
            use_sequential_pipeline=True,
            # Reduce concurrency for clearer logs
            prefetch_concurrency=16,
            image_processor_workers=4,
            ctc_workers=2,
        )
