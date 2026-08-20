"""Configuration for the PaddleOCR-VL OCR job(s).

One dataclass drives every PaddleOCR-VL job. ``paddleocr_v1`` uses the
defaults below verbatim; ``paddleocr_v2`` reuses the same checkpoint and
turns on ``layout_mask_enabled`` so header/footer regions from
``layout_detection_v1`` are painted with page-background colour before OCR.

Serving stack (see ``bec-ocr-training/docs/eval_in_production.md``):
    **vLLM + DRY-a12 + adaptive per-page resolution**, greedy decoding.
The checkpoint (`grow26-ep2`, depth-upscaled PaddleOCR-VL-1.6 with a warmed
3560-token Tibetan **unicode-stack** tokenizer) emits **Tibetan Unicode
directly** — there is NO Wylie/EWTS step; post-processing is canonical Unicode
normalization only.

All fields are overridable via the JSON ``config`` of ``bec jobs create``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Literal

Precision = Literal["bfloat16", "float16", "float32", "auto"]

# ---------------------------------------------------------------------------
# v1 defaults
# ---------------------------------------------------------------------------
# Fine-tuned, self-contained checkpoint (weights + processor + tokenizer),
# synced to local disk on worker startup (see model_sync.py). A ``paddleocr_v2``
# job simply points this at a different prefix.
DEFAULT_CHECKPOINT_S3_URI = (
    "s3://bec.bdrc.io/checkpoints/PaddleOCR/elie_v6_coarse_grow26_ep2/final/"
)

# Single user-turn prompt. Rendered via ``processor.apply_chat_template`` with
# ``add_generation_prompt=True`` (image + this text, one user message).
DEFAULT_PROMPT = "Extract all Tibetan text. Preserve line breaks."


def _default_model_cache_root() -> str:
    """Root dir under which each checkpoint is synced locally.

    Overridable via ``BEC_PADDLEOCR_MODEL_CACHE`` so the systemd unit can
    point it at a writable, systemd-managed cache dir.
    """
    return os.environ.get("BEC_PADDLEOCR_MODEL_CACHE", "/var/cache/bec-paddleocr/models")


@dataclass
class PaddleOCRConfig:
    """Tunable parameters for a PaddleOCR-VL OCR job (vLLM serving)."""

    # ------------------------------------------------------------------
    # Model / checkpoint
    # ------------------------------------------------------------------
    # S3 prefix of the fine-tuned checkpoint. Synced to disk on startup.
    checkpoint_s3_uri: str = DEFAULT_CHECKPOINT_S3_URI

    # Where checkpoints are cached locally. The actual model dir is derived
    # from the checkpoint prefix (see ``resolved_model_dir``) so different
    # versions never collide.
    model_cache_root: str = field(default_factory=_default_model_cache_root)

    # Hard override of the local model dir (skips the derived path). Usually
    # left None; handy for pointing at a pre-baked AMI location.
    model_local_dir: str | None = None

    # vLLM compute dtype. bf16 on Ampere+ (A10G/L40S/Blackwell). T4 (sm_75)
    # has no bf16 — set "float16" + enforce_eager there.
    dtype: Precision = "bfloat16"

    # Native vLLM path (PaddleOCRVLForConditionalGeneration). The synced
    # checkpoint is patched (image_token + processor size) by model_sync so
    # trust_remote_code stays False.
    trust_remote_code: bool = False

    # Image-token M-RoPE regime. grow26-ep2 was trained/validated with
    # ``mm_token_type_ids`` zeroed ("sequential"), but stock vLLM ALWAYS serves
    # the 2D image-grid regime ("grid") — it never reads mm_token_type_ids — so
    # serving a sequential checkpoint under stock vLLM is a train/serve mismatch
    # (line skips/merges on structured pages, large CER regression). Serving
    # "sequential" requires the vendored ``vllm_paddleocr_seqpos`` general plugin
    # installed into the vLLM venv; the worker sets
    # ``OCR_VLLM_IMAGE_TOKEN_POSITIONS`` before building the engine and refuses to
    # start "sequential" if the plugin is missing. "grid" = stock vLLM (no
    # plugin). "auto" reads the checkpoint's experiment_config.json
    # ``image_token_positions`` (defaulting to sequential for older checkpoints).
    image_token_positions: Literal["sequential", "grid", "auto"] = "sequential"

    # Checkpoint files NOT needed for inference (skip on sync). The training
    # state is large/irrelevant; the SentencePiece ``tokenizer.model`` is only
    # for llama.cpp (native/vLLM use ``tokenizer.json``) but is tiny, so we
    # keep it.
    sync_exclude_suffixes: tuple[str, ...] = (".pt",)
    sync_exclude_names: tuple[str, ...] = ("trainer_state.json",)

    # ------------------------------------------------------------------
    # vLLM engine knobs (see docs/eval_in_production.md "throughput tuning")
    # ------------------------------------------------------------------
    gpu_memory_utilization: float = 0.95
    max_model_len: int = 4096
    max_num_seqs: int = 512
    max_num_batched_tokens: int = 16384
    # CUDA graphs off only if RAM/toolkit is tight (e.g. T4). Default keeps them.
    enforce_eager: bool = False
    # OCR requests share no prefix and each image is unique, so prefix / mm
    # caching only waste memory.
    enable_prefix_caching: bool = False
    mm_processor_cache_gb: float = 0.0
    # On-the-fly quant (e.g. "fp8", needs sm_89+). None = bf16 (measured best).
    quantization: str | None = None
    # vLLM's flashinfer sampler JIT-compiles a kernel and needs nvcc; greedy
    # OCR doesn't need it, so disable it on toolkit-less boxes (sets
    # VLLM_USE_FLASHINFER_SAMPLER=0 before importing vllm).
    disable_flashinfer_sampler: bool = True

    # ------------------------------------------------------------------
    # DRY repetition guard (the ONLY loop guard that works on Tibetan OCR;
    # docs/experiments/loop_hallucination.md E7). Registered once on the vLLM
    # engine and enabled per request via SamplingParams.extra_args.
    #   multiplier=0.8, base=1.75, allowed_length=12, NO sequence breakers.
    # allowed_length=12 measured strictly better than 8 on grow26-ep2.
    # ------------------------------------------------------------------
    dry_multiplier: float = 0.8
    dry_base: float = 1.75
    dry_allowed_length: int = 12
    dry_window: int = 512
    dry_max_match: int = 50
    # Shad breakers BACKFIRE on Tibetan (loops are shad-delimited) — keep empty.
    dry_sequence_breakers: tuple[int, ...] = ()

    # ------------------------------------------------------------------
    # DRY fire telemetry + high-severity temperature retry
    # ------------------------------------------------------------------
    # DRY fires on ~57% of pages but almost always as a mild single-token nip;
    # only ~2% fire hard (fires>=100) and are worth re-decoding. The per-page
    # ``dry_fires`` count (+ ``dry_max_L``, ``retried``) is recorded in the
    # parquet. When ``dry_retry_temp > 0``, pages whose DRY fire count is
    # >= ``dry_retry_min_fires`` (OR whose leftover rep_score >=
    # ``dry_retry_min_rep``) are re-decoded at ``dry_retry_temp`` with
    # ``dry_retry_n`` samples (DRY still on); the sample with the lowest rep_score
    # (then fewest tokens) is kept and the row is flagged ``retried=True``.
    # ``fires>=100`` is the measured knee (docs/eval_in_production.md: retrying
    # only that ~2% moves micro-CER 0.0815 -> 0.0810; retrying every fire instead
    # REGRESSES to 0.0868 and doubles cost). Set ``dry_retry_temp=0`` to disable.
    dry_retry_temp: float = 0.4
    dry_retry_n: int = 3
    dry_retry_min_fires: int = 100
    # Leftover-loop safety floor: retry a page below the fire gate if its
    # rep_score >= this (0 disables). Mirrors rep_score_threshold (hard loop).
    dry_retry_min_rep: float = 0.5
    # Seed for the temperature retry pass (reproducible sampling).
    dry_retry_seed: int = 0

    # ------------------------------------------------------------------
    # Image preprocessing + adaptive per-page resolution
    # ------------------------------------------------------------------
    convert_to_rgb: bool = True

    # Cheap upper bound applied at decode (pyvips thumbnail) before the
    # adaptive router; keeps huge modern scans from blowing up memory.
    max_longest_side: int = 3500

    # Processor ``size.shortest_edge`` (matches the trainer eval min edge).
    processor_shortest_edge: int = 1024
    # 1x pixel budget = base_pixel_size * 28 * 28 (== processor longest_edge /
    # max_pixels). 1280 -> 1003520, the budget grow26-ep2 was scored at.
    base_pixel_size: int = 1280

    # Adaptive resolution: vision prefill dominates GPU time and vision-token
    # count ~= pixels/28^2, so the per-page pixel budget is the main speed
    # lever. The router downsizes each page to the smallest candidate budget
    # that keeps the glyph body (p75 connected-component height) >= res_tfloor
    # px. p75/tfloor=24 reproduces the near-free-lunch operating point
    # (GPU-confirmed: ~10% faster on A10G, ~4% on Blackwell, CER <= 1x).
    res_mode: Literal["adaptive", "fixed"] = "adaptive"
    res_scales: tuple[float, ...] = (0.6, 0.75, 1.0)
    res_tfloor: float = 24.0
    res_percentile: float = 75.0
    # ``fixed`` mode only: fraction of the 1x budget applied to every page.
    res_budget_scale: float = 1.0

    # ------------------------------------------------------------------
    # Prompt
    # ------------------------------------------------------------------
    prompt: str = DEFAULT_PROMPT

    # ------------------------------------------------------------------
    # Generation (greedy). DRY (above) is the loop guard — do NOT use
    # repetition_penalty / no_repeat_ngram_size (they wreck clean Tibetan).
    # ------------------------------------------------------------------
    # 2048 is a free cap: under DRY the token-length p99 is ~1783, max ~2133,
    # only ~1/1070 pages exceed 2048 (docs/eval_in_production.md).
    max_new_tokens: int = 2048
    # Greedy.
    temperature: float = 0.0
    # Keep at 1.0 (disabled). Only sent to vLLM when != 1.0.
    repetition_penalty: float = 1.0

    # ------------------------------------------------------------------
    # Post-processing (Unicode normalization) + loop self-flag
    # ------------------------------------------------------------------
    # The model emits Tibetan Unicode directly; postprocess.py applies the
    # canonical normalize_unicode_text (NFD reorder + graphical fold), matching
    # the training/eval scorer exactly. NO Wylie/EWTS conversion.

    # rep_score = 1 - unique(n-grams)/total(n-grams) on the raw prediction
    # (syllable tokens split on tsheg/whitespace).
    rep_ngram_size: int = 20
    # If rep_score >= threshold the page is flagged ``likely_loop`` for review.
    # DRY makes hard loops rare, so this is a belt-and-suspenders review flag.
    rep_score_threshold: float = 0.5

    # ------------------------------------------------------------------
    # Batching / IO
    # ------------------------------------------------------------------
    # Pages fetched+decoded and submitted to vLLM per cycle. vLLM does its own
    # continuous batching (up to max_num_seqs); this only bounds how many
    # decoded images are resident at once. Must be >= 1.
    ocr_batch_size: int = 128

    s3_fetch_concurrency: int = 32
    s3_get_timeout_s: int = 30
    s3_max_attempts: int = 3

    # ------------------------------------------------------------------
    # Output writer
    # ------------------------------------------------------------------
    parquet_flush_every: int = 256
    parquet_compression: Literal["zstd", "snappy", "gzip", "none"] = "zstd"
    write_errors_jsonl: bool = True

    # ------------------------------------------------------------------
    # script_classification_v2 pre-filter
    # ------------------------------------------------------------------
    # Skip OCR on pages whose already-computed script_classification_v2 label
    # is in ``filter_skip_labels`` (e.g. blank / non-Tibetan / non-plain-text).
    # The sibling job's parquet for the SAME volume+version is located
    # automatically next to this job's output prefix. Skipped pages are still
    # recorded in the parquet (``skipped=True`` + ``skip_reason``) and do NOT
    # count toward the failure rate.
    filter_enabled: bool = True

    # Sibling classification job whose output gates OCR. (For a future
    # classifier, only this + labels change.)
    filter_job_name: str = "script_classification_v2"

    # Labels to skip. The 8-class model emits: danyig_pedri, druma,
    # gyuyig_tsugdri, multiscript, non_tibetan, uchen, blank, nonplaintext.
    filter_skip_labels: tuple[str, ...] = ("blank", "non_tibetan", "nonplaintext")

    # Also skip pages with predicted ``prob`` below this value (0 = disabled).
    # Per the script_classification_v2 README, blank/pure-white scans collapse
    # to a near-uniform softmax (argmax prob ~0.13) and the ``blank`` label
    # does NOT reliably win; there is a wide empty gap between ~0.40 and ~0.84,
    # so 0.30 cleanly isolates blank/uncertain pages (only ~0.3% of real content
    # falls below it) and catches the blanks the ``blank`` label misses.
    filter_min_prob: float = 0.30

    # If True, a missing classification artifact makes the volume a terminal
    # failure. If False (default), the worker logs a warning and OCRs every
    # page (no filtering) so a not-yet-classified volume still gets processed.
    filter_required: bool = False

    # ------------------------------------------------------------------
    # layout_detection_v1 header/footer background fill (paddleocr_v2)
    # ------------------------------------------------------------------
    # When True, load the sibling ``layout_detection_v1`` parquet for the same
    # volume+version and paint detected header/footer boxes with an estimate of
    # the page background colour before OCR. Footnotes are never painted.
    # ``paddleocr_v1`` leaves this False; ``paddleocr_v2`` turns it on at
    # registration. Missing layout output is a warning (OCR proceeds unmasked)
    # unless ``layout_mask_required`` is True.
    layout_mask_enabled: bool = False

    layout_mask_job_name: str = "layout_detection_v1"

    # Class names to blank. ``layout_detection_v1`` emits header / text-area /
    # footnote / footer; we only want running titles and folio numbers gone.
    layout_mask_labels: tuple[str, ...] = ("header", "footer")

    # Regions that must never be blanked by the header/footer pass, even if a
    # header/footer box overlaps them. Footnote is listed so the H/F pass
    # cannot wipe notes before they are cropped for isolated OCR.
    layout_mask_protect_labels: tuple[str, ...] = ("text-area", "footnote")

    # Extra pixels grown around each painted box (helps slightly tight
    # detections). 0 = exact box.
    layout_mask_pad_px: int = 2

    # If True, a missing layout artifact makes the volume a terminal failure.
    # If False (default), the worker logs a warning and OCRs the raw pages.
    layout_mask_required: bool = False

    # ------------------------------------------------------------------
    # Two-column split (paddleocr_v2, uses the same layout parquet)
    # ------------------------------------------------------------------
    # When two ``text-area`` boxes overlap >= ``layout_column_min_vert_overlap``
    # vertically and < ``layout_column_max_horiz_overlap`` horizontally, crop
    # each (with a background-padded margin) and OCR them as separate requests.
    # Transcriptions are concatenated left-to-right with ``layout_column_join``
    # (two line breaks). ``paddleocr_v1`` leaves this off.
    layout_split_columns: bool = False

    layout_column_label: str = "text-area"
    layout_column_min_vert_overlap: float = 0.60
    layout_column_max_horiz_overlap: float = 0.05
    # Synthetic background margin around each cropped column, as a fraction
    # of min(page_w, page_h). The pad is not cropped from the scan and may
    # extend past the page edge.
    layout_column_margin_frac: float = 0.02
    layout_column_join: str = "\n\n"

    # Isolate footnotes: crop each ``footnote`` box, OCR it separately, and
    # write the merged transcription to ``footnote_text`` (not ``page_text``).
    # After cropping, the footnote regions are painted with background so they
    # do not leak into the body OCR. ``paddleocr_v1`` leaves this off.
    layout_isolate_footnotes: bool = False
    layout_footnote_label: str = "footnote"
    # Padding around each footnote crop. The pad is synthetic page
    # background (not cropped from the scan) and may extend past the page
    # edge. Size is the max of this fraction of min(page_w, page_h),
    # ``layout_footnote_margin_min_px``, and
    # ``layout_footnote_box_margin_frac`` of the box height.
    layout_footnote_margin_frac: float = 0.05
    layout_footnote_margin_min_px: int = 32
    layout_footnote_box_margin_frac: float = 0.5

    # ------------------------------------------------------------------
    # Failure handling
    # ------------------------------------------------------------------
    volume_timeout_s: float = 3600.0
    max_page_failure_rate: float = 0.05
    # A ``length``-truncated page keeps its (partial) text and is NOT an error
    # by default; it is recorded with ``truncated=True``.
    treat_truncation_as_failure: bool = False

    # ------------------------------------------------------------------
    # Derived helpers
    # ------------------------------------------------------------------
    def processor_longest_edge(self) -> int:
        """1x pixel budget (== processor size.longest_edge == max_pixels)."""
        return self.base_pixel_size * 28 * 28

    def resolved_model_dir(self) -> str:
        """Local directory the checkpoint is synced to."""
        if self.model_local_dir:
            return self.model_local_dir
        clean = self.checkpoint_s3_uri.replace("s3://", "").strip("/")
        parts = [p for p in clean.split("/") if p]
        name = "_".join(parts[-2:]) if len(parts) >= 2 else (parts[-1] if parts else "model")
        return os.path.join(self.model_cache_root, name)

    def model_id(self) -> str:
        """Human-readable model identifier recorded in the parquet output."""
        clean = self.checkpoint_s3_uri.replace("s3://", "").strip("/")
        parts = [p for p in clean.split("/") if p]
        return "/".join(parts[-2:]) if len(parts) >= 2 else self.checkpoint_s3_uri

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------
    def __post_init__(self) -> None:
        if not self.checkpoint_s3_uri.startswith("s3://"):
            raise ValueError(
                f"checkpoint_s3_uri must be an s3:// URI, got {self.checkpoint_s3_uri!r}"
            )
        if self.dtype not in ("bfloat16", "float16", "float32", "auto"):
            raise ValueError(f"unsupported dtype: {self.dtype}")
        if self.max_new_tokens < 16:
            raise ValueError(f"max_new_tokens too small: {self.max_new_tokens}")
        if self.temperature < 0.0:
            raise ValueError(f"temperature must be >= 0, got {self.temperature}")
        if self.repetition_penalty <= 0:
            raise ValueError(
                f"repetition_penalty must be > 0 (1.0 = no penalty), got {self.repetition_penalty}"
            )
        if self.dry_multiplier < 0:
            raise ValueError(f"dry_multiplier must be >= 0, got {self.dry_multiplier}")
        if self.dry_allowed_length < 1:
            raise ValueError(f"dry_allowed_length must be >= 1, got {self.dry_allowed_length}")
        if self.image_token_positions not in ("sequential", "grid", "auto"):
            raise ValueError(
                f"image_token_positions must be 'sequential', 'grid' or 'auto', "
                f"got {self.image_token_positions!r}"
            )
        if self.dry_retry_temp < 0.0:
            raise ValueError(f"dry_retry_temp must be >= 0, got {self.dry_retry_temp}")
        if self.dry_retry_n < 1:
            raise ValueError(f"dry_retry_n must be >= 1, got {self.dry_retry_n}")
        if self.dry_retry_min_fires < 0:
            raise ValueError(f"dry_retry_min_fires must be >= 0, got {self.dry_retry_min_fires}")
        if not 0.0 <= self.dry_retry_min_rep <= 1.0:
            raise ValueError(f"dry_retry_min_rep out of [0,1]: {self.dry_retry_min_rep}")
        if self.max_longest_side < 256:
            raise ValueError(f"max_longest_side too small (quality cliff): {self.max_longest_side}")
        if self.processor_shortest_edge < 1:
            raise ValueError(f"processor_shortest_edge must be >= 1, got {self.processor_shortest_edge}")
        if self.base_pixel_size < 1:
            raise ValueError(f"base_pixel_size must be >= 1, got {self.base_pixel_size}")
        if self.res_mode not in ("adaptive", "fixed"):
            raise ValueError(f"res_mode must be 'adaptive' or 'fixed', got {self.res_mode!r}")
        if not self.res_scales or any(s <= 0 for s in self.res_scales):
            raise ValueError(f"res_scales must be non-empty positive fractions, got {self.res_scales}")
        # Router expects ascending scales (cheapest first).
        self.res_scales = tuple(sorted(float(s) for s in self.res_scales))
        if not 0.0 < self.res_budget_scale <= 1.0:
            raise ValueError(f"res_budget_scale must be in (0,1], got {self.res_budget_scale}")
        if not 0.0 < self.gpu_memory_utilization <= 1.0:
            raise ValueError(
                f"gpu_memory_utilization must be in (0,1], got {self.gpu_memory_utilization}"
            )
        if self.max_model_len < 512:
            raise ValueError(f"max_model_len too small: {self.max_model_len}")
        if self.ocr_batch_size < 1:
            raise ValueError(f"ocr_batch_size must be >= 1, got {self.ocr_batch_size}")
        if self.rep_ngram_size < 1:
            raise ValueError(f"rep_ngram_size must be >= 1, got {self.rep_ngram_size}")
        if not 0.0 <= self.rep_score_threshold <= 1.0:
            raise ValueError(f"rep_score_threshold out of [0,1]: {self.rep_score_threshold}")
        if self.s3_fetch_concurrency < 1:
            raise ValueError(f"s3_fetch_concurrency must be >= 1, got {self.s3_fetch_concurrency}")
        if not 0.0 <= self.max_page_failure_rate <= 1.0:
            raise ValueError(f"max_page_failure_rate out of [0,1]: {self.max_page_failure_rate}")
        if self.layout_mask_pad_px < 0:
            raise ValueError(f"layout_mask_pad_px must be >= 0, got {self.layout_mask_pad_px}")
        if not self.layout_mask_labels:
            raise ValueError("layout_mask_labels must be non-empty")
        if not 0.0 <= self.layout_column_min_vert_overlap <= 1.0:
            raise ValueError(
                f"layout_column_min_vert_overlap out of [0,1]: "
                f"{self.layout_column_min_vert_overlap}"
            )
        if not 0.0 <= self.layout_column_max_horiz_overlap <= 1.0:
            raise ValueError(
                f"layout_column_max_horiz_overlap out of [0,1]: "
                f"{self.layout_column_max_horiz_overlap}"
            )
        if self.layout_column_margin_frac < 0.0:
            raise ValueError(
                f"layout_column_margin_frac must be >= 0, got {self.layout_column_margin_frac}"
            )
        if self.layout_footnote_margin_frac < 0.0:
            raise ValueError(
                f"layout_footnote_margin_frac must be >= 0, got {self.layout_footnote_margin_frac}"
            )
        if self.layout_footnote_margin_min_px < 0:
            raise ValueError(
                f"layout_footnote_margin_min_px must be >= 0, got "
                f"{self.layout_footnote_margin_min_px}"
            )
        if self.layout_footnote_box_margin_frac < 0.0:
            raise ValueError(
                f"layout_footnote_box_margin_frac must be >= 0, got "
                f"{self.layout_footnote_box_margin_frac}"
            )
