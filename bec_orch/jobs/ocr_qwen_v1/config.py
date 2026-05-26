"""Configuration for ``ocr_qwen_v1`` — vLLM offline OCR with Qwen3.5-VL 0.8B.

All defaults below come from the benchmark sweeps documented in
``scratch/findings.md`` (May 2026, on a single NVIDIA A10G with vLLM 0.20.2).
Override any field at job-creation time via the JSON config file or
``--config-text`` flag of ``bec jobs create``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Literal

Precision = Literal["bfloat16", "float16", "auto"]


def _default_mem_util() -> float:
    """GPU memory fraction reserved for vLLM (KV cache + weights + activations).

    Allow operator override via the ``BEC_OCR_QWEN_MEM_UTIL`` env var so the
    same job config can run on different GPU sizes without a job-update round
    trip. Default 0.85 assumes a dedicated GPU (what the BEC fleet runs);
    drop to ~0.55 if something else also uses the GPU on the same box.
    """
    raw = os.environ.get("BEC_OCR_QWEN_MEM_UTIL")
    if raw:
        try:
            return float(raw)
        except ValueError:
            pass
    return 0.85


@dataclass
class OCRQwenV1Config:
    """Tunable parameters for ``ocr_qwen_v1``.

    NOTE: the model itself is downloaded from the HuggingFace Hub at first run
    into the standard HF cache (``~/.cache/huggingface/hub`` by default; set
    ``HF_HOME`` to relocate). Total disk footprint ≈ 1.7 GB.
    """

    # ------------------------------------------------------------------
    # vLLM engine
    # ------------------------------------------------------------------
    # HuggingFace model id (or local path). Pinned to the Mitra fine-tune.
    model_id: str = "buddhist-nlp/bdrc-mitra-ocr-qwen35-0.8b"

    # Compute dtype. The model was published in BF16; using fp16 silently is
    # fine on most GPUs but BF16 is the safest match.
    dtype: Precision = "bfloat16"

    # Fraction of free GPU memory vLLM is allowed to claim. 0.85 is the
    # right value on a dedicated GPU and was validated up to 23 GB (A10G);
    # see ``_default_mem_util()`` for the env override on shared GPUs.
    gpu_memory_utilization: float = float(_default_mem_util())

    # Multi-GPU sharding. Stays at 1 — the model is small enough that
    # data-parallel (one worker per GPU) beats tensor-parallel splitting.
    tensor_parallel_size: int = 1

    # Maximum total context length (image tokens + prompt + output) per
    # request. 8192 is plenty: image ≤ ~1500 tok + prompt ~10 tok + output
    # ≤ ``max_tokens`` (3072). Reduce only if KV cache is tight.
    max_model_len: int = 8192

    # The model is loaded from a public HF repo and has no custom modeling
    # code, but vLLM still wants this flag for some Qwen3-VL paths.
    trust_remote_code: bool = True

    # Disabling CUDA graph capture saves ~10 s on cold start but loses
    # 10-20 % decode throughput. Keep ``False`` in production.
    enforce_eager: bool = False

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------
    # Per-page user prompt. The model was trained with an image-only user
    # message (no text content at all) — empty string here means we omit the
    # text part entirely from the chat message, matching training. Benchmark
    # on 687 Uchen pages: identical median CER and ~0.4pp mean CER difference
    # with vs. without a prompt — well within noise. Default = no prompt.
    prompt: str = ""

    # Hard cap on output tokens. Distribution on real BDRC pechas:
    # p50=2049, p90=2227, p99=2672. 3072 catches >99 % of legitimate pages
    # without letting runaway hallucinating pages eat all wall time. Pages
    # that hit the cap are recorded with ``truncated=True`` (NOT errors).
    max_tokens: int = 3072

    # Deterministic decoding for reproducibility + better batching cohesion.
    temperature: float = 0.0

    # Repetition / frequency / presence penalties (passed straight through to
    # ``vllm.SamplingParams``). The classic OCR-with-LLM failure mode is the
    # model getting stuck emitting the same syllable or line until it hits
    # ``max_tokens`` — these knobs (especially ``repetition_penalty``) suppress
    # that behaviour at decode time.
    #
    # ``repetition_penalty`` (multiplicative, default 1.0):
    #   > 1.0 penalises tokens already seen in prompt+generation so far,
    #   < 1.0 encourages repetition. 1.05–1.20 is the usable range for OCR;
    #   higher than ~1.25 starts hurting legitimately repeated content
    #   (refrains, glyph runs, mantra-style passages). Set to 1.15 by default
    #   — well above 1.0 to break stuck loops, but still safe for the
    #   repetition that's normal in Tibetan pecha text.
    #
    # ``frequency_penalty`` (additive, scales with count, default 0.0):
    #   logit -= count * frequency_penalty. > 0 discourages already-seen
    #   tokens proportionally to how many times they've been emitted.
    #   Kept at 0 by default — ``repetition_penalty`` is the right primary
    #   knob for OCR; bump this only if a specific run shows runaway
    #   repetition that the multiplicative penalty alone doesn't kill.
    #
    # ``presence_penalty`` (additive, fixed, default 0.0):
    #   logit -= presence_penalty for any token already emitted at least
    #   once. > 0 encourages topical drift. Almost never wanted for OCR
    #   (we explicitly DO want to repeat tokens — they're real glyphs);
    #   exposed for completeness.
    repetition_penalty: float = 1.0
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0

    # ------------------------------------------------------------------
    # Image preprocessing
    # ------------------------------------------------------------------
    # Convert decoded image to RGB before sending. The fine-tune was trained
    # on RGB pages; bilevel mode "1" or grayscale "L" must be converted.
    convert_to_rgb: bool = True

    # Cap longest side after decode. Downsizing below ~1600 px destroys
    # accuracy (see findings.md). 2800 is the safe ceiling.
    max_image_side: int = 2800

    # Cap on total pixel count. Required because a square modern page at
    # 2800×2800 would alone produce 7744 image tokens and overflow
    # ``max_model_len``. 4 000 000 px (≈ 2000×2000 square, or 3000×1333
    # landscape) keeps image tokens ≲ 3900, leaving room for output.
    max_image_pixels: int = 4_000_000

    # Resampling filter when downscaling. PIL.Image constants are not JSON
    # serialisable so we accept a string. "bicubic" is the default; "lanczos"
    # is slightly better quality at ~2× the cost; "bilinear" is faster but
    # softer.
    downscale_resample: Literal["bicubic", "bilinear", "lanczos", "nearest"] = "bicubic"

    # ------------------------------------------------------------------
    # I/O pipeline
    # ------------------------------------------------------------------
    # Number of pages processed per fetch → OCR → write cycle. We DO NOT
    # decode a whole volume into RAM up-front: at any moment only
    # ``ocr_batch_size`` decoded RGB images are resident. 128 keeps the
    # peak resident set near ~1.5 GB (128 × ~10 MB at 4 MP/page RGB) on
    # top of vLLM's ~8 GB anon-rss — comfortably under 16 GB on g5.xlarge.
    # vLLM still batches internally per step, so throughput is unchanged
    # vs a single huge ``LLM.chat`` call. Tune up on larger boxes.
    ocr_batch_size: int = 128

    # ThreadPoolExecutor size for parallel S3 GET + PIL decode. S3 fetch +
    # decode runs at ~110 pages/s with 32 threads on a c6i/A10G — way faster
    # than the GPU needs, so this rarely needs tuning.
    s3_fetch_concurrency: int = 32

    # Per-page S3 GET timeout (boto3 read/connect timeout). Pecha images are
    # tiny (~30 KB); long timeouts here just delay failure detection.
    s3_get_timeout_s: int = 30

    # Per-page S3 retry attempts (boto3 standard retry mode handles backoff).
    s3_max_attempts: int = 3

    # ------------------------------------------------------------------
    # Output writer
    # ------------------------------------------------------------------
    # Parquet flush threshold (records buffered before each ``write_table``).
    # 256 keeps the streaming write smooth without holding much memory.
    parquet_flush_every: int = 256

    # zstd is a great quality/speed tradeoff and is the project standard
    # (also used by LDV1 / OCRV1).
    parquet_compression: Literal["zstd", "snappy", "gzip", "none"] = "zstd"

    # Soft cap on total decoded-image bytes per ``LLM.chat()`` call. vLLM's
    # chat path copies every PIL image to the EngineCore subprocess during
    # chat rendering, transiently ~2× host RAM for the chunk; sizing chunks
    # by bytes (rather than page count) adapts to the wide spread between
    # small pecha folios (~3 MB after RGB+resize) and bigger modern book
    # pages (up to 12 MB hard ceiling, since max_image_pixels=4_000_000
    # caps decoded RGB at 4M × 3 = 12 MB).
    #
    # 1 GB keeps peak host RAM under ~2.5 GB on a 16 GB g5.xlarge worker
    # even with worst-case images, while still giving the GPU big-enough
    # batches to saturate (>~64 pages worth, well above vLLM's saturation
    # point at our 8K max_model_len). A chunk always contains at least one
    # image, even on the off chance someone later raises max_image_pixels
    # above this budget.
    #
    # Streaming chunked calls also lets the parquet writer drain throughout
    # the volume rather than at the end, so a mid-volume crash preserves
    # the rows from already-completed chunks.
    #
    # Set to 0 to disable chunking and send the whole volume in one
    # ``LLM.chat`` call (faster by a few %, but risks OOM on large volumes
    # or under-provisioned worker boxes).
    llm_chat_chunk_max_bytes: int = 1_000_000_000

    # Also write a sidecar ``<basename>-errors.jsonl`` with per-page failure
    # details. Cheap to keep on; turn off only if storage is precious.
    write_errors_jsonl: bool = True

    # ------------------------------------------------------------------
    # Failure handling
    # ------------------------------------------------------------------
    # Volume-level timeout. Should be much larger than the worst observed
    # volume time on the target GPU; ~30 min covers ~4000 pages on A10G.
    volume_timeout_s: float = 1800.0

    # Fraction of pages that may fail (S3 / decode / OCR) before the whole
    # volume is reported as a terminal failure. With 0.05 a volume of 200
    # pages can have up to 10 failures and still succeed.
    max_page_failure_rate: float = 0.05

    # Whether a `length`-truncated output counts toward the failure budget.
    # Default ``False`` — truncated pages are still useful (you keep the
    # first ~3072 tokens of text) and they're recorded with `truncated=True`.
    treat_truncation_as_failure: bool = False

    # ------------------------------------------------------------------
    # Engine lifecycle
    # ------------------------------------------------------------------
    # Reset vLLM's multi-modal + prefix caches after every volume. Costs
    # almost nothing and is the simplest way to keep long-running workers
    # memory-stable (see findings.md memory-leak test).
    reset_caches_between_volumes: bool = True

    def __post_init__(self) -> None:
        # Cheap validation to catch typos before we go through the whole
        # vLLM load (~46 s when cache is warm).
        if not 0.05 <= self.gpu_memory_utilization <= 0.98:
            raise ValueError(
                f"gpu_memory_utilization out of range: {self.gpu_memory_utilization}"
            )
        if self.tensor_parallel_size < 1:
            raise ValueError(f"tensor_parallel_size must be ≥ 1, got {self.tensor_parallel_size}")
        if self.max_model_len < 1024:
            raise ValueError(f"max_model_len too small: {self.max_model_len}")
        if self.max_tokens < 64 or self.max_tokens > self.max_model_len:
            raise ValueError(f"max_tokens out of range: {self.max_tokens}")
        # vLLM ranges: repetition_penalty must be > 0 (multiplicative);
        # frequency_penalty and presence_penalty must be in [-2, 2].
        if self.repetition_penalty <= 0:
            raise ValueError(
                f"repetition_penalty must be > 0 (1.0 = no penalty), "
                f"got {self.repetition_penalty}"
            )
        if not -2.0 <= self.frequency_penalty <= 2.0:
            raise ValueError(
                f"frequency_penalty out of [-2, 2]: {self.frequency_penalty}"
            )
        if not -2.0 <= self.presence_penalty <= 2.0:
            raise ValueError(
                f"presence_penalty out of [-2, 2]: {self.presence_penalty}"
            )
        if self.max_image_side < 256:
            raise ValueError(f"max_image_side too small (quality cliff): {self.max_image_side}")
        if self.max_image_pixels < 256 * 256:
            raise ValueError(f"max_image_pixels too small: {self.max_image_pixels}")
        if self.s3_fetch_concurrency < 1:
            raise ValueError(f"s3_fetch_concurrency must be ≥ 1, got {self.s3_fetch_concurrency}")
        if self.ocr_batch_size < 1:
            raise ValueError(f"ocr_batch_size must be ≥ 1, got {self.ocr_batch_size}")
        if not 0.0 <= self.max_page_failure_rate <= 1.0:
            raise ValueError(f"max_page_failure_rate out of [0,1]: {self.max_page_failure_rate}")
        if self.llm_chat_chunk_max_bytes < 0:
            raise ValueError(
                f"llm_chat_chunk_max_bytes must be ≥ 0 (0 disables chunking), "
                f"got {self.llm_chat_chunk_max_bytes}"
            )
