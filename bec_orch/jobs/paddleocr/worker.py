"""PaddleOCR-VL ``JobWorker`` (vLLM offline engine, GPU inference).

Serving stack (``bec-ocr-training/docs/eval_in_production.md``): **vLLM +
sequential image-token M-RoPE + DRY-a12 + adaptive per-page resolution**, greedy
decoding, with a high-severity temperature retry. The ``grow26-ep2`` checkpoint
was trained in the ``sequential`` regime (mm_token_type_ids zeroed); stock vLLM
only serves ``grid``, so the ``vllm_paddleocr_seqpos`` plugin +
``OCR_VLLM_IMAGE_TOKEN_POSITIONS=sequential`` are required (set here before the
engine starts). It emits Tibetan **Unicode** directly (warmed unicode-stack
tokenizer), so post-processing is canonical Unicode normalization — NOT
Wylie/EWTS.

Per volume:
  1. Walk the manifest in fetch batches of ``cfg.ocr_batch_size``; a ThreadPool
     fetches each page and decodes + resolution-routes it to RGB (only one
     batch of decoded images resident at a time).
  2. The whole fetch batch is submitted to the in-process vLLM engine, which
     does its own continuous batching (up to ``cfg.max_num_seqs``). DRY is
     registered once on the engine and enabled per request, with per-page fire
     telemetry routed through a temp-dir side-channel.
  3. Pages where DRY fired hard (``dry_fires >= dry_retry_min_fires``, or a
     leftover hard loop) are re-decoded at ``dry_retry_temp`` and the lowest-rep
     sample is kept.
  4. Predictions are Unicode-normalized, a loop self-flag + ``dry_fires`` are
     recorded, and rows are streamed to parquet.

The engine + processor are loaded once on construction and reused for every
volume. Everything model-specific comes from :class:`PaddleOCRConfig`, so
``paddleocr_v2`` reuses this class unchanged.
"""

from __future__ import annotations

import gc
import logging
import os
import shutil
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import boto3
from botocore.config import Config as BotoConfig
from PIL import Image

from bec_orch.core.models import TaskResult
from bec_orch.errors import RetryableTaskError, TerminalTaskError

from .config import PaddleOCRConfig
from .filter import load_skip_map
from .model_sync import sync_checkpoint
from .parquet_writer import StreamingPaddleOCRWriter
from .postprocess import normalize_text, rep_score
from .preprocess import ImageDecodeError, bytes_to_rgb

if TYPE_CHECKING:
    from bec_orch.jobs.base import JobContext

logger = logging.getLogger(__name__)

_SOURCE_BUCKET = os.environ.get("BEC_SOURCE_S3_BUCKET", "archive.tbrc.org")


@dataclass
class _FetchedPage:
    filename: str
    etag: str
    image: Image.Image | None
    res_scale: float = 1.0


@dataclass
class _FailedPage:
    filename: str
    stage: str  # "fetch" | "decode"
    etag: str | None
    error: str


@dataclass
class _OCRResult:
    raw_text: str
    output_tokens: int
    truncated: bool
    finish_reason: str
    # DRY fire telemetry (worker side-channel; 0 when DRY is off).
    dry_fires: int = 0
    dry_max_L: int = 0
    dry_sum_penalty: float = 0.0
    # True if this page was re-decoded at temperature (high DRY severity) and
    # ``raw_text`` is the retry pick rather than the greedy output.
    retried: bool = False


class PaddleOCRJobWorker:
    """JobWorker for any PaddleOCR-VL checkpoint (config-driven, vLLM engine)."""

    def __init__(self, cfg: PaddleOCRConfig) -> None:
        self.cfg = cfg
        self._s3 = self._build_s3_client(cfg)
        self.llm = None
        self.processor = None
        self.prompt_text: str = ""
        self._sp_cls: Any = None
        self._load_dry_stats_dir = None
        self._load_engine(cfg)

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_s3_client(cfg: PaddleOCRConfig):
        region = os.environ.get("BEC_REGION", "us-east-1")
        boto_cfg = BotoConfig(
            region_name=region,
            retries={"max_attempts": cfg.s3_max_attempts, "mode": "standard"},
            connect_timeout=cfg.s3_get_timeout_s,
            read_timeout=cfg.s3_get_timeout_s,
            max_pool_connections=max(cfg.s3_fetch_concurrency * 2, 32),
        )
        return boto3.client("s3", config=boto_cfg)

    def _load_engine(self, cfg: PaddleOCRConfig) -> None:
        # vLLM's flashinfer sampler JIT-compiles a kernel needing nvcc; greedy
        # OCR doesn't need it. Disable before importing vllm on toolkit-less
        # boxes. Set here so it also applies to worker subprocesses.
        if cfg.disable_flashinfer_sampler:
            os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")

        # Sync + patch the checkpoint (image_token + processor size) so the
        # native vLLM path works with trust_remote_code=False.
        model_dir = sync_checkpoint(cfg)

        # Import lazily so the worker registry can be imported without a GPU
        # stack / vLLM installed.
        from transformers import AutoProcessor
        from vllm import LLM, SamplingParams

        logger.info(
            f"[paddleocr] loading vLLM engine from {model_dir} dtype={cfg.dtype} "
            f"trust_remote_code={cfg.trust_remote_code} "
            f"max_model_len={cfg.max_model_len} max_num_seqs={cfg.max_num_seqs}"
        )
        t0 = time.time()

        llm_kwargs: dict[str, Any] = dict(
            model=model_dir,
            trust_remote_code=cfg.trust_remote_code,
            dtype=cfg.dtype,
            enforce_eager=cfg.enforce_eager,
            gpu_memory_utilization=cfg.gpu_memory_utilization,
            max_model_len=cfg.max_model_len,
            limit_mm_per_prompt={"image": 1},
            enable_prefix_caching=cfg.enable_prefix_caching,
            mm_processor_cache_gb=cfg.mm_processor_cache_gb,
            max_num_batched_tokens=cfg.max_num_batched_tokens,
        )
        if cfg.max_num_seqs and cfg.max_num_seqs > 0:
            llm_kwargs["max_num_seqs"] = cfg.max_num_seqs
        if cfg.quantization:
            llm_kwargs["quantization"] = cfg.quantization

        # Serve in the checkpoint's image-token M-RoPE regime. Stock vLLM only
        # serves "grid"; "sequential" (what grow26-ep2 was trained in) needs the
        # vllm_paddleocr_seqpos plugin. Set the env BEFORE constructing LLM so the
        # spawned EngineCore/worker processes inherit it and the plugin activates.
        self._ensure_seqpos(self._resolve_regime(cfg, model_dir))

        # Register the DRY per-request logits processor at engine init; it stays
        # a no-op for requests that don't set dry_multiplier in extra_args.
        if cfg.dry_multiplier and cfg.dry_multiplier > 0:
            from .dry import DRYLogitsProcessor, load_dry_stats_dir

            if DRYLogitsProcessor is None:
                raise RuntimeError(
                    "DRY requested (dry_multiplier>0) but DRYLogitsProcessor is "
                    "unavailable for this vLLM build; upgrade vLLM or set "
                    "dry_multiplier=0"
                )
            llm_kwargs["logits_processors"] = [DRYLogitsProcessor]
            self._load_dry_stats_dir = load_dry_stats_dir

        self.llm = LLM(**llm_kwargs)
        self._sp_cls = SamplingParams

        # Processor is only used to render the chat-template prompt string;
        # vLLM handles image processing internally from the patched checkpoint.
        self.processor = AutoProcessor.from_pretrained(
            model_dir, trust_remote_code=cfg.trust_remote_code
        )
        self.prompt_text = self._build_prompt(cfg.prompt)

        logger.info(f"[paddleocr] vLLM engine ready in {time.time() - t0:.1f}s")

    # ------------------------------------------------------------------
    # Image-token M-RoPE regime (sequential vs grid)
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_regime(cfg: PaddleOCRConfig, model_dir: str) -> str:
        """Resolve the effective regime; "auto" reads the checkpoint config."""
        requested = cfg.image_token_positions
        if requested in ("sequential", "grid"):
            return requested
        import json

        for name in ("experiment_config.json", "info.json"):
            path = os.path.join(model_dir, name)
            try:
                if os.path.isfile(path):
                    with open(path, encoding="utf-8") as f:
                        v = json.load(f).get("image_token_positions")
                    if v in ("sequential", "grid"):
                        return v
            except Exception:  # noqa: BLE001 — fall through to the default
                continue
        return "sequential"

    def _ensure_seqpos(self, regime: str) -> None:
        """Set OCR_VLLM_IMAGE_TOKEN_POSITIONS and refuse sequential w/o the plugin.

        Stock vLLM always builds 2D grid M-RoPE for PaddleOCR-VL and never reads
        mm_token_type_ids, so the ``sequential`` regime needs the
        ``vllm_paddleocr_seqpos`` general plugin installed in the same venv as
        vLLM (it overrides ``get_mrope_input_positions`` in every engine process).
        """
        os.environ["OCR_VLLM_IMAGE_TOKEN_POSITIONS"] = regime
        if regime != "sequential":
            logger.info(
                f"[paddleocr] image_token_positions={regime} "
                f"(stock vLLM grid path; seqpos plugin is a no-op)"
            )
            return
        try:
            from importlib.metadata import entry_points

            eps = entry_points()
            group = (
                eps.select(group="vllm.general_plugins")
                if hasattr(eps, "select")
                else eps.get("vllm.general_plugins", [])
            )
            names = {e.name for e in group}
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                f"cannot inspect vLLM general plugins ({exc}); install "
                "vllm_paddleocr_seqpos into the vLLM venv "
                "(pip install bec_orch/jobs/paddleocr/vllm_paddleocr_seqpos)"
            ) from exc
        if "paddleocr_seqpos" not in names:
            raise RuntimeError(
                "image_token_positions='sequential' requires the "
                "vllm_paddleocr_seqpos plugin — stock vLLM only serves the grid "
                "regime, which regresses a sequential-trained checkpoint. Install "
                "it into the vLLM venv:  pip install "
                "bec_orch/jobs/paddleocr/vllm_paddleocr_seqpos  (or set "
                "image_token_positions='grid')."
            )
        logger.info(
            "[paddleocr] vLLM seqpos plugin registered; "
            "OCR_VLLM_IMAGE_TOKEN_POSITIONS=sequential"
        )

    def _build_prompt(self, prompt_text: str) -> str:
        """Render the chat-template prompt (image placeholder + instruction)."""
        messages = [
            {
                "role": "user",
                "content": [{"type": "image"}, {"type": "text", "text": prompt_text}],
            }
        ]
        return self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    def _dry_extra(
        self, stats_id: str | None = None, stats_path: str | None = None
    ) -> dict[str, Any] | None:
        """DRY ``extra_args`` for one request (None when DRY is disabled).

        When ``stats_id``/``stats_path`` are given, the per-request fire summary
        is written to ``<stats_path>/<stats_id>.json`` by the engine worker.
        """
        cfg = self.cfg
        if not (cfg.dry_multiplier and cfg.dry_multiplier > 0):
            return None
        ea: dict[str, Any] = {
            "dry_multiplier": cfg.dry_multiplier,
            "dry_base": cfg.dry_base,
            "dry_allowed_length": cfg.dry_allowed_length,
            "dry_window": cfg.dry_window,
            "dry_max_match": cfg.dry_max_match,
        }
        if cfg.dry_sequence_breakers:
            ea["dry_sequence_breakers"] = list(cfg.dry_sequence_breakers)
        if stats_id is not None and stats_path:
            ea["dry_stats_id"] = str(stats_id)
            ea["dry_stats_path"] = stats_path
        return ea

    def _make_sampling(
        self,
        temperature: float,
        n: int = 1,
        seed: int | None = None,
        stats_id: str | None = None,
        stats_path: str | None = None,
    ):
        cfg = self.cfg
        kwargs: dict[str, Any] = dict(
            temperature=float(temperature),
            max_tokens=cfg.max_new_tokens,
            n=int(n),
        )
        if seed is not None:
            kwargs["seed"] = int(seed)
        if cfg.repetition_penalty != 1.0:
            kwargs["repetition_penalty"] = cfg.repetition_penalty
        ea = self._dry_extra(stats_id, stats_path)
        if ea:
            kwargs["extra_args"] = ea
        return self._sp_cls(**kwargs)

    # ------------------------------------------------------------------
    # JobWorker protocol
    # ------------------------------------------------------------------

    def run(self, ctx: JobContext) -> TaskResult:
        vol = ctx.volume
        manifest = ctx.volume_manifest.manifest
        n_expected = len(manifest)
        if n_expected == 0:
            raise TerminalTaskError(f"empty manifest for volume {vol.w_id}/{vol.i_id}")

        logger.info(
            f"[paddleocr] {vol.w_id}/{vol.i_id}: {n_expected} images "
            f"(source s3://{_SOURCE_BUCKET})"
        )
        t_start = time.time()
        return self._run_volume(ctx, manifest, t_start)

    # ------------------------------------------------------------------
    # Volume processing
    # ------------------------------------------------------------------

    def _run_volume(
        self,
        ctx: JobContext,
        manifest: list[dict[str, Any]],
        t_start: float,
    ) -> TaskResult:
        cfg = self.cfg
        vol = ctx.volume
        n_expected = len(manifest)

        from bec_orch.core.worker_runtime import get_s3_folder_prefix

        vol_prefix = get_s3_folder_prefix(vol.w_id, vol.i_id)

        parquet_uri, errors_uri = self._artifact_uris(ctx)
        writer = StreamingPaddleOCRWriter(
            parquet_uri=parquet_uri,
            errors_jsonl_uri=errors_uri,
            model_id=cfg.model_id(),
            flush_every=cfg.parquet_flush_every,
            compression=cfg.parquet_compression,
        )

        # Pre-filter via the sibling classification job. Skipped pages are
        # recorded (skipped=True) and dropped from the OCR manifest.
        ocr_manifest, n_skipped = self._apply_filter(ctx, manifest, writer)

        n_to_ocr = len(ocr_manifest)
        batch_size = max(1, cfg.ocr_batch_size)
        n_batches = (n_to_ocr + batch_size - 1) // batch_size

        n_ocr_done = 0
        ocr_failures = 0
        ocr_truncated = 0
        n_dry_fired = 0
        n_retried = 0
        errors_by_stage: dict[str, int] = {}
        t_fetch_total = 0.0
        t_ocr_total = 0.0

        try:
            for batch_idx in range(n_batches):
                start = batch_idx * batch_size
                stop = min(start + batch_size, n_to_ocr)
                batch_manifest = ocr_manifest[start:stop]

                t_fetch = time.time()
                fetched, failed = self._fetch_and_decode(vol_prefix, batch_manifest)
                t_fetch_total += time.time() - t_fetch

                for f in failed:
                    writer.write_error(
                        filename=f.filename,
                        source_etag=f.etag,
                        stage=f.stage,
                        error_message=f.error,
                    )
                    errors_by_stage[f.stage] = errors_by_stage.get(f.stage, 0) + 1

                if not fetched:
                    logger.warning(
                        f"[paddleocr] {vol.w_id}/{vol.i_id}: batch "
                        f"{batch_idx + 1}/{n_batches} ({start}:{stop}) — "
                        f"all {len(batch_manifest)} pages failed before OCR"
                    )
                    continue

                t_ocr = time.time()
                try:
                    results = self._ocr_batch(fetched)
                except Exception as e:  # noqa: BLE001 — surface as per-page OCR errors
                    logger.exception(
                        f"[paddleocr] {vol.w_id}/{vol.i_id}: generate failed for "
                        f"{len(fetched)} pages"
                    )
                    for page in fetched:
                        writer.write_error(
                            filename=page.filename,
                            source_etag=page.etag,
                            stage="ocr",
                            error_message=f"generate failed: {e}",
                        )
                        ocr_failures += 1
                        errors_by_stage["ocr"] = errors_by_stage.get("ocr", 0) + 1
                    results = None
                t_ocr_total += time.time() - t_ocr

                if results is not None:
                    for page, res in zip(fetched, results, strict=True):
                        # Safety net: postprocessing must never take down the
                        # batch. normalize_text is already hardened, but we
                        # guard the whole per-page path (rep_score + write) so
                        # any residual failure is isolated to one page.
                        try:
                            page_text = normalize_text(res.raw_text)
                            score = rep_score(res.raw_text, cfg.rep_ngram_size)
                            likely_loop = score >= cfg.rep_score_threshold
                            writer.write_success(
                                filename=page.filename,
                                source_etag=page.etag,
                                page_text=page_text,
                                raw_text=res.raw_text,
                                rep_score=score,
                                likely_loop=likely_loop,
                                output_tokens=res.output_tokens,
                                finish_reason=res.finish_reason,
                                truncated=res.truncated,
                                res_scale=page.res_scale,
                                dry_fires=res.dry_fires,
                                dry_max_L=res.dry_max_L,
                                retried=res.retried,
                            )
                            if res.truncated:
                                ocr_truncated += 1
                            if res.dry_fires > 0:
                                n_dry_fired += 1
                            if res.retried:
                                n_retried += 1
                        except Exception as e:  # noqa: BLE001 — isolate per page
                            logger.exception(
                                f"[paddleocr] {vol.w_id}/{vol.i_id}: postprocess/write "
                                f"failed for {page.filename}"
                            )
                            writer.write_error(
                                filename=page.filename,
                                source_etag=page.etag,
                                stage="postprocess",
                                error_message=f"postprocess failed: {e}",
                            )
                            ocr_failures += 1
                            errors_by_stage["postprocess"] = (
                                errors_by_stage.get("postprocess", 0) + 1
                            )

                # Free the decoded images for this batch.
                for page in fetched:
                    if page.image is not None:
                        try:
                            page.image.close()
                        except Exception:  # noqa: BLE001 — best-effort
                            pass
                        page.image = None

                n_ocr_done += len(fetched)
                del fetched
                gc.collect()

                logger.info(
                    f"[paddleocr] {vol.w_id}/{vol.i_id}: batch "
                    f"{batch_idx + 1}/{n_batches} done "
                    f"(ocr {n_ocr_done}/{n_to_ocr}, "
                    f"cum_fetch={t_fetch_total:.1f}s, cum_ocr={t_ocr_total:.1f}s)"
                )

            # Terminal only if nothing was OCR'd AND nothing was intentionally
            # skipped (i.e. every page failed before the GPU). A volume where
            # all pages were filtered out is a legitimate success.
            if n_ocr_done == 0 and n_skipped == 0:
                raise TerminalTaskError(
                    f"all {n_expected} pages failed before OCR for {vol.w_id}/{vol.i_id}"
                )
        finally:
            writer.close()

        elapsed_ms = (time.time() - t_start) * 1000

        n_fetch_decode_errors = sum(
            v for k, v in errors_by_stage.items() if k in ("fetch", "decode")
        )
        n_errors = n_fetch_decode_errors + ocr_failures
        if cfg.treat_truncation_as_failure:
            n_errors += ocr_truncated
            if ocr_truncated:
                errors_by_stage["truncation"] = ocr_truncated

        # Failure rate is over pages we actually attempted (skipped pages
        # excluded), so filtering a volume can't inflate the rate.
        n_attempted = max(n_expected - n_skipped, 1)
        failure_rate = n_errors / n_attempted
        logger.info(
            f"[paddleocr] {vol.w_id}/{vol.i_id}: done in {elapsed_ms / 1000:.1f}s "
            f"({n_expected} pages, success={writer.success_count}, skipped={n_skipped}, "
            f"errors={n_errors}, truncated={ocr_truncated}, likely_loop={writer.loop_count}, "
            f"dry_fired={n_dry_fired}, retried={n_retried}, "
            f"rate={failure_rate:.2%}, fetch={t_fetch_total:.1f}s, ocr={t_ocr_total:.1f}s)"
        )

        if failure_rate > cfg.max_page_failure_rate:
            raise RetryableTaskError(
                f"volume {vol.w_id}/{vol.i_id} failure rate {failure_rate:.2%} "
                f"> threshold {cfg.max_page_failure_rate:.2%} "
                f"(errors_by_stage={errors_by_stage})"
            )

        return TaskResult(
            total_images=n_expected,
            nb_errors=n_errors,
            total_duration_ms=elapsed_ms,
            avg_duration_per_page_ms=elapsed_ms / n_expected,
            nb_dropped_records=0,
            errors_by_stage=errors_by_stage or None,
        )

    # ------------------------------------------------------------------
    # Pre-filter (script_classification_v2)
    # ------------------------------------------------------------------

    def _apply_filter(
        self,
        ctx: JobContext,
        manifest: list[dict[str, Any]],
        writer: StreamingPaddleOCRWriter,
    ) -> tuple[list[dict[str, Any]], int]:
        """Drop pages the sibling classifier flagged; record them as skipped.

        Returns ``(ocr_manifest, n_skipped)``. When filtering is disabled or the
        classification artifact is absent (and not required), returns the full
        manifest unchanged.
        """
        cfg = self.cfg
        if not cfg.filter_enabled:
            return manifest, 0

        loc = ctx.artifacts_location
        vol = ctx.volume
        skip_map, found = load_skip_map(loc.bucket, loc.prefix, loc.basename, cfg)
        if not found:
            if cfg.filter_required:
                raise TerminalTaskError(
                    f"{cfg.filter_job_name} artifact missing for {vol.w_id}/{vol.i_id} "
                    f"(filter_required=True)"
                )
            logger.warning(
                f"[paddleocr] {vol.w_id}/{vol.i_id}: no {cfg.filter_job_name} output found — "
                f"OCR-ing all {len(manifest)} pages (filter_required=False)"
            )
            return manifest, 0

        ocr_manifest: list[dict[str, Any]] = []
        n_skipped = 0
        for item in manifest:
            fn = item.get("filename")
            reason = skip_map.get(fn) if fn else None
            if reason:
                writer.write_skipped(filename=fn, source_etag=None, skip_reason=reason)
                n_skipped += 1
            else:
                ocr_manifest.append(item)
        logger.info(
            f"[paddleocr] {vol.w_id}/{vol.i_id}: filter skipped {n_skipped}/{len(manifest)} "
            f"pages; {len(ocr_manifest)} to OCR"
        )
        return ocr_manifest, n_skipped

    # ------------------------------------------------------------------
    # Fetch + decode
    # ------------------------------------------------------------------

    def _fetch_and_decode(
        self, vol_prefix: str, manifest: list[dict[str, Any]]
    ) -> tuple[list[_FetchedPage], list[_FailedPage]]:
        cfg = self.cfg
        ok_by_filename: dict[str, _FetchedPage] = {}
        ko_by_filename: dict[str, _FailedPage] = {}

        def _one(filename: str) -> None:
            key = f"{vol_prefix}{filename}"
            try:
                resp = self._s3.get_object(Bucket=_SOURCE_BUCKET, Key=key)
            except Exception as e:  # noqa: BLE001 — boto3 ClientError subtypes
                ko_by_filename[filename] = _FailedPage(
                    filename=filename, stage="fetch", etag=None, error=f"s3 get failed: {e}"
                )
                return
            etag = resp.get("ETag", "").strip('"')
            try:
                data = resp["Body"].read()
            except Exception as e:
                ko_by_filename[filename] = _FailedPage(
                    filename=filename, stage="fetch", etag=etag or None,
                    error=f"s3 body read failed: {e}",
                )
                return
            try:
                img, res_scale = bytes_to_rgb(data, cfg)
            except ImageDecodeError as e:
                ko_by_filename[filename] = _FailedPage(
                    filename=filename, stage="decode", etag=etag, error=str(e),
                )
                return
            except Exception as e:  # noqa: BLE001 — be defensive
                ko_by_filename[filename] = _FailedPage(
                    filename=filename, stage="decode", etag=etag,
                    error=f"unexpected decode error: {e}",
                )
                return
            ok_by_filename[filename] = _FetchedPage(
                filename=filename, etag=etag, image=img, res_scale=res_scale
            )

        with ThreadPoolExecutor(max_workers=cfg.s3_fetch_concurrency) as pool:
            futures = [
                pool.submit(_one, item["filename"])
                for item in manifest
                if item.get("filename")
            ]
            for _ in as_completed(futures):
                pass

        ok: list[_FetchedPage] = []
        ko: list[_FailedPage] = []
        for item in manifest:
            fn = item.get("filename")
            if fn in ok_by_filename:
                ok.append(ok_by_filename[fn])
            elif fn in ko_by_filename:
                ko.append(ko_by_filename[fn])
        return ok, ko

    # ------------------------------------------------------------------
    # OCR (vLLM)
    # ------------------------------------------------------------------

    def _ocr_batch(self, pages: list[_FetchedPage]) -> list[_OCRResult]:
        """Submit ``pages`` to the vLLM engine (continuous batching).

        Greedy + DRY is the bulk decode. Per request we route DRY fire telemetry
        to a temp-dir side-channel (the logits processor runs in the engine
        worker), read it back, and — if ``dry_retry_temp>0`` — re-decode the
        high-severity pages (``dry_fires >= dry_retry_min_fires`` or leftover
        ``rep_score >= dry_retry_min_rep``) at temperature, keeping the lowest-rep
        sample.
        """
        cfg = self.cfg
        prompts = []
        for p in pages:
            assert p.image is not None, f"image for {p.filename} was cleared before OCR"
            prompts.append(
                {"prompt": self.prompt_text, "multi_modal_data": {"image": p.image}}
            )

        dry_on = bool(cfg.dry_multiplier and cfg.dry_multiplier > 0)
        stats_dir = tempfile.mkdtemp(prefix="bec_dry_") if dry_on else None
        try:
            sps = [
                self._make_sampling(
                    cfg.temperature,
                    n=1,
                    stats_id=(f"{i:06d}" if dry_on else None),
                    stats_path=stats_dir,
                )
                for i in range(len(prompts))
            ]
            outputs = self.llm.generate(prompts, sps, use_tqdm=False)

            dry_stats = (
                self._load_dry_stats_dir(stats_dir)
                if (dry_on and self._load_dry_stats_dir)
                else {}
            )

            results: list[_OCRResult] = []
            for i, o in enumerate(outputs):
                out = o.outputs[0]
                finish_reason = out.finish_reason or ""
                st = dry_stats.get(f"{i:06d}", {})
                results.append(
                    _OCRResult(
                        raw_text=out.text or "",
                        output_tokens=len(out.token_ids),
                        truncated=finish_reason == "length",
                        finish_reason=finish_reason,
                        dry_fires=int(st.get("fires", 0) or 0),
                        dry_max_L=int(st.get("max_L", 0) or 0),
                        dry_sum_penalty=float(st.get("sum_penalty", 0.0) or 0.0),
                    )
                )

            if dry_on and cfg.dry_retry_temp and cfg.dry_retry_temp > 0:
                self._apply_dry_retry(prompts, results)

            return results
        finally:
            if stats_dir:
                shutil.rmtree(stats_dir, ignore_errors=True)

    def _apply_dry_retry(
        self, prompts: list[dict[str, Any]], results: list[_OCRResult]
    ) -> None:
        """Re-decode high-DRY-severity pages at temperature; keep the best sample.

        Mutates ``results`` in place. The gate is an OR of: DRY fired at least
        ``dry_retry_min_fires`` times, or the greedy prediction's ``rep_score``
        is at least ``dry_retry_min_rep`` (leftover hard-loop safety floor).
        """
        cfg = self.cfg
        min_fires = int(cfg.dry_retry_min_fires)
        min_rep = float(cfg.dry_retry_min_rep)
        idxs: list[int] = []
        for i, res in enumerate(results):
            if min_fires > 0 and res.dry_fires >= min_fires:
                idxs.append(i)
            elif min_rep > 0 and rep_score(res.raw_text, cfg.rep_ngram_size) >= min_rep:
                idxs.append(i)
        if not idxs:
            return

        logger.info(
            f"[paddleocr] DRY-retry: re-decoding {len(idxs)}/{len(results)} pages "
            f"at temp={cfg.dry_retry_temp} n={cfg.dry_retry_n} "
            f"(gate: fires>={min_fires or 'off'} or rep>={min_rep or 'off'})"
        )
        r_prompts = [prompts[i] for i in idxs]
        r_sps = [
            self._make_sampling(cfg.dry_retry_temp, n=cfg.dry_retry_n, seed=cfg.dry_retry_seed)
            for _ in idxs
        ]
        r_outs = self.llm.generate(r_prompts, r_sps, use_tqdm=False)
        for i, o in zip(idxs, r_outs, strict=True):
            best = self._pick_retry_sample(o.outputs)
            if best is None:
                continue
            res = results[i]
            finish_reason = best.finish_reason or ""
            res.raw_text = best.text or ""
            res.output_tokens = len(best.token_ids)
            res.finish_reason = finish_reason
            res.truncated = finish_reason == "length"
            res.retried = True

    def _pick_retry_sample(self, outputs: list[Any]) -> Any | None:
        """Production pick: lowest leftover-loop ``rep_score``, then shortest."""
        cfg = self.cfg
        best = None
        best_key: tuple[float, int] | None = None
        for s in outputs:
            key = (rep_score(s.text or "", cfg.rep_ngram_size), len(s.token_ids))
            if best_key is None or key < best_key:
                best_key = key
                best = s
        return best

    # ------------------------------------------------------------------
    # Misc
    # ------------------------------------------------------------------

    def _artifact_uris(self, ctx: JobContext) -> tuple[str, str | None]:
        loc = ctx.artifacts_location
        prefix = loc.prefix.rstrip("/")
        parquet_uri = f"s3://{loc.bucket}/{prefix}/{loc.basename}.parquet"
        errors_uri: str | None = None
        if self.cfg.write_errors_jsonl:
            errors_uri = f"s3://{loc.bucket}/{prefix}/{loc.basename}-errors.jsonl.gz"
        return parquet_uri, errors_uri
