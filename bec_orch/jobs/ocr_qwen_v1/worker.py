"""``ocr_qwen_v1`` JobWorker.

Design (see ``scratch/findings.md``):

    1. ThreadPool fetches all volume images from S3 and decodes them to RGB
       (parallel, ~110 pages/s on a 32-thread pool, way faster than the GPU).
    2. A single synchronous ``LLM.chat(list[...])`` runs all decoded pages in
       one go — the engine packs the prefill optimally and beats the async
       streaming pattern by ~15 %.
    3. Results stream into a parquet writer.

The vLLM ``LLM`` is owned by the JobWorker instance and reused across all
volumes the worker processes (model load: ~46 s warm). Caches are reset
between volumes per ``OCRQwenV1Config.reset_caches_between_volumes``.
"""

from __future__ import annotations

import gc
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import boto3
from botocore.config import Config as BotoConfig
from PIL import Image

from bec_orch.core.models import TaskResult
from bec_orch.errors import RetryableTaskError, TerminalTaskError

from .config import OCRQwenV1Config
from .parquet_writer import StreamingOCRQwenV1Writer
from .preprocess import ImageDecodeError, bytes_to_rgb

if TYPE_CHECKING:
    from bec_orch.jobs.base import JobContext

logger = logging.getLogger(__name__)

# S3 source bucket for BDRC images. Worker runtime hardcodes the same default
# (``archive.tbrc.org``) and the LDV1 worker hardcodes the URI as well, so we
# follow that convention.
_SOURCE_BUCKET = os.environ.get("BEC_SOURCE_S3_BUCKET", "archive.tbrc.org")


@dataclass
class _FetchedPage:
    filename: str
    etag: str
    image: Image.Image


@dataclass
class _FailedPage:
    filename: str
    stage: str  # "fetch" | "decode"
    etag: str | None
    error: str


class OCRQwenV1JobWorker:
    """JobWorker for ``ocr_qwen_v1``.

    The vLLM engine is loaded once on construction and reused for every
    volume the worker processes.
    """

    def __init__(self, cfg: OCRQwenV1Config) -> None:
        self.cfg = cfg
        self._s3 = self._build_s3_client(cfg)
        self._llm = self._build_llm(cfg)
        self._sampling = self._build_sampling_params(cfg)

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_s3_client(cfg: OCRQwenV1Config):
        region = os.environ.get("BEC_REGION", "us-east-1")
        boto_cfg = BotoConfig(
            region_name=region,
            retries={"max_attempts": cfg.s3_max_attempts, "mode": "standard"},
            connect_timeout=cfg.s3_get_timeout_s,
            read_timeout=cfg.s3_get_timeout_s,
            max_pool_connections=max(cfg.s3_fetch_concurrency * 2, 32),
        )
        return boto3.client("s3", config=boto_cfg)

    @staticmethod
    def _build_llm(cfg: OCRQwenV1Config):
        # Import vLLM lazily so that a missing GPU / vLLM install does not
        # break the rest of the worker registry.
        from vllm import LLM

        logger.info(
            f"[ocr_qwen_v1] loading vLLM model={cfg.model_id} dtype={cfg.dtype} "
            f"gpu_mem_util={cfg.gpu_memory_utilization} max_model_len={cfg.max_model_len}"
        )
        t0 = time.time()
        llm = LLM(
            model=cfg.model_id,
            dtype=cfg.dtype,
            tensor_parallel_size=cfg.tensor_parallel_size,
            gpu_memory_utilization=cfg.gpu_memory_utilization,
            max_model_len=cfg.max_model_len,
            trust_remote_code=cfg.trust_remote_code,
            enforce_eager=cfg.enforce_eager,
        )
        logger.info(f"[ocr_qwen_v1] vLLM ready in {time.time() - t0:.1f}s")
        return llm

    @staticmethod
    def _build_sampling_params(cfg: OCRQwenV1Config):
        from vllm import SamplingParams

        return SamplingParams(
            temperature=cfg.temperature,
            max_tokens=cfg.max_tokens,
        )

    # ------------------------------------------------------------------
    # JobWorker protocol
    # ------------------------------------------------------------------

    def run(self, ctx: JobContext) -> TaskResult:
        cfg = self.cfg
        vol = ctx.volume
        manifest = ctx.volume_manifest.manifest
        n_expected = len(manifest)
        if n_expected == 0:
            raise TerminalTaskError(f"empty manifest for volume {vol.w_id}/{vol.i_id}")

        logger.info(
            f"[ocr_qwen_v1] {vol.w_id}/{vol.i_id}: {n_expected} images "
            f"(source s3://{_SOURCE_BUCKET})"
        )

        t_start = time.time()
        try:
            return self._run_volume(ctx, manifest, t_start)
        finally:
            if cfg.reset_caches_between_volumes:
                self._reset_caches()

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

        # 1) Fetch + decode in parallel. Import lazily so the package can be
        # introspected without the psycopg/DB stack.
        from bec_orch.core.worker_runtime import get_s3_folder_prefix

        vol_prefix = get_s3_folder_prefix(vol.w_id, vol.i_id)
        t_fetch = time.time()
        fetched, failed = self._fetch_and_decode(vol_prefix, manifest)
        fetch_dur = time.time() - t_fetch
        logger.info(
            f"[ocr_qwen_v1] {vol.w_id}/{vol.i_id}: fetch+decode "
            f"{len(fetched)} ok / {len(failed)} failed in {fetch_dur:.1f}s "
            f"({len(fetched) / max(fetch_dur, 1e-6):.1f} p/s)"
        )

        if not fetched:
            # All pages failed before reaching the GPU — definitely terminal.
            raise TerminalTaskError(
                f"all {n_expected} pages failed before OCR for {vol.w_id}/{vol.i_id}"
            )

        # 2) Single big LLM.chat call. The engine handles concurrency and
        # KV-cache scheduling internally. ``use_tqdm=False`` keeps logs clean.
        t_ocr = time.time()
        outputs = self._ocr_batch(fetched)
        ocr_dur = time.time() - t_ocr
        logger.info(
            f"[ocr_qwen_v1] {vol.w_id}/{vol.i_id}: OCR done in {ocr_dur:.1f}s "
            f"({len(fetched) / max(ocr_dur, 1e-6):.2f} p/s)"
        )

        # 3) Stream rows to parquet (success + failure rows go into the same
        # parquet file; failure detail also lands in the optional jsonl).
        parquet_uri, errors_uri = self._artifact_uris(ctx)
        writer = StreamingOCRQwenV1Writer(
            parquet_uri=parquet_uri,
            errors_jsonl_uri=errors_uri,
            model_id=cfg.model_id,
            flush_every=cfg.parquet_flush_every,
            compression=cfg.parquet_compression,
        )

        ocr_failures = 0
        ocr_truncated = 0
        per_page_durations_ms: list[float] = []

        try:
            for page, out in zip(fetched, outputs, strict=True):
                if not out.outputs:
                    writer.write_error(
                        filename=page.filename,
                        source_etag=page.etag,
                        stage="ocr",
                        error_message="empty vLLM output",
                    )
                    ocr_failures += 1
                    continue
                gen = out.outputs[0]
                finish_reason = gen.finish_reason or ""
                truncated = finish_reason == "length"
                writer.write_success(
                    filename=page.filename,
                    source_etag=page.etag,
                    page_text=gen.text or "",
                    output_tokens=len(gen.token_ids),
                    finish_reason=finish_reason,
                    truncated=truncated,
                )
                if truncated:
                    ocr_truncated += 1

            for f in failed:
                writer.write_error(
                    filename=f.filename,
                    source_etag=f.etag,
                    stage=f.stage,
                    error_message=f.error,
                )
        finally:
            writer.close()

        elapsed_ms = (time.time() - t_start) * 1000
        per_page_durations_ms = [elapsed_ms / n_expected] * n_expected  # best-effort

        n_errors = ocr_failures + len(failed)
        if cfg.treat_truncation_as_failure:
            n_errors += ocr_truncated

        errors_by_stage: dict[str, int] = {}
        for f in failed:
            errors_by_stage[f.stage] = errors_by_stage.get(f.stage, 0) + 1
        if ocr_failures:
            errors_by_stage["ocr"] = errors_by_stage.get("ocr", 0) + ocr_failures
        if cfg.treat_truncation_as_failure and ocr_truncated:
            errors_by_stage["truncation"] = ocr_truncated

        failure_rate = n_errors / n_expected if n_expected else 0.0
        logger.info(
            f"[ocr_qwen_v1] {vol.w_id}/{vol.i_id}: done in {elapsed_ms / 1000:.1f}s "
            f"({n_expected} pages, success={writer.success_count}, "
            f"errors={n_errors}, truncated={ocr_truncated}, rate={failure_rate:.2%})"
        )

        # Classify retryable vs terminal at volume level.
        if failure_rate > cfg.max_page_failure_rate:
            # High failure rate — likely transient (S3, GPU OOM, bad batch).
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
    # Fetch + decode
    # ------------------------------------------------------------------

    def _fetch_and_decode(
        self, vol_prefix: str, manifest: list[dict[str, Any]]
    ) -> tuple[list[_FetchedPage], list[_FailedPage]]:
        cfg = self.cfg
        ok: list[_FetchedPage] = []
        ko: list[_FailedPage] = []

        # Preserve manifest order in the output so parquet rows match the
        # natural folio sequence (handy for downstream consumers).
        order = {item.get("filename"): idx for idx, item in enumerate(manifest)}
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
                img = bytes_to_rgb(data, cfg)
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

            ok_by_filename[filename] = _FetchedPage(filename=filename, etag=etag, image=img)

        with ThreadPoolExecutor(max_workers=cfg.s3_fetch_concurrency) as pool:
            futures = [
                pool.submit(_one, item["filename"])
                for item in manifest
                if item.get("filename")
            ]
            for _ in as_completed(futures):
                pass  # _one writes into ok_by_filename / ko_by_filename

        # Stable, manifest-ordered output.
        for item in manifest:
            fn = item.get("filename")
            if fn in ok_by_filename:
                ok.append(ok_by_filename[fn])
            elif fn in ko_by_filename:
                ko.append(ko_by_filename[fn])

        return ok, ko

    # ------------------------------------------------------------------
    # OCR
    # ------------------------------------------------------------------

    def _ocr_batch(self, pages: list[_FetchedPage]):
        """Run one synchronous ``LLM.chat`` call over the whole volume.

        Returns the raw vLLM ``RequestOutput`` list in the same order as
        ``pages``.
        """
        prompt = self.cfg.prompt
        messages = [
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_pil", "image_pil": p.image},
                        {"type": "text", "text": prompt},
                    ],
                }
            ]
            for p in pages
        ]
        return self._llm.chat(messages, sampling_params=self._sampling, use_tqdm=False)

    # ------------------------------------------------------------------
    # Caches
    # ------------------------------------------------------------------

    def _reset_caches(self) -> None:
        try:
            self._llm.reset_mm_cache()
        except Exception as e:  # noqa: BLE001 — vLLM API may vary
            logger.debug(f"[ocr_qwen_v1] reset_mm_cache failed (non-fatal): {e}")
        try:
            self._llm.reset_prefix_cache()
        except Exception as e:  # noqa: BLE001
            logger.debug(f"[ocr_qwen_v1] reset_prefix_cache failed (non-fatal): {e}")
        gc.collect()

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
