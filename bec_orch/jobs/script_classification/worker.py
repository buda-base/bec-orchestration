"""``script_classification`` JobWorker.

Wraps the vendored ``tibetan-manuscript-classifier`` pipeline (orientation +
6-class script classification; blank-page pre-filter disabled — see
``vendor/pipeline.py``). Its entire integration surface is
``pipe.run(image_bytes) -> dict``: it decodes, preprocesses, and classifies
internally and never raises. So the worker's only job is: fetch raw bytes
per page from S3 (in fixed-size batches, to bound resident memory), hand
them straight to ``pipe.run()``, and stream results to parquet.

The pipeline (both HF checkpoints) is loaded once in ``__init__`` and
reused across every volume this worker process handles — there is no
per-volume cache to reset (unlike ``ocr_qwen_v1``'s vLLM KV/prefix cache):
the vendored ``Classifier`` instances hold no mutable per-call state.
"""

from __future__ import annotations

import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import boto3
from botocore.config import Config as BotoConfig

from bec_orch.core.models import TaskResult
from bec_orch.errors import RetryableTaskError, TerminalTaskError

from .config import ScriptClassificationConfig
from .parquet_writer import StreamingScriptClassificationWriter

if TYPE_CHECKING:
    from bec_orch.jobs.base import JobContext

logger = logging.getLogger(__name__)

# S3 source bucket for BDRC images — same convention as ldv1/ocr_qwen_v1.
_SOURCE_BUCKET = os.environ.get("BEC_SOURCE_S3_BUCKET", "archive.tbrc.org")


@dataclass
class _FetchedPage:
    filename: str
    etag: str
    data: bytes


@dataclass
class _FailedPage:
    filename: str
    stage: str  # "fetch" | "classify"
    etag: str | None
    error: str


class ScriptClassificationJobWorker:
    """JobWorker for ``script_classification``.

    The classification pipeline is loaded once on construction and reused
    for every volume the worker processes.
    """

    def __init__(self, cfg: ScriptClassificationConfig) -> None:
        self.cfg = cfg
        self._s3 = self._build_s3_client(cfg)

        from .vendor.loader import get_pipeline

        self._pipe = get_pipeline(cfg)
        # Pipeline already computes a real model_version internally (short
        # checkpoint hashes from hf_hub_download's resolved snapshot path) —
        # reuse it rather than re-deriving from repo-id strings.
        self._model_version = self._pipe.model_version

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_s3_client(cfg: ScriptClassificationConfig):
        region = os.environ.get("BEC_REGION", "us-east-1")
        boto_cfg = BotoConfig(
            region_name=region,
            retries={"max_attempts": cfg.s3_max_attempts, "mode": "standard"},
            connect_timeout=cfg.s3_get_timeout_s,
            read_timeout=cfg.s3_get_timeout_s,
            max_pool_connections=max(cfg.s3_fetch_concurrency * 2, 32),
        )
        return boto3.client("s3", config=boto_cfg)

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
            f"[script_classification] {vol.w_id}/{vol.i_id}: {n_expected} images "
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
        batch_size = max(1, cfg.classify_batch_size)
        n_batches = (n_expected + batch_size - 1) // batch_size

        # Import lazily so the package can be introspected without the
        # psycopg/DB stack.
        from bec_orch.core.worker_runtime import get_s3_folder_prefix

        vol_prefix = get_s3_folder_prefix(vol.w_id, vol.i_id)

        parquet_uri, errors_uri = self._artifact_uris(ctx)
        writer = StreamingScriptClassificationWriter(
            parquet_uri=parquet_uri,
            errors_jsonl_uri=errors_uri,
            model_version=self._model_version,
            flush_every=cfg.parquet_flush_every,
            compression=cfg.parquet_compression,
        )

        n_classified = 0
        errors_by_stage: dict[str, int] = {}
        t_fetch_total = 0.0
        t_classify_total = 0.0

        try:
            for batch_idx in range(n_batches):
                start = batch_idx * batch_size
                stop = min(start + batch_size, n_expected)
                batch_manifest = manifest[start:stop]

                t_fetch = time.time()
                fetched, failed = self._fetch_raw_bytes(vol_prefix, batch_manifest)
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
                        f"[script_classification] {vol.w_id}/{vol.i_id}: batch "
                        f"{batch_idx + 1}/{n_batches} ({start}:{stop}) — "
                        f"all {len(batch_manifest)} pages failed before classification"
                    )
                    continue

                t_classify = time.time()
                for page in fetched:
                    row = self._pipe.run(page.data)  # never raises
                    if row["status"] == "ok":
                        writer.write_success(
                            filename=page.filename,
                            source_etag=page.etag,
                            exif_orientation_tag=row["exif_orientation_tag"],
                            orientation_pred=row["orientation_pred"],
                            orientation_prob=row["orientation_prob"],
                            rotation_applied=row["rotation_applied"],
                            sixclass_label=row["sixclass_label"],
                            sixclass_probs=row["sixclass_probs"],
                            final_label=row["final_label"],
                            model_version=row["model_version"],
                        )
                    else:
                        writer.write_error(
                            filename=page.filename,
                            source_etag=page.etag,
                            stage="classify",
                            error_message=row["error"] or "unknown classification error",
                        )
                        errors_by_stage["classify"] = errors_by_stage.get("classify", 0) + 1
                t_classify_total += time.time() - t_classify

                n_classified += len(fetched)

                logger.info(
                    f"[script_classification] {vol.w_id}/{vol.i_id}: batch "
                    f"{batch_idx + 1}/{n_batches} done "
                    f"(classified {n_classified}/{n_expected}, "
                    f"cum_fetch={t_fetch_total:.1f}s, cum_classify={t_classify_total:.1f}s)"
                )

            if n_classified == 0:
                # Every single page failed before reaching the model — terminal.
                raise TerminalTaskError(
                    f"all {n_expected} pages failed before classification for {vol.w_id}/{vol.i_id}"
                )
        finally:
            writer.close()

        elapsed_ms = (time.time() - t_start) * 1000

        n_errors = sum(errors_by_stage.values())
        failure_rate = n_errors / n_expected if n_expected else 0.0
        logger.info(
            f"[script_classification] {vol.w_id}/{vol.i_id}: done in {elapsed_ms / 1000:.1f}s "
            f"({n_expected} pages, success={writer.success_count}, "
            f"errors={n_errors}, rate={failure_rate:.2%}, "
            f"fetch={t_fetch_total:.1f}s, classify={t_classify_total:.1f}s)"
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
    # Fetch (raw bytes only — the vendored pipeline decodes internally)
    # ------------------------------------------------------------------

    def _fetch_raw_bytes(
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

            ok_by_filename[filename] = _FetchedPage(filename=filename, etag=etag, data=data)

        with ThreadPoolExecutor(max_workers=cfg.s3_fetch_concurrency) as pool:
            futures = [
                pool.submit(_one, item["filename"])
                for item in manifest
                if item.get("filename")
            ]
            for _ in as_completed(futures):
                pass  # _one writes into ok_by_filename / ko_by_filename

        # Stable, manifest-ordered output.
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
