"""``script_classification_v2`` JobWorker.

Wraps the vendored single-head DINOv3 classifier pipeline
(``BDRC/8-class-tibetan-page-classifier``). The pipeline's integration surface
is ``pipe.run_batch(list[bytes]) -> list[dict]`` (plus single-image
``pipe.run(bytes) -> dict``): both decode, preprocess and classify internally
and never raise. So the worker only fetches raw bytes per page from S3 (in
fixed-size batches, to bound resident memory), hands each batch to
``pipe.run_batch()``, and streams results to parquet.

A background prefetch thread fetches the next batch(es) of raw bytes from S3
(``cfg.prefetch_batches`` deep, bounded queue) while the main thread runs the
CPU decode + GPU forward of the current batch — identical overlap strategy to
``script_classification``.

The model is loaded once in ``__init__`` and reused across every volume; there
is no per-volume cache to reset.
"""

from __future__ import annotations

import logging
import os
import queue
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import boto3
from botocore.config import Config as BotoConfig

from bec_orch.core.models import TaskResult
from bec_orch.errors import RetryableTaskError, TerminalTaskError

from .config import ScriptClassificationV2Config
from .parquet_writer import StreamingScriptClassificationV2Writer

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


class ScriptClassificationV2JobWorker:
    """JobWorker for ``script_classification_v2``.

    The classification model is loaded once on construction and reused for
    every volume the worker processes.
    """

    def __init__(self, cfg: ScriptClassificationV2Config) -> None:
        self.cfg = cfg
        self._s3 = self._build_s3_client(cfg)

        from .vendor.loader import get_pipeline

        self._pipe = get_pipeline(cfg)
        self._model_version = self._pipe.model_version

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_s3_client(cfg: ScriptClassificationV2Config):
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

        src_bucket = ctx.source_bucket or _SOURCE_BUCKET
        logger.info(
            f"[script_classification_v2] {vol.w_id}/{vol.i_id}: {n_expected} images "
            f"(source={ctx.source}, s3://{src_bucket}/{ctx.image_prefix})"
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

        # Image location is resolved by the runtime (source-aware) and passed on
        # the context; fall back to BDRC-style routing only if unset.
        src_bucket = ctx.source_bucket or _SOURCE_BUCKET
        if ctx.image_prefix:
            vol_prefix = ctx.image_prefix
        else:
            from bec_orch.core.worker_runtime import get_s3_folder_prefix

            vol_prefix = get_s3_folder_prefix(vol.w_id, vol.i_id)

        parquet_uri, errors_uri = self._artifact_uris(ctx)
        writer = StreamingScriptClassificationV2Writer(
            parquet_uri=parquet_uri,
            errors_jsonl_uri=errors_uri,
            model_version=self._model_version,
            labels=self._pipe.labels,
            model_repo=cfg.model_repo_id,
            flush_every=cfg.parquet_flush_every,
            compression=cfg.parquet_compression,
        )

        n_classified = 0
        errors_by_stage: dict[str, int] = {}
        t_classify_total = 0.0
        t_fetch_wait_total = 0.0

        prefetch_depth = max(1, getattr(cfg, "prefetch_batches", 1))
        fetch_q: queue.Queue = queue.Queue(maxsize=prefetch_depth)
        stop_event = threading.Event()
        producer_error: list[BaseException] = []
        _SENTINEL = None

        def _producer() -> None:
            try:
                for b_idx in range(n_batches):
                    if stop_event.is_set():
                        return
                    b_start = b_idx * batch_size
                    b_stop = min(b_start + batch_size, n_expected)
                    b_manifest = manifest[b_start:b_stop]
                    fetched, failed = self._fetch_raw_bytes(src_bucket, vol_prefix, b_manifest)
                    while not stop_event.is_set():
                        try:
                            fetch_q.put((b_idx, b_manifest, fetched, failed), timeout=0.5)
                            break
                        except queue.Full:
                            continue
            except BaseException as e:  # noqa: BLE001 - surfaced to consumer
                producer_error.append(e)
            finally:
                while not stop_event.is_set():
                    try:
                        fetch_q.put(_SENTINEL, timeout=0.5)
                        break
                    except queue.Full:
                        continue

        fetcher = threading.Thread(target=_producer, name="scriptclsv2-prefetch", daemon=True)
        fetcher.start()

        try:
            while True:
                t_wait = time.time()
                item = fetch_q.get()
                t_fetch_wait_total += time.time() - t_wait
                if item is _SENTINEL:
                    break
                batch_idx, batch_manifest, fetched, failed = item

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
                        f"[script_classification_v2] {vol.w_id}/{vol.i_id}: batch "
                        f"{batch_idx + 1}/{n_batches} — all "
                        f"{len(batch_manifest)} pages failed before classification"
                    )
                    continue

                t_classify = time.time()
                # never raises; len(rows) == len(fetched), order-aligned.
                rows = self._pipe.run_batch([page.data for page in fetched])
                for page, row in zip(fetched, rows):
                    if row["status"] == "ok":
                        writer.write_success(
                            filename=page.filename,
                            source_etag=page.etag,
                            exif_orientation_tag=row["exif_orientation_tag"],
                            label=row["label"],
                            prob=row["prob"],
                            probs=row["probs"],
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
                    f"[script_classification_v2] {vol.w_id}/{vol.i_id}: batch "
                    f"{batch_idx + 1}/{n_batches} done "
                    f"(classified {n_classified}/{n_expected}, "
                    f"cum_classify={t_classify_total:.1f}s, "
                    f"cum_fetch_wait={t_fetch_wait_total:.1f}s)"
                )

            if producer_error:
                raise producer_error[0]

            if n_classified == 0:
                raise TerminalTaskError(
                    f"all {n_expected} pages failed before classification for "
                    f"{vol.w_id}/{vol.i_id}"
                )
        finally:
            stop_event.set()
            try:
                while True:
                    fetch_q.get_nowait()
            except queue.Empty:
                pass
            fetcher.join(timeout=30)
            writer.close()

        elapsed_ms = (time.time() - t_start) * 1000

        n_errors = sum(errors_by_stage.values())
        failure_rate = n_errors / n_expected if n_expected else 0.0
        logger.info(
            f"[script_classification_v2] {vol.w_id}/{vol.i_id}: done in {elapsed_ms / 1000:.1f}s "
            f"({n_expected} pages, success={writer.success_count}, "
            f"errors={n_errors}, rate={failure_rate:.2%}, "
            f"classify={t_classify_total:.1f}s, fetch_wait={t_fetch_wait_total:.1f}s)"
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
        self, bucket: str, vol_prefix: str, manifest: list[dict[str, Any]]
    ) -> tuple[list[_FetchedPage], list[_FailedPage]]:
        cfg = self.cfg
        ok_by_filename: dict[str, _FetchedPage] = {}
        ko_by_filename: dict[str, _FailedPage] = {}

        def _one(filename: str) -> None:
            key = f"{vol_prefix}{filename}"
            try:
                resp = self._s3.get_object(Bucket=bucket, Key=key)
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
