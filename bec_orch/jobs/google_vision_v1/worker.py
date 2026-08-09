"""``google_vision_v1`` JobWorker.

No GPU. For a single volume, ``run(ctx)`` performs the whole Google Vision
async batch flow synchronously and keeps ALL state in memory (there is no
external DB tracking batches -- the runtime's SQS visibility extender keeps the
message alive while we wait):

    1. TRANSFER  stream-copy every page image S3 -> GCS staging (kept, not
                 deleted). Pages are grouped into lanes: ``images`` (jpg/png/jp2)
                 and ``files`` (tif/tiff).
    2. SUBMIT    per lane, chunk pages into batches (<= batch_size) and submit
                 an async DOCUMENT_TEXT_DETECTION operation for each, respecting
                 ``max_concurrent_ops`` and backing off on quota (429) errors.
    3. WAIT      poll the long-running operations until every batch is
                 completed/failed (or the volume timeout is hit).
    4. EXPORT    read the Vision JSON back from GCS, parse it into a per-volume
                 parquet, and write parquet + raw ``jsonl.zst`` to the dest S3
                 bucket under ``ctx.artifacts_location``. The runtime writes
                 ``success.json`` afterward.

The clients are built once in ``__init__`` and reused across volumes.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from bec_orch.core.models import TaskResult
from bec_orch.errors import RetryableTaskError, TerminalTaskError

from . import parsing
from .config import GoogleVisionV1Config
from .vision_io import GcsIO, VisionClient, build_s3_client, lane_and_media_type, parse_gs_uri

if TYPE_CHECKING:
    from bec_orch.jobs.base import JobContext

logger = logging.getLogger(__name__)

# Source S3 bucket for BDRC images (same convention as the other jobs).
_SOURCE_BUCKET = os.environ.get("BEC_SOURCE_S3_BUCKET", "archive.tbrc.org")

_GS_IN_ERROR_RE = re.compile(r"(gs://[^\s]+)")


@dataclass
class _Page:
    filename: str
    img_idx: int  # 0-based line index in the volume manifest
    lane: str
    media_type: str
    s3_key: str
    gcs_blob_name: str
    source_gcs_uri: str
    source_etag: str = ""


@dataclass
class _Batch:
    batch_id: str
    lane: str
    output_prefix: str  # gs://.../{lane}/{batch_id}/
    pages: list[_Page]
    status: str = "pending"  # pending | submitted | completed | failed
    operation_name: str | None = None
    submit_attempts: int = 0
    error: str | None = None


class GoogleVisionV1JobWorker:
    """JobWorker for ``google_vision_v1`` (S3 -> GCS -> Vision async -> S3)."""

    def __init__(self, cfg: GoogleVisionV1Config) -> None:
        self.cfg = cfg
        self._s3 = build_s3_client(cfg)
        self._vision = VisionClient(cfg.google_credentials_path, cfg.feature_type, cfg.model)
        self._gcs = GcsIO(cfg.google_credentials_path, cfg.gcp_project)

    # ------------------------------------------------------------------
    # JobWorker protocol
    # ------------------------------------------------------------------

    def run(self, ctx: JobContext) -> TaskResult:
        vol = ctx.volume
        manifest = ctx.volume_manifest.manifest
        if len(manifest) == 0:
            raise TerminalTaskError(f"empty manifest for volume {vol.w_id}/{vol.i_id}")

        t_start = time.time()
        pages = self._build_pages(ctx, manifest)
        if not pages:
            raise TerminalTaskError(f"no usable page filenames for volume {vol.w_id}/{vol.i_id}")

        logger.info(
            f"[google_vision_v1] {vol.w_id}/{vol.i_id}: {len(pages)} pages "
            f"(source=s3://{ctx.source_bucket or _SOURCE_BUCKET}/{self._vol_prefix(ctx)})"
        )

        errors_by_stage: dict[str, int] = {}

        ok_pages = self._transfer_pages(ctx, pages, errors_by_stage)
        batches = self._build_batches(ctx, ok_pages)
        self._run_batches(ctx, batches)

        rows_by_idx, raw_records, matched_uris = self._collect_results(batches)

        # Classify remaining errors.
        for b in batches:
            if b.status == "failed":
                errors_by_stage["operation"] = errors_by_stage.get("operation", 0) + len(b.pages)
        completed_pages = sum(len(b.pages) for b in batches if b.status == "completed")
        missing = completed_pages - len(matched_uris)
        if missing > 0:
            errors_by_stage["missing_response"] = errors_by_stage.get("missing_response", 0) + missing

        n_expected = len(pages)
        n_errors = sum(errors_by_stage.values())
        n_matched = len(matched_uris)

        # Write artifacts (parquet + raw jsonl.zst) for whatever we did get.
        if rows_by_idx:
            self._write_artifacts(ctx, rows_by_idx, raw_records)

        elapsed_ms = (time.time() - t_start) * 1000
        failure_rate = n_errors / n_expected if n_expected else 0.0
        logger.info(
            f"[google_vision_v1] {vol.w_id}/{vol.i_id}: done in {elapsed_ms / 1000:.1f}s "
            f"({n_expected} pages, ocr={n_matched}, errors={n_errors}, rate={failure_rate:.2%}, "
            f"errors_by_stage={errors_by_stage or {}})"
        )

        if n_matched == 0:
            raise RetryableTaskError(
                f"no Vision responses obtained for {vol.w_id}/{vol.i_id} "
                f"({n_expected} pages; errors_by_stage={errors_by_stage})"
            )
        if failure_rate > self.cfg.max_page_failure_rate:
            raise RetryableTaskError(
                f"volume {vol.w_id}/{vol.i_id} failure rate {failure_rate:.2%} "
                f"> threshold {self.cfg.max_page_failure_rate:.2%} "
                f"(errors_by_stage={errors_by_stage})"
            )

        return TaskResult(
            total_images=n_expected,
            nb_errors=n_errors,
            total_duration_ms=elapsed_ms,
            avg_duration_per_page_ms=elapsed_ms / n_expected,
            errors_by_stage=errors_by_stage or None,
        )

    # ------------------------------------------------------------------
    # Step helpers
    # ------------------------------------------------------------------

    def _vol_prefix(self, ctx: JobContext) -> str:
        if ctx.image_prefix:
            return ctx.image_prefix
        from bec_orch.core.worker_runtime import get_s3_folder_prefix

        return get_s3_folder_prefix(ctx.volume.w_id, ctx.volume.i_id)

    @staticmethod
    def _version_segment(ctx: JobContext) -> str:
        # artifacts prefix is "{job_name}/{w_id}/{i_id}/{version}"
        parts = ctx.artifacts_location.prefix.strip("/").split("/")
        return parts[-1] if parts else "000000"

    def _build_pages(self, ctx: JobContext, manifest: list[dict[str, Any]]) -> list[_Page]:
        cfg = self.cfg
        vol = ctx.volume
        vol_prefix = self._vol_prefix(ctx)
        version = self._version_segment(ctx)
        vol_seg = f"{vol.w_id}/{vol.i_id}/{version}"

        pages: list[_Page] = []
        for idx, item in enumerate(manifest):
            fn = item.get("filename")
            if not fn:
                continue
            lane, media = lane_and_media_type(fn)
            gcs_blob = f"{cfg.staging_prefix}{vol_seg}/{fn}"
            pages.append(
                _Page(
                    filename=fn,
                    img_idx=idx,
                    lane=lane,
                    media_type=media,
                    s3_key=f"{vol_prefix}{fn}",
                    gcs_blob_name=gcs_blob,
                    source_gcs_uri=f"gs://{cfg.staging_bucket}/{gcs_blob}",
                )
            )
        return pages

    def _transfer_pages(
        self, ctx: JobContext, pages: list[_Page], errors_by_stage: dict[str, int]
    ) -> list[_Page]:
        """Stream-copy pages S3 -> GCS in parallel. Returns the successful pages."""
        cfg = self.cfg
        src_bucket = ctx.source_bucket or _SOURCE_BUCKET
        failed: dict[str, str] = {}

        def _one(page: _Page) -> None:
            try:
                if cfg.skip_existing_gcs and self._gcs.blob_exists(cfg.staging_bucket, page.gcs_blob_name):
                    # Already staged (e.g. reprocessing); grab the source ETag cheaply.
                    try:
                        head = self._s3.head_object(Bucket=src_bucket, Key=page.s3_key)
                        page.source_etag = head.get("ETag", "").strip('"')
                    except Exception:  # noqa: BLE001 - etag is best-effort here
                        pass
                    return
                page.source_etag = self._gcs.upload_from_s3(
                    self._s3,
                    src_bucket,
                    page.s3_key,
                    cfg.staging_bucket,
                    page.gcs_blob_name,
                    content_type=page.media_type,
                )
            except Exception as e:  # noqa: BLE001 - boto3/gcs error subtypes
                failed[page.filename] = f"transfer failed: {e}"

        with ThreadPoolExecutor(max_workers=cfg.transfer_concurrency) as pool:
            for _ in as_completed([pool.submit(_one, p) for p in pages]):
                pass

        if failed:
            errors_by_stage["transfer"] = errors_by_stage.get("transfer", 0) + len(failed)
            logger.warning(
                f"[google_vision_v1] {ctx.volume.w_id}/{ctx.volume.i_id}: "
                f"{len(failed)}/{len(pages)} pages failed S3->GCS transfer"
            )
        return [p for p in pages if p.filename not in failed]

    def _build_batches(self, ctx: JobContext, pages: list[_Page]) -> list[_Batch]:
        cfg = self.cfg
        vol = ctx.volume
        version = self._version_segment(ctx)
        vol_seg = f"{vol.w_id}/{vol.i_id}/{version}"

        by_lane: dict[str, list[_Page]] = defaultdict(list)
        for p in pages:
            by_lane[p.lane].append(p)

        batches: list[_Batch] = []
        for lane in sorted(by_lane):
            lane_pages = by_lane[lane]
            for start in range(0, len(lane_pages), cfg.batch_size):
                chunk = lane_pages[start : start + cfg.batch_size]
                n = start // cfg.batch_size + 1
                batch_id = f"{vol.w_id}-{vol.i_id}-{version}-{lane}-{n:04d}"
                out_prefix = f"gs://{cfg.output_bucket}/{cfg.output_prefix}{vol_seg}/{lane}/{batch_id}/"
                batches.append(_Batch(batch_id=batch_id, lane=lane, output_prefix=out_prefix, pages=chunk))
        return batches

    def _run_batches(self, ctx: JobContext, batches: list[_Batch]) -> None:
        """Submit + poll all batches in memory until every one is terminal."""
        if not batches:
            return
        cfg = self.cfg
        vol = ctx.volume
        deadline = time.time() + cfg.volume_timeout_s
        pause_until = 0.0
        consecutive_429 = 0

        def _terminal() -> bool:
            return all(b.status in ("completed", "failed") for b in batches)

        while not _terminal():
            now = time.time()
            if now > deadline:
                raise RetryableTaskError(
                    f"volume {vol.w_id}/{vol.i_id} timed out after {cfg.volume_timeout_s:.0f}s "
                    f"waiting for Vision (statuses: {self._status_counts(batches)})"
                )

            # --- submit pending, respecting concurrency + quota pause ---
            if now >= pause_until:
                in_flight = sum(1 for b in batches if b.status == "submitted")
                for b in [b for b in batches if b.status == "pending"]:
                    if in_flight >= cfg.max_concurrent_ops:
                        break
                    try:
                        b.operation_name = self._vision.submit_batch(
                            [p.source_gcs_uri for p in b.pages],
                            b.output_prefix,
                            cfg.output_shard_size,
                        )
                        b.status = "submitted"
                        in_flight += 1
                        consecutive_429 = 0
                        logger.info(
                            f"[google_vision_v1] {vol.w_id}/{vol.i_id}: submitted {b.batch_id} "
                            f"({len(b.pages)} images) -> {b.operation_name}"
                        )
                    except Exception as e:  # noqa: BLE001
                        if self._is_quota_error(e):
                            consecutive_429 += 1
                            pause = min(
                                cfg.quota_pause_s * (cfg.quota_pause_multiplier ** (consecutive_429 - 1)),
                                cfg.max_quota_pause_s,
                            )
                            pause_until = time.time() + pause
                            logger.warning(
                                f"[google_vision_v1] {vol.w_id}/{vol.i_id}: quota error on "
                                f"{b.batch_id}, pausing {pause:.0f}s ({e})"
                            )
                            break
                        b.submit_attempts += 1
                        if b.submit_attempts >= cfg.max_submit_retries:
                            b.status = "failed"
                            b.error = f"submit failed after {b.submit_attempts} attempts: {e}"
                            logger.error(f"[google_vision_v1] {vol.w_id}/{vol.i_id}: {b.error}")
                        else:
                            logger.warning(
                                f"[google_vision_v1] {vol.w_id}/{vol.i_id}: submit error on "
                                f"{b.batch_id} (attempt {b.submit_attempts}): {e}"
                            )

            # --- poll submitted ---
            for b in [b for b in batches if b.status == "submitted"]:
                try:
                    done, err = self._vision.check_operation(b.operation_name or "")
                except Exception as e:  # noqa: BLE001 - transient; retry next cycle
                    logger.warning(
                        f"[google_vision_v1] {vol.w_id}/{vol.i_id}: poll error on {b.batch_id}: {e}"
                    )
                    continue
                if not done:
                    continue
                if err and not self._output_written(err):
                    b.status = "failed"
                    b.error = err
                    logger.warning(f"[google_vision_v1] {vol.w_id}/{vol.i_id}: {b.batch_id} failed: {err}")
                else:
                    if err:
                        logger.warning(
                            f"[google_vision_v1] {vol.w_id}/{vol.i_id}: {b.batch_id} returned "
                            f"'{err}' but output exists; treating as completed"
                        )
                    b.status = "completed"
                    logger.info(f"[google_vision_v1] {vol.w_id}/{vol.i_id}: {b.batch_id} completed")

            if _terminal():
                break
            time.sleep(cfg.poll_interval_s)

    def _collect_results(
        self, batches: list[_Batch]
    ) -> tuple[dict[int, dict], list[dict], set[str]]:
        """Download Vision JSON for completed batches and parse into rows."""
        cfg = self.cfg
        completed = [b for b in batches if b.status == "completed"]
        uri_to_page: dict[str, _Page] = {}
        for b in completed:
            for p in b.pages:
                uri_to_page[p.source_gcs_uri] = p

        rows_by_idx: dict[int, dict] = {}
        raw_records: list[dict] = []
        matched_uris: set[str] = set()
        if not completed:
            return rows_by_idx, raw_records, matched_uris

        blob_names: list[str] = []
        for b in completed:
            _, prefix = parse_gs_uri(b.output_prefix)
            try:
                blob_names.extend(self._gcs.list_output_blob_names(cfg.output_bucket, prefix))
            except Exception as e:  # noqa: BLE001
                logger.warning(f"[google_vision_v1] failed to list output for {b.batch_id}: {e}")

        def _dl(name: str) -> tuple[str, bytes | None]:
            try:
                return name, self._gcs.download_blob_bytes(
                    cfg.output_bucket, name, cfg.gcs_download_timeout_s
                )
            except Exception as e:  # noqa: BLE001
                logger.warning(f"[google_vision_v1] failed to download {name}: {e}")
                return name, None

        with ThreadPoolExecutor(max_workers=cfg.gcs_download_concurrency) as pool:
            for fut in as_completed([pool.submit(_dl, n) for n in blob_names]):
                name, data = fut.result()
                if data is None:
                    continue
                try:
                    parsed = json.loads(data)
                except Exception as e:  # noqa: BLE001
                    logger.warning(f"[google_vision_v1] failed to parse {name}: {e}")
                    continue
                for resp in parsed.get("responses", []):
                    uri = resp.get("context", {}).get("uri", "")
                    page = uri_to_page.get(uri)
                    if page is None or uri in matched_uris:
                        continue
                    matched_uris.add(uri)
                    rows_by_idx[page.img_idx] = parsing.parse_response(
                        page.filename, page.img_idx, page.source_etag, resp
                    )
                    raw_records.append(
                        {"img_file_name": page.filename, "img_idx": page.img_idx, "response": resp}
                    )

        return rows_by_idx, raw_records, matched_uris

    def _write_artifacts(self, ctx: JobContext, rows_by_idx: dict[int, dict], raw_records: list[dict]) -> None:
        cfg = self.cfg
        rows = [rows_by_idx[k] for k in sorted(rows_by_idx)]
        raw_records = sorted(raw_records, key=lambda r: r["img_idx"])

        parquet_bytes = parsing.rows_to_parquet_bytes(rows, cfg.parquet_compression)
        jsonl_bytes = parsing.responses_to_jsonl_zst_bytes(raw_records, cfg.jsonl_zstd_level)

        loc = ctx.artifacts_location
        base = f"{loc.basename}{cfg.artifact_suffix}"
        # Write the data artifacts under a dedicated top-level prefix (default
        # ``gv/``) so they sit alongside the other Google Vision runs, keeping
        # the ``{w}/{i}/{version}`` tail (and thus the exact per-volume path +
        # filenames) that the runtime derived. If ``s3_artifact_prefix`` is
        # empty, fall back to the runtime's ``{job_name}/...`` location.
        if cfg.s3_artifact_prefix:
            tail = loc.prefix.strip("/").split("/", 1)[1]  # strip the job-name root
            prefix = f"{cfg.s3_artifact_prefix}{tail}"
        else:
            prefix = loc.prefix.rstrip("/")
        parquet_key = f"{prefix}/{base}.parquet"
        jsonl_key = f"{prefix}/{base}.jsonl.zst"

        self._s3.put_object(
            Bucket=loc.bucket, Key=parquet_key, Body=parquet_bytes, ContentType="application/octet-stream"
        )
        self._s3.put_object(
            Bucket=loc.bucket, Key=jsonl_key, Body=jsonl_bytes, ContentType="application/octet-stream"
        )
        logger.info(
            f"[google_vision_v1] {ctx.volume.w_id}/{ctx.volume.i_id}: wrote {len(rows)} rows to "
            f"s3://{loc.bucket}/{parquet_key} (+ raw jsonl.zst)"
        )

    # ------------------------------------------------------------------
    # Misc
    # ------------------------------------------------------------------

    @staticmethod
    def _is_quota_error(exc: Exception) -> bool:
        s = str(exc)
        return "429" in s or "RESOURCE_EXHAUSTED" in s

    def _output_written(self, error: str) -> bool:
        """Vision 'Error 7' write races: treat as success if the output exists."""
        if "Error 7" not in error:
            return False
        m = _GS_IN_ERROR_RE.search(error)
        if not m:
            return False
        gs_uri = m.group(1).rstrip(".")
        try:
            return self._gcs.gs_uri_exists(gs_uri)
        except Exception:  # noqa: BLE001
            return False

    @staticmethod
    def _status_counts(batches: list[_Batch]) -> dict[str, int]:
        counts: dict[str, int] = {}
        for b in batches:
            counts[b.status] = counts.get(b.status, 0) + 1
        return counts
