"""Streaming parquet writer for PaddleOCR-VL output.

Schema (one row per processed page):

    img_file_name   string    base filename as in the manifest
    source_etag     string    S3 ETag of the source image
    ok              bool      True if OCR produced text (even if truncated)
    truncated       bool      True if generation hit ``max_new_tokens``
    finish_reason   string    "stop" | "length"
    page_text       string    Unicode transcription (normalized, canonical form)
    raw_text        string    raw model output (Unicode, pre-normalization)
    rep_score       float64   repeated-n-gram fraction on the raw prediction
    likely_loop     bool      rep_score >= threshold (flag for review)
    output_tokens   int32     number of generated tokens
    res_scale       float64   adaptive resolution budget fraction used (1.0=1x)
    dry_fires       int32     times the DRY guard fired while decoding this page
    dry_max_L       int32     longest repeated-suffix match DRY penalised (severity)
    retried         bool      True if the page was re-decoded at temperature
                              (DRY fired >= threshold); page_text is the retry pick
    skipped         bool      True if OCR was skipped (see skip_reason)
    skip_reason     string    classifier label / rule that caused the skip
    error_stage     string    "" / "fetch" / "decode" / "ocr"
    error_message   string    short error description (<=512 chars)
    model_id        string    checkpoint identifier that produced the row

Errors are also optionally mirrored to ``<basename>-errors.jsonl.gz``.
"""

from __future__ import annotations

import gzip
import io
import json
import logging

import pyarrow as pa
import pyarrow.parquet as pq
import s3fs

logger = logging.getLogger(__name__)

_MAX_ERROR_MSG_LEN = 512


def paddleocr_build_schema() -> pa.Schema:
    return pa.schema(
        [
            pa.field("img_file_name", pa.string()),
            pa.field("source_etag", pa.string()),
            pa.field("ok", pa.bool_()),
            pa.field("truncated", pa.bool_()),
            pa.field("finish_reason", pa.string()),
            pa.field("page_text", pa.string()),
            pa.field("raw_text", pa.string()),
            pa.field("rep_score", pa.float64()),
            pa.field("likely_loop", pa.bool_()),
            pa.field("output_tokens", pa.int32()),
            pa.field("res_scale", pa.float64()),
            pa.field("dry_fires", pa.int32()),
            pa.field("dry_max_L", pa.int32()),
            pa.field("retried", pa.bool_()),
            pa.field("skipped", pa.bool_()),
            pa.field("skip_reason", pa.string()),
            pa.field("error_stage", pa.string()),
            pa.field("error_message", pa.string()),
            pa.field("model_id", pa.string()),
        ]
    )


def _short(msg: str | None) -> str:
    if not msg:
        return ""
    msg = msg.replace("\n", " ").strip()
    return msg if len(msg) <= _MAX_ERROR_MSG_LEN else msg[: _MAX_ERROR_MSG_LEN - 1] + "…"


class StreamingPaddleOCRWriter:
    """Streaming parquet writer, plus optional gzipped jsonl errors file."""

    def __init__(
        self,
        parquet_uri: str,
        errors_jsonl_uri: str | None,
        model_id: str,
        *,
        flush_every: int = 256,
        compression: str = "zstd",
    ) -> None:
        self.parquet_uri = parquet_uri
        self.errors_jsonl_uri = errors_jsonl_uri
        self.model_id = model_id
        self.flush_every = flush_every
        self.compression = compression if compression != "none" else None
        self.schema = paddleocr_build_schema()

        self._records: list[dict] = []
        self._error_records: list[dict] = []
        self._total_written = 0
        self._success_count = 0
        self._error_count = 0
        self._truncated_count = 0
        self._loop_count = 0
        self._skipped_count = 0

        self._s3 = s3fs.S3FileSystem()
        self._writer: pq.ParquetWriter | None = None
        self._file = None

    # ---------------- public API ----------------

    def write_success(
        self,
        *,
        filename: str,
        source_etag: str,
        page_text: str,
        raw_text: str,
        rep_score: float,
        likely_loop: bool,
        output_tokens: int,
        finish_reason: str,
        truncated: bool,
        res_scale: float = 1.0,
        dry_fires: int = 0,
        dry_max_L: int = 0,
        retried: bool = False,
    ) -> None:
        self._records.append(
            {
                "img_file_name": filename,
                "source_etag": source_etag or "",
                "ok": True,
                "truncated": truncated,
                "finish_reason": finish_reason or "",
                "page_text": page_text or "",
                "raw_text": raw_text or "",
                "rep_score": float(rep_score),
                "likely_loop": bool(likely_loop),
                "output_tokens": int(output_tokens),
                "res_scale": float(res_scale),
                "dry_fires": int(dry_fires),
                "dry_max_L": int(dry_max_L),
                "retried": bool(retried),
                "skipped": False,
                "skip_reason": "",
                "error_stage": "",
                "error_message": "",
                "model_id": self.model_id,
            }
        )
        self._success_count += 1
        if truncated:
            self._truncated_count += 1
        if likely_loop:
            self._loop_count += 1
        self._maybe_flush()

    def write_skipped(
        self,
        *,
        filename: str,
        source_etag: str | None,
        skip_reason: str,
    ) -> None:
        """Record a page that OCR intentionally skipped (not an error)."""
        self._records.append(
            {
                "img_file_name": filename,
                "source_etag": source_etag or "",
                "ok": False,
                "truncated": False,
                "finish_reason": "skipped",
                "page_text": "",
                "raw_text": "",
                "rep_score": 0.0,
                "likely_loop": False,
                "output_tokens": 0,
                "res_scale": 0.0,
                "dry_fires": 0,
                "dry_max_L": 0,
                "retried": False,
                "skipped": True,
                "skip_reason": skip_reason or "",
                "error_stage": "",
                "error_message": "",
                "model_id": self.model_id,
            }
        )
        self._skipped_count += 1
        self._maybe_flush()

    def write_error(
        self,
        *,
        filename: str,
        source_etag: str | None,
        stage: str,
        error_message: str,
    ) -> None:
        short = _short(error_message)
        self._records.append(
            {
                "img_file_name": filename,
                "source_etag": source_etag or "",
                "ok": False,
                "truncated": False,
                "finish_reason": "",
                "page_text": "",
                "raw_text": "",
                "rep_score": 0.0,
                "likely_loop": False,
                "output_tokens": 0,
                "res_scale": 0.0,
                "dry_fires": 0,
                "dry_max_L": 0,
                "retried": False,
                "skipped": False,
                "skip_reason": "",
                "error_stage": stage,
                "error_message": short,
                "model_id": self.model_id,
            }
        )
        self._error_records.append(
            {
                "img_file_name": filename,
                "source_etag": source_etag or "",
                "stage": stage,
                "error_message": short,
            }
        )
        self._error_count += 1
        self._maybe_flush()

    def close(self) -> None:
        self._flush()
        if self._writer is not None:
            self._writer.close()
            self._writer = None
        if self._file is not None:
            self._file.close()
            self._file = None
        if self.errors_jsonl_uri and self._error_records:
            self._write_errors_jsonl()
        logger.info(
            f"[paddleocr.writer] wrote {self._total_written} rows to {self.parquet_uri} "
            f"(success={self._success_count}, errors={self._error_count}, "
            f"skipped={self._skipped_count}, truncated={self._truncated_count}, "
            f"likely_loop={self._loop_count})"
        )

    # ---------------- stats ----------------

    @property
    def success_count(self) -> int:
        return self._success_count

    @property
    def error_count(self) -> int:
        return self._error_count

    @property
    def truncated_count(self) -> int:
        return self._truncated_count

    @property
    def loop_count(self) -> int:
        return self._loop_count

    @property
    def skipped_count(self) -> int:
        return self._skipped_count

    # ---------------- internals ----------------

    def _maybe_flush(self) -> None:
        if len(self._records) >= self.flush_every:
            self._flush()

    def _flush(self) -> None:
        if not self._records:
            return
        if self._writer is None:
            self._open_writer()
        table = pa.Table.from_pylist(self._records, schema=self.schema)
        assert self._writer is not None
        self._writer.write_table(table)
        self._total_written += len(self._records)
        logger.debug(
            f"[paddleocr.writer] flushed {len(self._records)} rows (total={self._total_written})"
        )
        self._records = []

    def _open_writer(self) -> None:
        s3_path = self.parquet_uri.replace("s3://", "")
        self._file = self._s3.open(s3_path, "wb")
        kw = {}
        if self.compression is not None:
            kw["compression"] = self.compression
        self._writer = pq.ParquetWriter(self._file, self.schema, **kw)

    def _write_errors_jsonl(self) -> None:
        buf = io.BytesIO()
        with gzip.GzipFile(fileobj=buf, mode="wb") as gz:
            for rec in self._error_records:
                gz.write(json.dumps(rec, ensure_ascii=False).encode("utf-8"))
                gz.write(b"\n")
        s3_path = self.errors_jsonl_uri.replace("s3://", "")
        with self._s3.open(s3_path, "wb") as f:
            f.write(buf.getvalue())
        logger.info(
            f"[paddleocr.writer] wrote {len(self._error_records)} error rows to "
            f"{self.errors_jsonl_uri}"
        )
