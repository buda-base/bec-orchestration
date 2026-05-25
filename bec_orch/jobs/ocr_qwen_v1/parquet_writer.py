"""Streaming parquet writer for ``ocr_qwen_v1`` output.

Schema (one row per processed page):

    img_file_name   string    base filename as in the manifest (e.g. ``I1CZ350113.tif``)
    source_etag     string    S3 ETag of the source image (for cache invalidation)
    ok              bool      True if OCR produced text (even if truncated)
    truncated       bool      True if vLLM hit ``max_tokens`` for this page
    finish_reason   string    "stop" | "length" | other
    page_text       string    full transcription (empty on error)
    output_tokens   int32     number of output tokens generated
    error_stage     string    "" / "fetch" / "decode" / "ocr"
    error_message   string    short error description (truncated to 512 chars)
    model_id        string    HF id of the model that produced the row

Errors are also optionally mirrored to ``<basename>-errors.jsonl`` via
``write_errors_jsonl=True`` in the config.
"""

from __future__ import annotations

import gzip
import io
import json
import logging
from typing import TYPE_CHECKING

import pyarrow as pa
import pyarrow.parquet as pq
import s3fs

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

_MAX_ERROR_MSG_LEN = 512


def ocr_qwen_v1_build_schema() -> pa.Schema:
    return pa.schema(
        [
            pa.field("img_file_name", pa.string()),
            pa.field("source_etag", pa.string()),
            pa.field("ok", pa.bool_()),
            pa.field("truncated", pa.bool_()),
            pa.field("finish_reason", pa.string()),
            pa.field("page_text", pa.string()),
            pa.field("output_tokens", pa.int32()),
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


class StreamingOCRQwenV1Writer:
    """Streaming parquet writer for OCR results, plus optional jsonl errors file."""

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
        self.schema = ocr_qwen_v1_build_schema()

        self._records: list[dict] = []
        self._error_records: list[dict] = []
        self._total_written = 0
        self._success_count = 0
        self._error_count = 0
        self._truncated_count = 0

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
        output_tokens: int,
        finish_reason: str,
        truncated: bool,
    ) -> None:
        self._records.append(
            {
                "img_file_name": filename,
                "source_etag": source_etag or "",
                "ok": True,
                "truncated": truncated,
                "finish_reason": finish_reason or "",
                "page_text": page_text or "",
                "output_tokens": int(output_tokens),
                "error_stage": "",
                "error_message": "",
                "model_id": self.model_id,
            }
        )
        self._success_count += 1
        if truncated:
            self._truncated_count += 1
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
                "output_tokens": 0,
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
            f"[ocr_qwen_v1.writer] wrote {self._total_written} rows to {self.parquet_uri} "
            f"(success={self._success_count}, errors={self._error_count}, "
            f"truncated={self._truncated_count})"
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
            f"[ocr_qwen_v1.writer] flushed {len(self._records)} rows (total={self._total_written})"
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
            f"[ocr_qwen_v1.writer] wrote {len(self._error_records)} error rows to "
            f"{self.errors_jsonl_uri}"
        )
