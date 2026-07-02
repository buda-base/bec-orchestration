"""Streaming parquet writer for ``script_classification`` output.

Schema (one row per processed page):

    img_file_name          string          base filename as in the manifest
    source_etag             string          S3 ETag of the source image
    status                  string          "ok" | "error"
    error                    string          message when status="error" (else "")
    exif_orientation_tag    int32 (nullable) raw EXIF tag, read but never applied to pixels
    orientation_pred         string          "non_flipped" | "flipped"
    orientation_prob         float32         probability of the predicted orientation class
    rotation_applied         int32           degrees actually applied before 6-class scoring: 0 or 180
    sixclass_label           string          argmax script class
    sixclass_probs           list<float32>   full probability vector, checkpoint's idx_to_label order
    final_label              string          equal to sixclass_label
    model_version            string          short checkpoint hashes for both models
    error_stage              string          "" / "fetch" / "classify"
    error_message            string          short error description (truncated to 512 chars)

No ``blank`` column: the upstream blank-page pre-filter is disabled (see
``vendor/pipeline.py``), so every image is scored — a permanently-constant
column would misleadingly look like a real filter result. No
``flip_applied`` column either: it's a 1:1 boolean encoding of
``rotation_applied`` already in this schema.

Errors are also optionally mirrored to ``<basename>-errors.jsonl`` via
``write_errors_jsonl=True`` in the config.
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


def script_classification_build_schema() -> pa.Schema:
    return pa.schema(
        [
            pa.field("img_file_name", pa.string()),
            pa.field("source_etag", pa.string()),
            pa.field("status", pa.string()),
            pa.field("error", pa.string()),
            pa.field("exif_orientation_tag", pa.int32()),
            pa.field("orientation_pred", pa.string()),
            pa.field("orientation_prob", pa.float32()),
            pa.field("rotation_applied", pa.int32()),
            pa.field("sixclass_label", pa.string()),
            pa.field("sixclass_probs", pa.list_(pa.float32())),
            pa.field("final_label", pa.string()),
            pa.field("model_version", pa.string()),
            pa.field("error_stage", pa.string()),
            pa.field("error_message", pa.string()),
        ]
    )


def _short(msg: str | None) -> str:
    if not msg:
        return ""
    msg = msg.replace("\n", " ").strip()
    return msg if len(msg) <= _MAX_ERROR_MSG_LEN else msg[: _MAX_ERROR_MSG_LEN - 1] + "…"


class StreamingScriptClassificationWriter:
    """Streaming parquet writer for classification results, plus optional jsonl errors file."""

    def __init__(
        self,
        parquet_uri: str,
        errors_jsonl_uri: str | None,
        model_version: str,
        *,
        flush_every: int = 256,
        compression: str = "zstd",
    ) -> None:
        self.parquet_uri = parquet_uri
        self.errors_jsonl_uri = errors_jsonl_uri
        self.model_version = model_version
        self.flush_every = flush_every
        self.compression = compression if compression != "none" else None
        self.schema = script_classification_build_schema()

        self._records: list[dict] = []
        self._error_records: list[dict] = []
        self._total_written = 0
        self._success_count = 0
        self._error_count = 0

        self._s3 = s3fs.S3FileSystem()
        self._writer: pq.ParquetWriter | None = None
        self._file = None

    # ---------------- public API ----------------

    def write_success(
        self,
        *,
        filename: str,
        source_etag: str,
        exif_orientation_tag: int | None,
        orientation_pred: str | None,
        orientation_prob: float | None,
        rotation_applied: int | None,
        sixclass_label: str | None,
        sixclass_probs: list[float] | None,
        final_label: str | None,
        model_version: str,
    ) -> None:
        self._records.append(
            {
                "img_file_name": filename,
                "source_etag": source_etag or "",
                "status": "ok",
                "error": "",
                "exif_orientation_tag": exif_orientation_tag,
                "orientation_pred": orientation_pred,
                "orientation_prob": orientation_prob,
                "rotation_applied": rotation_applied,
                "sixclass_label": sixclass_label,
                "sixclass_probs": sixclass_probs,
                "final_label": final_label,
                "model_version": model_version,
                "error_stage": "",
                "error_message": "",
            }
        )
        self._success_count += 1
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
                "status": "error",
                "error": short,
                "exif_orientation_tag": None,
                "orientation_pred": None,
                "orientation_prob": None,
                "rotation_applied": None,
                "sixclass_label": None,
                "sixclass_probs": None,
                "final_label": None,
                "model_version": self.model_version,
                "error_stage": stage,
                "error_message": short,
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
            f"[script_classification.writer] wrote {self._total_written} rows to {self.parquet_uri} "
            f"(success={self._success_count}, errors={self._error_count})"
        )

    # ---------------- stats ----------------

    @property
    def success_count(self) -> int:
        return self._success_count

    @property
    def error_count(self) -> int:
        return self._error_count

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
            f"[script_classification.writer] flushed {len(self._records)} rows (total={self._total_written})"
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
            f"[script_classification.writer] wrote {len(self._error_records)} error rows to "
            f"{self.errors_jsonl_uri}"
        )
