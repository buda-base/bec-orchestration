"""Streaming parquet writer for ``script_classification_v2`` output.

Schema (one row per processed page):

    img_file_name          string          base filename as in the manifest
    source_etag             string          S3 ETag of the source image
    status                  string          "ok" | "error"
    error                    string          message when status="error" (else "")
    exif_orientation_tag    int32 (nullable) raw EXIF tag, read but never applied to pixels
    label                    string          argmax class label
    prob                     float32         probability of the predicted class
    probs                    list<float32>   full softmax vector, checkpoint idx_to_label order
    model_version            string          "<repo-name>:<short-checkpoint-sha>"
    error_stage              string          "" / "fetch" / "classify"
    error_message            string          short error description (truncated to 512 chars)

The ``probs`` vector is the model's full softmax output. To make it
self-describing, the per-class label ordering is stored in the parquet file's
schema metadata under ``labels`` (JSON array; ``probs[i]`` <-> ``labels[i]``),
alongside ``model_version`` and ``model_repo``.

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


def script_classification_v2_build_schema(
    labels: list[str] | None = None,
    model_version: str | None = None,
    model_repo: str | None = None,
) -> pa.Schema:
    metadata: dict[bytes, bytes] = {}
    if labels is not None:
        metadata[b"labels"] = json.dumps(labels).encode("utf-8")
    if model_version is not None:
        metadata[b"model_version"] = model_version.encode("utf-8")
    if model_repo is not None:
        metadata[b"model_repo"] = model_repo.encode("utf-8")
    return pa.schema(
        [
            pa.field("img_file_name", pa.string()),
            pa.field("source_etag", pa.string()),
            pa.field("status", pa.string()),
            pa.field("error", pa.string()),
            pa.field("exif_orientation_tag", pa.int32()),
            pa.field("label", pa.string()),
            pa.field("prob", pa.float32()),
            pa.field("probs", pa.list_(pa.float32())),
            pa.field("model_version", pa.string()),
            pa.field("error_stage", pa.string()),
            pa.field("error_message", pa.string()),
        ],
        metadata=metadata or None,
    )


def _short(msg: str | None) -> str:
    if not msg:
        return ""
    msg = msg.replace("\n", " ").strip()
    return msg if len(msg) <= _MAX_ERROR_MSG_LEN else msg[: _MAX_ERROR_MSG_LEN - 1] + "…"


class StreamingScriptClassificationV2Writer:
    """Streaming parquet writer for classification results, plus optional jsonl errors file."""

    def __init__(
        self,
        parquet_uri: str,
        errors_jsonl_uri: str | None,
        model_version: str,
        *,
        labels: list[str] | None = None,
        model_repo: str | None = None,
        flush_every: int = 256,
        compression: str = "zstd",
    ) -> None:
        self.parquet_uri = parquet_uri
        self.errors_jsonl_uri = errors_jsonl_uri
        self.model_version = model_version
        self.flush_every = flush_every
        self.compression = compression if compression != "none" else None
        self.schema = script_classification_v2_build_schema(
            labels=labels,
            model_version=model_version,
            model_repo=model_repo,
        )

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
        label: str | None,
        prob: float | None,
        probs: list[float] | None,
        model_version: str,
    ) -> None:
        self._records.append(
            {
                "img_file_name": filename,
                "source_etag": source_etag or "",
                "status": "ok",
                "error": "",
                "exif_orientation_tag": exif_orientation_tag,
                "label": label,
                "prob": prob,
                "probs": probs,
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
                "label": None,
                "prob": None,
                "probs": None,
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
            f"[script_classification_v2.writer] wrote {self._total_written} rows to "
            f"{self.parquet_uri} (success={self._success_count}, errors={self._error_count})"
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
            f"[script_classification_v2.writer] wrote {len(self._error_records)} error rows to "
            f"{self.errors_jsonl_uri}"
        )
