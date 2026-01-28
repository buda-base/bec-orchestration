"""
Dual output writer for OCR results - Parquet for bulk queries + JSONL.gz for detailed data.

Writes OCR results in two formats:
1. Parquet: Compact page-level data for bulk text queries (streaming)
2. JSONL.gz: Full structured data with lines/segments/syllables (written once at end)
"""

import gzip
import json
import logging
from collections import defaultdict
from typing import Any, Union

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import s3fs

from .data_structures import PageOCRResult, PageResult, PipelineError


def _convert_numpy_types(obj: Any) -> Any:
    """Convert numpy types to native Python types for JSON serialization."""
    if isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: _convert_numpy_types(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_convert_numpy_types(item) for item in obj]
    return obj


logger = logging.getLogger(__name__)


def ocr_build_schema() -> pa.Schema:
    """Build the schema for OCR output parquet files."""
    return pa.schema(
        [
            pa.field("img_file_name", pa.string()),
            pa.field("source_etag", pa.string()),
            pa.field("ok", pa.bool_()),
            pa.field("error_message", pa.string()),
            pa.field("page_text", pa.string()),
            pa.field("page_confidence", pa.float64()),
            pa.field("rotation_angle", pa.float64()),
            pa.field("tps_points", pa.binary()),
        ]
    )


class ParquetWriter:
    """Streaming Parquet writer for compact OCR page data."""

    def __init__(self, uri: str, batch_size: int = 100) -> None:
        self.uri = uri
        self.batch_size = batch_size
        self.schema = ocr_build_schema()
        self.records: list[dict] = []
        self.total_written = 0

        self._s3 = s3fs.S3FileSystem()
        self._writer: pq.ParquetWriter | None = None
        self._file = None

    def _open_writer(self) -> None:
        """Open the parquet writer lazily on first write."""
        s3_path = self.uri.replace("s3://", "")
        self._file = self._s3.open(s3_path, "wb")
        self._writer = pq.ParquetWriter(self._file, self.schema)

    def write_page(self, result: PageOCRResult) -> None:
        """Write a single OCR result page to parquet."""
        record = {
            "img_file_name": result.img_file_name,
            "source_etag": result.source_etag if not result.error else "",
            "ok": result.error is None,
            "error_message": result.error or "",
            "page_text": result.page_text if not result.error else "",
            "page_confidence": result.page_confidence if not result.error else 0.0,
            "rotation_angle": result.rotation_angle,
            "tps_points": json.dumps(result.tps_points).encode() if result.tps_points else b"",
        }

        self.records.append(record)

        if len(self.records) >= self.batch_size:
            self._flush_batch()

    def write_error(self, error: PipelineError) -> None:
        """Write a pipeline error as a failed page record."""
        record = {
            "img_file_name": error.filename,
            "source_etag": error.source_etag or "",
            "ok": False,
            "error_message": f"[{error.stage}] {error.error_type}: {error.message}",
            "page_text": "",
            "page_confidence": 0.0,
            "rotation_angle": 0.0,
            "tps_points": b"",
        }

        self.records.append(record)

        if len(self.records) >= self.batch_size:
            self._flush_batch()

    def _flush_batch(self) -> None:
        """Flush accumulated records to parquet."""
        if not self.records:
            return

        if self._writer is None:
            self._open_writer()

        table = pa.Table.from_pylist(self.records, schema=self.schema)
        if self._writer is not None:
            self._writer.write_table(table)
        self.total_written += len(self.records)
        logger.debug(f"Flushed {len(self.records)} records (total: {self.total_written})")
        self.records = []

    def close(self) -> None:
        """Flush remaining records and close the writer."""
        self._flush_batch()

        if self._writer:
            self._writer.close()
            self._writer = None

        if self._file:
            self._file.close()
            self._file = None

        logger.info(f"Wrote {self.total_written} records to {self.uri}")


class OutputWriter:
    """
    Coordinates writing to both output formats (Parquet + JSONL.gz).

    Parquet is written in streaming batches.
    JSONL.gz is collected in memory and written once at close().
    PipelineErrors are tracked and reported.
    """

    def __init__(self, parquet_uri: str, jsonl_uri: str) -> None:
        self.parquet_uri = parquet_uri
        self.jsonl_uri = jsonl_uri

        self.parquet_writer = ParquetWriter(self.parquet_uri)
        self._s3 = s3fs.S3FileSystem()

        # Collect JSONL records in memory, write at end
        self._jsonl_records: list[dict] = []

        # Error tracking
        self._errors: list[PipelineError] = []
        self._error_count_by_stage: dict[str, int] = defaultdict(int)
        self._success_count = 0

    def write_page(self, result: Union[PageOCRResult, "PageResult"]) -> None:
        """Write page to both output formats."""
        # Convert legacy PageResult to PageOCRResult if needed
        if not hasattr(result, "img_file_name"):
            if isinstance(result, PageResult):
                result = PageOCRResult(
                    img_file_name=result.filename,
                    source_etag=result.source_etag,
                    rotation_angle=0.0,
                    tps_points=None,
                    lines=[],
                    error=result.error,
                )
            else:
                raise TypeError(f"Expected PageResult or PageOCRResult, got {type(result)}")

        # Write to Parquet (streaming)
        self.parquet_writer.write_page(result)

        # Collect JSONL record (write at end)
        self._jsonl_records.append(self._build_jsonl_record(result))

        self._success_count += 1

    def write_error(self, error: PipelineError) -> None:
        """Track and write a pipeline error."""
        self._errors.append(error)
        self._error_count_by_stage[error.stage] += 1

        # Write error to Parquet (streaming)
        self.parquet_writer.write_error(error)

        # Collect error record for JSONL
        self._jsonl_records.append({
            "img_file_name": error.filename,
            "source_etag": error.source_etag,
            "error": f"[{error.stage}] {error.error_type}: {error.message}",
            "lines": [],
        })

    def _build_jsonl_record(self, result: PageOCRResult) -> dict:
        """Build a JSONL record from PageOCRResult."""
        if result.error:
            return {
                "img_file_name": result.img_file_name,
                "source_etag": result.source_etag,
                "rotation_angle": result.rotation_angle,
                "tps_points": result.tps_points,
                "error": result.error,
                "lines": [],
            }

        return {
            "img_file_name": result.img_file_name,
            "source_etag": result.source_etag,
            "rotation_angle": result.rotation_angle,
            "tps_points": result.tps_points,
            "lines": [
                {
                    "line_idx": line.line_idx,
                    "bbox": line.bbox.as_list(),
                    "text": line.text,
                    "confidence": line.confidence,
                    "syllables": [
                        {
                            "px": [s.start_pixel, s.end_pixel],
                            "t": s.text,
                            "c": s.confidence,
                        }
                        for s in line.syllables
                    ],
                }
                for line in result.lines
            ],
        }

    def _write_jsonl(self) -> None:
        """Write all collected JSONL records to S3."""
        if not self._jsonl_records:
            logger.info(f"No records to write to {self.jsonl_uri}")
            return

        s3_path = self.jsonl_uri.replace("s3://", "")
        with self._s3.open(s3_path, "wb") as f:
            with gzip.GzipFile(fileobj=f, mode="wb") as gz:
                for record in self._jsonl_records:
                    json_line = json.dumps(_convert_numpy_types(record), ensure_ascii=False) + "\n"
                    gz.write(json_line.encode("utf-8"))

        logger.info(f"Wrote {len(self._jsonl_records)} records to {self.jsonl_uri}")

    def get_error_stats(self) -> dict:
        """Get error statistics."""
        return {
            "success_count": self._success_count,
            "error_count": len(self._errors),
            "errors_by_stage": dict(self._error_count_by_stage),
        }

    def close(self) -> None:
        """Close Parquet writer and write JSONL."""
        # Close Parquet (flushes remaining records)
        self.parquet_writer.close()

        # Write all JSONL records at once
        self._write_jsonl()

        # Log summary
        if self._errors:
            logger.warning(
                f"Pipeline completed with {len(self._errors)} errors: {dict(self._error_count_by_stage)}"
            )


# Legacy compatibility - keep old class name
StreamingParquetWriter = ParquetWriter
