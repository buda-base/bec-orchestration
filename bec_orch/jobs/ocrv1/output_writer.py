"""
Dual output writer for OCR results - Parquet for bulk queries + JSONL.gz for detailed data.

Writes OCR results in two formats:
1. Parquet: Compact page-level data for bulk text queries
2. JSONL.gz: Full structured data with lines/segments/syllables
"""

import gzip
import json
import logging
from typing import Any, Union

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import s3fs

from .data_structures import PageOCRResult, PageResult


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


class GzipJsonlWriter:
    """Streaming gzipped JSONL writer for structured page data."""

    def __init__(self, uri: str) -> None:
        self.uri = uri
        self._s3 = s3fs.S3FileSystem()
        self._file = None
        self._gzip = None
        self.total_written = 0

    def _open_writer(self) -> None:
        """Open the gzip writer lazily on first write."""
        s3_path = self.uri.replace("s3://", "")
        self._file = self._s3.open(s3_path, "wb")
        self._gzip = gzip.GzipFile(fileobj=self._file, mode="wb")

    def write_page(self, result: PageOCRResult) -> None:
        """Write one page with all lines/syllables as a single JSON line."""
        if self._gzip is None:
            self._open_writer()

        if result.error:
            # Write error record with minimal structure
            record = {
                "img_file_name": result.img_file_name,
                "source_etag": result.source_etag,
                "rotation_angle": result.rotation_angle,
                "tps_points": result.tps_points,
                "error": result.error,
                "lines": [],
            }
        else:
            record = {
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

        json_line = json.dumps(_convert_numpy_types(record), ensure_ascii=False) + "\n"
        self._gzip.write(json_line.encode("utf-8"))  # type: ignore[arg-type]
        self.total_written += 1

    def close(self) -> None:
        """Close the gzip writer."""
        if self._gzip:
            self._gzip.close()
            self._gzip = None

        if self._file:
            self._file.close()
            self._file = None

        logger.info(f"Wrote {self.total_written} records to {self.uri}")


class OutputWriter:
    """Coordinates writing to both output formats (Parquet + JSONL.gz)."""

    def __init__(self, volume_id: str, output_prefix: str) -> None:
        self.parquet_writer = ParquetWriter(f"{output_prefix}/{volume_id}_ocr.parquet")
        self.jsonl_writer = GzipJsonlWriter(f"{output_prefix}/{volume_id}_ocr.jsonl.gz")

    def write_page(self, result: Union[PageOCRResult, "PageResult"]) -> None:
        """Write page to both output formats. Handles both PageResult (legacy) and PageOCRResult (new)."""
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

        # Write to both formats
        self.parquet_writer.write_page(result)  # type: ignore[arg-type]
        self.jsonl_writer.write_page(result)  # type: ignore[arg-type]

    def close(self) -> None:
        """Close both writers."""
        self.parquet_writer.close()
        self.jsonl_writer.close()


# Legacy compatibility - keep old class name
StreamingParquetWriter = ParquetWriter
