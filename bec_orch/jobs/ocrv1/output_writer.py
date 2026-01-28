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
    """Build the schema for OCR output parquet files.
    
    Schema design follows ldv1 conventions:
    - tps_points: list<list<float32>> for TPS transformation points
    - line_bboxes: list<struct<x:int16, y:int16, w:int16, h:int16>> for bounding boxes
    - line_texts: list<string> for per-line OCR text (more efficient than page_text + line_texts)
    - nb_lines: int32 count of lines (like ldv1's nb_contours)
    - Error fields: ok, error_stage, error_type, error_message (like ldv1)
    """
    # TPS points: list of [input_points, output_points] where each is list of [x, y] pairs
    tps_points_type = pa.list_(pa.list_(pa.float32()))
    
    # Bounding box struct (x, y, w, h) - use int16 like ldv1 for efficiency
    bbox_struct = pa.struct([
        ("x", pa.int16()),
        ("y", pa.int16()),
        ("w", pa.int16()),
        ("h", pa.int16()),
    ])
    line_bboxes_type = pa.list_(bbox_struct)
    
    # Line texts as list of strings (native parquet array, more efficient than JSON in binary)
    line_texts_type = pa.list_(pa.string())
    
    return pa.schema(
        [
            pa.field("img_file_name", pa.string()),
            pa.field("source_etag", pa.string()),
            pa.field("rotation_angle", pa.float32()),
            pa.field("tps_points", tps_points_type),
            pa.field("tps_alpha", pa.float32()),
            pa.field("line_bboxes", line_bboxes_type),
            pa.field("line_texts", line_texts_type),
            pa.field("nb_lines", pa.int32()),
            # Error fields (like ldv1)
            pa.field("ok", pa.bool_()),
            pa.field("error_stage", pa.string()),    # null when ok=True
            pa.field("error_type", pa.string()),     # null when ok=True
            pa.field("error_message", pa.string()),  # null when ok=True
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
        """Write a single OCR result page to parquet.
        
        Writes line-level data in native parquet types following ldv1 conventions.
        """
        if result.error:
            # Error record - extract stage/type from error message if formatted as "[Stage] Type: Message"
            error_stage = ""
            error_type = ""
            error_message = result.error
            
            # Try to parse structured error format: "[Stage] Type: Message"
            if result.error.startswith("[") and "]" in result.error:
                bracket_end = result.error.index("]")
                error_stage = result.error[1:bracket_end]
                remainder = result.error[bracket_end + 1:].strip()
                if ":" in remainder:
                    error_type, error_message = remainder.split(":", 1)
                    error_type = error_type.strip()
                    error_message = error_message.strip()
                else:
                    error_message = remainder
            
            record = {
                "img_file_name": result.img_file_name,
                "source_etag": result.source_etag,
                "rotation_angle": float(result.rotation_angle),
                "tps_points": None,
                "tps_alpha": None,
                "line_bboxes": [],
                "line_texts": [],
                "nb_lines": 0,
                "ok": False,
                "error_stage": error_stage,
                "error_type": error_type,
                "error_message": error_message,
            }
        else:
            # Success record - parse TPS points to match ldv1 format
            tps_points_list = None
            tps_alpha = None
            if result.tps_points:
                # result.tps_points is ((input_pts, output_pts), alpha)
                (input_pts, output_pts), alpha = result.tps_points
                # Convert to list of lists of floats
                tps_points_list = [input_pts, output_pts]
                tps_alpha = float(alpha) if alpha is not None else None
            
            # Build line bboxes as list of dicts matching the struct schema
            line_bboxes = [
                {
                    "x": int(line.bbox.x),
                    "y": int(line.bbox.y),
                    "w": int(line.bbox.w),
                    "h": int(line.bbox.h),
                }
                for line in result.lines
            ]
            
            # Build line texts as simple list of strings
            line_texts = [line.text for line in result.lines]
            
            record = {
                "img_file_name": result.img_file_name,
                "source_etag": result.source_etag,
                "rotation_angle": float(result.rotation_angle),
                "tps_points": tps_points_list,
                "tps_alpha": tps_alpha,
                "line_bboxes": line_bboxes,
                "line_texts": line_texts,
                "nb_lines": len(result.lines),
                "ok": True,
                "error_stage": "",
                "error_type": "",
                "error_message": "",
            }

        self.records.append(record)

        if len(self.records) >= self.batch_size:
            self._flush_batch()

    def write_error(self, error: PipelineError) -> None:
        """Write a pipeline error as a failed page record."""
        record = {
            "img_file_name": error.filename,
            "source_etag": error.source_etag or "",
            "rotation_angle": 0.0,
            "tps_points": None,
            "tps_alpha": None,
            "line_bboxes": [],
            "line_texts": [],
            "nb_lines": 0,
            "ok": False,
            "error_stage": error.stage,
            "error_type": error.error_type,
            "error_message": error.message,
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

    def __init__(self, parquet_uri: str, jsonl_uri: str | None = None) -> None:
        self.parquet_uri = parquet_uri
        self.jsonl_uri = jsonl_uri

        self.parquet_writer = ParquetWriter(self.parquet_uri)
        self._s3 = s3fs.S3FileSystem() if jsonl_uri else None

        # Collect JSONL records in memory, write at end (only if enabled)
        self._jsonl_records: list[dict] = [] if jsonl_uri else []

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

        # Collect JSONL record (write at end) - only if jsonl_uri is set
        if self.jsonl_uri:
            self._jsonl_records.append(self._build_jsonl_record(result))

        self._success_count += 1

    def write_error(self, error: PipelineError) -> None:
        """Track and write a pipeline error."""
        self._errors.append(error)
        self._error_count_by_stage[error.stage] += 1

        # Write error to Parquet (streaming)
        self.parquet_writer.write_error(error)

        # Collect error record for JSONL (only if enabled)
        if self.jsonl_uri:
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
        if not self.jsonl_uri:
            return  # JSONL output is disabled
        
        if not self._jsonl_records:
            logger.info(f"No records to write to {self.jsonl_uri}")
            return

        s3_path = self.jsonl_uri.replace("s3://", "")
        if self._s3 is None:
            logger.warning("S3 filesystem not initialized for JSONL writing")
            return
            
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
