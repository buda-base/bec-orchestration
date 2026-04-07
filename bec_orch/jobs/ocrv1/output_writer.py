"""
Parquet output writer for OCR results.

Writes OCR results in Parquet format for bulk text queries with streaming writes.
"""

import logging
from collections import defaultdict
from typing import Union

import pyarrow as pa
import pyarrow.parquet as pq
import s3fs

from .data_structures import PageOCRResult, PageResult, PipelineError

logger = logging.getLogger(__name__)


def ocr_build_schema() -> pa.Schema:
    """Build the schema for OCR output parquet files.

    Schema design follows ldv1 conventions:
    - tps_points: list<list<float32>> for TPS transformation points
      Format matches ldv1: [[in_y, in_x, out_y, out_x], ...] - one 4-float list per point
    - line_bboxes: list<struct<x:int16, y:int16, w:int16, h:int16>> for bounding boxes
    - line_texts: list<string> for per-line OCR text (more efficient than page_text + line_texts)
    - nb_lines: int32 count of lines (like ldv1's nb_contours)
    - Error fields: ok, error_stage, error_type, error_message (like ldv1)
    """
    # TPS points: list of [in_y, in_x, out_y, out_x] per point (matches ldv1 format)
    # = list<list<float32>> where inner list has 4 floats
    tps_points_type = pa.list_(pa.list_(pa.float32()))

    # Bounding box struct (x, y, w, h) - use int16 like ldv1 for efficiency
    bbox_struct = pa.struct(
        [
            ("x", pa.int16()),
            ("y", pa.int16()),
            ("w", pa.int16()),
            ("h", pa.int16()),
        ]
    )
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
            pa.field("error_stage", pa.string()),  # null when ok=True
            pa.field("error_type", pa.string()),  # null when ok=True
            pa.field("error_message", pa.string()),  # null when ok=True
        ]
    )


class ParquetWriter:
    """Parquet writer for compact OCR page data.

    Accumulates all records in memory and writes once at close() to avoid
    blocking the async event loop with S3 I/O during processing.
    """

    def __init__(self, uri: str) -> None:
        self.uri = uri
        self.schema = ocr_build_schema()
        self.records: list[dict] = []

    def write_page(self, result: PageOCRResult) -> None:
        """Accumulate a single OCR result page (no I/O until close).

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
                remainder = result.error[bracket_end + 1 :].strip()
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
            # Success record - serialize TPS points to match ldv1 format
            tps_points_list = None
            tps_alpha = None
            if result.tps_points:
                # result.tps_points is ((input_pts, output_pts), alpha)
                # input_pts and output_pts are lists of [x, y] pairs
                (input_pts, output_pts), alpha = result.tps_points
                # Convert to ldv1 format: [[in_y, in_x, out_y, out_x], ...]
                # Note: ldv1 stores as [y, x] order, not [x, y]
                tps_points_list = []
                n = min(len(input_pts), len(output_pts))
                for i in range(n):
                    # input_pts[i] is [x, y], we need [y, x, y, x] format
                    ix, iy = float(input_pts[i][0]), float(input_pts[i][1])
                    ox, oy = float(output_pts[i][0]), float(output_pts[i][1])
                    tps_points_list.append([iy, ix, oy, ox])
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

    def write_error(self, error: PipelineError) -> None:
        """Accumulate a pipeline error as a failed page record (no I/O until close)."""
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

    def close(self) -> None:
        """Write all accumulated records to S3 as a single Parquet file."""
        if not self.records:
            logger.warning(f"No records to write to {self.uri}")
            return

        # Build table from all accumulated records
        table = pa.Table.from_pylist(self.records, schema=self.schema)

        # Write to S3 in one operation
        s3 = s3fs.S3FileSystem()
        s3_path = self.uri.replace("s3://", "")

        with s3.open(s3_path, "wb") as f:
            pq.write_table(table, f)

        logger.info(f"Wrote {len(self.records)} records to {self.uri}")


class OutputWriter:
    """
    Coordinates writing to Parquet output format.

    Accumulates all records in memory during processing, then writes
    a single Parquet file to S3 when close() is called. This avoids
    blocking the async event loop with S3 I/O during processing.
    """

    def __init__(self, parquet_uri: str) -> None:
        self.parquet_uri = parquet_uri
        self.parquet_writer = ParquetWriter(self.parquet_uri)

        # Error tracking
        self._errors: list[PipelineError] = []
        self._error_count_by_stage: dict[str, int] = defaultdict(int)
        self._success_count = 0

    def write_page(self, result: Union[PageOCRResult, "PageResult"]) -> None:
        """Accumulate page for Parquet output (no I/O until close)."""
        # Convert legacy PageResult to PageOCRResult if needed
        if isinstance(result, PageResult):
            ocr_result = PageOCRResult(
                img_file_name=result.filename,
                source_etag=result.source_etag,
                rotation_angle=0.0,
                tps_points=None,
                lines=[],
                error=result.error,
            )
        else:
            ocr_result = result

        # Accumulate record (no I/O)
        self.parquet_writer.write_page(ocr_result)
        self._success_count += 1

    def write_error(self, error: PipelineError) -> None:
        """Accumulate pipeline error (no I/O until close)."""
        self._errors.append(error)
        self._error_count_by_stage[error.stage] += 1

        # Accumulate error record (no I/O)
        self.parquet_writer.write_error(error)

    def close(self) -> None:
        """Write all accumulated records to S3 and log summary."""
        self.parquet_writer.close()

        # Log summary
        if self._errors:
            logger.warning(f"Pipeline completed with {len(self._errors)} errors: {dict(self._error_count_by_stage)}")
        else:
            logger.info(f"Pipeline completed successfully with {self._success_count} pages processed")
