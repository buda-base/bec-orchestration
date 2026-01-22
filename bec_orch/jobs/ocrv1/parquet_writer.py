"""
Streaming Parquet writer for OCR results.

Writes records in batches to avoid accumulating everything in memory.
"""

import logging

import pyarrow as pa
import pyarrow.parquet as pq
import s3fs

logger = logging.getLogger(__name__)


def ocr_build_schema() -> pa.Schema:
    """Build the schema for OCR output parquet files."""
    return pa.schema(
        [
            pa.field("img_file_name", pa.string()),
            pa.field("source_etag", pa.string()),
            pa.field("ok", pa.bool_()),
            pa.field("error_message", pa.string()),
            pa.field("texts", pa.list_(pa.string())),
        ]
    )


class StreamingParquetWriter:
    """
    Streaming parquet writer that flushes records in batches.

    This avoids accumulating all records in memory before writing.
    """

    def __init__(self, uri: str, batch_size: int = 100):
        self.uri = uri
        self.batch_size = batch_size
        self.schema = ocr_build_schema()
        self.records: list[dict] = []
        self.total_written = 0

        self._s3 = s3fs.S3FileSystem()
        self._writer: pq.ParquetWriter | None = None
        self._file = None

    def _open_writer(self):
        """Open the parquet writer lazily on first write."""
        s3_path = self.uri.replace("s3://", "")
        self._file = self._s3.open(s3_path, "wb")
        self._writer = pq.ParquetWriter(self._file, self.schema)

    def write_record(
        self,
        filename: str,
        source_etag: str,
        texts: list[str],
        error: str | None = None,
    ) -> None:
        """Write a single OCR result record."""
        record = {
            "img_file_name": filename,
            "source_etag": source_etag if not error else "",
            "ok": error is None,
            "error_message": error or "",
            "texts": texts if not error else [],
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
