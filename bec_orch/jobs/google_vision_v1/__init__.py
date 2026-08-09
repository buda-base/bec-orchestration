"""``google_vision_v1`` job package.

Runs Google Cloud Vision ``DOCUMENT_TEXT_DETECTION`` (async batch) over each
BDRC volume. No GPU: the worker stream-copies page images from S3 to GCS
staging, submits async Vision batches, waits for them in memory (no external
state DB), and writes a per-volume ``-gv.parquet`` + raw ``-gv.jsonl.zst`` to
the destination S3 bucket. See ``README.md`` for details.
"""

from .config import GoogleVisionV1Config
from .worker import GoogleVisionV1JobWorker

__all__ = ["GoogleVisionV1Config", "GoogleVisionV1JobWorker"]
