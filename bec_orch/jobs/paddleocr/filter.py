"""Pre-filter OCR pages using a sibling classification job's output.

The ``script_classification_v2`` job (or any compatible classifier configured
via ``PaddleOCRConfig.filter_job_name``) writes one parquet per volume with a
per-page ``label`` + ``prob``. We use that to skip OCR on pages that are e.g.
blank, non-Tibetan, or non-plain-text.

The classifier's parquet for a given volume+version lives right next to this
job's output, under the same ``{W}/{I}/{version}/{basename}`` layout — only the
top-level job-name segment differs — so we can locate it from this job's
``ArtifactLocation`` without any extra bookkeeping.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .config import PaddleOCRConfig

logger = logging.getLogger(__name__)


def sibling_parquet_uri(bucket: str, ocr_prefix: str, basename: str, sibling_job: str) -> str:
    """Locate the sibling job's parquet for the same volume+version.

    ``ocr_prefix`` looks like ``paddleocr_v1/<W>/<I>/<version>``; we swap the
    leading job-name segment for ``sibling_job`` and keep the same basename.
    """
    parts = [p for p in ocr_prefix.strip("/").split("/") if p]
    sibling_prefix = "/".join([sibling_job] + parts[1:])
    return f"s3://{bucket}/{sibling_prefix}/{basename}.parquet"


def load_skip_map(
    bucket: str, ocr_prefix: str, basename: str, cfg: PaddleOCRConfig
) -> tuple[dict[str, str], bool]:
    """Build ``{filename: skip_reason}`` from the sibling classifier parquet.

    Returns ``(skip_map, found)``. When the artifact is missing, ``found`` is
    False and ``skip_map`` is empty (the caller decides whether that's fatal
    based on ``cfg.filter_required``).
    """
    import pyarrow.parquet as pq
    import s3fs

    uri = sibling_parquet_uri(bucket, ocr_prefix, basename, cfg.filter_job_name)
    path = uri.replace("s3://", "")
    fs = s3fs.S3FileSystem()

    try:
        exists = fs.exists(path)
    except Exception as e:  # noqa: BLE001 — treat listing errors as "not found"
        logger.warning(f"[paddleocr.filter] could not stat {uri}: {e}")
        return {}, False
    if not exists:
        logger.warning(f"[paddleocr.filter] no classification artifact at {uri}")
        return {}, False

    with fs.open(path, "rb") as f:
        table = pq.read_table(f, columns=["img_file_name", "label", "prob", "status"])

    names = table.column("img_file_name").to_pylist()
    labels = table.column("label").to_pylist()
    probs = table.column("prob").to_pylist()
    statuses = table.column("status").to_pylist()

    skip_labels = set(cfg.filter_skip_labels)
    min_prob = cfg.filter_min_prob
    skip_map: dict[str, str] = {}
    for name, label, prob, status in zip(names, labels, probs, statuses, strict=False):
        if not name or status != "ok":
            # Leave classifier errors to OCR (don't skip on missing signal).
            continue
        if label in skip_labels:
            skip_map[name] = label
        elif min_prob > 0.0 and prob is not None and prob < min_prob:
            skip_map[name] = f"low_prob<{min_prob:g}"

    logger.info(
        f"[paddleocr.filter] {uri}: {len(skip_map)}/{len(names)} pages will be skipped "
        f"(labels={sorted(skip_labels)}, min_prob={min_prob})"
    )
    return skip_map, True
