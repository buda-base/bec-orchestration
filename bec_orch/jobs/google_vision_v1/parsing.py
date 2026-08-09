"""Vision-response parsing + per-volume output writing for ``google_vision_v1``.

The parquet schema and JSON parsing intentionally match the standalone
``export_volume_ocr.py`` so the artifacts this job writes are byte-compatible
with the existing google-vision exports:

    img_file_name  string        base filename (as in the manifest)
    img_idx        int16         0-based page index (manifest order)
    source_etag    string        S3 ETag of the source image
    nb_pages       int16         pages in the Vision response (multi-page TIFF)
    languages      list<string>  detected language codes, best-confidence first
    confidence     float16       median word confidence on the first page
    text_len       int32         len(full text) in characters
    nb_lines_tib   int16         lines containing >=1 Tibetan char (U+0F00-0FFF)
    text           string        full page text

A raw sidecar ``<basename>-gv.jsonl.zst`` stores one
``{"img_file_name", "img_idx", "response"}`` object per line (zstd level 1).
"""

from __future__ import annotations

import io
import json
import logging
from statistics import median
from typing import Any

logger = logging.getLogger(__name__)


def _pa():  # lazy import; pyarrow is a job-only dep
    import pyarrow as pa

    return pa


def build_schema() -> Any:
    pa = _pa()
    return pa.schema(
        [
            ("img_file_name", pa.string()),
            ("img_idx", pa.int16()),
            ("source_etag", pa.string()),
            ("nb_pages", pa.int16()),
            ("languages", pa.list_(pa.string())),
            ("confidence", pa.float16()),
            ("text_len", pa.int32()),
            ("nb_lines_tib", pa.int16()),
            ("text", pa.string()),
        ]
    )


# ---------------------------------------------------------------------------
# Response field extraction (identical semantics to export_volume_ocr.py)
# ---------------------------------------------------------------------------

def _extract_languages(response: dict) -> list[str]:
    pages = response.get("fullTextAnnotation", {}).get("pages", [])
    if not pages:
        return []
    page = pages[0]
    langs = page.get("property", {}).get("detectedLanguages", [])
    if not langs:
        lang_scores: dict[str, float] = {}
        for block in page.get("blocks", []):
            for lang in block.get("property", {}).get("detectedLanguages", []):
                code = lang.get("languageCode", "")
                if code:
                    lang_scores[code] = max(lang_scores.get(code, 0.0), lang.get("confidence", 0.0))
            for para in block.get("paragraphs", []):
                for lang in para.get("property", {}).get("detectedLanguages", []):
                    code = lang.get("languageCode", "")
                    if code:
                        lang_scores[code] = max(lang_scores.get(code, 0.0), lang.get("confidence", 0.0))
        return [code for code, _ in sorted(lang_scores.items(), key=lambda x: -x[1])]
    return [l["languageCode"] for l in sorted(langs, key=lambda x: -x.get("confidence", 0.0))]


def _extract_word_confidences(response: dict) -> list[float]:
    pages = response.get("fullTextAnnotation", {}).get("pages", [])
    if not pages:
        return []
    confs: list[float] = []
    for block in pages[0].get("blocks", []):
        for para in block.get("paragraphs", []):
            for word in para.get("words", []):
                confs.append(float(word.get("confidence") or 0.0))
    return confs


def _count_tibetan_lines(text: str) -> int:
    if not text:
        return 0
    return sum(1 for line in text.split("\n") if any("\u0f00" <= ch <= "\u0fff" for ch in line))


def parse_response(img_file_name: str, img_idx: int, source_etag: str, response: dict) -> dict:
    """Turn one Vision API response into a parquet row dict."""
    text = response.get("fullTextAnnotation", {}).get("text", "")
    confs = _extract_word_confidences(response)
    nb_pages = len(response.get("fullTextAnnotation", {}).get("pages", []))
    return {
        "img_file_name": img_file_name,
        "img_idx": img_idx,
        "source_etag": source_etag or "",
        "nb_pages": nb_pages,
        "languages": _extract_languages(response),
        "confidence": float(median(confs)) if confs else 0.0,
        "text_len": len(text),
        "nb_lines_tib": _count_tibetan_lines(text),
        "text": text,
    }


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------

def rows_to_parquet_bytes(rows: list[dict], compression: str) -> bytes:
    import pyarrow.parquet as pq

    pa = _pa()
    schema = build_schema()
    columns: dict[str, list] = {f.name: [] for f in schema}
    for row in rows:
        for f in schema:
            columns[f.name].append(row.get(f.name))
    table = pa.table(columns, schema=schema)
    buf = io.BytesIO()
    codec = None if compression == "none" else compression
    pq.write_table(table, buf, compression=codec)
    return buf.getvalue()


def responses_to_jsonl_zst_bytes(records: list[dict], level: int) -> bytes:
    """``records`` are ``{"img_file_name", "img_idx", "response"}`` dicts."""
    import zstandard as zstd

    raw = "\n".join(json.dumps(rec, ensure_ascii=False) for rec in records).encode("utf-8")
    return zstd.ZstdCompressor(level=level).compress(raw)
