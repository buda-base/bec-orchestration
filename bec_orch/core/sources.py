"""Image-source routing for BEC volumes.

A "source" is a logical origin for a volume's images. Each source knows:

- which S3 bucket + key prefix the page images live under,
- how to discover the page list (a ``dimensions.json`` manifest vs. listing a
  flat folder),
- how the volume should be keyed in the ``volumes`` table (namespacing so a
  non-BDRC volume never collides with a real BDRC one), and
- how to derive the artifact version + idempotency etag when there is no
  ``dimensions.json`` etag to lean on.

This is the single place that maps ``VolumeRef`` -> concrete S3 locations, so
the worker runtime and every job worker can stay source-agnostic.

Adding a new source = add one branch/entry here; nothing else needs to know the
path convention.
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from typing import Optional

from bec_orch.core.models import VolumeRef

# Canonical source identifiers (stored lowercase everywhere).
SOURCE_BDRC = "bdrc"
SOURCE_OCR_BENCHMARK = "ocr_benchmark"

KNOWN_SOURCES = frozenset({SOURCE_BDRC, SOURCE_OCR_BENCHMARK})

# Manifest discovery strategies.
MANIFEST_DIMENSIONS_JSON = "dimensions_json"  # gzip JSON at {prefix}dimensions.json
MANIFEST_LIST_OBJECTS = "list_objects"  # list image files under {prefix}


class UnknownSourceError(ValueError):
    """Raised when a message references a source we don't know how to route."""


class MissingSourceFieldError(ValueError):
    """Raised when a source needs a field (e.g. i_version) that wasn't provided."""


@dataclass(frozen=True)
class ResolvedVolume:
    """Fully-resolved, network-free routing info for a volume."""

    source: str
    # Page images.
    image_bucket: str
    image_prefix: str  # object key = image_prefix + filename
    # DB identity (namespaced so sources never collide in the volumes table).
    db_w_id: str
    db_i_id: str
    # Manifest discovery.
    manifest_mode: str
    manifest_bucket: str
    manifest_prefix: str  # dimensions.json lives at manifest_prefix + "dimensions.json"
    # Versioning overrides for sources without a real dimensions.json etag.
    # When set, forced_version drives the artifact path segment and
    # forced_etag_hex drives the DB idempotency key.
    forced_version: Optional[str] = None
    forced_etag_hex: Optional[str] = None

    @property
    def manifest_key(self) -> str:
        return f"{self.manifest_prefix}dimensions.json"


def normalize_source(source: Optional[str]) -> str:
    """Return the canonical lowercase source id. ``None``/empty -> BDRC."""
    if not source:
        return SOURCE_BDRC
    s = source.strip().lower()
    if s not in KNOWN_SOURCES:
        raise UnknownSourceError(
            f"unknown source '{source}' (known: {', '.join(sorted(KNOWN_SOURCES))})"
        )
    return s


def bdrc_folder_prefix(w_id: str, i_id: str) -> str:
    """Canonical BDRC layout: Works/{md5(w_id)[:2]}/{w_id}/images/{w_id}-{suffix}/."""
    md5_hash = hashlib.md5(w_id.encode()).hexdigest()[:2]
    if i_id.startswith("I") and i_id[1:].isdigit() and len(i_id) == 5:
        suffix = i_id[1:]
    else:
        suffix = i_id
    return f"Works/{md5_hash}/{w_id}/images/{w_id}-{suffix}/"


def _ocr_benchmark_bucket() -> str:
    return os.environ.get("BEC_OCR_BENCHMARK_S3_BUCKET", "bec.bdrc.io")


def _synthetic_etag_hex(*parts: str) -> str:
    """Stable 32-char hex (16-byte MD5) used as a synthetic manifest etag."""
    return hashlib.md5("|".join(parts).encode()).hexdigest()


def resolve_volume(volume: VolumeRef, default_bucket: str) -> ResolvedVolume:
    """Resolve a ``VolumeRef`` to concrete S3 routing info.

    Args:
        volume: the volume to route (its ``source``/``i_version`` drive routing).
        default_bucket: bucket to use for the BDRC source (the runtime's
            configured source bucket), keeping manifest + image fetch unified.

    Raises:
        UnknownSourceError: unrecognized source.
        MissingSourceFieldError: a required field for the source is absent.
    """
    source = normalize_source(volume.source)

    if source == SOURCE_BDRC:
        prefix = bdrc_folder_prefix(volume.w_id, volume.i_id)
        return ResolvedVolume(
            source=source,
            image_bucket=default_bucket,
            image_prefix=prefix,
            db_w_id=volume.w_id,
            db_i_id=volume.i_id,
            manifest_mode=MANIFEST_DIMENSIONS_JSON,
            manifest_bucket=default_bucket,
            manifest_prefix=prefix,
        )

    if source == SOURCE_OCR_BENCHMARK:
        if not volume.i_version:
            raise MissingSourceFieldError(
                f"source '{source}' requires i_version (volume {volume.w_id}/{volume.i_id})"
            )
        bucket = _ocr_benchmark_bucket()
        prefix = f"ocr_benchmark/images/{volume.w_id}/{volume.i_id}/{volume.i_version}/"
        return ResolvedVolume(
            source=source,
            image_bucket=bucket,
            image_prefix=prefix,
            # Namespace the DB w_id so benchmark volumes never collide with the
            # real BDRC volume of the same id (chosen over a schema change).
            db_w_id=f"{source}:{volume.w_id}",
            db_i_id=volume.i_id,
            manifest_mode=MANIFEST_LIST_OBJECTS,
            manifest_bucket=bucket,
            manifest_prefix=prefix,
            forced_version=volume.i_version,
            forced_etag_hex=_synthetic_etag_hex(
                source, volume.w_id, volume.i_id, volume.i_version
            ),
        )

    # normalize_source already guards this, but keep it total.
    raise UnknownSourceError(f"unhandled source '{source}'")
