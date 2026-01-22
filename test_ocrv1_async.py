#!/usr/bin/env python3
"""
Test script for async OCRV1JobWorker.

Usage:
    BEC_OCR_MODEL_DIR=ocr_models/Woodblock python test_ocrv1_async.py
"""

import gzip
import hashlib
import json
import logging
import os

import s3fs

from bec_orch.core.models import ArtifactLocation, VolumeManifest, VolumeRef
from bec_orch.jobs.base import JobContext

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("test_ocrv1_async")


def get_volume_manifest_from_s3(w_id: str, i_id: str, bucket: str) -> VolumeManifest:
    s3 = s3fs.S3FileSystem()

    w_prefix = hashlib.md5(w_id.encode()).hexdigest()[:2]
    manifest_path = f"{bucket}/Works/{w_prefix}/{w_id}/images/{w_id}-{i_id}/dimensions.json"
    logger.info(f"Fetching manifest from s3://{manifest_path}")

    info = s3.info(manifest_path)
    etag = info.get("ETag", "").strip('"')
    last_modified = info.get("LastModified", "")
    if hasattr(last_modified, "isoformat"):
        last_modified_iso = last_modified.isoformat()
    else:
        last_modified_iso = str(last_modified)

    with s3.open(manifest_path, "rb") as f:
        raw = f.read()
        if raw[:2] == b"\x1f\x8b":
            raw = gzip.decompress(raw)
        manifest_data = json.loads(raw.decode("utf-8"))

    return VolumeManifest(
        manifest=manifest_data,
        s3_etag=etag,
        last_modified_iso=last_modified_iso,
    )


def main():
    w_id = "W1KG4313"
    i_id = "I1KG17496"
    version = "8eba7b"

    ld_bucket = "bec.bdrc.io"
    ocr_dest_bucket = "tests-bec.bdrc.io"
    source_image_bucket = "archive.tbrc.org"

    os.environ.setdefault("BEC_LD_BUCKET", ld_bucket)
    os.environ.setdefault("BEC_SOURCE_IMAGE_BUCKET", source_image_bucket)

    logger.info(f"Testing OCRV1JobWorkerAsync for {w_id}/{i_id} version {version}")
    logger.info(f"LD bucket: {ld_bucket}")
    logger.info(f"OCR dest bucket: {ocr_dest_bucket}")
    logger.info(f"Source image bucket: {source_image_bucket}")

    max_images = 100  # Limit for testing
    manifest = get_volume_manifest_from_s3(w_id, i_id, source_image_bucket)
    logger.info(f"Manifest has {len(manifest.manifest)} images, etag={manifest.s3_etag}")

    # Limit manifest to first N images for testing
    if max_images and len(manifest.manifest) > max_images:
        logger.info(f"Limiting to first {max_images} images for testing")
        manifest = VolumeManifest(
            manifest=manifest.manifest[:max_images],
            s3_etag=manifest.s3_etag,
            last_modified_iso=manifest.last_modified_iso,
        )

    artifacts_location = ArtifactLocation(
        bucket=ocr_dest_bucket,
        prefix=f"ocrv1/{w_id}/{i_id}/{version}",
        basename=f"{w_id}-{i_id}-{version}",
    )

    ctx = JobContext(
        job_id=1,
        volume=VolumeRef(w_id=w_id, i_id=i_id),
        job_name="ocrv1",
        job_config={},
        config_str="{}",
        volume_manifest=manifest,
        artifacts_location=artifacts_location,
    )

    logger.info("Initializing OCRV1JobWorkerAsync...")
    from bec_orch.jobs.ocrv1.worker_async import OCRV1JobWorkerAsync

    worker = OCRV1JobWorkerAsync()

    logger.info("Running async OCR...")
    result = worker.run(ctx)

    logger.info(f"Result: {result}")
    logger.info("Done!")


if __name__ == "__main__":
    main()
