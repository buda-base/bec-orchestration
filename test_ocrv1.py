#!/usr/bin/env python3
"""
Test script for OCRV1JobWorker.

Usage:
    export BEC_OCR_MODEL_DIR=/path/to/ocr_model_directory
    python test_ocrv1.py

The model directory must contain model_config.json with the model configuration.
"""
import gzip
import json
import logging
import os
from pathlib import Path

import s3fs

from bec_orch.core.models import ArtifactLocation, VolumeManifest, VolumeRef
from bec_orch.core.worker_runtime import get_s3_folder_prefix
from bec_orch.jobs.base import JobContext

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("test_ocrv1")


def get_volume_manifest_from_s3(w_id: str, i_id: str, source_bucket: str) -> VolumeManifest:
    """Fetch volume manifest from S3."""
    s3 = s3fs.S3FileSystem()
    prefix = get_s3_folder_prefix(w_id, i_id)
    manifest_key = f"{source_bucket}/{prefix}dimensions.json"

    logger.info(f"Fetching manifest from s3://{manifest_key}")

    with s3.open(manifest_key, "rb") as f:
        body_bytes = f.read()

    uncompressed = gzip.decompress(body_bytes)
    data = json.loads(uncompressed.decode("utf-8"))

    image_extensions = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".jp2"}
    manifest = []
    for item in data:
        filename = item.get("filename")
        if not filename:
            continue
        if "." not in filename or "/" in filename:
            continue
        ext = Path(filename).suffix.lower()
        if ext in image_extensions:
            manifest.append(item)

    info = s3.info(manifest_key)
    etag = info.get("ETag", "").strip('"')
    last_modified = info.get("LastModified", "")

    return VolumeManifest(
        manifest=manifest,
        s3_etag=etag,
        last_modified_iso=str(last_modified),
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

    logger.info(f"Testing OCRV1JobWorker for {w_id}/{i_id} version {version}")
    logger.info(f"LD bucket: {ld_bucket}")
    logger.info(f"OCR dest bucket: {ocr_dest_bucket}")
    logger.info(f"Source image bucket: {source_image_bucket}")

    manifest = get_volume_manifest_from_s3(w_id, i_id, source_image_bucket)
    logger.info(f"Manifest has {len(manifest.manifest)} images, etag={manifest.s3_etag}")

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

    from bec_orch.jobs.ocrv1.worker import OCRV1JobWorker

    logger.info("Initializing OCRV1JobWorker...")
    worker = OCRV1JobWorker()

    logger.info("Running OCR...")
    result = worker.run(ctx)

    logger.info(f"Result: {result}")
    logger.info("Done!")


if __name__ == "__main__":
    main()
