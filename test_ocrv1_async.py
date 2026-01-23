#!/usr/bin/env python3
"""
Test script for async OCRV1JobWorker.

Usage:
    BEC_OCR_MODEL_DIR=ocr_models/Woodblock python test_ocrv1_async.py

To generate a reference parquet with max accuracy settings:
    BEC_OCR_MODEL_DIR=ocr_models/Woodblock python test_ocrv1_async.py --reference
"""

import argparse
import gzip
import hashlib
import json
import logging
import os
import shutil
from difflib import SequenceMatcher
from pathlib import Path

import pandas as pd
import s3fs

from bec_orch.core.models import ArtifactLocation, VolumeManifest, VolumeRef
from bec_orch.jobs.base import JobContext

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("test_ocrv1_async")

# Path to reference parquet for accuracy comparison
REFERENCE_PARQUET_PATH = Path("test/reference_output.parquet")


def compare_with_reference(current_parquet_path: str) -> float | None:
    """
    Compare current OCR output with reference parquet.

    Returns character-level accuracy as a percentage (0-100),
    or None if reference file doesn't exist.
    """
    if not REFERENCE_PARQUET_PATH.exists():
        logger.warning(
            f"Reference parquet not found at {REFERENCE_PARQUET_PATH}. Run with --reference first to generate it."
        )
        return None

    try:
        reference_df = pd.read_parquet(REFERENCE_PARQUET_PATH)
        current_df = pd.read_parquet(current_parquet_path)
    except Exception as e:
        logger.error(f"Failed to load parquet files for comparison: {e}")
        return None

    # Match by img_file_name
    ref_by_file = {row.img_file_name: row for row in reference_df.itertuples()}
    cur_by_file = {row.img_file_name: row for row in current_df.itertuples()}

    total_chars = 0
    matching_chars = 0
    pages_compared = 0
    pages_with_diff = 0

    for filename in ref_by_file:
        if filename not in cur_by_file:
            continue

        ref_row = ref_by_file[filename]
        cur_row = cur_by_file[filename]

        # Get text content (handle list/array of lines or single string)
        ref_texts = ref_row.texts if hasattr(ref_row, "texts") else []
        cur_texts = cur_row.texts if hasattr(cur_row, "texts") else []

        # Convert to list if numpy array
        if hasattr(ref_texts, "tolist"):
            ref_texts = ref_texts.tolist()
        if hasattr(cur_texts, "tolist"):
            cur_texts = cur_texts.tolist()

        # Join all lines into single string for comparison
        if isinstance(ref_texts, list):
            ref_text = "\n".join(str(t) for t in ref_texts if t)
        else:
            ref_text = str(ref_texts) if ref_texts else ""

        if isinstance(cur_texts, list):
            cur_text = "\n".join(str(t) for t in cur_texts if t)
        else:
            cur_text = str(cur_texts) if cur_texts else ""

        if not ref_text:
            continue

        # Character-level similarity using SequenceMatcher
        matcher = SequenceMatcher(None, ref_text, cur_text)
        ratio = matcher.ratio()

        total_chars += len(ref_text)
        matching_chars += int(len(ref_text) * ratio)
        pages_compared += 1

        if ratio < 1.0:
            pages_with_diff += 1

    if total_chars == 0:
        logger.warning("No characters to compare")
        return None

    accuracy = (matching_chars / total_chars) * 100

    logger.info("=== ACCURACY COMPARISON ===")
    logger.info(f"Pages compared: {pages_compared}")
    logger.info(f"Pages with differences: {pages_with_diff}")
    logger.info(f"Total characters: {total_chars}")
    logger.info(f"Matching characters: {matching_chars}")
    logger.info(f"Character accuracy: {accuracy:.2f}%")

    return accuracy


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
    parser = argparse.ArgumentParser(description="Test async OCRV1 job worker")
    parser.add_argument(
        "--reference",
        action="store_true",
        help="Generate reference parquet with max accuracy settings (beam=100, no pruning)",
    )
    args = parser.parse_args()

    # Reference mode settings (will be applied to worker after initialization)
    reference_beam_width = 100 if args.reference else None
    reference_token_min_logp = -20.0 if args.reference else None

    if args.reference:
        logger.info("=== REFERENCE MODE: Max accuracy settings ===")
        logger.info(f"  beam_width: {reference_beam_width}")
        logger.info(f"  token_min_logp: {reference_token_min_logp}")

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

    # Apply reference mode settings to worker (these get passed to worker processes)
    if args.reference:
        worker.beam_width = reference_beam_width
        worker.token_min_logp = reference_token_min_logp
        worker.vocab_prune_threshold = None  # Explicitly disable pruning for max accuracy

    # Greedy decode is 17x faster but loses ~1% accuracy - use beam search for production
    worker.use_greedy_decode = False

    # k2 GPU decoder - requires k2 installed (pip install k2)
    # Set to True to use GPU-accelerated CTC decoding instead of pyctcdecode
    worker.use_k2_decoder = False
    # Sequential pipeline: complete GPU inference first, then CTC decode
    worker.use_sequential_pipeline = False

    # Log actual settings that will be used
    logger.info("=== CTC Decoder Settings ===")
    logger.info(f"  beam_width: {worker.beam_width} (None = module default 50)")
    logger.info(f"  token_min_logp: {worker.token_min_logp} (None = module default -5.0)")
    logger.info(f"  vocab_prune_threshold: {worker.vocab_prune_threshold} (None = module default)")
    logger.info(f"  vocab_prune_mode: {worker.vocab_prune_mode} (None = module default 'line')")
    logger.info(f"  use_greedy_decode: {worker.use_greedy_decode}")
    logger.info(f"  use_k2_decoder: {worker.use_k2_decoder}")
    logger.info(f"  use_sequential_pipeline: {worker.use_sequential_pipeline}")

    logger.info("Running async OCR...")
    result = worker.run(ctx)

    logger.info(f"Result: {result}")

    # Download output parquet from S3
    s3 = s3fs.S3FileSystem()
    current_parquet_s3 = f"{ocr_dest_bucket}/{artifacts_location.prefix}/{artifacts_location.basename}.parquet"
    temp_parquet = "temp_current_output.parquet"

    try:
        with s3.open(current_parquet_s3, "rb") as f_in:
            with open(temp_parquet, "wb") as f_out:
                shutil.copyfileobj(f_in, f_out)

        if args.reference:
            # Ensure directory exists and save as reference parquet
            REFERENCE_PARQUET_PATH.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(temp_parquet, REFERENCE_PARQUET_PATH)
            logger.info(f"Reference parquet saved to: {REFERENCE_PARQUET_PATH}")
        else:
            # Compare with reference if available
            accuracy = compare_with_reference(temp_parquet)
            if accuracy is not None:
                logger.info(f"\n>>> ACCURACY vs REFERENCE: {accuracy:.2f}% <<<\n")
    except Exception as e:
        logger.warning(f"Could not process output parquet: {e}")
    finally:
        if os.path.exists(temp_parquet):
            os.remove(temp_parquet)

    logger.info("Done!")


if __name__ == "__main__":
    main()
