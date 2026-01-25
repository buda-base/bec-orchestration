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
import sys
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

# Path to reference output for line-by-line comparison
REFERENCE_OUTPUT_PATH = Path("test/reference_output_lines.txt")


def extract_lines_from_parquet(parquet_path: str) -> list[str]:
    """
    Extract all OCR text lines from a parquet file, ordered consistently.
    Returns a list of lines (one per text line across all images).
    Preserves empty lines to maintain consistent line counts.
    """
    df = pd.read_parquet(parquet_path)
    
    # Sort by img_file_name for consistent ordering
    df = df.sort_values('img_file_name')
    
    all_lines = []
    for row in df.itertuples():
        texts = row.texts if hasattr(row, 'texts') else []
        
        # Convert to list if numpy array
        if hasattr(texts, 'tolist'):
            texts = texts.tolist()
        
        # Add each text line, including empty ones
        if isinstance(texts, list):
            for line in texts:
                # Convert to string, use empty string for None/null values
                all_lines.append(str(line) if line is not None else "")
        elif texts is not None:
            all_lines.append(str(texts))
        else:
            # Handle None case for non-list texts
            all_lines.append("")
    
    return all_lines


def save_reference_lines(parquet_path: str):
    """
    Save reference output lines to a text file.
    """
    lines = extract_lines_from_parquet(parquet_path)
    
    REFERENCE_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(REFERENCE_OUTPUT_PATH, 'w', encoding='utf-8') as f:
        for line in lines:
            f.write(line + '\n')
    
    logger.info(f"Reference output saved: {len(lines)} lines -> {REFERENCE_OUTPUT_PATH}")


def show_char_diff(ref_line: str, cur_line: str, max_context: int = 40) -> str:
    """
    Show character-level differences between two lines in a compact format.
    Highlights where changes occur with context.
    """
    matcher = SequenceMatcher(None, ref_line, cur_line)
    diff_parts = []
    
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == 'replace':
            ref_part = ref_line[i1:i2]
            cur_part = cur_line[j1:j2]
            # Show context around the change
            context_start = max(0, i1 - max_context)
            context_end = min(len(ref_line), i2 + max_context)
            context_before = ref_line[context_start:i1]
            context_after = ref_line[i2:context_end]
            
            diff_parts.append(
                f"...{context_before}[{ref_part}→{cur_part}]{context_after}..."
            )
        elif tag == 'delete':
            ref_part = ref_line[i1:i2]
            context_start = max(0, i1 - max_context)
            context_end = min(len(ref_line), i2 + max_context)
            context_before = ref_line[context_start:i1]
            context_after = ref_line[i2:context_end]
            
            diff_parts.append(
                f"...{context_before}[-{ref_part}]{context_after}..."
            )
        elif tag == 'insert':
            cur_part = cur_line[j1:j2]
            # Use position in current line for context
            context_start = max(0, j1 - max_context)
            context_end = min(len(cur_line), j2 + max_context)
            context_before = cur_line[context_start:j1]
            context_after = cur_line[j2:context_end]
            
            diff_parts.append(
                f"...{context_before}[+{cur_part}]{context_after}..."
            )
    
    return " ".join(diff_parts) if diff_parts else "No visible differences"


def compare_with_reference(current_parquet_path: str):
    """
    Compare current OCR output with reference line-by-line.
    Exits if reference file doesn't exist.
    Shows detailed character-level differences with examples.
    """
    if not REFERENCE_OUTPUT_PATH.exists():
        logger.error(
            f"Reference output not found at {REFERENCE_OUTPUT_PATH}. "
            f"Run with --reference first to generate it."
        )
        sys.exit(1)
    
    # Load reference lines
    with open(REFERENCE_OUTPUT_PATH, 'r', encoding='utf-8') as f:
        ref_lines = [line.rstrip('\n') for line in f]
    
    # Extract current lines
    try:
        cur_lines = extract_lines_from_parquet(current_parquet_path)
    except Exception as e:
        logger.error(f"Failed to load current parquet: {e}")
        sys.exit(1)
    
    # Check if line counts match
    if len(ref_lines) != len(cur_lines):
        logger.error(
            f"Line count mismatch! Reference has {len(ref_lines)} lines, "
            f"current has {len(cur_lines)} lines. "
            f"The input images may have changed."
        )
        sys.exit(1)
    
    # Compare line by line
    total_chars = 0
    diff_chars = 0
    lines_with_diff = 0
    diff_examples = []
    
    for i, (ref_line, cur_line) in enumerate(zip(ref_lines, cur_lines)):
        total_chars += len(ref_line)
        
        if ref_line != cur_line:
            lines_with_diff += 1
            
            # Count character differences
            matcher = SequenceMatcher(None, ref_line, cur_line)
            # Count characters that differ
            for tag, i1, i2, j1, j2 in matcher.get_opcodes():
                if tag != 'equal':
                    diff_chars += max(i2 - i1, j2 - j1)
            
            # Store example for display (limit to first 10)
            if len(diff_examples) < 10:
                # Special handling for empty line cases
                if not ref_line and cur_line:
                    diff_display = f"[EMPTY LINE→{cur_line[:80]}...]"
                elif ref_line and not cur_line:
                    diff_display = f"[{ref_line[:80]}...→EMPTY LINE]"
                else:
                    diff_display = show_char_diff(ref_line, cur_line)
                diff_examples.append((i + 1, diff_display))
    
    # Report results
    logger.info("=== LINE-BY-LINE COMPARISON WITH REFERENCE ===")
    logger.info(f"Total lines: {len(ref_lines)}")
    logger.info(f"Lines with differences: {lines_with_diff}")
    logger.info(f"Total characters: {total_chars}")
    logger.info(f"Characters different: {diff_chars}")
    
    if total_chars > 0:
        accuracy = ((total_chars - diff_chars) / total_chars) * 100
        logger.info(f"Character accuracy: {accuracy:.2f}%")
    
    if diff_examples:
        logger.info(f"\n=== SHOWING {len(diff_examples)} DIFFERENCE EXAMPLES ===")
        for line_num, diff_display in diff_examples:
            logger.info(f"\nLine {line_num}:")
            logger.info(f"  {diff_display}")
    else:
        logger.info("\nNo differences found! Output matches reference perfectly.")
    
    return lines_with_diff, diff_chars


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
    # Uses beam search only (no greedy/hybrid) with wider beam and more lenient pruning
    reference_beam_width = 80 if args.reference else None
    reference_token_min_logp = -5.0 if args.reference else None

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

    max_images = 50  # Limit for testing
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
        worker.use_hybrid_decode = False  # Only use beam search for reference (no greedy shortcut)
    else:
        # Normal mode - use same beam search params as reference for consistency
        worker.beam_width = reference_beam_width  # Same as reference
        worker.token_min_logp = reference_token_min_logp  # Same as reference
        # Keep pruning enabled in normal mode for performance

    # Greedy decode is 17x faster but loses ~1% accuracy - use beam search for production
    worker.use_greedy_decode = False
    # Hybrid decode: uses greedy first, falls back to beam search for low-confidence lines
    # Disabled to ensure deterministic results across GPU/CPU platforms
    # To enable with higher confidence threshold (only greedy for very confident lines):
    worker.greedy_confidence_threshold = -0.2  # Higher = more selective (default: -0.5)

    # NeMo GPU decoder - requires nemo_toolkit installed (pip install nemo_toolkit[asr])
    # Set to True to use GPU-accelerated CTC decoding instead of pyctcdecode
    worker.use_nemo_decoder = False
    worker.use_sequential_pipeline = True
    worker.kenlm_path = os.path.join(os.environ.get("BEC_OCR_MODEL_DIR", "ocr_models"), "tibetan_5gram.binary")

    # Log actual settings that will be used
    logger.info("=== CTC Decoder Settings ===")
    logger.info(f"  beam_width: {worker.beam_width} (None = module default 64)")
    logger.info(f"  token_min_logp: {worker.token_min_logp} (None = module default -3.0)")
    logger.info(f"  vocab_prune_threshold: {worker.vocab_prune_threshold} (None = module default)")
    logger.info(f"  vocab_prune_mode: {worker.vocab_prune_mode} (None = module default 'line')")
    logger.info(f"  use_greedy_decode: {worker.use_greedy_decode}")
    logger.info(f"  use_hybrid_decode: {worker.use_hybrid_decode}")
    logger.info(f"  use_nemo_decoder: {worker.use_nemo_decoder}")
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
            # Save line-by-line reference output
            save_reference_lines(temp_parquet)
        else:
            # Compare with reference (exits if reference doesn't exist)
            compare_with_reference(temp_parquet)
    except Exception as e:
        logger.warning(f"Could not process output parquet: {e}")
    finally:
        if os.path.exists(temp_parquet):
            os.remove(temp_parquet)

    logger.info("Done!")


if __name__ == "__main__":
    main()
