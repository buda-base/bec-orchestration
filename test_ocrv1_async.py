"""
Test script for async OCRV1JobWorker.

Usage:
    BEC_OCR_MODEL_DIR=ocr_models/Woodblock python test_ocrv1_async.py

To generate a reference parquet with max accuracy settings:
    BEC_OCR_MODEL_DIR=ocr_models/Woodblock python test_ocrv1_async.py --reference

To limit number of pages:
    BEC_OCR_MODEL_DIR=ocr_models/Woodblock python test_ocrv1_async.py --max-pages 50
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
from bec_orch.jobs.ocrv1.config import OCRV1Config
from bec_orch.jobs.ocrv1.debug_visualizer import create_debug_visualizations
from bec_orch.jobs.ocrv1.worker_async import OCRV1JobWorkerAsync

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
    df = df.sort_values("img_file_name")

    all_lines = []
    for row in df.itertuples():
        # New schema uses page_text (single string) instead of texts (list)
        if hasattr(row, "page_text"):
            page_text = row.page_text if row.page_text else ""
            # Split page text into individual lines
            lines = page_text.split("\n") if page_text else []
            all_lines.extend(lines)
        elif hasattr(row, "texts"):
            # Legacy format (backwards compatibility)
            texts = row.texts if row.texts else []
            
            # Convert to list if needed
            if isinstance(texts, list):
                pass  # Already a list
            elif texts is None:
                texts = []
            elif hasattr(texts, "tolist") and callable(getattr(texts, "tolist", None)):
                # Numpy array or similar with tolist method
                texts = texts.tolist()
            elif hasattr(texts, "__iter__") and not isinstance(texts, (str, bytes)):
                # Convert iterable to list (but not strings)
                texts = list(texts)
            else:
                # Single value, wrap in list
                texts = [texts]

            # Add each text line, including empty ones
            all_lines.extend(str(line) if line is not None else "" for line in texts)
        else:
            # No text data found, skip this row
            pass

    return all_lines


def save_reference_lines(parquet_path: str) -> None:
    """
    Save reference output lines to a text file.
    """
    lines = extract_lines_from_parquet(parquet_path)

    REFERENCE_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with REFERENCE_OUTPUT_PATH.open("w", encoding="utf-8") as f:
        f.writelines(line + "\n" for line in lines)

    logger.info(f"Reference output saved: {len(lines)} lines -> {REFERENCE_OUTPUT_PATH}")


def show_char_diff(ref_line: str, cur_line: str, max_context: int = 40) -> str:
    """
    Show character-level differences between two lines in a compact format.
    Highlights where changes occur with context.
    """
    matcher = SequenceMatcher(None, ref_line, cur_line)
    diff_parts = []

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "replace":
            ref_part = ref_line[i1:i2]
            cur_part = cur_line[j1:j2]
            # Show context around the change
            context_start = max(0, i1 - max_context)
            context_end = min(len(ref_line), i2 + max_context)
            context_before = ref_line[context_start:i1]
            context_after = ref_line[i2:context_end]

            diff_parts.append(f"...{context_before}[{ref_part}→{cur_part}]{context_after}...")
        elif tag == "delete":
            ref_part = ref_line[i1:i2]
            context_start = max(0, i1 - max_context)
            context_end = min(len(ref_line), i2 + max_context)
            context_before = ref_line[context_start:i1]
            context_after = ref_line[i2:context_end]

            diff_parts.append(f"...{context_before}[-{ref_part}]{context_after}...")
        elif tag == "insert":
            cur_part = cur_line[j1:j2]
            # Use position in current line for context
            context_start = max(0, j1 - max_context)
            context_end = min(len(cur_line), j2 + max_context)
            context_before = cur_line[context_start:j1]
            context_after = cur_line[j2:context_end]

            diff_parts.append(f"...{context_before}[+{cur_part}]{context_after}...")

    return " ".join(diff_parts) if diff_parts else "No visible differences"


def compare_with_reference(current_parquet_path: str) -> tuple[int, int]:
    """
    Compare current OCR output with reference line-by-line.
    Exits if reference file doesn't exist.
    Shows detailed character-level differences with examples.
    """
    if not REFERENCE_OUTPUT_PATH.exists():
        logger.error(
            f"Reference output not found at {REFERENCE_OUTPUT_PATH}. Run with --reference first to generate it."
        )
        sys.exit(1)

    # Load reference lines
    with REFERENCE_OUTPUT_PATH.open(encoding="utf-8") as f:
        ref_lines = [line.rstrip("\n") for line in f]

    # Extract current lines
    try:
        cur_lines = extract_lines_from_parquet(current_parquet_path)
    except Exception:
        logger.exception("Failed to load current parquet")
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

    for i, (ref_line, cur_line) in enumerate(zip(ref_lines, cur_lines, strict=True)):
        total_chars += len(ref_line)

        if ref_line != cur_line:
            lines_with_diff += 1

            # Count character differences
            matcher = SequenceMatcher(None, ref_line, cur_line)
            # Count characters that differ
            for tag, i1, i2, j1, j2 in matcher.get_opcodes():
                if tag != "equal":
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

    w_prefix = hashlib.md5(w_id.encode()).hexdigest()[:2]  # noqa: S324
    manifest_path = f"{bucket}/Works/{w_prefix}/{w_id}/images/{w_id}-{i_id}/dimensions.json"
    logger.info(f"Fetching manifest from s3://{manifest_path}")

    info = s3.info(manifest_path)
    etag = info.get("ETag", "").strip('"')
    last_modified = info.get("LastModified", "")
    last_modified_iso = last_modified.isoformat() if hasattr(last_modified, "isoformat") else str(last_modified)

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


def main() -> None:
    parser = argparse.ArgumentParser(description="Test async OCRV1 job worker")
    parser.add_argument(
        "--reference",
        action="store_true",
        help="Generate reference parquet with max accuracy settings (beam=100, no pruning)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Create visual debug output showing images with line and syllable bboxes",
    )
    parser.add_argument(
        "--max-pages",
        type=int,
        default=50,
        help="Maximum number of pages to process (default: 50)",
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

    max_images = args.max_pages  # Use command-line argument
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

    # Build config
    cfg = OCRV1Config(
        input_width=0,  # Will be set from model
        input_height=0,  # Will be set from model
        # CTC decoder settings
        beam_width=reference_beam_width if reference_beam_width else 64,
        token_min_logp=reference_token_min_logp if reference_token_min_logp else -3.0,
        use_greedy_decode=False,
        use_hybrid_decode=not args.reference,  # Disable hybrid for reference mode
        greedy_confidence_threshold=-0.2,  # Higher = more selective
        use_sequential_pipeline=True,
        kenlm_path=str(Path(os.environ.get("BEC_OCR_MODEL_DIR", "ocr_models")) / "tibetan_5gram.binary"),
    )

    if args.debug:
        logger.info("Debug mode enabled - will create visual debug output after processing")

    worker = OCRV1JobWorkerAsync(cfg=cfg)

    # Log actual settings that will be used
    logger.info("=== CTC Decoder Settings ===")
    logger.info(f"  beam_width: {cfg.beam_width}")
    logger.info(f"  token_min_logp: {cfg.token_min_logp}")
    logger.info(f"  vocab_prune_threshold: {cfg.vocab_prune_threshold}")
    logger.info(f"  vocab_prune_mode: {cfg.vocab_prune_mode}")
    logger.info(f"  use_greedy_decode: {cfg.use_greedy_decode}")
    logger.info(f"  use_hybrid_decode: {cfg.use_hybrid_decode}")
    logger.info(f"  use_sequential_pipeline: {cfg.use_sequential_pipeline}")

    logger.info("Running async OCR...")
    result = worker.run(ctx)

    logger.info(f"Result: {result}")

    # Download output parquet from S3
    s3 = s3fs.S3FileSystem()
    current_parquet_s3 = f"{ocr_dest_bucket}/{artifacts_location.prefix}/{artifacts_location.basename}_ocrv1.parquet"
    current_jsonl_s3 = f"{ocr_dest_bucket}/{artifacts_location.prefix}/{artifacts_location.basename}_ocrv1.jsonl.gz"
    temp_parquet = "temp_current_output.parquet"

    try:
        with s3.open(current_parquet_s3, "rb") as f_in, Path(temp_parquet).open("wb") as f_out:
            shutil.copyfileobj(f_in, f_out)

        if args.reference:
            # Save line-by-line reference output
            save_reference_lines(temp_parquet)
        else:
            # Compare with reference (exits if reference doesn't exist)
            compare_with_reference(temp_parquet)
        
        # Create debug visualizations if requested
        if args.debug:
            logger.info("=== Creating debug visualizations ===")
            debug_output_dir = Path("debug_output_ocr")
            try:
                # Check if JSONL output exists (may be disabled in config)
                jsonl_exists = s3.exists(current_jsonl_s3)
                
                if jsonl_exists:
                    logger.info("Using JSONL for detailed syllable-level visualizations")
                    create_debug_visualizations(
                        jsonl_uri=f"s3://{current_jsonl_s3}",
                        source_image_bucket=source_image_bucket,
                        w_id=w_id,
                        i_id=i_id,
                        output_dir=debug_output_dir,
                        max_images=10,  # Limit to first 10 images for quick debugging
                    )
                else:
                    logger.info("JSONL output not found (may be disabled in config)")
                    logger.info("Using parquet for page-level visualizations only")
                    from bec_orch.jobs.ocrv1.debug_visualizer import create_debug_visualizations_from_parquet
                    create_debug_visualizations_from_parquet(
                        parquet_uri=f"s3://{current_parquet_s3}",
                        jsonl_uri=None,
                        source_image_bucket=source_image_bucket,
                        w_id=w_id,
                        i_id=i_id,
                        output_dir=debug_output_dir,
                        max_images=10,
                    )
                logger.info(f"Debug visualizations saved to {debug_output_dir}/")
            except Exception as e:
                logger.error(f"Failed to create debug visualizations: {e}", exc_info=True)
    
    except Exception as e:
        logger.warning(f"Could not process output parquet: {e}")
    finally:
        if Path(temp_parquet).exists():
            Path(temp_parquet).unlink()

    logger.info("Done!")


if __name__ == "__main__":
    main()
