"""
Visual debugging tool for OCR pipeline output.

This module provides visualization utilities to debug OCR results by:
- Displaying images after rotation + TPS transformations
- Drawing line bounding boxes
- Drawing syllable bounding boxes with pixel ranges
- Overlaying text and confidence scores

The visualizations help verify that:
1. Image transformations are correct
2. Line detection bounding boxes align with actual text
3. Syllable segmentation pixel ranges are accurate
"""

import json
import logging
from pathlib import Path
from typing import Any

import cv2
import gzip
import numpy as np
import numpy.typing as npt
import pandas as pd
import s3fs

from bec_orch.jobs.shared.decoder import bytes_to_frame

logger = logging.getLogger(__name__)

# Try to import pyewts for Tibetan text conversion
try:
    from pyewts import pyewts
    converter = pyewts()
    HAS_PYEWTS = True
except ImportError:
    HAS_PYEWTS = False
    logger.warning("pyewts not available - Tibetan text will not render correctly")


def _load_jsonl_gz(jsonl_path: str) -> list[dict[str, Any]]:
    """Load JSONL.gz file from S3 or local filesystem."""
    if jsonl_path.startswith("s3://"):
        s3 = s3fs.S3FileSystem()
        s3_path = jsonl_path.replace("s3://", "")
        with s3.open(s3_path, "rb") as f:
            with gzip.GzipFile(fileobj=f, mode="rb") as gz:
                lines = gz.read().decode("utf-8").strip().split("\n")
                return [json.loads(line) for line in lines if line.strip()]
    else:
        with gzip.open(jsonl_path, "rt", encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]


def _load_parquet(parquet_path: str) -> pd.DataFrame:
    """Load parquet file from S3 or local filesystem."""
    if parquet_path.startswith("s3://"):
        s3 = s3fs.S3FileSystem()
        s3_path = parquet_path.replace("s3://", "")
        with s3.open(s3_path, "rb") as f:
            return pd.read_parquet(f)
    else:
        return pd.read_parquet(parquet_path)


def _fetch_image_from_s3(bucket: str, filename: str, w_id: str, i_id: str) -> bytes:
    """Fetch source image bytes from S3."""
    import hashlib
    s3 = s3fs.S3FileSystem()
    w_prefix = hashlib.md5(w_id.encode()).hexdigest()[:2]
    image_path = f"{bucket}/Works/{w_prefix}/{w_id}/images/{w_id}-{i_id}/{filename}"
    with s3.open(image_path, "rb") as f:
        return f.read()


def _apply_image_transforms(
    image: npt.NDArray,
    rotation_angle: float,
    tps_points: tuple | None,
) -> npt.NDArray:
    """
    Apply rotation and TPS transforms to image (same as pipeline preprocessing).
    
    The returned image is in the same coordinate space as the line bboxes stored
    in the OCR output (transformed but NOT resized).
    
    Returns:
        Transformed image in the same coordinate space as line bboxes.
    """
    # Parse TPS points
    tps_input_pts = None
    tps_output_pts = None
    tps_alpha = None
    
    if tps_points:
        # tps_points is ((input_pts, output_pts), alpha)
        (tps_input_pts, tps_output_pts), tps_alpha = tps_points
    
    # Ensure image is uint8
    if image.dtype != np.uint8:
        image = (np.clip(image, 0, 255)).astype(np.uint8)
    
    # Apply rotation + TPS using the same function as the pipeline
    if image.ndim == 2:
        # Grayscale
        from bec_orch.jobs.ldv1.img_helpers import apply_transform_1
        transformed = apply_transform_1(
            image,
            rotation_angle,
            tps_input_pts,
            tps_output_pts,
            tps_alpha,
        )
    elif image.ndim == 3 and image.shape[2] == 3:
        # Color
        from bec_orch.jobs.ldv1.img_helpers import apply_transform_3
        transformed = apply_transform_3(
            image,
            rotation_angle,
            tps_input_pts,
            tps_output_pts,
            tps_alpha,
        )
    else:
        raise ValueError(f"Unsupported image shape: {image.shape}")
    
    return transformed


def _draw_line_bbox(image: npt.NDArray, bbox: list[int], color: tuple[int, int, int], thickness: int = 2) -> None:
    """Draw a line bounding box on the image."""
    x, y, w, h = bbox
    cv2.rectangle(image, (x, y), (x + w, y + h), color, thickness)


def _draw_syllable_bbox(
    image: npt.NDArray,
    line_bbox: list[int],
    syllable: dict[str, Any],
    line_bbox_scale_factor: float,
    color: tuple[int, int, int],
    thickness: int = 1,
) -> None:
    """
    Draw a syllable bounding box on the image.
    
    Syllable bbox is computed from:
    - Line bbox (x, y, w, h) - already scaled to displayed coords
    - Syllable pixel range (start_pixel, end_pixel) - relative to LINE width in original coords
    - Box extends full line height
    
    Args:
        line_bbox_scale_factor: Scale factor used for the line bbox (to scale syllable positions too)
    """
    line_x, line_y, line_w, line_h = line_bbox
    start_pixel, end_pixel = syllable["px"]
    
    # Syllable pixel ranges are relative to the line width (not page width)
    # They're in original line coordinates, so scale them to match displayed line
    scaled_start = int(start_pixel * line_bbox_scale_factor)
    scaled_end = int(end_pixel * line_bbox_scale_factor)
    
    # Syllable bbox: relative to line bbox
    syl_x = line_x + scaled_start
    syl_y = line_y
    syl_w = scaled_end - scaled_start
    syl_h = line_h
    
    cv2.rectangle(image, (syl_x, syl_y), (syl_x + syl_w, syl_y + syl_h), color, thickness)


def visualize_ocr_result(
    image_bytes: bytes,
    ocr_record: dict[str, Any],
    output_path: Path,
    draw_text: bool = True,
    draw_confidence: bool = True,
    draw_syllables: bool = True,
) -> None:
    """
    Create a visualization of OCR result on the transformed image.
    
    Args:
        image_bytes: Raw image bytes
        ocr_record: OCR result from JSONL (with rotation_angle, tps_points, lines, syllables)
        output_path: Path to save visualization image
        draw_text: Whether to draw decoded text on lines
        draw_confidence: Whether to draw confidence scores
        draw_syllables: Whether to draw syllable bounding boxes
    """
    # Decode image
    result = bytes_to_frame(ocr_record['img_file_name'], image_bytes)
    if result is None:
        logger.error(f"Failed to decode image for {ocr_record['img_file_name']}")
        return
    
    # bytes_to_frame returns (image, is_binary, orig_h, orig_w)
    image, is_binary, orig_h, orig_w = result
    
    # Apply transforms (rotation + TPS) - same as pipeline
    rotation_angle = ocr_record.get("rotation_angle", 0.0)
    tps_points = ocr_record.get("tps_points")
    
    transformed = _apply_image_transforms(image, rotation_angle, tps_points)
    
    # Calculate scale factor between displayed image and original image
    # bytes_to_frame may have downscaled the image
    actual_h, actual_w = transformed.shape[:2]
    scale_factor = min(actual_w / orig_w, actual_h / orig_h) if orig_w > 0 and orig_h > 0 else 1.0
    
    logger.debug(
        f"Image {ocr_record['img_file_name']}: orig={orig_w}x{orig_h}, "
        f"transformed={actual_w}x{actual_h}, scale={scale_factor:.3f}"
    )
    
    # Convert to BGR for color drawing
    if len(transformed.shape) == 2:
        vis_image = cv2.cvtColor(transformed, cv2.COLOR_GRAY2BGR)
    else:
        vis_image = transformed.copy()
    
    # Draw lines and syllables
    for line in ocr_record.get("lines", []):
        # Bboxes in JSONL are in original image coordinates
        # Scale them to match the displayed (possibly downscaled) image
        orig_bbox = line["bbox"]
        bbox = [
            int(orig_bbox[0] * scale_factor),
            int(orig_bbox[1] * scale_factor),
            int(orig_bbox[2] * scale_factor),
            int(orig_bbox[3] * scale_factor),
        ]
        
        # Draw line bbox in red
        _draw_line_bbox(vis_image, bbox, color=(0, 0, 255), thickness=2)
        
        # Draw syllables in blue (if enabled and available)
        if draw_syllables:
            for syllable in line.get("syllables", []):
                _draw_syllable_bbox(vis_image, bbox, syllable, scale_factor, color=(255, 0, 0), thickness=1)
        
        # Optionally draw text and confidence
        if draw_text or draw_confidence:
            x, y, w, h = bbox
            text_parts = []
            
            if draw_text:
                # Convert Tibetan to Latin if possible
                tibetan_text = line["text"]
                if HAS_PYEWTS:
                    try:
                        latin_text = converter.toWylie(tibetan_text)
                        text_parts.append(latin_text[:50])  # Truncate long text
                    except Exception:
                        text_parts.append("[conv error]")
                else:
                    text_parts.append("[no pyewts]")
            
            if draw_confidence:
                conf = line.get("confidence", 0.0)
                text_parts.append(f"conf={conf:.3f}")
            
            text = " | ".join(text_parts)
            
            # Draw text above the line bbox
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.4
            thickness = 1
            
            # Get text size to draw background
            (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, thickness)
            
            # Draw white background for text
            text_y = max(y - 5, text_h + 5)
            cv2.rectangle(
                vis_image,
                (x, text_y - text_h - baseline),
                (x + text_w, text_y + baseline),
                (255, 255, 255),
                -1,  # Filled
            )
            
            # Draw text
            cv2.putText(
                vis_image,
                text,
                (x, text_y),
                font,
                font_scale,
                (0, 0, 0),  # Black text
                thickness,
                cv2.LINE_AA,
            )
    
    # Save visualization
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), vis_image)
    logger.info(f"Saved visualization to {output_path}")


def create_debug_visualizations(
    jsonl_uri: str,
    source_image_bucket: str,
    w_id: str,
    i_id: str,
    output_dir: Path,
    max_images: int | None = None,
) -> None:
    """
    Create debug visualizations for all images in the JSONL output.
    
    Args:
        jsonl_uri: S3 URI or local path to JSONL.gz file
        source_image_bucket: S3 bucket containing source images
        w_id: Work ID
        i_id: Image group ID
        output_dir: Directory to save visualizations
        max_images: Optional limit on number of images to visualize
    """
    logger.info(f"Loading OCR results from {jsonl_uri}")
    records = _load_jsonl_gz(jsonl_uri)
    
    if max_images:
        records = records[:max_images]
    
    logger.info(f"Creating visualizations for {len(records)} images in {output_dir}")
    
    for i, record in enumerate(records):
        filename = record["img_file_name"]
        
        # Skip error records
        if record.get("error"):
            logger.warning(f"Skipping {filename} - has error: {record['error']}")
            continue
        
        try:
            # Fetch source image
            logger.info(f"[{i+1}/{len(records)}] Processing {filename}")
            image_bytes = _fetch_image_from_s3(source_image_bucket, filename, w_id, i_id)
            
            # Create visualization
            output_path = output_dir / f"{filename}_debug.jpg"
            visualize_ocr_result(
                image_bytes,
                record,
                output_path,
                draw_text=True,
                draw_confidence=True,
            )
        except Exception as e:
            logger.error(f"Failed to visualize {filename}: {e}", exc_info=True)
    
    logger.info(f"Completed {len(records)} visualizations in {output_dir}")


def create_debug_visualizations_from_parquet(
    parquet_uri: str,
    jsonl_uri: str | None,
    source_image_bucket: str,
    w_id: str,
    i_id: str,
    output_dir: Path,
    max_images: int | None = None,
) -> None:
    """
    Create debug visualizations using Parquet output (and optionally JSONL for syllable details).
    
    If jsonl_uri is None, creates line-level visualizations from parquet only.
    If jsonl_uri is provided, uses JSONL for detailed syllable data.
    
    Args:
        parquet_uri: S3 URI or local path to parquet file
        jsonl_uri: S3 URI or local path to JSONL.gz file (None if JSONL output disabled)
        source_image_bucket: S3 bucket containing source images
        w_id: Work ID
        i_id: Image group ID
        output_dir: Directory to save visualizations
        max_images: Optional limit on number of images to visualize
    """
    if jsonl_uri:
        # JSONL available - use it for detailed syllable-level visualization
        create_debug_visualizations(
            jsonl_uri=jsonl_uri,
            source_image_bucket=source_image_bucket,
            w_id=w_id,
            i_id=i_id,
            output_dir=output_dir,
            max_images=max_images,
        )
    else:
        # JSONL not available - use parquet for line-level visualization only
        _create_debug_visualizations_from_parquet_only(
            parquet_uri=parquet_uri,
            source_image_bucket=source_image_bucket,
            w_id=w_id,
            i_id=i_id,
            output_dir=output_dir,
            max_images=max_images,
        )


def _create_debug_visualizations_from_parquet_only(
    parquet_uri: str,
    source_image_bucket: str,
    w_id: str,
    i_id: str,
    output_dir: Path,
    max_images: int | None = None,
) -> None:
    """
    Create debug visualizations from parquet only (line-level, no syllables).
    
    This is used when JSONL output is disabled. It reads the parquet file
    and creates visualizations with line bounding boxes and text.
    
    Args:
        parquet_uri: S3 URI or local path to parquet file
        source_image_bucket: S3 bucket containing source images
        w_id: Work ID
        i_id: Image group ID
        output_dir: Directory to save visualizations
        max_images: Optional limit on number of images to visualize
    """
    logger.info(f"Loading OCR results from parquet: {parquet_uri}")
    df = _load_parquet(parquet_uri)
    
    # Filter to successful results only
    df = df[df["ok"] == True]  # noqa: E712
    
    if max_images:
        df = df.head(max_images)
    
    logger.info(f"Creating line-level visualizations for {len(df)} images in {output_dir}")
    
    for i, row in df.iterrows():
        filename = row["img_file_name"]
        
        try:
            # Fetch source image
            logger.info(f"[{i+1}/{len(df)}] Processing {filename}")
            image_bytes = _fetch_image_from_s3(source_image_bucket, filename, w_id, i_id)
            
            # Parse TPS points from parquet (list of lists format like ldv1)
            tps_points = None
            if row.get("tps_points") is not None and len(row["tps_points"]) == 2:
                try:
                    input_pts = row["tps_points"][0]  # list of [x, y] pairs
                    output_pts = row["tps_points"][1]  # list of [x, y] pairs
                    tps_alpha = row.get("tps_alpha")
                    tps_points = ((input_pts, output_pts), tps_alpha)
                except Exception as e:
                    logger.warning(f"Failed to parse TPS points for {filename}: {e}")
            
            # Parse line-level data from parquet (native types, not JSON)
            line_bboxes = []
            line_texts = []
            
            if row.get("line_bboxes") is not None:
                # line_bboxes is a list of structs with x, y, w, h
                for bbox_struct in row["line_bboxes"]:
                    line_bboxes.append([
                        int(bbox_struct["x"]),
                        int(bbox_struct["y"]),
                        int(bbox_struct["w"]),
                        int(bbox_struct["h"]),
                    ])
            
            if row.get("line_texts") is not None:
                # line_texts is already a list of strings
                line_texts = list(row["line_texts"])
            
            # Build OCR record from parquet data (mimics JSONL structure but without syllables)
            # Note: We don't have per-line confidence in parquet, so we omit it
            ocr_record = {
                "img_file_name": filename,
                "rotation_angle": float(row.get("rotation_angle", 0.0)),
                "tps_points": tps_points,
                "lines": [
                    {
                        "bbox": bbox,
                        "text": text,
                        "confidence": 0.0,  # Not available in parquet (no line_confidences or page_confidence)
                        "syllables": [],  # No syllable data in parquet
                    }
                    for bbox, text in zip(line_bboxes, line_texts, strict=False)
                ],
            }
            
            # Create visualization using the standard function (but syllables will be empty)
            output_path = output_dir / f"{filename}_debug.jpg"
            visualize_ocr_result(
                image_bytes,
                ocr_record,
                output_path,
                draw_text=True,
                draw_confidence=False,  # Don't draw since we're using page confidence as proxy
                draw_syllables=False,  # No syllable data in parquet
            )
        except Exception as e:
            logger.error(f"Failed to visualize {filename}: {e}", exc_info=True)
    
    logger.info(f"Completed {len(df)} visualizations in {output_dir}")

