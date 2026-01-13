#!/usr/bin/env python3
"""
Debug script to visualize image processing results from parquet files.

Reads a parquet file (local or S3) and generates debug images showing:
- Original image
- Rotated image (if rotation_angle exists)
- Rotated + TPS image (if TPS exists)
- Contours overlay on final processed image
"""

import argparse
import io
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
from urllib.parse import urlparse, unquote

import cv2
import numpy as np
import pyarrow.parquet as pq
import pyarrow.fs as pafs

try:
    import boto3
    import botocore
except ImportError:
    boto3 = None
    botocore = None

# Import image processing functions from the pipeline
try:
    # Try relative import first (when used as module)
    from .img_helpers import _apply_rotation_3, _apply_tps_3
    from .utils import (
        get_s3_folder_prefix,
        gets3blob,
        get_image_list_and_version_s3,
        _normalize_uri,
        _join_uri,
        _get_local_image_tasks,
    )
    from .types_common import ImageTask
except ImportError:
    # Fallback for when run as script directly
    import sys
    from pathlib import Path
    # Add parent directory to path if needed
    script_dir = Path(__file__).parent
    if str(script_dir) not in sys.path:
        sys.path.insert(0, str(script_dir))
    from img_helpers import _apply_rotation_3, _apply_tps_3
    from utils import (
        get_s3_folder_prefix,
        gets3blob,
        get_image_list_and_version_s3,
        _normalize_uri,
        _join_uri,
        _get_local_image_tasks,
    )
    from types_common import ImageTask


def _open_filesystem_and_path(uri: str) -> Tuple[Any, str]:
    """Returns (filesystem, path_within_fs) for a given uri."""
    if uri.startswith("s3://"):
        if pafs is None:
            raise RuntimeError("S3 support requires pyarrow.fs")
        fs = pafs.S3FileSystem()
        path = uri[len("s3://"):]
        return fs, path

    if uri.startswith("file://"):
        u = urlparse(uri)
        local_path = unquote(u.path)
        if pafs is None:
            return None, local_path
        fs = pafs.LocalFileSystem()
        return fs, local_path

    # Plain local path
    if pafs is None:
        return None, uri
    fs = pafs.LocalFileSystem()
    return fs, uri


def _parse_s3_uri(uri: str) -> Tuple[str, str]:
    """Parse s3://bucket/key into (bucket, key)."""
    u = urlparse(uri)
    if u.scheme != "s3":
        raise ValueError(f"Not an s3:// URI: {uri}")
    bucket = u.netloc
    key = u.path.lstrip("/")
    return bucket, key


def _parse_file_uri(uri: str) -> Path:
    """Parse file:// URI to Path."""
    u = urlparse(uri)
    if u.scheme != "file":
        raise ValueError(f"Not a file:// URI: {uri}")
    p = unquote(u.path)
    return Path(p)


def load_image_from_uri(uri: str) -> Optional[np.ndarray]:
    """Load image from S3 or local filesystem."""
    if uri.startswith("s3://"):
        if boto3 is None:
            raise RuntimeError("S3 support requires boto3")
        bucket, key = _parse_s3_uri(uri)
        try:
            s3 = boto3.client("s3")
            obj = s3.get_object(Bucket=bucket, Key=key)
            img_bytes = obj["Body"].read()
        except botocore.exceptions.ClientError as e:
            if e.response['Error']['Code'] == '404':
                return None
            raise
        # Decode image
        nparr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        return img
    else:
        # Local file
        if uri.startswith("file://"):
            path = _parse_file_uri(uri)
        else:
            path = Path(uri)
        if not path.exists():
            return None
        img = cv2.imread(str(path))
        return img


# Helper functions are now imported from main.py


def extract_w_i_from_parquet_filename(parquet_uri: str) -> Optional[Tuple[str, str]]:
    """
    Extract W and I IDs from parquet filename if it matches pattern:
    W[0-9A-Za-z_]+-I[0-9A-Za-z_]+-[0-9A-Za-z]+.parquet
    
    Returns (w_id, i_id) if pattern matches, None otherwise.
    """
    # Extract filename from URI
    if parquet_uri.startswith("s3://"):
        bucket, key = _parse_s3_uri(parquet_uri)
        filename = key.split("/")[-1]
    elif parquet_uri.startswith("file://"):
        filename = _parse_file_uri(parquet_uri).name
    else:
        filename = Path(parquet_uri).name
    
    # Match pattern: W<id>-I<id>-<version>.parquet
    pattern = r'^W([0-9A-Za-z_]+)-I([0-9A-Za-z_]+)-[0-9A-Za-z]+\.parquet$'
    match = re.match(pattern, filename)
    if match:
        w_id = f"W{match.group(1)}"
        i_id = f"I{match.group(2)}"
        return (w_id, i_id)
    
    return None


def infer_image_uri(parquet_uri: str, img_filename: str) -> str:
    """Infer image URI from parquet location and filename (fallback method)."""
    if parquet_uri.startswith("s3://"):
        # Remove filename from parquet URI and append image filename
        bucket, key = _parse_s3_uri(parquet_uri)
        # Remove the parquet filename
        key_parts = key.split("/")
        key_parts[-1] = img_filename
        new_key = "/".join(key_parts)
        return f"s3://{bucket}/{new_key}"
    else:
        # Local file
        if parquet_uri.startswith("file://"):
            parquet_path = _parse_file_uri(parquet_uri)
        else:
            parquet_path = Path(parquet_uri)
        # Replace parquet filename with image filename
        img_path = parquet_path.parent / img_filename
        return str(img_path.absolute())


def deserialize_tps_points(tps_points: list) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """
    Deserialize TPS points from parquet format.
    
    Parquet format: list of [in_y, in_x, out_y, out_x] (float32)
    - Points are stored in original image coordinate system (after scaling from processed frame)
    - Format matches what ld_postprocessor.py stores: [y, x] order
    
    Returns: (input_pts, output_pts) as (N,2) arrays in [y,x] format (float64)
    - input_pts: source points in original image coordinates (where the line is curved)
    - output_pts: target points in original image coordinates (where we want the line to be straight)
    
    Note: These points are meant to be applied to the original image AFTER rotation (if any).
    The coordinate system matches the rotated original image dimensions.
    """
    if tps_points is None or len(tps_points) == 0:
        return None
    
    input_pts = []
    output_pts = []
    for pt in tps_points:
        if len(pt) >= 4:
            # Parquet format: [in_y, in_x, out_y, out_x]
            # Convert to [y, x] format arrays as expected by _apply_tps_3
            input_pts.append([pt[0], pt[1]])  # [y, x] - source points
            output_pts.append([pt[2], pt[3]])  # [y, x] - target points
    
    if len(input_pts) == 0:
        return None
    
    return np.array(input_pts, dtype=np.float64), np.array(output_pts, dtype=np.float64)


def deserialize_contours(contours: list) -> list:
    """
    Deserialize contours from parquet format.
    Parquet format: list<list<struct{x:int16,y:int16}>>
    Returns: list of numpy arrays in (N,2) format with (x,y) coordinates
    """
    if contours is None:
        return []
    
    result = []
    for contour in contours:
        # Check if contour is empty (handle both list and numpy array cases)
        if contour is None:
            continue
        if isinstance(contour, (list, tuple)) and len(contour) == 0:
            continue
        if hasattr(contour, '__len__') and len(contour) == 0:
            continue
        
        pts = []
        for pt in contour:
            pts.append([pt["x"], pt["y"]])
        if len(pts) > 0:
            result.append(np.array(pts, dtype=np.int32))
    
    return result


def draw_contours_on_image(img: np.ndarray, contours: list, color: Tuple[int, int, int] = (0, 255, 255), alpha: float = 0.5) -> np.ndarray:
    """
    Draw filled contours on a grayscale version of the image with transparency.
    Args:
        img: BGR image (H, W, 3)
        contours: list of numpy arrays with (x,y) points
        color: BGR color tuple (default yellow)
        alpha: transparency factor (0.0-1.0)
    Returns:
        Grayscale image with filled yellow contours overlaid at specified transparency
    """
    # Convert to grayscale as base
    if img.ndim == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Convert back to 3-channel for drawing
        base = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    else:
        base = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    
    # Create overlay image for contours
    overlay = base.copy()
    
    # Draw filled contours on overlay (thickness=-1 fills the contour)
    for contour in contours:
        if len(contour.shape) == 2 and contour.shape[1] == 2:
            # Reshape to OpenCV format: (N, 1, 2)
            cnt = contour.reshape(-1, 1, 2)
            # Fill the contour (thickness=-1) and also draw edges (thickness=2)
            cv2.drawContours(overlay, [cnt], -1, color, -1)  # Fill
            cv2.drawContours(overlay, [cnt], -1, color, 2)   # Edge
    
    # Blend overlay (with contours) with base (grayscale) at specified alpha
    result = cv2.addWeighted(overlay, alpha, base, 1.0 - alpha, 0)
    return result


def process_row(row: dict, parquet_uri: str, output_folder: Path, image_uri_map: Optional[Dict[str, str]] = None, tps_only: bool = False) -> None:
    """Process a single row from the parquet file."""
    img_filename = row.get("img_file_name")
    if not img_filename:
        print(f"Warning: Skipping row with no img_file_name", file=sys.stderr)
        return
    
    # Skip error rows
    if not row.get("ok", True):
        print(f"Warning: Skipping error row for {img_filename}", file=sys.stderr)
        return
    
    # Skip rows without TPS data if --tps-only is set
    if tps_only:
        tps_points = row.get("tps_points")
        if tps_points is None or len(tps_points) == 0:
            return
    
    # Determine image location
    if image_uri_map and img_filename in image_uri_map:
        img_uri = image_uri_map[img_filename]
    else:
        # Fallback: infer from parquet location
        img_uri = infer_image_uri(parquet_uri, img_filename)
    
    # Load original image
    img = load_image_from_uri(img_uri)
    if img is None:
        error_msg = f"Error: Could not load image from {img_uri}"
        if image_uri_map is None:
            error_msg += " (no --input-folder or --w/--i provided, and inference failed)"
        print(error_msg, file=sys.stderr)
        raise FileNotFoundError(error_msg)
    
    # Ensure image is 3-channel BGR
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.ndim == 3 and img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    elif img.ndim == 3 and img.shape[2] != 3:
        print(f"Warning: Unexpected image shape {img.shape} for {img_filename}", file=sys.stderr)
        return
    
    # Extract processing parameters
    rotation_angle = row.get("rotation_angle")
    if rotation_angle is not None and (np.isnan(rotation_angle) or abs(rotation_angle) < 1e-6):
        rotation_angle = None
    
    tps_points = row.get("tps_points")
    tps_alpha = row.get("tps_alpha")
    if tps_alpha is not None and np.isnan(tps_alpha):
        tps_alpha = None
    
    contours = row.get("contours")
    
    # Base filename without extension
    base_name = Path(img_filename).stem
    ext = Path(img_filename).suffix or ".jpg"
    
    # 1. Write original image with new naming scheme
    original_path = output_folder / f"{base_name}_00_orig{ext}"
    cv2.imwrite(str(original_path), img)
    print(f"Written: {original_path}", file=sys.stderr)
    
    # 2. Apply transformations and determine transformation suffix
    current_img = img
    transformation_suffix = ""
    
    # Apply rotation if needed
    if rotation_angle is not None:
        current_img = _apply_rotation_3(current_img, rotation_angle)
        transformation_suffix = "_rotated"
    
    # Apply TPS if needed (on rotated image or base image)
    if tps_points is not None:
        tps_data = deserialize_tps_points(tps_points)
        if tps_data is not None:
            input_pts, output_pts = tps_data
            alpha = float(tps_alpha) if tps_alpha is not None else 0.5
            # NOTE: _apply_tps_3 now swaps input_pts and output_pts internally to get
            # the correct inverse transformation for backward mapping
            current_img = _apply_tps_3(current_img, input_pts, output_pts, alpha)
            transformation_suffix = "_rotated_tps"
    
    # 3. Draw contours on final transformed image
    if contours is not None:
        deserialized_contours = deserialize_contours(contours)
        if deserialized_contours:
            contours_img = draw_contours_on_image(current_img, deserialized_contours, color=(0, 255, 255), alpha=0.5)
            contours_path = output_folder / f"{base_name}_01_contours{transformation_suffix}{ext}"
            cv2.imwrite(str(contours_path), contours_img)
            print(f"Written: {contours_path}", file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(
        description="Debug parquet files by generating visualization images"
    )
    parser.add_argument(
        "parquet_uri",
        type=str,
        help="URI or path to parquet file (local or s3://)"
    )
    parser.add_argument(
        "output_folder",
        type=str,
        help="Local output folder for debug images"
    )
    
    # Input mode arguments (same as main.py)
    parser.add_argument(
        "--input-folder",
        type=str,
        help="Local folder containing image files (for local mode)"
    )
    s3_group = parser.add_argument_group("S3 mode (requires both --w and --i)")
    s3_group.add_argument(
        "--w",
        type=str,
        help="Work ID (e.g., W22084) for S3 mode"
    )
    s3_group.add_argument(
        "--i",
        type=str,
        help="Image group ID (e.g., I0886) for S3 mode"
    )
    
    parser.add_argument(
        "--tps-only",
        action="store_true",
        help="Only output images for rows that have non-null TPS data"
    )
    
    args = parser.parse_args()
    
    # Validate input mode arguments
    if args.input_folder:
        if args.w or args.i:
            parser.error("--input-folder cannot be used with --w or --i")
    elif args.w or args.i:
        if not (args.w and args.i):
            parser.error("Both --w and --i must be provided for S3 mode")
    
    # Create output folder
    output_folder = Path(args.output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)
    
    # Build image URI mapping if input arguments provided
    image_uri_map: Optional[Dict[str, str]] = None
    
    if args.input_folder:
        # Local mode
        image_tasks = _get_local_image_tasks(args.input_folder)
        image_uri_map = {task.img_filename: task.source_uri for task in image_tasks}
        print(f"Using local input folder: {args.input_folder} ({len(image_uri_map)} images)", file=sys.stderr)
    elif args.w and args.i:
        # S3 mode (explicit)
        image_tasks, i_version = get_image_list_and_version_s3(args.w, args.i)
        if image_tasks is None or i_version is None:
            raise ValueError(f"Failed to fetch image list for W{args.w} I{args.i}")
        image_uri_map = {task.img_filename: task.source_uri for task in image_tasks}
        print(f"Using S3 mode: W{args.w} I{args.i} ({len(image_uri_map)} images)", file=sys.stderr)
    else:
        # Try to infer from parquet filename
        w_i = extract_w_i_from_parquet_filename(args.parquet_uri)
        if w_i:
            w_id, i_id = w_i
            print(f"Inferred {w_id} {i_id} from parquet filename. Fetching images from S3...", file=sys.stderr)
            image_tasks, i_version = get_image_list_and_version_s3(w_id, i_id)
            if image_tasks is None or i_version is None:
                print(f"Warning: Failed to fetch image list for {w_id} {i_id}. Will attempt to infer from parquet location.", file=sys.stderr)
                image_uri_map = None
            else:
                image_uri_map = {task.img_filename: task.source_uri for task in image_tasks}
                print(f"Using S3 mode (inferred): {w_id} {i_id} ({len(image_uri_map)} images)", file=sys.stderr)
        else:
            print("Warning: No --input-folder or --w/--i provided, and parquet filename doesn't match W*-I*-*.parquet pattern. Will attempt to infer image locations from parquet location.", file=sys.stderr)
            image_uri_map = None
    
    # Read parquet file
    parquet_uri = args.parquet_uri
    fs, path = _open_filesystem_and_path(parquet_uri)
    
    if fs is None:
        # Local filesystem
        table = pq.read_table(path)
    else:
        # Use pyarrow filesystem
        with fs.open_input_file(path) as f:
            table = pq.read_table(f)
    
    # Convert to pandas for easier iteration
    df = table.to_pandas()
    
    print(f"Processing {len(df)} rows from {parquet_uri}", file=sys.stderr)
    
    # Process each row
    errors = []
    for idx, row in df.iterrows():
        try:
            process_row(row.to_dict(), parquet_uri, output_folder, image_uri_map, tps_only=args.tps_only)
        except FileNotFoundError as e:
            # Re-raise FileNotFoundError if we don't have explicit input (inference failed)
            if image_uri_map is None:
                errors.append(str(e))
            else:
                img_name = row.get("img_file_name", "unknown")
                print(f"Error processing {img_name}: {e}", file=sys.stderr)
        except Exception as e:
            img_name = row.get("img_file_name", "unknown")
            print(f"Error processing {img_name}: {e}", file=sys.stderr)
            import traceback
            traceback.print_exc()
    
    if errors and image_uri_map is None:
        print("\nErrors occurred while inferring image locations:", file=sys.stderr)
        for error in errors[:5]:  # Show first 5 errors
            print(f"  {error}", file=sys.stderr)
        if len(errors) > 5:
            print(f"  ... and {len(errors) - 5} more errors", file=sys.stderr)
        print("\nPlease provide --input-folder (for local) or --w/--i (for S3) to specify image locations.", file=sys.stderr)
        sys.exit(1)
    
    print(f"Done. Output written to {output_folder}", file=sys.stderr)


if __name__ == "__main__":
    main()

