"""
Line decoder module for OCR pipeline.

Decodes images and extracts preprocessed line tensors with coordinates.

Input: FetchedBytes (raw image bytes + LD metadata)
Output: ProcessedPage (line tensors + coordinates on transformed image)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import cv2
import numpy as np
import numpy.typing as npt

from bec_orch.jobs.ldv1.img_helpers import adaptive_binarize, apply_transform_1
from bec_orch.jobs.shared.decoder import bytes_to_frame

from .line import get_line_image

if TYPE_CHECKING:
    from .config import OCRV1Config

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Data structures
# -----------------------------------------------------------------------------


@dataclass
class FetchedBytes:
    """Input for LineDecoder - raw image bytes with LD metadata."""

    page_idx: int
    filename: str
    source_etag: str
    file_bytes: bytes
    ld_row: dict  # Contains contours, rotation_angle, tps_points, etc.


@dataclass
class ProcessedLine:
    """A single preprocessed line ready for GPU inference."""

    tensor: npt.NDArray  # (1, H, W) float32 normalized [-1, 1]
    content_width: int  # Width of content region in tensor
    left_pad_width: int  # Width of left padding in tensor
    # Coordinates on the TRANSFORMED image (post-rotation, post-TPS, pre-model-resize)
    bbox: tuple[int, int, int, int]  # (x, y, w, h) bounding box on transformed image
    # Original contours that make up this line (can be multiple if merged)
    contours: list[list[dict]]  # List of contours, each contour is list of {x, y} dicts


@dataclass
class ProcessedPage:
    """Output of LineDecoder for one page."""

    page_idx: int
    filename: str
    source_etag: str
    lines: list[ProcessedLine]
    # Page-level metadata for coordinate reconstruction
    orig_width: int  # Original image width before any scaling
    orig_height: int  # Original image height before any scaling
    transformed_width: int  # Width after transforms (rotation + TPS)
    transformed_height: int  # Height after transforms (rotation + TPS)
    # Transform parameters applied to get from original to transformed coordinates
    rotation_angle: float = 0.0  # Rotation angle in degrees
    tps_points: tuple | None = None  # ((input_pts, output_pts), alpha) or None
    error: str | None = None


# -----------------------------------------------------------------------------
# Line segment merging utilities
# -----------------------------------------------------------------------------


@dataclass
class LineSegment:
    """A line segment with its contour and bounding box info."""

    contour: npt.NDArray  # NumPy array of points
    contour_dict: list[dict[str, int]]  # Original dict format [{x, y}, ...]
    bbox: tuple[int, int, int, int]  # (x, y, w, h)
    center: tuple[float, float]  # (cx, cy) center of bbox


def _build_line_segment(contour: list[dict[str, int]] | npt.NDArray) -> LineSegment:
    """Build a LineSegment from a contour (dict list or numpy array)."""
    # Convert to numpy if needed
    if isinstance(contour, list):
        contour_dict: list[dict[str, int]] = [{"x": p["x"], "y": p["y"]} for p in contour]
        pts = np.array([[p["x"], p["y"]] for p in contour], dtype=np.int32)
    else:
        # Normalize shape to (N, 1, 2) for cv2 compatibility
        if contour.ndim == 2:
            contour = contour.reshape(-1, 1, 2)
        pts = contour.reshape(-1, 2)
        contour_dict: list[dict[str, int]] = [{"x": int(pt[0]), "y": int(pt[1])} for pt in pts]

    x, y, w, h = cv2.boundingRect(pts)
    cx = x + w / 2.0
    cy = y + h / 2.0

    return LineSegment(
        contour=pts.reshape(-1, 1, 2),
        contour_dict=contour_dict,
        bbox=(x, y, w, h),
        center=(cx, cy),
    )


def _sort_bbox_centers(centers: list[tuple[float, float]]) -> list[float]:
    """Sort y-coordinates of centers."""
    return sorted([c[1] for c in centers])


def _group_line_segments(segments: list[LineSegment], line_threshold: float) -> list[list[LineSegment]]:
    """Group segments into lines based on vertical proximity."""
    if not segments:
        return []

    # Sort by y-center
    sorted_segments = sorted(segments, key=lambda s: s.center[1])

    groups: list[list[LineSegment]] = []
    current_group: list[LineSegment] = [sorted_segments[0]]
    current_y = sorted_segments[0].center[1]

    for seg in sorted_segments[1:]:
        if abs(seg.center[1] - current_y) <= line_threshold:
            current_group.append(seg)
        else:
            groups.append(current_group)
            current_group = [seg]
            current_y = seg.center[1]

    if current_group:
        groups.append(current_group)

    # Sort segments within each group by x-center (left to right)
    for group in groups:
        group.sort(key=lambda s: s.center[0])

    # Reverse to get bottom-to-top order (matching original behavior)
    groups.reverse()

    return groups


def _estimate_line_threshold(segments: list[LineSegment]) -> float:
    """Estimate line threshold from median line height."""
    if not segments:
        return 20.0

    heights = [seg.bbox[3] for seg in segments]
    median_height = float(np.median(heights))
    return median_height * 0.5


def _merge_line_segments(
    contours: list[list[dict]], line_threshold: float | None = None
) -> list[tuple[list[dict], list[list[dict]]]]:
    """
    Merge contours that belong to the same line.

    Args:
        contours: List of contours, each is a list of {x, y} dicts
        line_threshold: Max vertical distance to consider same line.
                       If None, estimated from median line height.

    Returns:
        List of (merged_contour, original_contours) tuples.
        merged_contour is the convex hull of all original contours in the group.
        original_contours is the list of original contours that were merged.
    """
    if not contours:
        return []

    # Build LineSegment objects
    segments = [_build_line_segment(c) for c in contours]

    # Estimate threshold if not provided
    if line_threshold is None:
        line_threshold = _estimate_line_threshold(segments)

    # Group into lines
    groups = _group_line_segments(segments, line_threshold)

    # Merge each group
    result: list[tuple[list[dict], list[list[dict]]]] = []
    for group in groups:
        if len(group) == 1:
            # Single segment - no merge needed
            result.append((group[0].contour_dict, [group[0].contour_dict]))
        else:
            # Merge: compute convex hull of all points
            all_points = np.vstack([seg.contour for seg in group])
            hull = cv2.convexHull(all_points)
            hull_dict = [{"x": int(pt[0][0]), "y": int(pt[0][1])} for pt in hull]
            original_contours = [seg.contour_dict for seg in group]
            result.append((hull_dict, original_contours))

    return result


# -----------------------------------------------------------------------------
# LineDecoder class
# -----------------------------------------------------------------------------


class LineDecoder:
    """
    Decodes images and extracts preprocessed line tensors with coordinates.

    Thread-safe: all processing is stateless per call.
    """

    def __init__(self, cfg: OCRV1Config) -> None:
        """
        Initialize LineDecoder.

        Args:
            cfg: OCRV1Config with all configuration options
        """
        self.cfg = cfg

    def process(self, fetched: FetchedBytes) -> ProcessedPage:
        """
        Process a fetched image and extract lines.

        Args:
            fetched: FetchedBytes containing raw image and LD metadata

        Returns:
            ProcessedPage with line tensors and coordinates
        """
        try:
            return self._process_impl(fetched)
        except Exception as e:
            logger.warning(f"[LineDecoder] Failed to process {fetched.filename}: {e}")
            return ProcessedPage(
                page_idx=fetched.page_idx,
                filename=fetched.filename,
                source_etag=fetched.source_etag,
                lines=[],
                orig_width=0,
                orig_height=0,
                transformed_width=0,
                transformed_height=0,
                error=str(e),
            )

    def _process_impl(self, fetched: FetchedBytes) -> ProcessedPage:
        """Internal implementation of process()."""
        ld_row = fetched.ld_row

        # Decode image
        image, _, orig_h, orig_w = bytes_to_frame(
            fetched.filename,
            fetched.file_bytes,
            max_width=self.cfg.max_image_width,
            max_height=self.cfg.max_image_height,
            patch_size=self.cfg.max_image_width,  # No patch consideration
            linearize=True,
        )

        # Compute scale factor (for coordinate scaling)
        actual_h, actual_w = image.shape[:2]
        scale_factor = 1.0
        if orig_h > 0 and orig_w > 0:
            scale_h = actual_h / orig_h
            scale_w = actual_w / orig_w
            if abs(scale_h - 1.0) > 0.001 or abs(scale_w - 1.0) > 0.001:
                scale_factor = min(scale_h, scale_w)
                logger.debug(
                    f"Image {fetched.filename} was downscaled: "
                    f"{orig_w}x{orig_h} -> {actual_w}x{actual_h} (scale={scale_factor:.3f})"
                )

        # Extract and scale TPS points if present
        rotation_angle = ld_row.get("rotation_angle", 0.0) or 0.0
        tps_points = ld_row.get("tps_points")
        tps_alpha = ld_row.get("tps_alpha", 0.5)

        tps_input_pts = None
        tps_output_pts = None
        if tps_points:
            tps_input_pts, tps_output_pts = tps_points
            if scale_factor != 1.0:
                if tps_input_pts is not None:
                    tps_input_pts = [[p[0] * scale_factor, p[1] * scale_factor] for p in tps_input_pts]
                if tps_output_pts is not None:
                    tps_output_pts = [[p[0] * scale_factor, p[1] * scale_factor] for p in tps_output_pts]

        # Apply transforms (rotation + TPS)
        image = apply_transform_1(image, rotation_angle, tps_input_pts, tps_output_pts, tps_alpha)
        transformed_h, transformed_w = image.shape[:2]

        # Build TPS info for output (scaled points + alpha)
        tps_info = None
        if tps_input_pts is not None or tps_output_pts is not None:
            tps_info = ((tps_input_pts, tps_output_pts), tps_alpha)

        # Get contours from LD row
        contours = ld_row.get("contours", [])
        if not contours:
            return ProcessedPage(
                page_idx=fetched.page_idx,
                filename=fetched.filename,
                source_etag=fetched.source_etag,
                lines=[],
                orig_width=orig_w,
                orig_height=orig_h,
                transformed_width=transformed_w,
                transformed_height=transformed_h,
                rotation_angle=rotation_angle,
                tps_points=tps_info,
            )

        # Scale contours to match resized image
        if scale_factor != 1.0:
            scaled_contours = []
            for contour_points in contours:
                scaled_points = [
                    {"x": int(p["x"] * scale_factor), "y": int(p["y"] * scale_factor)} for p in contour_points
                ]
                scaled_contours.append(scaled_points)
            contours = scaled_contours

        # Merge line segments if enabled
        if self.cfg.merge_line_segments:
            original_count = len(contours)
            merged = _merge_line_segments(contours, line_threshold=self.cfg.line_merge_threshold)
            if len(merged) != original_count:
                logger.debug(
                    f"[LineDecoder] {fetched.filename}: merged {original_count} segments into {len(merged)} lines"
                )
            # merged is list of (merged_contour, original_contours)
        else:
            # No merging - each contour is its own line
            merged = [(c, [c]) for c in contours]

        # Process each line
        lines: list[ProcessedLine] = []
        mask_buffer = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
        current_k = 1.7

        for line_idx, (merged_contour, original_contours) in enumerate(merged):
            # Extract line image using merged contour
            line_img, current_k, bbox = self._extract_line(image, merged_contour, current_k, mask_buffer)

            if line_img is None:
                # Create empty line entry
                empty_tensor = np.zeros((1, self.cfg.input_height, self.cfg.input_width), dtype=np.float32)
                lines.append(
                    ProcessedLine(
                        tensor=empty_tensor,
                        content_width=1,
                        left_pad_width=0,
                        bbox=(0, 0, 0, 0),
                        contours=original_contours,
                    )
                )
                continue

            # Preprocess to tensor
            tensor, content_width, left_pad_width = self._preprocess_line(
                line_img,
                debug_filename=fetched.filename if self.cfg.debug_output_dir else None,
                debug_line_idx=line_idx if self.cfg.debug_output_dir else None,
            )

            lines.append(
                ProcessedLine(
                    tensor=tensor,
                    content_width=content_width,
                    left_pad_width=left_pad_width,
                    bbox=bbox,
                    contours=original_contours,
                )
            )

        return ProcessedPage(
            page_idx=fetched.page_idx,
            filename=fetched.filename,
            source_etag=fetched.source_etag,
            lines=lines,
            orig_width=orig_w,
            orig_height=orig_h,
            transformed_width=transformed_w,
            transformed_height=transformed_h,
            rotation_angle=rotation_angle,
            tps_points=tps_info,
        )

    def _extract_line(
        self,
        image: npt.NDArray,
        contour_points: list[dict],
        k_factor: float,
        mask_buffer: npt.NDArray,
    ) -> tuple[npt.NDArray | None, float, tuple[int, int, int, int]]:
        """
        Extract line image from contour.

        Returns:
            Tuple of (line_image, adapted_k_factor, bbox).
            line_image is None if extraction failed.
            bbox is (x, y, w, h) on the transformed image.
        """
        if not contour_points:
            return None, k_factor, (0, 0, 0, 0)

        pts = np.array([[p["x"], p["y"]] for p in contour_points], dtype=np.int32)
        x, y, w, h = cv2.boundingRect(pts)
        if h <= 0:
            return None, k_factor, (0, 0, 0, 0)

        mask_buffer.fill(0)
        cv2.drawContours(mask_buffer, [pts], -1, 255, -1)

        line_img, adapted_k = get_line_image(image, mask_buffer, h, bbox_tolerance=3.0, k_factor=k_factor)

        if line_img.size == 0:
            return None, adapted_k, (x, y, w, h)

        return line_img, adapted_k, (x, y, w, h)

    def _preprocess_line(
        self,
        image: npt.NDArray,
        debug_filename: str | None = None,
        debug_line_idx: int | None = None,
    ) -> tuple[npt.NDArray, int, int]:
        """
        Preprocess line image to tensor.

        Args:
            image: Line image (grayscale, may be 2D or 3D with 1 channel)
            debug_filename: If provided, save debug images
            debug_line_idx: Line index for debug filename

        Returns:
            Tuple of (tensor, content_width, left_pad_width)
        """
        # Ensure 2D grayscale
        if image.ndim == 3:
            image = image.squeeze(axis=-1)

        h, w = image.shape[:2]
        target_h = self.cfg.input_height
        target_w = self.cfg.input_width

        if self.cfg.use_line_prepadding:
            # Add square padding (h x h) on left and right before resizing
            left_pad = h
            right_pad = h
            padded_w = w + left_pad + right_pad

            # Create padded image with white (255) padding
            with_lr_pad = np.ones((h, padded_w), dtype=np.uint8) * 255
            with_lr_pad[:, left_pad : left_pad + w] = image

            # Calculate resize dimensions to fit target while maintaining aspect ratio
            aspect = padded_w / h
            if aspect > (target_w / target_h):
                new_w = target_w
                new_h = max(1, int(target_w / aspect))
            else:
                new_h = target_h
                new_w = max(1, int(target_h * aspect))

            resized = cv2.resize(with_lr_pad, (new_w, new_h), interpolation=cv2.INTER_AREA)

            # Calculate content and left_pad widths in resized image coordinates
            scale = new_w / padded_w
            left_pad_resized = int(left_pad * scale)
            content_width_resized = int(w * scale)
        else:
            # No prepadding - just resize to fit target
            left_pad_resized = 0
            aspect = w / h
            if aspect > (target_w / target_h):
                new_w = target_w
                new_h = max(1, int(target_w / aspect))
            else:
                new_h = target_h
                new_w = max(1, int(target_h * aspect))

            resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
            content_width_resized = new_w

        # Pad to target size with white (255)
        padded = np.ones((target_h, target_w), dtype=np.uint8) * 255
        y_offset = (target_h - new_h) // 2
        x_offset = 0
        padded[y_offset : y_offset + new_h, x_offset : x_offset + new_w] = resized

        # Binarize
        binary = adaptive_binarize(padded)

        # Debug: save preprocessed line images
        if self.cfg.debug_output_dir and debug_filename is not None and debug_line_idx is not None:
            debug_dir = Path(self.cfg.debug_output_dir)
            debug_dir.mkdir(parents=True, exist_ok=True)
            safe_name = Path(debug_filename).stem.replace("/", "_").replace("\\", "_")
            out_path = debug_dir / f"{safe_name}_L{debug_line_idx:02d}.png"
            cv2.imwrite(str(out_path), binary)

        # Normalize
        tensor = binary.reshape((1, target_h, target_w)).astype(np.float32)
        tensor = (tensor / 127.5) - 1.0

        return tensor, content_width_resized, left_pad_resized
