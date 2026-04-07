"""
Line detection and sorting utilities for OCR processing.

This module contains functions for:
- Line detection and contour analysis
- Line sorting and grouping algorithms
- Line image extraction and processing
- Rotation angle calculation from line orientations
"""

import logging
from dataclasses import dataclass

import cv2
import numpy as np
import numpy.typing as npt

# Constants
GRAYSCALE_NDIM = 2
MIN_K_FACTOR = 0.1

logger = logging.getLogger(__name__)


@dataclass
class BBox:
    """Bounding box with coordinates and dimensions."""

    x: int
    y: int
    w: int
    h: int

    def as_list(self) -> list[int]:
        """Return bbox as [x, y, w, h] list."""
        return [self.x, self.y, self.w, self.h]


@dataclass
class Line:
    """Line representation with contour and bounding box."""

    contour: npt.NDArray  # NumPy array of points
    bbox: BBox
    center: tuple[float, float]  # (cx, cy) center of bbox


def mask_n_crop(image: npt.NDArray, mask: npt.NDArray) -> npt.NDArray:
    """
    Apply mask to image and crop to non-zero regions.

    Args:
        image: Input image array
        mask: Binary mask array

    Returns:
        Masked and cropped image
    """
    image = image.astype(np.uint8)
    mask = mask.astype(np.uint8)

    if len(image.shape) == GRAYSCALE_NDIM:
        image = np.expand_dims(image, axis=-1)

    image_masked = cv2.bitwise_and(image, image, mask, mask)
    image_masked = np.delete(image_masked, np.where(~image_masked.any(axis=1))[0], axis=0)
    return np.delete(image_masked, np.where(~image_masked.any(axis=0))[0], axis=1)


def extract_line(image: npt.NDArray, mask: npt.NDArray, bbox_h: int, k_factor: float = 1.2) -> npt.NDArray:
    """
    Extract line region using morphological operations.

    Args:
        image: Input image array
        mask: Binary mask of line region
        bbox_h: Height of bounding box
        k_factor: Scaling factor for morphological kernel

    Returns:
        Extracted line image
    """
    k_size = int(bbox_h * k_factor)
    morph_multiplier = k_factor

    morph_rect = cv2.getStructuringElement(shape=cv2.MORPH_RECT, ksize=(k_size, int(k_size * morph_multiplier)))
    iterations = 1
    dilated_mask = cv2.dilate(mask, kernel=morph_rect, iterations=iterations)
    return mask_n_crop(image, dilated_mask)


def get_line_image(
    image: npt.NDArray, mask: npt.NDArray, bbox_h: int, bbox_tolerance: float = 2.5, k_factor: float = 1.2
) -> tuple[npt.NDArray, float]:
    """
    Extract line image with adaptive height tolerance.

    Args:
        image: Input image array
        mask: Binary mask of line region
        bbox_h: Height of bounding box
        bbox_tolerance: Height tolerance multiplier
        k_factor: Initial scaling factor for morphological kernel

    Returns:
        Tuple of (line_image, adapted_k_factor)
    """
    try:
        tmp_k = k_factor
        line_img = extract_line(image, mask, bbox_h, k_factor=tmp_k)

        # Add a safety check to prevent infinite loop
        max_attempts = 10
        attempts = 0

        while line_img.shape[0] > bbox_h * bbox_tolerance and attempts < max_attempts:
            tmp_k = tmp_k - 0.1
            if tmp_k <= MIN_K_FACTOR:  # Prevent k_factor from becoming too small
                break
            line_img = extract_line(image, mask, bbox_h, k_factor=tmp_k)
            attempts += 1

        return line_img, tmp_k  # noqa: TRY300
    except (cv2.error, ValueError):
        # Return a minimal valid image and the original k_factor in case of error
        logger.exception("Error in get_line_image")
        # Create a small blank image as fallback
        fallback_img = np.zeros((bbox_h, bbox_h * 2, 3), dtype=np.uint8)
        return fallback_img, k_factor


# =============================================================================
# Line grouping and sorting functions (moved from line_decoder.py)
# =============================================================================


def build_line_data(contour: list[dict[str, int]] | npt.NDArray) -> Line:
    """Build a Line object from a contour (dict list or numpy array)."""
    # Convert to numpy if needed
    if isinstance(contour, list):
        pts = np.array([[p["x"], p["y"]] for p in contour], dtype=np.int32)
    else:
        # Normalize shape to (N, 1, 2) for cv2 compatibility
        if contour.ndim == 2:
            contour = contour.reshape(-1, 1, 2)
        pts = contour.reshape(-1, 2)

    x, y, w, h = cv2.boundingRect(pts)
    cx = x + w / 2.0
    cy = y + h / 2.0

    return Line(
        contour=pts.reshape(-1, 1, 2),
        bbox=BBox(x=x, y=y, w=w, h=h),
        center=(cx, cy),
    )


def sort_bbox_centers(
    bbox_centers: list[tuple[float, float]], line_threshold: float
) -> list[list[tuple[float, float]]]:
    """Group bbox centers by vertical proximity to form lines."""
    if not bbox_centers:
        return []

    # Sort by y-coordinate
    sorted_centers = sorted(bbox_centers, key=lambda c: c[1])

    groups: list[list[tuple[float, float]]] = []
    current_group: list[tuple[float, float]] = [sorted_centers[0]]
    current_y = sorted_centers[0][1]

    for center in sorted_centers[1:]:
        if abs(center[1] - current_y) <= line_threshold:
            current_group.append(center)
        else:
            groups.append(current_group)
            current_group = [center]
            current_y = center[1]

    if current_group:
        groups.append(current_group)

    # Sort centers within each group by x-coordinate (left to right)
    for group in groups:
        group.sort(key=lambda c: c[0])

    # Reverse to get bottom-to-top order (matching reading order)
    groups.reverse()

    return groups


def group_line_chunks(sorted_centers: list[list[tuple[float, float]]], lines: list[Line]) -> list[list[Line]]:
    """Group line chunks into logical lines based on sorted centers."""
    if not sorted_centers or not lines:
        return []

    # Create mapping from center to line
    center_to_line = {line.center: line for line in lines}

    # Build groups of lines
    return [
        [center_to_line[center] for center in center_group if center in center_to_line]
        for center_group in sorted_centers
        if any(center in center_to_line for center in center_group)
    ]


def get_line_threshold(line_prediction: list[Line]) -> float:
    """Auto-calculate line grouping threshold from median line height."""
    if not line_prediction:
        return 20.0

    heights = [line.bbox.h for line in line_prediction]
    median_height = float(np.median(heights))
    return median_height * 0.5


def sort_lines_by_threshold(line_prediction: list[Line]) -> list[list[Line]]:
    """Main entry point: combines threshold calculation + sorting + grouping."""
    if not line_prediction:
        return []

    # Calculate threshold
    line_threshold = get_line_threshold(line_prediction)

    # Get bbox centers
    bbox_centers = [line.center for line in line_prediction]

    # Sort and group centers
    sorted_centers = sort_bbox_centers(bbox_centers, line_threshold)

    # Group lines and return
    return group_line_chunks(sorted_centers, line_prediction)
