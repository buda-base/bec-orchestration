"""
Line detection and sorting utilities for OCR processing.

This module contains functions for:
- Line detection and contour analysis
- Line sorting and grouping algorithms
- Line image extraction and processing
- Rotation angle calculation from line orientations
"""

import cv2
import numpy as np
import numpy.typing as npt

# Constants
GRAYSCALE_NDIM = 2
MIN_K_FACTOR = 0.1


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
    except (cv2.error, ValueError) as e:
        # Return a minimal valid image and the original k_factor in case of error
        print(f"Error in get_line_image: {e}")
        # Create a small blank image as fallback (grayscale)
        fallback_img = np.zeros((bbox_h, bbox_h * 2), dtype=np.uint8)
        return fallback_img, k_factor
