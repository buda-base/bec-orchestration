from __future__ import annotations

from typing import Any, Dict, Optional

import cv2
import numpy as np
import numpy.typing as npt
import scipy.ndimage

from tps import ThinPlateSpline

DEFAULT_ALPHA = 0.5


def _rotation_matrix(angle_deg: float, h: int, w: int) -> npt.NDArray[np.float32]:
    center = (w / 2.0, h / 2.0)
    return cv2.getRotationMatrix2D(center, float(angle_deg), 1.0).astype(np.float32)


def _apply_rotation_1(img: npt.NDArray[np.uint8], angle_deg: float) -> npt.NDArray[np.uint8]:
    """
    Grayscale rotation. No binarization assumptions.
    """
    if angle_deg is None or abs(float(angle_deg)) < 1e-12:
        return img

    h, w = img.shape
    M = _rotation_matrix(angle_deg, h, w)

    rotated = cv2.warpAffine(
        img, M, (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    return np.ascontiguousarray(rotated)


def _apply_rotation_3(img: npt.NDArray[np.uint8], angle_deg: float) -> npt.NDArray[np.uint8]:
    if angle_deg is None or abs(float(angle_deg)) < 1e-12:
        return img

    h, w, _ = img.shape
    M = _rotation_matrix(angle_deg, h, w)

    rotated = cv2.warpAffine(
        img, M, (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )
    return np.ascontiguousarray(rotated)


def _apply_tps_1(img: npt.NDArray[np.uint8], input_pts, output_pts, alpha=DEFAULT_ALPHA) -> npt.NDArray[np.uint8]:
    """
    Grayscale TPS (bilinear sampling).
    """
    if input_pts is None:
        return img

    h, w = img.shape
    input_pts = np.asarray(input_pts, dtype=np.float64)
    output_pts = np.asarray(output_pts, dtype=np.float64)

    tps = ThinPlateSpline(alpha)
    # For backward mapping (output -> input) needed by map_coordinates, we need to fit
    # the inverse transformation. If transform() does forward mapping, we swap the arguments.
    # input_pts = where the line is (curved), output_pts = where we want it (straight)
    # We need: for each output pixel, where to sample from in input -> fit(output_pts, input_pts)
    tps.fit(output_pts, input_pts)

    out_grid = np.indices((h, w), dtype=np.float64).transpose(1, 2, 0)  # (h,w,2) in [y,x]
    in_coords = tps.transform(out_grid.reshape(-1, 2)).reshape(h, w, 2)  # (h,w,2) [y,x]
    coords = in_coords.transpose(2, 0, 1)  # (2,h,w)

    warped = scipy.ndimage.map_coordinates(img, coords, order=1, mode="constant", cval=0)
    warped = np.clip(warped, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(warped)


def _apply_tps_3(img: npt.NDArray[np.uint8], input_pts, output_pts, alpha=DEFAULT_ALPHA) -> npt.NDArray[np.uint8]:
    if input_pts is None:
        return img

    h, w, c = img.shape
    if c != 3:
        raise ValueError(f"Expected 3 channels, got shape={img.shape}")

    input_pts = np.asarray(input_pts, dtype=np.float64)
    output_pts = np.asarray(output_pts, dtype=np.float64)
    
    # we consider that corners should be in the list if wanted, no function argument

    tps = ThinPlateSpline(alpha)
    # For backward mapping (output -> input) needed by map_coordinates, we need to fit
    # the inverse transformation. If transform() does forward mapping, we swap the arguments.
    # input_pts = where the line is (curved), output_pts = where we want it (straight)
    # We need: for each output pixel, where to sample from in input -> fit(output_pts, input_pts)
    tps.fit(output_pts, input_pts)

    out_grid = np.indices((h, w), dtype=np.float64).transpose(1, 2, 0)  # (h,w,2) [y,x]
    in_coords = tps.transform(out_grid.reshape(-1, 2)).reshape(h, w, 2)  # (h,w,2) [y,x]
    coords = in_coords.transpose(2, 0, 1)  # (2,h,w)

    out = np.empty_like(img)
    for ch in range(3):
        warped = scipy.ndimage.map_coordinates(
            img[..., ch],
            coords,
            order=1,
            mode="constant",
            cval=0,
        )
        out[..., ch] = np.clip(warped, 0, 255).astype(np.uint8)

    return np.ascontiguousarray(out)


def apply_transform_1(
    img: npt.NDArray[np.uint8],
    rotation: float,
    tps_input_pts, tps_output_pts, tps_alpha=DEFAULT_ALPHA,
) -> npt.NDArray[np.uint8]:
    """
    Apply rotation + TPS to a 1-channel uint8 grayscale image.
    Returns a contiguous uint8 grayscale image.
    """
    if img.dtype != np.uint8 or img.ndim != 2:
        raise ValueError(f"img must be 2D uint8, got dtype={img.dtype}, shape={img.shape}")

    out = _apply_rotation_1(img, rotation)
    out = _apply_tps_1(out, tps_input_pts, tps_output_pts, tps_alpha)
    return out


def apply_transform_3(
    img: npt.NDArray[np.uint8],
    rotation: float,
    tps_input_pts, tps_output_pts, tps_alpha=DEFAULT_ALPHA,
) -> npt.NDArray[np.uint8]:
    """
    Apply rotation + TPS to a 3-channel uint8 image.
    Returns a contiguous uint8 image with exactly 3 channels.
    """
    if img.dtype != np.uint8 or img.ndim != 3 or img.shape[2] != 3:
        raise ValueError(f"img must be (H,W,3) uint8, got dtype={img.dtype}, shape={img.shape}")

    out = _apply_rotation_3(img, rotation)
    out = _apply_tps_3(out, tps_input_pts, tps_output_pts, tps_alpha)
    return out


def adaptive_binarize(gray: np.ndarray, block_size = 31, c = 15) -> np.ndarray:
    # in app v1, block_size = 31, c = 15
    # chatgpt says block_size = 61 and c = 7 are potential good candidates but only after background normalization
    # says rule of thumb should be:
    #   block_size ≈ 14–20 × stroke_width (must be odd)
    #   c ≈ 1.5–2.5 × stroke_width (if background-normalized, use the low end)
    if gray.ndim != 2 or gray.dtype != np.uint8:
        raise ValueError("Adaptive binarization requires grayscale uint8")
    # block_size must be odd and >= 3
    if block_size < 3:
        block_size = 3
    if (block_size & 1) == 0:
        block_size += 1

    # adaptiveThreshold returns 0/255 uint8
    return cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        block_size,
        c,
    )
