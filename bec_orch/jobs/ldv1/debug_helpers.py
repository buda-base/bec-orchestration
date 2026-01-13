"""
Helper functions for debug mode output.
All functions are designed to be non-blocking and have minimal performance impact.
"""

import asyncio
from pathlib import Path
from typing import Optional, List
import numpy as np
import numpy.typing as npt
import cv2


def should_debug_image(cfg, img_filename: str) -> bool:
    """
    Check if debug output should be generated for a specific image.
    
    Returns True if:
    - debug_mode is enabled AND
    - (debug_images is None OR img_filename is in debug_images)
    """
    if not cfg.debug_mode or not cfg.debug_folder:
        return False
    if cfg.debug_images is None:
        return True
    return img_filename in cfg.debug_images


def get_debug_path(cfg, img_filename: str, suffix: str) -> Path:
    """
    Get the debug output path for an image with a specific suffix.
    
    Example: img_filename="I123.jpg", suffix="01_decoded" -> "I123_01_decoded.jpg"
    """
    base = Path(img_filename).stem
    ext = Path(img_filename).suffix.lower()
    # Use .jpg for all outputs (except original which keeps original extension)
    if suffix.startswith("00_orig"):
        output_ext = ext if ext else ".jpg"
    else:
        output_ext = ".jpg"
    return Path(cfg.debug_folder) / f"{base}_{suffix}{output_ext}"


async def save_debug_bytes(cfg, img_filename: str, file_bytes: bytes) -> None:
    """
    Save original image bytes to debug folder.
    Non-blocking: uses asyncio.to_thread for file I/O.
    """
    if not should_debug_image(cfg, img_filename):
        return
    
    debug_path = get_debug_path(cfg, img_filename, "00_orig")
    await asyncio.to_thread(_write_bytes_sync, debug_path, file_bytes)


def _write_bytes_sync(path: Path, data: bytes) -> None:
    """Synchronous helper for writing bytes."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


def save_debug_bytes_sync(cfg, img_filename: str, file_bytes: bytes) -> None:
    """
    Synchronous version of save_debug_bytes for use in thread pools.
    Saves original image bytes to debug folder.
    """
    if not should_debug_image(cfg, img_filename):
        return
    
    debug_path = get_debug_path(cfg, img_filename, "00_orig")
    try:
        _write_bytes_sync(debug_path, file_bytes)
    except Exception:
        pass  # Silently ignore debug save errors


def save_debug_image_sync(cfg, img_filename: str, suffix: str, img: npt.NDArray[np.uint8]) -> None:
    """
    Synchronous version of save_debug_image for use in thread pools.
    Saves a numpy array as a JPEG image to debug folder.
    """
    if not should_debug_image(cfg, img_filename):
        return
    
    debug_path = get_debug_path(cfg, img_filename, suffix)
    try:
        _encode_and_write_sync(debug_path, img)
    except Exception:
        pass  # Silently ignore debug save errors


async def save_debug_image(cfg, img_filename: str, suffix: str, img: npt.NDArray[np.uint8]) -> None:
    """
    Save a numpy array as a JPEG image to debug folder.
    Non-blocking: uses asyncio.to_thread for encoding and file I/O.
    
    Args:
        cfg: PipelineConfig
        img_filename: Original image filename
        suffix: Debug suffix (e.g., "01_decoded", "10_pass1_line_mask")
        img: 2D uint8 numpy array (grayscale) or 3D uint8 (BGR)
    """
    if not should_debug_image(cfg, img_filename):
        return
    
    debug_path = get_debug_path(cfg, img_filename, suffix)
    await asyncio.to_thread(_encode_and_write_sync, debug_path, img)


def _encode_and_write_sync(path: Path, img: npt.NDArray[np.uint8]) -> None:
    """Synchronous helper for encoding and writing image."""
    path.parent.mkdir(parents=True, exist_ok=True)
    
    # Ensure uint8
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)
    
    # Handle grayscale vs BGR
    if img.ndim == 2:
        # Grayscale
        success = cv2.imwrite(str(path), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
    elif img.ndim == 3:
        # BGR (3 or 4 channels)
        if img.shape[2] == 4:
            # RGBA -> BGR (drop alpha)
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
        elif img.shape[2] == 3:
            # Already BGR or RGB - assume BGR
            pass
        else:
            raise ValueError(f"Unsupported image shape: {img.shape}")
        success = cv2.imwrite(str(path), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
    else:
        raise ValueError(f"Unsupported image ndim: {img.ndim}")
    
    if not success:
        raise RuntimeError(f"Failed to write debug image: {path}")


async def save_debug_contours(
    cfg,
    img_filename: str,
    suffix: str,
    base_img: npt.NDArray[np.uint8],
    contours: List[npt.NDArray[np.int32]],
    contour_color: tuple[int, int, int] = (0, 255, 0),
    fill_alpha: float = 0.3,
) -> None:
    """
    Draw contours on a base image with transparency and save to debug folder.
    Non-blocking: uses asyncio.to_thread for drawing and file I/O.
    
    Args:
        cfg: PipelineConfig
        img_filename: Original image filename
        suffix: Debug suffix (e.g., "11_pass1_contours")
        base_img: 2D uint8 grayscale image
        contours: List of contour arrays (OpenCV format)
        contour_color: BGR color tuple for contours (default: green)
        fill_alpha: Alpha value for filled contours (0.0 = transparent, 1.0 = opaque)
    """
    if not should_debug_image(cfg, img_filename):
        return
    
    debug_path = get_debug_path(cfg, img_filename, suffix)
    await asyncio.to_thread(_draw_contours_and_write_sync, debug_path, base_img, contours, contour_color, fill_alpha)


def _draw_contours_and_write_sync(
    path: Path,
    base_img: npt.NDArray[np.uint8],
    contours: List[npt.NDArray[np.int32]],
    contour_color: tuple[int, int, int],
    fill_alpha: float,
) -> None:
    """Synchronous helper for drawing contours and writing image."""
    path.parent.mkdir(parents=True, exist_ok=True)
    
    # Convert grayscale to BGR for colored contours
    if base_img.ndim == 2:
        img_bgr = cv2.cvtColor(base_img, cv2.COLOR_GRAY2BGR)
    else:
        img_bgr = base_img.copy()
    
    # Create overlay for transparent fill
    overlay = img_bgr.copy()
    
    # Draw filled contours with transparency
    if fill_alpha > 0.0:
        cv2.drawContours(overlay, contours, contourIdx=-1, color=contour_color, thickness=-1)
        cv2.addWeighted(overlay, fill_alpha, img_bgr, 1.0 - fill_alpha, 0, img_bgr)
    
    # Draw contour outlines
    cv2.drawContours(img_bgr, contours, contourIdx=-1, color=contour_color, thickness=2)
    
    success = cv2.imwrite(str(path), img_bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])
    if not success:
        raise RuntimeError(f"Failed to write debug contours image: {path}")

