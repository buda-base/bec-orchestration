"""
LineProcessor component for OCRv1 pipeline.

Processes fetched bytes into preprocessed line tensors:
1. Decode image with page-level resize (4096×2048) using shared decoder
2. Apply transformations (rotation/TPS) using ldv1 functions
3. Extract line images from grayscale page
4. Preprocess each line: resize + conditional binarization + normalization
"""
import asyncio
import logging
import time
import traceback as tb_mod
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

import cv2
import numpy as np
import numpy.typing as npt

from ..shared.decoder import bytes_to_frame
from ..ldv1.img_helpers import adaptive_binarize, apply_transform_1
from .line import get_line_image
from .types_ocrv1 import (
    EndOfStream,
    FetchedBytes,
    FetchedBytesMsg,
    LineTensor,
    PipelineError,
    ProcessedPage,
    ProcessedPageMsg,
)

logger = logging.getLogger(__name__)


class LineProcessor:
    """
    Processes raw image bytes into preprocessed line tensors.
    
    Pipeline stage: FetchedBytes → ProcessedPage
    
    Uses ThreadPoolExecutor for CPU-bound work (decode, transform, extract, preprocess).
    """

    def __init__(
        self,
        input_width: int,
        input_height: int,
        q_in: asyncio.Queue[FetchedBytesMsg],
        q_out: asyncio.Queue[ProcessedPageMsg],
        num_workers: int = 4,
        debug_output_dir: Optional[str] = None,
    ):
        """
        Initialize LineProcessor.
        
        Args:
            input_width: Model input width (for padding)
            input_height: Model input height (for padding)
            q_in: Input queue (FetchedBytesMsg)
            q_out: Output queue (ProcessedPageMsg)
            num_workers: Number of thread pool workers
            debug_output_dir: Directory to save debug images (None = disabled)
        """
        self.input_width = input_width
        self.input_height = input_height
        self.q_in = q_in
        self.q_out = q_out
        self.num_workers = num_workers
        self.debug_output_dir = debug_output_dir
        
        self._executor = ThreadPoolExecutor(max_workers=num_workers, thread_name_prefix="lineproc")
        self._processed_count = 0
        self._error_count = 0
        
        # Create debug output directory if enabled
        if self.debug_output_dir:
            import os
            os.makedirs(self.debug_output_dir, exist_ok=True)
            logger.info(f"[LineProcessor] Debug mode enabled - saving images to {self.debug_output_dir}")

    async def run(self) -> None:
        """
        Main loop: consume FetchedBytes, process in thread pool, emit ProcessedPage.
        """
        logger.info(f"[LineProcessor] Starting with {self.num_workers} workers")
        loop = asyncio.get_event_loop()
        
        # Track pending processing tasks
        pending_tasks: set[asyncio.Task] = set()
        
        try:
            while True:
                msg = await self.q_in.get()
                
                if isinstance(msg, EndOfStream):
                    # Wait for all pending tasks to complete
                    if pending_tasks:
                        await asyncio.gather(*pending_tasks, return_exceptions=True)
                    await self.q_out.put(EndOfStream(stream="processed", producer="LineProcessor"))
                    break
                
                if isinstance(msg, PipelineError):
                    # Pass errors through
                    await self.q_out.put(msg)
                    self._error_count += 1
                    continue
                
                fetched: FetchedBytes = msg
                
                # Process in thread pool (non-blocking)
                task = asyncio.create_task(self._process_one(loop, fetched))
                pending_tasks.add(task)
                task.add_done_callback(pending_tasks.discard)
        
        except asyncio.CancelledError:
            logger.info("[LineProcessor] Cancelled, cleaning up...")
            raise
        
        finally:
            self._executor.shutdown(wait=False)
            logger.info(
                f"[LineProcessor] Done - processed={self._processed_count}, errors={self._error_count}"
            )

    async def _process_one(self, loop: asyncio.AbstractEventLoop, fetched: FetchedBytes) -> None:
        """Process one page asynchronously (runs processing in thread pool)."""
        try:
            processed = await loop.run_in_executor(
                self._executor,
                self._process_page_sync,
                fetched,
            )
            await self.q_out.put(processed)
            
            if processed.error is None:
                self._processed_count += 1
            else:
                self._error_count += 1
        
        except Exception as e:
            logger.error(f"[LineProcessor] Failed to process {fetched.filename}: {e}", exc_info=True)
            error = PipelineError(
                stage="LineProcessor",
                page_idx=fetched.page_idx,
                filename=fetched.filename,
                source_etag=fetched.source_etag,
                error_type=type(e).__name__,
                message=str(e),
                traceback=tb_mod.format_exc(),
                retryable=False,
            )
            await self.q_out.put(error)
            self._error_count += 1

    def _process_page_sync(self, fetched: FetchedBytes) -> ProcessedPage:
        """
        Synchronous processing (runs in thread pool).
        
        Steps:
        1. Decode with bytes_to_frame (4096×2048 resize) → grayscale + is_binary
        2. Apply transforms (rotation/TPS) if needed → still grayscale
        3. Extract lines from grayscale → list of grayscale line images
        4. Preprocess each line: resize + conditional binarize + normalize → tensor
        """
        ld_data = fetched.ld_data
        
        try:
            # 1. Decode at full resolution (like original code)
            #    Use large max dimensions to prevent downscaling for typical images
            #    The _downscale functions won't upscale (they check s >= 1.0)
            frame, is_binary, orig_h, orig_w = bytes_to_frame(
                fetched.filename,
                fetched.file_bytes,
                max_width=6000,
                max_height=3000,
                patch_size=6000,  # Large value prevents wide-image special case from shrinking
                linearize=True,
                normalize_background=False,
                patch_vertical_overlap_px=0,
                snap_extra_patch_row_threshold_px=0,
                max_patch_rows=100,
            )
            # frame: 2D grayscale uint8, no resize for images under 10000×5000
            # is_binary: True if source was binary (TIFF Group4, etc.)
            # orig_h, orig_w: Original image dimensions (same as frame dimensions unless resized)
            
            # Calculate scale factor for contours
            # Contours are in original image coordinates, need to scale to match resized frame
            decoded_h, decoded_w = frame.shape[:2]
            scale_x = decoded_w / orig_w if orig_w > 0 else 1.0
            scale_y = decoded_h / orig_h if orig_h > 0 else 1.0
            
        except Exception as e:
            return ProcessedPage(
                page_idx=fetched.page_idx,
                filename=fetched.filename,
                source_etag=fetched.source_etag,
                lines=[],
                error=f"Decode failed: {e}",
            )
        
        # 2. Apply transforms if needed (rotation/TPS)
        was_transformed = False
        rotation_angle = ld_data.rotation_angle
        tps_points = ld_data.tps_points
        tps_alpha = ld_data.tps_alpha
        
        if (rotation_angle and abs(rotation_angle) > 0.01) or tps_points:
            was_transformed = True
            
            try:
                # Parse TPS points if present
                tps_input_pts = None
                tps_output_pts = None
                if tps_points:
                    tps_input_pts, tps_output_pts = tps_points
                
                # Apply transforms using ldv1 function (grayscale → grayscale)
                frame = apply_transform_1(
                    frame,
                    rotation_angle,
                    tps_input_pts,
                    tps_output_pts,
                    tps_alpha,
                )
            except Exception as e:
                return ProcessedPage(
                    page_idx=fetched.page_idx,
                    filename=fetched.filename,
                    source_etag=fetched.source_etag,
                    lines=[],
                    error=f"Transform failed: {e}",
                )
        
        # 3. Extract lines from grayscale
        contours = ld_data.contours
        if not contours:
            return ProcessedPage(
                page_idx=fetched.page_idx,
                filename=fetched.filename,
                source_etag=fetched.source_etag,
                lines=[],
            )
        
        lines = []
        mask_buffer = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)
        current_k = 1.7
        
        for line_idx, contour_points in enumerate(contours):
            try:
                # Scale contour points from original image coordinates to resized frame coordinates
                scaled_contour = [
                    {"x": int(p["x"] * scale_x), "y": int(p["y"] * scale_y)}
                    for p in contour_points
                ]
                
                line_img, current_k = self._extract_line(frame, scaled_contour, current_k, mask_buffer)
                if line_img is None:
                    # Empty line - create zero tensor
                    lines.append(
                        LineTensor(
                            tensor=np.zeros((1, self.input_height, self.input_width), dtype=np.float32),
                            original_width=1,
                        )
                    )
                    continue
                
                # line_img is grayscale (2D array)
                original_width = line_img.shape[1]
                
                # 4. Preprocess line: resize + conditional binarization + normalize
                tensor = self._preprocess_line(line_img, is_binary, was_transformed)
                
                # Debug: save line images if enabled
                if self.debug_output_dir:
                    self._save_debug_image(
                        fetched.filename,
                        line_idx,
                        line_img,
                        tensor,
                        is_binary,
                        was_transformed,
                        scale_x,
                        scale_y,
                    )
                
                lines.append(LineTensor(tensor=tensor, original_width=original_width))
            
            except Exception as e:
                logger.warning(f"[LineProcessor] Failed to process line {line_idx} in {fetched.filename}: {e}")
                # Add empty line on error
                lines.append(
                    LineTensor(
                        tensor=np.zeros((1, self.input_height, self.input_width), dtype=np.float32),
                        original_width=1,
                    )
                )
        
        return ProcessedPage(
            page_idx=fetched.page_idx,
            filename=fetched.filename,
            source_etag=fetched.source_etag,
            lines=lines,
        )

    def _extract_line(
        self,
        image: npt.NDArray,
        contour_points: list[dict],
        k_factor: float,
        mask_buffer: npt.NDArray,
    ) -> tuple[Optional[npt.NDArray], float]:
        """
        Extract line image from contour.
        
        Args:
            image: Grayscale page image
            contour_points: List of {x, y} dicts defining line contour
            k_factor: Morphological kernel scaling factor
            mask_buffer: Reusable buffer for mask
        
        Returns:
            Tuple of (line_image, adapted_k_factor) or (None, k_factor)
        """
        if not contour_points:
            return None, k_factor
        
        pts = np.array([[p["x"], p["y"]] for p in contour_points], dtype=np.int32)
        _, _, _, bbox_h = cv2.boundingRect(pts)
        if bbox_h <= 0:
            return None, k_factor
        
        mask_buffer.fill(0)
        cv2.drawContours(mask_buffer, [pts], -1, 255, -1)
        
        line_img, adapted_k = get_line_image(
            image, mask_buffer, bbox_h, bbox_tolerance=3.0, k_factor=k_factor
        )
        
        if line_img.size == 0:
            return None, adapted_k
        
        return line_img, adapted_k

    def _preprocess_line(
        self,
        line_img: npt.NDArray,
        is_binary: bool,
        was_transformed: bool,
    ) -> npt.NDArray:
        """
        Preprocess line: resize to model dims, conditionally binarize, normalize.
        
        Args:
            line_img: Grayscale line image (2D uint8)
            is_binary: Whether source page was binary
            was_transformed: Whether rotation/TPS was applied to page
        
        Returns:
            Normalized tensor (1, H, W) ready for model
        """
        # Ensure grayscale
        if line_img.ndim == 3:
            # Shouldn't happen, but handle gracefully
            line_img = cv2.cvtColor(line_img, cv2.COLOR_RGB2GRAY)
        
        h, w = line_img.shape
        
        # 1. Resize to fit model dimensions (aspect-preserving)
        target_h = self.input_height
        target_w = self.input_width
        aspect = w / h
        
        if aspect > (target_w / target_h):
            new_w = target_w
            new_h = max(1, int(target_w / aspect))
        else:
            new_h = target_h
            new_w = max(1, int(target_h * aspect))
        
        resized = cv2.resize(line_img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
        # 2. Pad to model size (white background = 255)
        padded = np.ones((target_h, target_w), dtype=np.uint8) * 255
        y_offset = (target_h - new_h) // 2
        x_offset = 0
        padded[y_offset : y_offset + new_h, x_offset : x_offset + new_w] = resized
        
        # 3. Conditional binarization
        if is_binary and not was_transformed:
            # Binary source, no transform → already binary, keep as-is
            binary = padded
        elif is_binary and was_transformed:
            # Binary source, transformed → quick re-binarization
            _, binary = cv2.threshold(padded, 127, 255, cv2.THRESH_BINARY)
        else:
            # Grayscale/RGB source → proper adaptive binarization
            binary = adaptive_binarize(padded, block_size=31, c=15)
        
        # 4. Normalize to tensor: [0, 255] → [-1, 1]
        tensor = binary.reshape((1, target_h, target_w)).astype(np.float32)
        tensor = (tensor / 127.5) - 1.0
        
        return tensor

    def _save_debug_image(
        self,
        filename: str,
        line_idx: int,
        line_img: npt.NDArray,
        tensor: npt.NDArray,
        is_binary: bool,
        was_transformed: bool,
        scale_x: float,
        scale_y: float,
    ) -> None:
        """
        Save debug images showing line extraction and preprocessing.
        
        Saves 3 images per line:
        1. Original extracted line (grayscale)
        2. After preprocessing (binarized + padded)
        3. Metadata text file
        """
        import os
        
        # Create safe filename
        base_name = filename.replace('/', '_').replace('.', '_')
        prefix = f"{base_name}_line{line_idx:03d}"
        
        # Convert tensor back to uint8 for visualization
        # Tensor is (1, H, W) with values in [-1, 1]
        preprocessed = ((tensor[0] + 1.0) * 127.5).astype(np.uint8)
        
        # Save original line
        orig_path = os.path.join(self.debug_output_dir, f"{prefix}_1_original.jpg")
        cv2.imwrite(orig_path, line_img)
        
        # Save preprocessed (after resize + binarize + pad)
        prep_path = os.path.join(self.debug_output_dir, f"{prefix}_2_preprocessed.jpg")
        cv2.imwrite(prep_path, preprocessed)
        
        # Save metadata
        meta_path = os.path.join(self.debug_output_dir, f"{prefix}_meta.txt")
        with open(meta_path, 'w') as f:
            f.write(f"Filename: {filename}\n")
            f.write(f"Line index: {line_idx}\n")
            f.write(f"Contour scale: {scale_x:.4f}x, {scale_y:.4f}y\n")
            f.write(f"Original size: {line_img.shape[1]}x{line_img.shape[0]}\n")
            f.write(f"Model size: {self.input_width}x{self.input_height}\n")
            f.write(f"Is binary source: {is_binary}\n")
            f.write(f"Was transformed: {was_transformed}\n")
            binarize_method = (
                "none (kept as-is)" if (is_binary and not was_transformed) else
                "quick threshold" if (is_binary and was_transformed) else
                "adaptive binarization"
            )
            f.write(f"Binarization: {binarize_method}\n")
