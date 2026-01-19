import asyncio
import concurrent.futures as futures
from collections import deque
from ..ldv1.config import PipelineConfig
import io
import logging
import os
import time
from typing import Tuple, Optional, Union
import numpy as np
import cv2
from functools import lru_cache
import math

from ..ldv1.types_common import ImageTask, DecodedFrame, FetchedBytesMsg, DecodedFrameMsg, FetchedBytes, PipelineError, EndOfStream
from ..ldv1.debug_helpers import save_debug_bytes_sync, save_debug_image_sync

# Pre-import PIL at module level (avoid repeated import overhead)
try:
    from PIL import Image as PILImage
    _PIL_AVAILABLE = True
except ImportError:
    PILImage = None
    _PIL_AVAILABLE = False

# Optional: pyvips / libvips fast-path (very fast decode + can resize before NumPy materialization)
try:
    import pyvips  # type: ignore
    _VIPS_AVAILABLE = True
    # Suppress vips warnings about missing TIFF tags
    # Set both Python logger and libvips internal log level to ERROR
    vips_logger = logging.getLogger('pyvips')
    vips_logger.setLevel(logging.ERROR)
    # libvips uses GLib logging - this disables WARNING and INFO messages
    import os as _os_vips
    if 'VIPS_WARNING' not in _os_vips.environ:
        _os_vips.environ['VIPS_WARNING'] = '0'
except ImportError:
    pyvips = None
    _VIPS_AVAILABLE = False

logger = logging.getLogger(__name__)
# Separate logger for timing/performance warnings (ERROR by default)
timings_logger = logging.getLogger("bec_timings")


class ImageDecodeError(RuntimeError):
    """Raised when an image cannot be decoded or processed safely."""

def _is_likely_binary_u8(gray_u8: np.ndarray) -> bool:
    """
    Cheap heuristic: check a small strided sample for values in {0,1,255}.
    Avoids np.unique() over huge images.
    """
    if gray_u8.dtype != np.uint8 or gray_u8.ndim != 2 or gray_u8.size == 0:
        return False
    h, w = gray_u8.shape
    step_y = max(1, h // 512)
    step_x = max(1, w // 512)
    sample = gray_u8[::step_y, ::step_x]
    u = np.unique(sample)
    if u.size > 2:
        return False
    s = set(map(int, u.tolist()))
    return s.issubset({0, 1, 255})


def _decode_via_vips(
    filename: str,
    image_bytes: bytes,
    *,
    max_width: int,
    max_height: int,
    patch_size: int,
    patch_vertical_overlap_px: int,
    snap_extra_patch_row_threshold_px: int,
    max_patch_rows: int,
) -> tuple[np.ndarray, bool, int, int, bool]:
    """
    Decode to 2D uint8 grayscale via libvips, and (critically) downscale BEFORE converting to NumPy.

    Returns: (gray_u8, likely_binary, orig_h, orig_w, did_downscale)
    """
    if not _VIPS_AVAILABLE:
        raise ImageDecodeError("pyvips not available")

    try:
        # access="sequential" helps throughput on large images
        img = pyvips.Image.new_from_buffer(image_bytes, "", access="sequential")

        # Multi-page guard (best-effort; not all loaders provide n-pages)
        try:
            if img.get_typeof("n-pages") != 0:
                n_pages = int(img.get("n-pages"))
                if n_pages != 1:
                    raise ImageDecodeError(f"Multi-page image not supported ({n_pages} frames)")
        except ImageDecodeError:
            raise
        except Exception:
            pass

        orig_w, orig_h = int(img.width), int(img.height)

        # Ensure grayscale, 1 band
        if img.bands > 1:
            img = img.colourspace("b-w")

        # Track likely-binary from vips pixel format if possible
        binary_hint = (img.format == "onebit")
        if binary_hint:
            img = img.cast("uchar")

        # Compute downscale factor using the same logic as OpenCV path
        s = _compute_downscale(
            orig_w,
            orig_h,
            max_width,
            max_height,
            patch_size,
            patch_vertical_overlap_px,
            snap_extra_patch_row_threshold_px,
            max_patch_rows,
        )

        did_downscale = False
        if s < 1.0:
            # For bilevel docs, nearest preserves 0/255 best; for grayscale, linear is a good speed/quality tradeoff.
            img = img.resize(s, kernel=("nearest" if binary_hint else "linear"))
            did_downscale = True

        if img.format != "uchar":
            # Be strict; fallback lanes will handle exotic formats more predictably.
            raise ImageDecodeError(f"Unsupported vips pixel format '{img.format}' for '{filename}'")

        mem = img.write_to_memory()
        gray = np.frombuffer(mem, dtype=np.uint8).reshape(int(img.height), int(img.width))

        likely_binary = binary_hint or _is_likely_binary_u8(gray)
        if likely_binary and gray.size and int(gray.max()) == 1:
            gray = gray.copy()
            gray *= 255  # in-place scale to {0,255}

        return np.ascontiguousarray(gray), bool(likely_binary), orig_h, orig_w, did_downscale

    except ImageDecodeError:
        raise
    except Exception as e:
        raise ImageDecodeError(f"VIPS decode failed for '{filename}': {e}") from e


class Decoder:
    """Decode stage (thread pool).

    Input: FetchedBytesMsg
    Output: DecodedFrameMsg pushed to q_decoder_to_gpu_pass_1.
    """

    def __init__(self, cfg, q_prefetcher_to_decoder: asyncio.Queue[FetchedBytesMsg], q_decoder_to_gpu_pass_1: asyncio.Queue[DecodedFrameMsg]):
        self.cfg = cfg
        self.q_prefetcher_to_decoder = q_prefetcher_to_decoder
        self.q_decoder_to_gpu_pass_1 = q_decoder_to_gpu_pass_1
        self.pool = futures.ThreadPoolExecutor(
            max_workers=cfg.decode_threads,
            thread_name_prefix="decode",
        )

    def _decode_one(self, item: FetchedBytes) -> DecodedFrame:
        # Debug: save original bytes (only if debug mode enabled - check before call to avoid overhead)
        if self.cfg.debug_mode:
            save_debug_bytes_sync(self.cfg, item.task.img_filename, item.file_bytes)
        
        frame, is_binary, orig_h, orig_w = bytes_to_frame(
            item.task.img_filename,
            item.file_bytes,
            max_width=self.cfg.frame_max_width,
            max_height=self.cfg.frame_max_height,
            linearize=self.cfg.linearize,
            normalize_background=self.cfg.normalize_background,
            patch_size=self.cfg.patch_size,
            patch_vertical_overlap_px=self.cfg.patch_vertical_overlap_px,
            snap_extra_patch_row_threshold_px=self.cfg.snap_extra_patch_row_threshold_px,
            max_patch_rows=self.cfg.max_patch_rows
        )
        
        # Debug: save decoded image (only if debug mode enabled)
        if self.cfg.debug_mode:
            save_debug_image_sync(self.cfg, item.task.img_filename, "01_decoded", frame)
        
        decoded = DecodedFrame(task=item.task, source_etag=item.source_etag, orig_h=orig_h, orig_w=orig_w, frame=frame, is_binary=is_binary, first_pass=True, rotation_angle=None, tps_data=None)
        return decoded

    def _decode_one_with_timing(self, item: FetchedBytes) -> Tuple[Union[DecodedFrame, PipelineError], float, FetchedBytes]:
        """
        Decode and return (result, decode_time_seconds, original_item).
        
        On error, returns a PipelineError instead of raising, so the caller
        can track the failed frame and emit it to downstream stages.
        """
        import traceback as tb_mod
        t0 = time.perf_counter()
        try:
            result = self._decode_one(item)
            return result, time.perf_counter() - t0, item
        except Exception as e:
            decode_time = time.perf_counter() - t0
            tb_str = tb_mod.format_exc()
            error = PipelineError(
                stage="Decoder",
                task=item.task,
                source_etag=item.source_etag,
                error_type=type(e).__name__,
                message=str(e),
                traceback=tb_str,
                retryable=False,
                attempt=1,
            )
            return error, decode_time, item

    async def run(self) -> None:
        """
        Main loop with TRUE parallel decoding.
        
        Key optimization: Fire off multiple decode tasks without awaiting each one.
        This keeps all thread pool workers busy instead of just 1.
        
        Also uses non-blocking puts to avoid blocking on output queue.
        """
        loop = asyncio.get_running_loop()
        
        # Timing stats
        decoded_count = 0
        error_count = 0
        total_decode_time = 0.0
        total_wait_time = 0.0
        total_output_wait_time = 0.0
        run_start = time.perf_counter()
        
        # Pending decode futures (asyncio.Future from run_in_executor)
        pending_futures: list[asyncio.Future] = []
        max_inflight = self.cfg.decode_threads * 2  # Keep 2x workers busy
        
        # Output buffer for completed decodes (to avoid blocking puts)
        # Use deque for O(1) popleft instead of list's O(n) pop(0)
        output_buffer: deque = deque()
        max_output_buffer = self.cfg.decode_threads * 2
        
        # Track when input is exhausted
        input_done = False
        
        logger.info(f"[Decoder] Starting with {self.cfg.decode_threads} threads (parallel mode)")

        try:
            while True:
                # --- Drain output buffer first (non-blocking) ---
                while output_buffer:
                    try:
                        self.q_decoder_to_gpu_pass_1.put_nowait(output_buffer[0])
                        output_buffer.popleft()  # O(1) with deque
                    except asyncio.QueueFull:
                        # Queue full, stop draining
                        break
                
                # --- Collect completed decode futures ---
                still_pending = []
                for fut in pending_futures:
                    if fut.done():
                        try:
                            result, decode_time, original_item = fut.result()
                            total_decode_time += decode_time
                            
                            if isinstance(result, PipelineError):
                                # Decode failed - emit the error so downstream tracks the frame
                                error_count += 1
                                logger.error(
                                    f"[Decoder] Decode failed for {original_item.task.img_filename}: "
                                    f"{result.error_type}: {result.message}"
                                )
                                output_buffer.append(result)
                            else:
                                # Success
                                decoded_count += 1
                                
                                if decode_time > 0.5:
                                    timings_logger.warning(
                                        f"[Decoder] Slow decode: {result.task.img_filename} took {decode_time:.2f}s"
                                    )
                                
                                # Add to output buffer instead of blocking put
                                output_buffer.append(result)
                            
                        except Exception as e:
                            # Unexpected error from the future itself (not from decode)
                            error_count += 1
                            logger.error(f"[Decoder] Unexpected error from decode future: {e}", exc_info=True)
                    else:
                        still_pending.append(fut)
                pending_futures = still_pending
                
                # --- Check termination ---
                if input_done and len(pending_futures) == 0 and len(output_buffer) == 0:
                    run_time = time.perf_counter() - run_start
                    avg_decode = total_decode_time / max(1, decoded_count)
                    throughput = decoded_count / run_time if run_time > 0 else 0
                    
                    logger.info(
                        f"[Decoder] DONE - {decoded_count} decoded, {error_count} errors, "
                        f"run_time={run_time:.2f}s ({throughput:.1f} img/s), "
                        f"avg_decode={avg_decode*1000:.1f}ms, total_wait={total_wait_time:.2f}s, "
                        f"output_wait={total_output_wait_time:.2f}s"
                    )
                    
                    await self.q_decoder_to_gpu_pass_1.put(EndOfStream(stream="decoded", producer="Decoder"))
                    return
                
                # --- If output buffer is getting full, wait for downstream to consume ---
                if len(output_buffer) >= max_output_buffer:
                    wait_start = time.perf_counter()
                    # Wait for space in output queue
                    await self.q_decoder_to_gpu_pass_1.put(output_buffer.popleft())  # O(1)
                    total_output_wait_time += time.perf_counter() - wait_start
                    continue  # Go back to draining output buffer
                
                # --- Fetch more items if we have capacity ---
                # Use non-blocking get first, then timeout
                while len(pending_futures) < max_inflight and not input_done:
                    msg = None
                    try:
                        msg = self.q_prefetcher_to_decoder.get_nowait()
                    except asyncio.QueueEmpty:
                        # Queue empty, use short timeout
                        wait_start = time.perf_counter()
                        try:
                            msg = await asyncio.wait_for(
                                self.q_prefetcher_to_decoder.get(),
                                timeout=0.005  # 5ms (shorter timeout)
                            )
                        except asyncio.TimeoutError:
                            total_wait_time += time.perf_counter() - wait_start
                            break
                        total_wait_time += time.perf_counter() - wait_start
                    
                    if isinstance(msg, EndOfStream):
                        input_done = True
                        break
                    
                    if isinstance(msg, PipelineError):
                        output_buffer.append(msg)  # Errors go through same path
                        error_count += 1
                        continue
                    
                    # Fire off decode in thread pool (returns Future, don't await!)
                    fut = loop.run_in_executor(
                        self.pool,
                        self._decode_one_with_timing,
                        msg
                    )
                    pending_futures.append(fut)
                
                # Yield to event loop (let futures make progress)
                await asyncio.sleep(0)

        except asyncio.CancelledError:
            logger.info("[Decoder] Cancelled, cleaning up...")
            for fut in pending_futures:
                fut.cancel()
            raise
        finally:
            self.pool.shutdown(wait=False)



def _ext_lower(filename: str) -> str:
    return os.path.splitext(filename)[1].lower().lstrip(".")

def _jpeg_dimensions(image_bytes: bytes) -> Optional[tuple[int, int]]:
    """
    Best-effort JPEG dimension probe to get (width, height) without full decode.

    Prefer Pillow if available (simple/robust). Note: Pillow may emit/raise a DecompressionBomb
    warning/error based only on the header dimensions; we suppress the warning and fall back to
    the header parser if Pillow refuses the image.
    """
    if _PIL_AVAILABLE:
        try:
            import warnings
            with warnings.catch_warnings():
                # We only want dimensions; suppress bomb warning.
                warnings.simplefilter("ignore", category=getattr(PILImage, "DecompressionBombWarning", Warning))
                with PILImage.open(io.BytesIO(image_bytes)) as im:
                    w, h = im.size
                    if w > 0 and h > 0:
                        return int(w), int(h)
        except Exception:
            # Fall back to header parsing below
            pass

    # Fallback: minimal JPEG header parse (no full decode).
    # JPEG starts with SOI 0xFFD8
    if len(image_bytes) < 4 or image_bytes[0] != 0xFF or image_bytes[1] != 0xD8:
        return None

    i = 2
    n = len(image_bytes)

    # Markers that contain width/height (Start Of Frame, excluding DHT/DAC/etc.)
    sof_markers = {
        0xC0, 0xC1, 0xC2, 0xC3,
        0xC5, 0xC6, 0xC7,
        0xC9, 0xCA, 0xCB,
        0xCD, 0xCE, 0xCF,
    }

    try:
        while i + 4 <= n:
            # Find next marker (0xFF ??). Skip fill bytes.
            if image_bytes[i] != 0xFF:
                i += 1
                continue
            while i < n and image_bytes[i] == 0xFF:
                i += 1
            if i >= n:
                return None
            marker = image_bytes[i]
            i += 1

            # Start of Scan: after this, the stream is entropy-coded; stop.
            if marker == 0xDA:
                return None
            # EOI
            if marker == 0xD9:
                return None
            # Restart markers and standalone markers have no length; continue.
            if 0xD0 <= marker <= 0xD7 or marker == 0x01:
                continue

            if i + 2 > n:
                return None
            seg_len = (image_bytes[i] << 8) | image_bytes[i + 1]
            i += 2
            if seg_len < 2 or i + (seg_len - 2) > n:
                return None

            if marker in sof_markers:
                # Segment layout: [precision:1][height:2][width:2]...
                if seg_len < 7:
                    return None
                p = i
                height = (image_bytes[p + 1] << 8) | image_bytes[p + 2]
                width = (image_bytes[p + 3] << 8) | image_bytes[p + 4]
                if width <= 0 or height <= 0:
                    return None
                return width, height

            # Skip segment payload
            i += seg_len - 2
    except Exception:
        return None

    return None

def _compute_downscale(
    w: int,
    h: int,
    max_w: int,
    max_h: int,
    patch_size: int,
    patch_vertical_overlap_px: int = 78,
    snap_extra_patch_row_threshold_px: int = 78,
    max_patch_rows: int = 2
) -> float:
    """
    Compute a resize scale factor for patch-based inference of line detection.

    Pipeline logic:
      1) Downscale to fit within (max_w, max_h) (never upscale in this step).
      2) Ensure at least one full patch in height (may upscale).
      3) Snap height *down* if it barely crosses a patch-row boundary (works for any row count).
      4) Optionally cap the number of patch rows by shrinking height to the maximum allowed.

    Definitions (vertical tiling with overlap):
      stride_y = patch_size - patch_vertical_overlap_px
      Row boundaries happen at: patch_size + k * stride_y   (k >= 0)
    """
    if w <= 0 or h <= 0:
        raise ImageDecodeError(f"Invalid image dimensions: {w}x{h}")

    if patch_size <= 0:
        raise ValueError(f"Invalid patch_size: {patch_size}")

    if patch_vertical_overlap_px < 0 or patch_vertical_overlap_px >= patch_size:
        raise ValueError(
            f"patch_vertical_overlap_px must be in [0, patch_size-1], got {patch_vertical_overlap_px}"
        )

    stride_y = patch_size - patch_vertical_overlap_px  # vertical step between rows

    # -----------------------------
    # Special case: very wide images (width > 2*height) -> fit exactly one patch row
    # -----------------------------
    if w > 2.5 * h:
        # Target: final height = patch_size (one row of patches)
        scale_for_height = patch_size / float(h)
        # But also respect max_w constraint
        scale_for_width = max_w / float(w)
        s = min(scale_for_height, scale_for_width)
        return s

    # -----------------------------
    # Step 1) Fit within max box (no upscaling)
    # -----------------------------
    scale_to_max_w = max_w / float(w)
    scale_to_max_h = max_h / float(h)
    s = min(scale_to_max_w, scale_to_max_h, 1.0)

    scaled_h = h * s

    # -----------------------------
    # Step 2) Ensure at least one patch in height (if close)
    # -----------------------------
    if scaled_h < patch_size and scaled_h > 0.75 * patch_size:
        s = patch_size / float(h)
        scaled_h = patch_size

    # -----------------------------
    # Step 3) Snap down if we're just barely above ANY row boundary
    #
    # Boundaries: H = patch_size + k * stride_y
    # If scaled_h is in (boundary, boundary + threshold], snap down to boundary.
    # -----------------------------
    if snap_extra_patch_row_threshold_px > 0:
        if scaled_h > patch_size:
            excess = scaled_h - patch_size

            # k is the largest integer such that boundary(k) <= scaled_h
            k = int(math.floor(excess / float(stride_y)))
            boundary_h = patch_size + k * stride_y

            extra_px = scaled_h - boundary_h
            if 0.0 < extra_px <= float(snap_extra_patch_row_threshold_px):
                scaled_h = boundary_h
                s = scaled_h / float(h)

    # -----------------------------
    # Step 4) Cap patch rows (soft cap)
    #
    # Max height allowed for R rows: patch_size + (R - 1) * stride_y
    # -----------------------------
    if max_patch_rows is not None and max_patch_rows > 0:
        max_allowed_h = patch_size + (max_patch_rows - 1) * stride_y
        if scaled_h > max_allowed_h:
            scaled_h = max_allowed_h
            s = scaled_h / float(h)

    return s



def _downscale_gray(gray: np.ndarray, max_w: int, max_h: int, patch_wh: int, patch_vertical_overlap_px: int, snap_extra_patch_row_threshold_px: int, max_patch_rows: int) -> np.ndarray:
    h, w = gray.shape[:2]
    s = _compute_downscale(w, h, max_w, max_h, patch_wh, patch_vertical_overlap_px, snap_extra_patch_row_threshold_px, max_patch_rows)
    if s >= 1.0:
        return gray
    new_w = max(1, int(w * s))
    new_h = max(1, int(h * s))
    # INTER_AREA is best for downscaling continuous-tone grayscale
    return cv2.resize(gray, (new_w, new_h), interpolation=cv2.INTER_AREA)

def _downscale_binary(binary: np.ndarray, max_w: int, max_h: int, patch_wh: int, patch_vertical_overlap_px: int, snap_extra_patch_row_threshold_px: int, max_patch_rows: int) -> np.ndarray:
    h, w = binary.shape[:2]
    s = _compute_downscale(w, h, max_w, max_h, patch_wh, patch_vertical_overlap_px, snap_extra_patch_row_threshold_px, max_patch_rows)
    if s >= 1.0:
        return binary
    new_w = max(1, int(w * s))
    new_h = max(1, int(h * s))
    # Preserve 0/255 exactly for already-binary inputs
    return cv2.resize(binary, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

def _to_gray_cv(img: np.ndarray) -> np.ndarray:
    """
    Convert cv2-decoded image to single-channel uint8 grayscale with minimal overhead.
    """
    if img is None:
        raise ImageDecodeError("cv2.imdecode returned None")

    if img.dtype != np.uint8:
        # cv2.imdecode should yield uint8; if not, normalize safely.
        img = np.clip(img, 0, 255).astype(np.uint8, copy=False)

    if img.ndim == 2:
        return img  # already gray

    if img.ndim != 3:
        raise ImageDecodeError(f"Unexpected cv2 image ndim={img.ndim}")

    ch = img.shape[2]
    if ch == 3:
        # OpenCV uses BGR
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if ch == 4:
        # BGRA
        return cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)

    raise ImageDecodeError(f"Unsupported channel count: {ch}")

def _decode_via_cv2(image_bytes: bytes, likely_jpeg: bool) -> np.ndarray:
    """
    Decodes bytes as a uint8 grayscale frame with cv2 (mostly used for jpegs)

    Uses IMREAD_GRAYSCALE in imdecode when relevant for performance gains

    Returns the frame
    """
    buf = np.frombuffer(image_bytes, dtype=np.uint8)

    if likely_jpeg:
        # Fast path: grayscale at decode time
        img = cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            return img

        # Fallback
        img = cv2.imdecode(buf, cv2.IMREAD_UNCHANGED)
    else:
        img = cv2.imdecode(buf, cv2.IMREAD_UNCHANGED)

    if img is None:
        raise ImageDecodeError("OpenCV failed to decode image")

    return img

def _decode_via_pil(filename: str, image_bytes: bytes) -> tuple[np.ndarray, bool]:
    """
    Decodes bytes as a uint8 grayscale frame with PIL (mostly used for tiffs)

    Returns the frame and a boolean True if the image is already binary (no need to re-binarize it)
    """
    if not _PIL_AVAILABLE:
        raise ImageDecodeError("PIL not available")

    try:
        with PILImage.open(io.BytesIO(image_bytes)) as im:
            n_frames = getattr(im, "n_frames", 1)
            if n_frames != 1:
                raise ImageDecodeError(f"Multi-page image not supported ({n_frames} frames)")

            mode = im.mode

            # ---- binary-fast-path modes ----
            if mode == "1":
                # Avoid an extra temporary for huge bilevel pages
                arr = np.array(im, dtype=np.uint8)  # 0/1
                arr *= 255  # in-place -> 0/255
                return arr, True

            if mode == "L":
                # Already grayscale - no conversion needed
                arr = np.array(im, dtype=np.uint8)
                return arr, False

            if mode == "P":
                # Palette mode - convert to grayscale
                im = im.convert("L")
                arr = np.array(im, dtype=np.uint8)
                return arr, False

            # ---- definitely non-binary ----
            if mode in ("RGB", "RGBA", "CMYK"):
                im = im.convert("RGB")
                rgb = np.array(im, dtype=np.uint8)
                gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
                return gray, False

            # ---- fallback ----
            im = im.convert("L")
            arr = np.array(im, dtype=np.uint8)
            return arr, False

    except ImageDecodeError:
        raise
    except Exception as e:
        raise ImageDecodeError(f"PIL decode failed for '{filename}': {e}") from e

# Functions to go from sRGB to linear light values

@lru_cache(maxsize=1)
def _srgb_to_linear_u8_lut() -> np.ndarray:
    """
    Returns a 256x1 uint8 LUT suitable for cv2.LUT, mapping sRGB-encoded [0..255]
    to linear-light [0..255].
    """
    x = np.arange(256, dtype=np.float32) / 255.0
    lin = np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)
    lut = (lin * 255.0 + 0.5).astype(np.uint8)
    return np.ascontiguousarray(lut.reshape(256, 1))  # OpenCV expects (256,1) or (1,256)

def _srgb_u8_to_linear_u8(gray_u8: np.ndarray) -> np.ndarray:
    """
    Very fast sRGB->linear for uint8 grayscale (or uint8 multi-channel too).
    Uses cv2.LUT with a precomputed 256-entry table.
    """
    if gray_u8.dtype != np.uint8:
        raise ValueError("_srgb_u8_to_linear_u8 expects uint8 input")
    lut = _srgb_to_linear_u8_lut()
    # cv2.LUT supports 1-channel or multi-channel uint8; mapping is applied per channel.
    return cv2.LUT(gray_u8, lut)

# Function to normalize the background

def _normalize_background_u8(
    gray_u8: np.ndarray,
    *,
    sigma: float | None = None,
    border_type: int = cv2.BORDER_REPLICATE,
) -> np.ndarray:
    """
    Fast document background/shading normalization.

    Steps:
      1) convert to float32 [0..1]
      2) large Gaussian blur to estimate background (low-frequency shading)
      3) subtract background (high-pass)
      4) min/max normalize to uint8 [0..255] (in OpenCV, output dtype set to CV_8U)

    Input:  2D uint8
    Output: 2D uint8
    """
    if gray_u8.dtype != np.uint8 or gray_u8.ndim != 2:
        raise ValueError("_normalize_background_u8_fast expects a 2D uint8 grayscale image")

    h, w = gray_u8.shape

    # Choose sigma based on image size (tuned for text documents)
    # possibly good defaults for pechas
    if sigma is None:
        m = h if h < w else w
        sigma = 0.03 * float(m)
        if sigma < 10.0: sigma = 10.0
        if sigma > 60.0: sigma = 60.0

    # Convert once to float32 in [0..1]
    # (Using numpy for scaling is typically faster than calling multiple cv2 ops.)
    f = gray_u8.astype(np.float32) * (1.0 / 255.0)

    # Background estimate (low-frequency shading)
    bg = cv2.GaussianBlur(f, (0, 0), sigmaX=sigma, sigmaY=sigma, borderType=border_type)

    # Division (flat-field). eps prevents blow-ups in dark regions.
    #norm = cv2.divide(f, bg + 1e-6)
    norm = cv2.subtract(f, bg)  # stays float32

    # Normalize to uint8
    dst = np.empty(norm.shape, dtype=np.uint8)
    return cv2.normalize(norm, dst, 0, 255, cv2.NORM_MINMAX)

def bytes_to_frame(
    filename: str,
    image_bytes: bytes,
    *,
    max_width: int = 4096,
    max_height: int = 2048,
    patch_size: int = 512,
    linearize = True, # convert to linear rgb
    normalize_background: bool = False,
    patch_vertical_overlap_px: int = 78,
    snap_extra_patch_row_threshold_px: int = 78,
    max_patch_rows: int = 2
) -> Tuple[np.ndarray, bool, int, int]:
    """
    Decode image bytes into a uint8 OpenCV frame (2D array),
    downscaled to fit within (max_width, max_height).

    Result is either linear light value or binary {0,255}.

    Optimized fast paths (99% of the data):
      - VIPS (if available): tried first for all formats (can downscale before NumPy materialization).
      - JPEG: OpenCV decode; PIL fallback for CMYK/odd cases.
      - TIFF (Group4, already binary): PIL; OpenCV fallback.

    Important notes:
      - (approximately) linearizes channel for intensity math by default
      - has an option to normalize the background, not active by default
      - does not read / apply ICC profiles (apparently not a common practice for OCR)

    Raises ImageDecodeError on unrecoverable failures.
    """
    if not isinstance(filename, str) or not filename:
        raise ImageDecodeError("filename must be a non-empty string")
    if not isinstance(image_bytes, (bytes, bytearray, memoryview)):
        raise ImageDecodeError("image_bytes must be bytes-like")

    ext = _ext_lower(filename)

    # Decide preferred decoder lane based on extension and typical reliability
    likely_tiff = ext in ("tif", "tiff")
    likely_jpeg = ext in ("jpg", "jpeg")

    gray: Optional[np.ndarray] = None
    likely_binary = False
    orig_h = 0
    orig_w = 0
    vips_used = False
    vips_did_downscale = False

    # Decoder lane 0: VIPS first (best-effort), but keep OpenCV's JPEG grayscale fast-path for normal-size JPEGs.
    # VIPS stays preferred for TIFFs and for any JPEG that would be downscaled (VIPS can resize before NumPy materialization).
    prefer_vips = _VIPS_AVAILABLE and (not likely_jpeg or likely_tiff)
    if _VIPS_AVAILABLE and likely_jpeg and not likely_tiff:
        dims = _jpeg_dimensions(image_bytes)
        if dims is not None:
            jw, jh = dims
            try:
                s = _compute_downscale(
                    jw,
                    jh,
                    max_width,
                    max_height,
                    patch_size,
                    patch_vertical_overlap_px,
                    snap_extra_patch_row_threshold_px,
                    max_patch_rows,
                )
                prefer_vips = (s < 1.0)
            except Exception:
                # If anything is odd, fall back to OpenCV-first for safety.
                prefer_vips = False

    if prefer_vips:
        try:
            gray, likely_binary, orig_h, orig_w, vips_did_downscale = _decode_via_vips(
                filename,
                image_bytes,
                max_width=max_width,
                max_height=max_height,
                patch_size=patch_size,
                patch_vertical_overlap_px=patch_vertical_overlap_px,
                snap_extra_patch_row_threshold_px=snap_extra_patch_row_threshold_px,
                max_patch_rows=max_patch_rows,
            )
            vips_used = True
        except Exception:
            gray = None
            likely_binary = False
            vips_used = False

    # Decoder lanes 1/2: existing OpenCV/PIL logic (fallback)
    if not vips_used:
        try:
            if likely_tiff:
                gray, likely_binary = _decode_via_pil(filename, image_bytes)
            else:
                img = _decode_via_cv2(image_bytes, likely_jpeg)
                gray = _to_gray_cv(img)
        except Exception as e1:
            try:
                if gray is None:
                    if likely_tiff:
                        # If PIL-first failed, try OpenCV
                        img = _decode_via_cv2(image_bytes, likely_jpeg)
                        gray = _to_gray_cv(img)
                    else:
                        # If OpenCV-first failed, try VIPS (if available), then PIL
                        if _VIPS_AVAILABLE:
                            try:
                                gray, likely_binary, orig_h, orig_w, vips_did_downscale = _decode_via_vips(
                                    filename,
                                    image_bytes,
                                    max_width=max_width,
                                    max_height=max_height,
                                    patch_size=patch_size,
                                    patch_vertical_overlap_px=patch_vertical_overlap_px,
                                    snap_extra_patch_row_threshold_px=snap_extra_patch_row_threshold_px,
                                    max_patch_rows=max_patch_rows,
                                )
                                vips_used = True
                            except Exception:
                                gray = None
                                likely_binary = False
                                vips_used = False
                        if gray is None:
                            gray, likely_binary = _decode_via_pil(filename, image_bytes)
            except Exception as e2:
                raise ImageDecodeError(
                    f"Failed to decode '{filename}': "
                    f"primary_error={type(e1).__name__}: {e1}; "
                    f"fallback_error={type(e2).__name__}: {e2}"
                ) from e2

    if gray is None:
        raise ImageDecodeError("Decoded image is None after decoding attempts")

    # Ensure 2D uint8 (may be a bit too defensive)
    if gray.ndim != 2:
        # Extremely defensive: force to gray if something slipped through
        gray = _to_gray_cv(gray)

    if gray.dtype != np.uint8:
        gray = np.clip(gray, 0, 255).astype(np.uint8, copy=False)

    if orig_h == 0 or orig_w == 0:
        orig_h, orig_w = gray.shape[:2]

    # If already binary, preserve it and just downscale with nearest-neighbor
    if likely_binary:
        if not vips_used:
            binary = _downscale_binary(
                gray,
                max_width,
                max_height,
                patch_size,
                patch_vertical_overlap_px,
                snap_extra_patch_row_threshold_px,
                max_patch_rows,
            )
            # Enforce exactly {0,255}
            if binary.size and binary.max() == 1:
                binary = (binary * 255).astype(np.uint8, copy=False)
            return np.ascontiguousarray(binary), True, orig_h, orig_w

        # VIPS path already applied downscale before NumPy materialization (if needed)
        if gray.size and gray.max() == 1:
            gray = gray.copy()
            gray *= 255
        return np.ascontiguousarray(gray), True, orig_h, orig_w

    # IMPORTANT: downscale BEFORE linearize to avoid huge temporary allocations on very large pages.
    # (VIPS already downscaled before NumPy materialization when used.)
    if not vips_used or not vips_did_downscale:
        gray = _downscale_gray(
            gray,
            max_width,
            max_height,
            patch_size,
            patch_vertical_overlap_px,
            snap_extra_patch_row_threshold_px,
            max_patch_rows,
        )

    if linearize:
        gray = _srgb_u8_to_linear_u8(gray)

    if normalize_background:
        gray = _normalize_background_u8(gray)

    # Enforce contiguous uint8 frame (skip if already contiguous - common after cv2 ops)
    if not gray.flags['C_CONTIGUOUS'] or gray.dtype != np.uint8:
        gray = np.ascontiguousarray(gray, dtype=np.uint8)
    return gray, False, orig_h, orig_w
