import logging
import os
import time
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from bec_orch.core.models import TaskResult
    from bec_orch.jobs.base import JobContext

import cv2
import numpy as np
import numpy.typing as npt
import pyarrow as pa
import pyarrow.parquet as pq
import s3fs

from .ctc_decoder import CTCDecoder
from .line import get_line_image
from .model import OCRModel

logger = logging.getLogger(__name__)


def ocr_build_schema() -> pa.Schema:
    """Build a PyArrow schema for Parquet output of OCR.

    Output columns:
    - img_file_name: string
    - source_etag: string
    - contours: list<list<struct<x:int16,y:int16>>>
    - texts: list<string>
    - ok: bool
    - error_stage: string (nullable)
    - error_type: string (nullable)
    - error_message: string (nullable)
    """
    point_struct = pa.struct([("x", pa.int16()), ("y", pa.int16())])
    contour = pa.list_(point_struct)
    contours = pa.list_(contour)
    schema = pa.schema(
        [
            ("img_file_name", pa.string()),
            ("source_etag", pa.string()),
            ("contours", contours),
            ("texts", pa.list_(pa.string())),
            ("ok", pa.bool_()),
            ("error_stage", pa.string()),
            ("error_type", pa.string()),
            ("error_message", pa.string()),
        ]
    )
    return schema


class OCRV1JobWorker:
    """
    Synchronous adapter for OCR inference to integrate with BEC orchestration.

    Implements the JobWorker protocol expected by BECWorkerRuntime.

    The model is loaded once in __init__ and reused across all volumes to avoid
    reloading overhead.
    """

    def __init__(self):
        """Initialize the job worker and load the OCR model.

        Reads model configuration from model_config.json in the model directory.
        Required env var: BEC_OCR_MODEL_DIR (path to directory containing model_config.json)
        """
        import json

        model_dir = os.environ.get("BEC_OCR_MODEL_DIR")
        if not model_dir:
            raise ValueError(
                "BEC_OCR_MODEL_DIR environment variable not set. "
                "Set it to the path of the OCR model directory containing model_config.json."
            )

        model_dir = model_dir.strip("\"'")
        model_dir_path = Path(model_dir)
        if not model_dir_path.exists():
            raise FileNotFoundError(
                f"OCR model directory not found: {model_dir}\nPlease check that BEC_OCR_MODEL_DIR is set correctly."
            )

        if not model_dir_path.is_dir():
            raise ValueError(
                f"BEC_OCR_MODEL_DIR is not a directory: {model_dir}\n"
                f"BEC_OCR_MODEL_DIR must point to a directory containing model_config.json."
            )

        # TODO: the model should come from the JobContext
        config_path = model_dir_path / "model_config.json"
        if not config_path.exists():
            raise FileNotFoundError(
                f"model_config.json not found in {model_dir}\nExpected config file at: {config_path}"
            )

        logger.info(f"Loading OCR model config from: {config_path}")

        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)

        onnx_model_file = model_dir_path / config["onnx-model"]
        if not onnx_model_file.exists():
            raise FileNotFoundError(
                f"ONNX model file not found: {onnx_model_file}\n"
                f"Referenced in model_config.json as: {config['onnx-model']}"
            )

        input_width = config["input_width"]
        input_height = config["input_height"]
        input_layer = config["input_layer"]
        output_layer = config["output_layer"]
        charset = config["charset"]
        squeeze_channel = config["squeeze_channel_dim"] == "yes"
        swap_hw = config["swap_hw"] == "yes"
        add_blank = config["add_blank"] == "yes"

        logger.info(f"Loading OCR model: {onnx_model_file}")
        logger.info(
            f"  Architecture: {config.get('architecture', 'unknown')}, Version: {config.get('version', 'unknown')}"
        )
        logger.info(f"  Input: {input_width}x{input_height}, Charset length: {len(charset)}")

        self.ocr_model = OCRModel(
            model_file=str(onnx_model_file),
            input_layer=input_layer,
            output_layer=output_layer,
            squeeze_channel=squeeze_channel,
            swap_hw=swap_hw,
        )

        self.ctc_decoder = CTCDecoder(charset=charset, add_blank=add_blank)

        self._input_width = input_width
        self._input_height = input_height

        self.ld_bucket = os.environ.get("BEC_LD_BUCKET", "bec.bdrc.io")
        self.source_image_bucket = os.environ.get("BEC_SOURCE_IMAGE_BUCKET", "archive.tbrc.org")

        self._s3 = s3fs.S3FileSystem()

        logger.info("OCR model loaded successfully")

    def run(self, ctx: "JobContext") -> "TaskResult":
        """
        Run OCR on a volume.

        Flow:
        1. Check if LD success.json exists for this volume/version
        2. Read LD parquet file
        3. Validate all manifest images exist in parquet and are ok
        4. For each image, apply transforms and run OCR
        5. Write output parquet

        Args:
            ctx: Job context with volume info, config, and artifact location

        Returns:
            TaskResult with metrics
        """
        from bec_orch.core.models import TaskResult
        from bec_orch.errors import TerminalTaskError
        from bec_orch.jobs.shared.memory_monitor import log_memory_snapshot

        logger.info(f"Starting OCRV1 job for volume {ctx.volume.w_id}/{ctx.volume.i_id}")
        start_time = time.time()

        log_memory_snapshot(f"[OCRV1] Start volume {ctx.volume.w_id}/{ctx.volume.i_id}")

        ld_success_uri = self._get_ld_success_uri(ctx)
        if not self._check_s3_exists(ld_success_uri):
            raise TerminalTaskError(
                f"Line detection not completed for volume {ctx.volume.w_id}/{ctx.volume.i_id}. "
                f"Expected success marker at: {ld_success_uri}"
            )
        logger.info(f"LD success marker found: {ld_success_uri}")

        ld_parquet_uri = self._get_ld_parquet_uri(ctx)
        output_parquet_uri = self._get_output_parquet_uri(ctx)

        total_images = 0
        nb_errors = 0
        errors_by_stage: dict[str, int] = {}
        records = []

        try:
            logger.info(f"Reading LD parquet from {ld_parquet_uri}")
            input_table = self._read_input_parquet(ld_parquet_uri)
            logger.info(f"Read {len(input_table)} rows from parquet")

            parquet_rows_by_filename: dict[str, dict] = {row["img_file_name"]: row for row in input_table.to_pylist()}
            logger.info(f"Indexed {len(parquet_rows_by_filename)} files from parquet")

            manifest_filenames: set[str] = {
                str(item["filename"]) for item in ctx.volume_manifest.manifest if item.get("filename")
            }

            missing_in_parquet = manifest_filenames - set(parquet_rows_by_filename.keys())
            if missing_in_parquet:
                raise TerminalTaskError(
                    f"LD parquet missing {len(missing_in_parquet)} images from manifest: "
                    f"{sorted(missing_in_parquet)[:5]}{'...' if len(missing_in_parquet) > 5 else ''}"
                )

            total_images = len(manifest_filenames)
            for idx, filename in enumerate(sorted(manifest_filenames), 1):
                row = parquet_rows_by_filename[filename]

                if not row.get("ok", True):
                    raise TerminalTaskError(
                        f"Image {filename} has LD error: {row.get('error_type', 'Unknown')} - {row.get('error_message', '')}"
                    )

                try:
                    logger.info(f"[{idx}/{total_images}] Processing {filename}")
                    result = self._process_image(row, ctx)
                    records.append(result)
                    nb_lines = len(result.get("texts", []))
                    logger.info(f"[{idx}/{total_images}] {filename}: {nb_lines} lines recognized")
                except Exception as e:
                    logger.warning(f"[{idx}/{total_images}] OCR failed for {filename}: {e}")
                    records.append(self._build_error_record(row, "OCR", type(e).__name__, str(e)))
                    nb_errors += 1
                    errors_by_stage["OCR"] = errors_by_stage.get("OCR", 0) + 1

            self._write_output_parquet(records, output_parquet_uri)

        except Exception as e:
            log_memory_snapshot(f"[OCRV1] FAILED volume {ctx.volume.w_id}/{ctx.volume.i_id}", level=logging.ERROR)

            error_type = type(e).__name__
            logger.error(
                f"OCRV1 pipeline failed for volume {ctx.volume.w_id}/{ctx.volume.i_id}: {error_type}: {e}",
                exc_info=True,
            )

            from bec_orch.errors import RetryableTaskError, TerminalTaskError

            if "CUDA out of memory" in str(e) or "OOM" in str(e):
                logger.error("GPU out of memory error detected - task will be retried")
                raise RetryableTaskError(f"GPU OOM error: {e}") from e
            elif "NotFound" in error_type or "404" in str(e):
                logger.error("Resource not found - task is terminal")
                raise TerminalTaskError(f"Resource not found: {e}") from e
            elif error_type == "FileNotFoundError":
                logger.error("File not found - task is terminal")
                raise TerminalTaskError(f"File not found: {e}") from e
            else:
                logger.warning(f"Unclassified error ({error_type}) - task will be retried")
                raise RetryableTaskError(f"Pipeline error ({error_type}): {e}") from e

        elapsed_ms = (time.time() - start_time) * 1000
        avg_duration_per_page_ms = elapsed_ms / max(1, total_images)

        log_memory_snapshot(f"[OCRV1] End volume {ctx.volume.w_id}/{ctx.volume.i_id}")

        logger.info(
            f"OCRV1 job completed: {total_images} images, {nb_errors} errors, avg {avg_duration_per_page_ms:.2f}ms/page"
        )

        from bec_orch.errors import TerminalTaskError

        if nb_errors > 0:
            raise TerminalTaskError(f"Volume had {nb_errors} processing errors. Errors by stage: {errors_by_stage}")

        return TaskResult(
            total_images=total_images,
            nb_errors=nb_errors,
            total_duration_ms=elapsed_ms,
            avg_duration_per_page_ms=avg_duration_per_page_ms,
            errors_by_stage=errors_by_stage if errors_by_stage else None,
        )

    def _get_ld_artifacts_prefix(self, ctx: "JobContext") -> str:
        """Get the LD artifacts prefix for this volume/version.

        LD artifacts are at: ldv1/{w_id}/{i_id}/{version}/
        OCR artifacts are at: ocrv1/{w_id}/{i_id}/{version}/
        We derive LD path from OCR path by replacing job name.
        """
        ocr_prefix = ctx.artifacts_location.prefix
        ld_prefix = ocr_prefix.replace("ocrv1/", "ldv1/", 1)
        return ld_prefix

    def _get_ld_success_uri(self, ctx: "JobContext") -> str:
        ld_prefix = self._get_ld_artifacts_prefix(ctx).rstrip("/")
        return f"s3://{self.ld_bucket}/{ld_prefix}/success.json"

    def _get_ld_parquet_uri(self, ctx: "JobContext") -> str:
        ld_prefix = self._get_ld_artifacts_prefix(ctx).rstrip("/")
        ld_basename = ctx.artifacts_location.basename
        return f"s3://{self.ld_bucket}/{ld_prefix}/{ld_basename}.parquet"

    def _get_output_parquet_uri(self, ctx: "JobContext") -> str:
        prefix = ctx.artifacts_location.prefix.rstrip("/")
        basename = ctx.artifacts_location.basename
        return f"s3://{ctx.artifacts_location.bucket}/{prefix}/{basename}.parquet"

    def _check_s3_exists(self, uri: str) -> bool:
        """Check if an S3 object exists."""
        s3_path = uri.replace("s3://", "")
        return self._s3.exists(s3_path)

    def _read_input_parquet(self, uri: str) -> pa.Table:
        return pq.read_table(uri)

    def _write_output_parquet(self, records: list[dict], uri: str) -> None:
        schema = ocr_build_schema()
        table = pa.Table.from_pylist(records, schema=schema)

        s3_path = uri.replace("s3://", "")
        with self._s3.open(s3_path, "wb") as f:
            pq.write_table(table, f)

        logger.info(f"Wrote {len(records)} records to {uri}")

    def _process_image(self, row: dict, ctx: "JobContext") -> dict:
        img_file_name = row.get("img_file_name", "")
        source_etag = row.get("source_etag", "")
        contours = row.get("contours", [])

        image = self._load_image(img_file_name, ctx)
        logger.debug(f"Loaded image shape={image.shape}, dtype={image.dtype}")

        image = self._transform_image(image, row)

        texts = self._extract_and_ocr_lines(image, contours)

        return self._build_record(img_file_name, source_etag, contours, texts)

    def _transform_image(self, image: npt.NDArray, row: dict) -> npt.NDArray:
        rotation_angle = row.get("rotation_angle")
        if rotation_angle is not None and (np.isnan(rotation_angle) or abs(rotation_angle) < 1e-6):
            rotation_angle = None

        tps_points = row.get("tps_points")
        tps_alpha = row.get("tps_alpha")
        if tps_alpha is not None and np.isnan(tps_alpha):
            tps_alpha = None

        logger.debug(f"Applying transforms (rotation={rotation_angle}, tps={tps_points is not None})")
        image = self._apply_transforms(image, rotation_angle, tps_points, tps_alpha)
        logger.debug(f"Transformed image shape={image.shape}")
        return image

    def _extract_and_ocr_lines(self, image: npt.NDArray, contours: list) -> list[str]:
        nb_lines = len(contours)
        logger.debug(f"Processing {nb_lines} line contours")

        if nb_lines == 0:
            return []

        texts = [""] * nb_lines
        current_k = 1.7  # Adaptive k_factor for morphological operations
        mask_buffer = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)

        # Process in batches to limit memory usage
        batch_size = 8
        batch_data: list[tuple[int, npt.NDArray, int]] = []

        for idx, contour_points in enumerate(contours):
            line_img, current_k = self._extract_line_from_contour(image, contour_points, current_k, mask_buffer)
            if line_img is None:
                continue

            original_width = line_img.shape[1]
            tensor = self._preprocess_line(line_img)
            batch_data.append((idx, tensor, original_width))

            # Process batch when full
            if len(batch_data) >= batch_size:
                self._process_batch(batch_data, texts)
                batch_data = []

        # Process remaining
        if batch_data:
            self._process_batch(batch_data, texts)

        return texts

    def _process_batch(self, batch_data: list[tuple[int, npt.NDArray, int]], texts: list[str]) -> None:
        """Process a batch of line tensors through GPU and CTC decoder."""
        if not batch_data:
            return

        # Stack tensors
        tensors = np.concatenate([t for _, t, _ in batch_data], axis=0)

        # Run batched inference
        batch_logits = self.ocr_model.predict(tensors)

        # Handle single item batch
        if len(batch_data) == 1:
            batch_logits = [batch_logits]

        # Decode each result
        for (idx, _, original_width), logits in zip(batch_data, batch_logits):
            cropped_logits = self._crop_logits(logits, original_width)
            text = self.ctc_decoder.decode(cropped_logits)
            texts[idx] = text.strip().replace("§", " ")

    def _extract_line_from_contour(
        self, image: npt.NDArray, contour_points: list[dict], k_factor: float, mask_buffer: npt.NDArray
    ) -> tuple[npt.NDArray | None, float]:
        """Extract line image from contour polygon using mask-based extraction."""
        if not contour_points:
            return None, k_factor

        # Convert list of {x, y} dicts to numpy array for cv2
        pts = np.array([[p["x"], p["y"]] for p in contour_points], dtype=np.int32)

        # Get bounding box height for adaptive extraction
        _, _, _, bbox_h = cv2.boundingRect(pts)
        if bbox_h <= 0:
            return None, k_factor

        # Clear and reuse mask buffer
        mask_buffer.fill(0)
        cv2.drawContours(mask_buffer, [pts], -1, 255, -1)

        # Extract line using morphological mask-based method
        line_img, adapted_k = get_line_image(image, mask_buffer, bbox_h, bbox_tolerance=3.0, k_factor=k_factor)

        if line_img.size == 0:
            return None, adapted_k

        return line_img, adapted_k

    def _ocr_line(self, line_img: npt.NDArray) -> str:
        """Preprocess line image and run OCR inference."""
        original_width = line_img.shape[1]
        tensor = self._preprocess_line(line_img)
        logits = self.ocr_model.predict(tensor)
        logits = self._crop_logits(logits, original_width)
        text = self.ctc_decoder.decode(logits)
        return text.strip().replace("§", " ")

    def _crop_logits(self, logits: npt.NDArray, original_width: int) -> npt.NDArray:
        """Crop logits timesteps to avoid CTC decoding on padding."""
        vocab_size = len(self.ctc_decoder.ctc_vocab)
        time_axis = 1 if logits.shape[0] == vocab_size else 0
        total_timesteps = logits.shape[time_axis]
        crop_timesteps = max(1, int(total_timesteps * original_width / self._input_width))

        if time_axis == 1:
            return logits[:, :crop_timesteps]
        return logits[:crop_timesteps, :]

    def _preprocess_line(self, image: npt.NDArray) -> npt.NDArray:
        """Convert line image to model input tensor.

        Steps: pad to aspect ratio -> binarize -> grayscale -> normalize
        """
        image = self._pad_to_model_size(image)
        image = self._binarize(image)

        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        image = image.reshape((1, self._input_height, self._input_width))
        image = (image / 127.5) - 1.0
        return image.astype(np.float32)

    def _pad_to_model_size(self, img: npt.NDArray, padding: str = "black") -> npt.NDArray:
        """Resize and pad image to model input dimensions."""
        width_ratio = self._input_width / img.shape[1]
        height_ratio = self._input_height / img.shape[0]

        if width_ratio <= height_ratio:
            img = self._pad_to_width(img, padding)
        else:
            img = self._pad_to_height(img, padding)

        return cv2.resize(img, (self._input_width, self._input_height), interpolation=cv2.INTER_LINEAR)

    def _pad_to_width(self, img: npt.NDArray, padding: str) -> npt.NDArray:
        h, w = img.shape[:2]
        scale = self._input_width / w
        new_h = int(h * scale)
        resized = cv2.resize(img, (self._input_width, new_h), interpolation=cv2.INTER_LINEAR)

        if new_h >= self._input_height:
            return resized

        pad_top = (self._input_height - new_h) // 2
        pad_bottom = self._input_height - new_h - pad_top
        pad_value = 0 if padding == "black" else 255

        if len(resized.shape) == 3:
            return np.pad(resized, ((pad_top, pad_bottom), (0, 0), (0, 0)), mode="constant", constant_values=pad_value)
        return np.pad(resized, ((pad_top, pad_bottom), (0, 0)), mode="constant", constant_values=pad_value)

    def _pad_to_height(self, img: npt.NDArray, padding: str) -> npt.NDArray:
        h, w = img.shape[:2]
        scale = self._input_height / h
        new_w = int(w * scale)
        resized = cv2.resize(img, (new_w, self._input_height), interpolation=cv2.INTER_LINEAR)

        if new_w >= self._input_width:
            return resized

        pad_left = (self._input_width - new_w) // 2
        pad_right = self._input_width - new_w - pad_left
        pad_value = 0 if padding == "black" else 255

        if len(resized.shape) == 3:
            return np.pad(resized, ((0, 0), (pad_left, pad_right), (0, 0)), mode="constant", constant_values=pad_value)
        return np.pad(resized, ((0, 0), (pad_left, pad_right)), mode="constant", constant_values=pad_value)

    def _binarize(self, img: npt.NDArray) -> npt.NDArray:
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        else:
            gray = img
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        if len(img.shape) == 3:
            return cv2.cvtColor(binary, cv2.COLOR_GRAY2RGB)
        return binary

    def _build_record(
        self,
        img_file_name: str,
        source_etag: str,
        contours: list,
        texts: list[str],
    ) -> dict:
        logger.debug(f"Building output record with {len(texts)} texts")
        return {
            "img_file_name": img_file_name,
            "source_etag": source_etag,
            "contours": contours,
            "texts": texts,
            "ok": True,
            "error_stage": None,
            "error_type": None,
            "error_message": None,
        }

    def _load_image(self, img_file_name: str, ctx: "JobContext") -> npt.NDArray:
        from bec_orch.core.worker_runtime import get_s3_folder_prefix

        vol_prefix = get_s3_folder_prefix(ctx.volume.w_id, ctx.volume.i_id)
        s3_path = f"{self.source_image_bucket}/{vol_prefix}{img_file_name}"

        with self._s3.open(s3_path, "rb") as f:
            img_bytes = f.read()

        img_array = np.frombuffer(img_bytes, dtype=np.uint8)
        image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Failed to decode image: {img_file_name}")

        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    def _apply_transforms(
        self,
        image: npt.NDArray,
        rotation_angle: float | None,
        tps_points: list | None,
        tps_alpha: float | None,
    ) -> npt.NDArray:
        if rotation_angle is not None:
            image = self._apply_rotation(image, rotation_angle)

        if tps_points is not None:
            tps_data = self._deserialize_tps_points(tps_points)
            if tps_data is not None:
                input_pts, output_pts = tps_data
                alpha = float(tps_alpha) if tps_alpha is not None else 0.5
                image = self._apply_tps(image, input_pts, output_pts, alpha)

        return image

    def _apply_rotation(self, img: npt.NDArray, angle_deg: float) -> npt.NDArray:
        if angle_deg is None or abs(float(angle_deg)) < 1e-12:
            return img

        h, w = img.shape[:2]
        center = (w / 2.0, h / 2.0)
        M = cv2.getRotationMatrix2D(center, float(angle_deg), 1.0).astype(np.float32)

        if len(img.shape) == 3:
            rotated = cv2.warpAffine(
                img,
                M,
                (w, h),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=(0, 0, 0),
            )
        else:
            rotated = cv2.warpAffine(
                img,
                M,
                (w, h),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0,
            )
        return np.ascontiguousarray(rotated)

    def _apply_tps(
        self,
        img: npt.NDArray,
        input_pts: npt.NDArray,
        output_pts: npt.NDArray,
        alpha: float = 0.5,
    ) -> npt.NDArray:
        import scipy.ndimage
        from tps import ThinPlateSpline

        h, w = img.shape[:2]
        input_pts = np.asarray(input_pts, dtype=np.float64)
        output_pts = np.asarray(output_pts, dtype=np.float64)

        tps = ThinPlateSpline(alpha)
        tps.fit(output_pts, input_pts)

        out_grid = np.indices((h, w), dtype=np.float64).transpose(1, 2, 0)
        in_coords = tps.transform(out_grid.reshape(-1, 2)).reshape(h, w, 2)
        coords = in_coords.transpose(2, 0, 1)

        if len(img.shape) == 3:
            out = np.empty_like(img)
            for ch in range(img.shape[2]):
                warped = scipy.ndimage.map_coordinates(img[..., ch], coords, order=1, mode="constant", cval=0)
                out[..., ch] = np.clip(warped, 0, 255).astype(np.uint8)
            return np.ascontiguousarray(out)
        else:
            warped = scipy.ndimage.map_coordinates(img, coords, order=1, mode="constant", cval=0)
            warped = np.clip(warped, 0, 255).astype(np.uint8)
            return np.ascontiguousarray(warped)

    def _deserialize_tps_points(self, tps_points: list) -> tuple[npt.NDArray, npt.NDArray] | None:
        if tps_points is None or len(tps_points) == 0:
            return None

        input_pts = []
        output_pts = []
        for pt in tps_points:
            if len(pt) >= 4:
                input_pts.append([pt[0], pt[1]])
                output_pts.append([pt[2], pt[3]])

        if len(input_pts) == 0:
            return None

        return np.array(input_pts, dtype=np.float64), np.array(output_pts, dtype=np.float64)

    def _build_error_record(self, row: dict, stage: str, error_type: str, error_message: str) -> dict:
        return {
            "img_file_name": row.get("img_file_name", ""),
            "source_etag": row.get("source_etag", ""),
            "contours": row.get("contours", []),
            "texts": [],
            "ok": False,
            "error_stage": stage,
            "error_type": error_type,
            "error_message": error_message[:1000] if error_message else None,
        }
