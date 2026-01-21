import logging
import os
import time
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from bec_orch.jobs.base import JobContext
    from bec_orch.core.models import TaskResult

import cv2
import numpy as np
import numpy.typing as npt
import onnxruntime as ort
import pyarrow as pa
import pyarrow.parquet as pq
import pyewts
import s3fs

from .utils import get_execution_providers
from pyctcdecode.decoder import build_ctcdecoder

logger = logging.getLogger(__name__)


class OCRInference:
    def __init__(
        self,
        model_file: str,
        input_width: int,
        input_height: int,
        input_layer: str,
        output_layer: str,
        charset: str | list[str],
        squeeze_channel: bool,
        swap_hw: bool,
        add_blank: bool,
    ) -> None:
        self._onnx_model_file = model_file
        self._input_width = input_width
        self._input_height = input_height
        self._input_layer = input_layer
        self._output_layer = output_layer
        self._characters = charset
        self._squeeze_channel_dim = squeeze_channel
        self._swap_hw = swap_hw
        self._add_blank = add_blank
        self._execution_providers = get_execution_providers()
        self.ocr_session = ort.InferenceSession(self._onnx_model_file, providers=self._execution_providers)


        if isinstance(charset, str):
            self.charset = list(charset)
        else:
            self.charset = charset

        self.ctc_vocab = self.charset.copy()
        if add_blank and " " not in self.ctc_vocab:
            self.ctc_vocab.insert(0, " ")

        self.ctc_decoder = build_ctcdecoder(self.ctc_vocab)

    def _pad_ocr_line(
        self,
        img: npt.NDArray,
        padding: str = "black",
    ) -> npt.NDArray:
        width_ratio = self._input_width / img.shape[1]
        height_ratio = self._input_height / img.shape[0]

        if width_ratio < height_ratio:
            out_img = self._pad_to_width(img, self._input_width, self._input_height, padding)
        elif width_ratio > height_ratio:
            out_img = self._pad_to_height(img, self._input_width, self._input_height, padding)
        else:
            out_img = self._pad_to_width(img, self._input_width, self._input_height, padding)

        return cv2.resize(
            out_img,
            (self._input_width, self._input_height),
            interpolation=cv2.INTER_LINEAR,
        )

    def _pad_to_width(
        self, img: npt.NDArray, target_width: int, target_height: int, padding: str
    ) -> npt.NDArray:
        h, w = img.shape[:2]
        scale = target_width / w
        new_h = int(h * scale)
        resized = cv2.resize(img, (target_width, new_h), interpolation=cv2.INTER_LINEAR)

        if new_h >= target_height:
            return resized

        pad_top = (target_height - new_h) // 2
        pad_bottom = target_height - new_h - pad_top
        pad_value = 0 if padding == "black" else 255
        if len(resized.shape) == 3:
            padded = np.pad(
                resized,
                ((pad_top, pad_bottom), (0, 0), (0, 0)),
                mode="constant",
                constant_values=pad_value,
            )
        else:
            padded = np.pad(
                resized,
                ((pad_top, pad_bottom), (0, 0)),
                mode="constant",
                constant_values=pad_value,
            )
        return padded

    def _pad_to_height(
        self, img: npt.NDArray, target_width: int, target_height: int, padding: str
    ) -> npt.NDArray:
        h, w = img.shape[:2]
        scale = target_height / h
        new_w = int(w * scale)
        resized = cv2.resize(img, (new_w, target_height), interpolation=cv2.INTER_LINEAR)

        if new_w >= target_width:
            return resized

        pad_left = (target_width - new_w) // 2
        pad_right = target_width - new_w - pad_left
        pad_value = 0 if padding == "black" else 255
        if len(resized.shape) == 3:
            padded = np.pad(
                resized,
                ((0, 0), (pad_left, pad_right), (0, 0)),
                mode="constant",
                constant_values=pad_value,
            )
        else:
            padded = np.pad(
                resized,
                ((0, 0), (pad_left, pad_right)),
                mode="constant",
                constant_values=pad_value,
            )
        return padded

    def _binarize(self, img: npt.NDArray) -> npt.NDArray:
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        else:
            gray = img
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        if len(img.shape) == 3:
            return cv2.cvtColor(binary, cv2.COLOR_GRAY2RGB)
        return binary

    def _prepare_ocr_line(self, image: npt.NDArray) -> npt.NDArray:
        line_image = self._pad_ocr_line(image)
        line_image = self._binarize(line_image)

        if len(line_image.shape) == 3:
            line_image = cv2.cvtColor(line_image, cv2.COLOR_RGB2GRAY)

        line_image = line_image.reshape((1, self._input_height, self._input_width))
        line_image = (line_image / 127.5) - 1.0
        return line_image.astype(np.float32)

    def _pre_pad(self, image: npt.NDArray) -> npt.NDArray:
        h, w = image.shape[:2]
        if len(image.shape) == 3:
            c = image.shape[2]
            patch = np.ones(shape=(h, h, c), dtype=np.uint8) * 255
        else:
            patch = np.ones(shape=(h, h), dtype=np.uint8) * 255
        return np.hstack(tup=[patch, image, patch])

    def _predict(self, image_batch: npt.NDArray) -> npt.NDArray:
        image_batch = image_batch.astype(np.float32)
        ort_batch = ort.OrtValue.ortvalue_from_numpy(image_batch)
        ocr_results = self.ocr_session.run_with_ort_values(
            [self._output_layer], {self._input_layer: ort_batch}
        )
        logits = ocr_results[0].numpy()
        return np.squeeze(logits)

    def _decode(self, logits: npt.NDArray) -> str:
        if logits.shape[0] == len(self.ctc_vocab):
            logits = np.transpose(logits, axes=[1, 0])
        return self.ctc_decoder.decode(logits).replace(" ", "")

    def run(self, line_image: npt.NDArray, *, pre_pad: bool = True) -> str:
        if pre_pad:
            line_image = self._pre_pad(line_image)
        line_image = self._prepare_ocr_line(line_image)

        if self._swap_hw:
            line_image = np.transpose(line_image, axes=[0, 2, 1])

        if not self._squeeze_channel_dim:
            line_image = np.expand_dims(line_image, axis=1)

        logits = self._predict(line_image)
        return self._decode(logits)


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
    schema = pa.schema([
        ("img_file_name", pa.string()),
        ("source_etag", pa.string()),
        ("contours", contours),
        ("texts", pa.list_(pa.string())),
        ("ok", pa.bool_()),
        ("error_stage", pa.string()),
        ("error_type", pa.string()),
        ("error_message", pa.string()),
    ])
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
                f"OCR model directory not found: {model_dir}\n"
                f"Please check that BEC_OCR_MODEL_DIR is set correctly."
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
                f"model_config.json not found in {model_dir}\n"
                f"Expected config file at: {config_path}"
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
        logger.info(f"  Architecture: {config.get('architecture', 'unknown')}, Version: {config.get('version', 'unknown')}")
        logger.info(f"  Input: {input_width}x{input_height}, Charset length: {len(charset)}")

        self.ocr_inference = OCRInference(
            model_file=str(onnx_model_file),
            input_width=input_width,
            input_height=input_height,
            input_layer=input_layer,
            output_layer=output_layer,
            charset=charset,
            squeeze_channel=squeeze_channel,
            swap_hw=swap_hw,
            add_blank=add_blank,
        )

        self.converter = pyewts.pyewts()
        # TODO: make this configurable
        self.use_line_prepadding = False

        self.ld_bucket = os.environ.get("BEC_LD_BUCKET", "bec.bdrc.io")
        self.source_image_bucket = os.environ.get("BEC_SOURCE_IMAGE_BUCKET", "archive.tbrc.org")

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
            input_table = self._read_input_parquet(ld_parquet_uri)

            parquet_filenames = set()
            parquet_rows_by_filename: dict[str, dict] = {}
            for row_idx in range(len(input_table)):
                row = {col: input_table[col][row_idx].as_py() for col in input_table.column_names}
                filename = row.get("img_file_name", "")
                parquet_filenames.add(filename)
                parquet_rows_by_filename[filename] = row

            manifest_filenames: set[str] = {
                str(item["filename"])
                for item in ctx.volume_manifest.manifest
                if item.get("filename")
            }
            total_images = len(manifest_filenames)

            missing_in_parquet = manifest_filenames - parquet_filenames
            if missing_in_parquet:
                raise TerminalTaskError(
                    f"LD parquet missing {len(missing_in_parquet)} images from manifest: "
                    f"{sorted(missing_in_parquet)[:5]}{'...' if len(missing_in_parquet) > 5 else ''}"
                )

            for filename in sorted(manifest_filenames):
                row = parquet_rows_by_filename[filename]

                if not row.get("ok", True):
                    raise TerminalTaskError(
                        f"Image {filename} has LD error: {row.get('error_type', 'Unknown')} - {row.get('error_message', '')}"
                    )

                try:
                    result = self._process_image(row, ctx)
                    records.append(result)
                except Exception as e:
                    logger.warning(f"OCR failed for {filename}: {e}")
                    records.append(self._build_error_record(row, "OCR", type(e).__name__, str(e)))
                    nb_errors += 1
                    errors_by_stage["OCR"] = errors_by_stage.get("OCR", 0) + 1

            self._write_output_parquet(records, output_parquet_uri)

        except Exception as e:
            log_memory_snapshot(f"[OCRV1] FAILED volume {ctx.volume.w_id}/{ctx.volume.i_id}", level=logging.ERROR)

            error_type = type(e).__name__
            logger.error(
                f"OCRV1 pipeline failed for volume {ctx.volume.w_id}/{ctx.volume.i_id}: "
                f"{error_type}: {e}",
                exc_info=True
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
            f"OCRV1 job completed: {total_images} images, {nb_errors} errors, "
            f"avg {avg_duration_per_page_ms:.2f}ms/page"
        )

        from bec_orch.errors import TerminalTaskError

        if nb_errors > 0:
            raise TerminalTaskError(
                f"Volume had {nb_errors} processing errors. "
                f"Errors by stage: {errors_by_stage}"
            )

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
        s3 = s3fs.S3FileSystem()
        s3_path = uri.replace("s3://", "")
        return s3.exists(s3_path)

    def _read_input_parquet(self, uri: str) -> pa.Table:
        s3 = s3fs.S3FileSystem()
        s3_path = uri.replace("s3://", "")
        with s3.open(s3_path, "rb") as f:
            table = pq.read_table(f)
        return table

    def _write_output_parquet(self, records: list[dict], uri: str) -> None:
        schema = ocr_build_schema()
        table = pa.Table.from_pylist(records, schema=schema)

        s3 = s3fs.S3FileSystem()
        s3_path = uri.replace("s3://", "")
        with s3.open(s3_path, "wb") as f:
            pq.write_table(table, f)

        logger.info(f"Wrote {len(records)} records to {uri}")

    def _process_image(self, row: dict, ctx: "JobContext") -> dict:
        img_file_name = row.get("img_file_name", "")
        source_etag = row.get("source_etag", "")
        contours = row.get("contours", [])
        contours_bboxes = row.get("contours_bboxes", [])

        image = self._load_image(img_file_name, ctx)

        rotation_angle = row.get("rotation_angle")
        if rotation_angle is not None and (np.isnan(rotation_angle) or abs(rotation_angle) < 1e-6):
            rotation_angle = None

        tps_points = row.get("tps_points")
        tps_alpha = row.get("tps_alpha")
        if tps_alpha is not None and np.isnan(tps_alpha):
            tps_alpha = None

        image = self._apply_transforms(image, rotation_angle, tps_points, tps_alpha)

        texts = []
        for idx, bbox in enumerate(contours_bboxes):
            x = bbox.get("x", 0)
            y = bbox.get("y", 0)
            w = bbox.get("w", 0)
            h = bbox.get("h", 0)

            if w <= 0 or h <= 0:
                texts.append("")
                continue

            line_img = image[y : y + h, x : x + w]
            if line_img.size == 0:
                texts.append("")
                continue

            text = self.ocr_inference.run(line_img, pre_pad=self.use_line_prepadding)
            text = text.strip().replace("§", " ")
            texts.append(text)

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
        s3_uri = f"s3://{self.source_image_bucket}/{vol_prefix}{img_file_name}"

        s3 = s3fs.S3FileSystem()
        s3_path = s3_uri.replace("s3://", "")
        with s3.open(s3_path, "rb") as f:
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
                img, M, (w, h),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=(0, 0, 0),
            )
        else:
            rotated = cv2.warpAffine(
                img, M, (w, h),
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
                warped = scipy.ndimage.map_coordinates(
                    img[..., ch], coords, order=1, mode="constant", cval=0
                )
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
