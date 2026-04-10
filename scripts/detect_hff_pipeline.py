"""HFF Detection Pipeline

Stream images directly from S3, detect header/footer/footnote/body regions
using the Surya layout model, and write per-volume detection results as parquet.

No images are written to disk.  Only the detection bboxes (~150 KB per volume)
are persisted — roughly 3 GB total across 18,000 volumes.

Architecture
------------
    S3Prefetcher  →  DecodeStage  →  DetectStage  →  ParquetWriterStage
    (S3 bytes)       (bytes→BGR)     (Surya GPU)      (bbox rows → parquet)

    Bounded queues between every stage enforce backpressure so a slow
    DetectStage never causes the prefetcher to consume unbounded RAM.

    The Surya model is loaded ONCE before the volume loop and reused
    across all volumes (avoids the ~30 s reload cost per volume).

Usage
-----
    # single volume
    python scripts/detect_hff_pipeline.py \\
        --w-id W22084 --i-id I0886 --output-dir ./output

    # batch from CSV (columns: w_id, i_id)
    python scripts/detect_hff_pipeline.py \\
        --csv volumes.csv --output-dir ./output

    # write parquet directly to S3
    python scripts/detect_hff_pipeline.py \\
        --w-id W22084 --i-id I0886 \\
        --output-s3 s3://my-bucket/hff-detections/

    # dry-run (print what would be processed, no S3 or model calls)
    python scripts/detect_hff_pipeline.py \\
        --w-id W22084 --i-id I0886 --output-dir ./output --dry-run

Output parquet schema (one row per image)
-----------------------------------------
    w_id, i_id, filename
    header_boxes    : JSON  [{bbox:[x1,y1,x2,y2], confidence:float}, ...]
    footer_boxes    : JSON
    footnote_boxes  : JSON
    body_boxes      : JSON  (text-area detections)
    n_header, n_footer, n_footnote, n_body   : int
    has_header, has_footer, has_body         : bool
    duration_ms     : float
    error           : str   (empty on success)
"""
from __future__ import annotations

import sys
import os

# Make `bec_orch` importable when running as a script without installing the package.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import asyncio
import contextlib
import csv
import gzip
import hashlib
import io
import json
import logging
import re
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


# ── Config ────────────────────────────────────────────────────────────────────

@dataclass
class DetectConfig:
    """Configuration for the HFF detection pipeline."""

    # S3 source
    s3_region: str = "us-east-1"
    s3_bucket: str = "archive.tbrc.org"
    aws_profile: str = "default"

    # S3Prefetcher tuning (reuses existing bulk / streaming modes)
    bulk_prefetch: bool = True
    bulk_prefetch_concurrency: int = 32
    inflight_per_worker: int = 8          # used only in streaming mode

    # Surya detection
    confidence_threshold: float = 0.5
    detect_batch_size: int = 8            # images accumulated before one GPU call
    margin: int = 0                       # merge-nearby-detections gap (px)
    filter_to_hff_only: bool = False      # False → include text-area (body)

    # Safety
    volume_timeout_s: float = 3600.0


# ── Internal message types ────────────────────────────────────────────────────

class _EOS:
    """End-of-stream sentinel for the decode / detect / writer queues."""


_END_OF_STREAM = _EOS()


@dataclass
class DecodedImage:
    task: Any           # ImageTask
    bgr: np.ndarray     # H × W × 3, uint8 BGR


@dataclass
class DetectionResult:
    task: Any           # ImageTask
    detections: List[Dict[str, Any]]   # [{bbox, class_id, class_name, confidence}]
    duration_ms: float
    error: str          # empty string on success


# ── S3 / manifest helpers ─────────────────────────────────────────────────────

def _build_s3_prefix(w_id: str, i_id: str) -> str:
    """Return the S3 key prefix for a BDRC volume's images."""
    md5_2 = hashlib.md5(w_id.encode("utf-8")).hexdigest()[:2]
    i_id_m = i_id[1:] if re.fullmatch(r"I\d{4}", i_id) else i_id
    return f"Works/{md5_2}/{w_id}/images/{w_id}-{i_id_m}/"


def _load_manifest(s3_client, bucket: str, prefix: str) -> List[str]:
    """Fetch dimensions.json and return ordered list of image filenames."""
    key = f"{prefix}dimensions.json"
    logger.info(f"[Manifest] Fetching s3://{bucket}/{key}")

    resp = s3_client.get_object(Bucket=bucket, Key=key)
    payload = resp["Body"].read()

    try:
        data = json.loads(gzip.decompress(payload).decode("utf-8"))
    except OSError:
        data = json.loads(payload.decode("utf-8"))

    if not isinstance(data, list):
        raise RuntimeError(f"dimensions.json must be a JSON array, got {type(data)}")

    image_re = re.compile(r".*\.(jpg|jpeg|tif|tiff|png)$", re.IGNORECASE)
    kept: List[Tuple[Optional[int], str]] = []
    for item in data:
        fn = item.get("filename") if isinstance(item, dict) else None
        if not fn or not image_re.fullmatch(fn):
            continue
        stem = Path(fn).stem
        m = re.search(r"(\d+)$", stem) or re.search(r"(\d+)", stem)
        kept.append((int(m.group(1)) if m else None, fn))

    kept.sort(key=lambda r: (r[0] is None, r[0] or 0, r[1]))
    filenames = [fn for _, fn in kept]
    logger.info(f"[Manifest] {len(filenames)} image(s) found")
    return filenames


# ── Stage 1: DecodeStage ──────────────────────────────────────────────────────

class DecodeStage:
    """Reads FetchedBytes from q_in, decodes bytes → BGR numpy, pushes to q_out.

    Prefetch errors (PipelineError) are forwarded as DetectionResult(error=…)
    so the ParquetWriterStage still records one row per image.
    """

    def __init__(self, q_in: asyncio.Queue, q_out: asyncio.Queue) -> None:
        self.q_in = q_in
        self.q_out = q_out
        self._decoded = 0
        self._errors = 0

    @staticmethod
    def _decode_bytes(file_bytes: bytes) -> np.ndarray:
        pil = Image.open(io.BytesIO(file_bytes)).convert("RGB")
        return np.array(pil)[:, :, ::-1]   # RGB → BGR (OpenCV / HFF convention)

    async def run(self) -> None:
        from bec_orch.jobs.ldv1.types_common import EndOfStream, FetchedBytes, PipelineError

        while True:
            msg = await self.q_in.get()

            if isinstance(msg, EndOfStream):
                logger.info(f"[Decode] Done — decoded={self._decoded} errors={self._errors}")
                await self.q_out.put(_END_OF_STREAM)
                return

            if isinstance(msg, PipelineError):
                self._errors += 1
                logger.warning(f"[Decode] Prefetch error {msg.task.img_filename}: {msg.message}")
                await self.q_out.put(DetectionResult(
                    task=msg.task, detections=[], duration_ms=0.0,
                    error=f"prefetch:{msg.error_type}:{msg.message}",
                ))
                continue

            assert isinstance(msg, FetchedBytes)
            try:
                bgr = await asyncio.to_thread(self._decode_bytes, msg.file_bytes)
                self._decoded += 1
                await self.q_out.put(DecodedImage(task=msg.task, bgr=bgr))
            except Exception as exc:
                self._errors += 1
                logger.warning(f"[Decode] Failed {msg.task.img_filename}: {exc}")
                await self.q_out.put(DetectionResult(
                    task=msg.task, detections=[], duration_ms=0.0,
                    error=f"decode:{exc}",
                ))

            if (self._decoded + self._errors) % 200 == 0:
                logger.info(f"[Decode] Progress: decoded={self._decoded} errors={self._errors}")


# ── Stage 2: DetectStage ──────────────────────────────────────────────────────

class DetectStage:
    """Accumulates images into batches and runs Surya detection.

    The blocking Surya call runs in a dedicated ThreadPoolExecutor(1) so the
    asyncio event loop stays free for I/O during inference.

    The model is injected (pre-loaded outside the volume loop) so it is
    loaded only once regardless of how many volumes are processed.
    """

    def __init__(
        self,
        cfg: DetectConfig,
        detector: Any,      # SuryaLayoutDetector (pre-loaded)
        processor: Any,     # HFFProcessor (pre-loaded)
        q_in: asyncio.Queue,
        q_out: asyncio.Queue,
    ) -> None:
        self.cfg = cfg
        self._detector = detector
        self._processor = processor
        self.q_in = q_in
        self.q_out = q_out
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="surya")
        self._detected = 0
        self._errors = 0

    async def run(self) -> None:
        loop = asyncio.get_event_loop()
        batch: List[DecodedImage] = []

        while True:
            msg = await self.q_in.get()

            if isinstance(msg, _EOS):
                # Flush remaining images
                if batch:
                    results = await loop.run_in_executor(
                        self._executor, self._run_batch, list(batch)
                    )
                    for r in results:
                        await self.q_out.put(r)
                logger.info(
                    f"[Detect] Done — detected={self._detected} errors={self._errors}"
                )
                await self.q_out.put(_END_OF_STREAM)
                self._executor.shutdown(wait=False)
                return

            if isinstance(msg, DetectionResult):
                # Error forwarded from DecodeStage — pass straight through
                await self.q_out.put(msg)
                continue

            assert isinstance(msg, DecodedImage)
            batch.append(msg)

            if len(batch) >= self.cfg.detect_batch_size:
                results = await loop.run_in_executor(
                    self._executor, self._run_batch, list(batch)
                )
                for r in results:
                    await self.q_out.put(r)
                batch = []

    def _run_batch(self, batch: List[DecodedImage]) -> List[DetectionResult]:
        """Blocking: run Surya detection on each image in the batch.

        Runs the full batch in a single thread so I/O stages (prefetcher,
        decoder) can continue filling their queues while inference runs.
        """
        results: List[DetectionResult] = []
        t_batch = time.perf_counter()

        for item in batch:
            t0 = time.perf_counter()
            try:
                dets = self._detector.detect(
                    item.bgr,
                    filter_to_hff_only=self.cfg.filter_to_hff_only,
                    normalize_bbox=False,
                )
                print("dets>>>>>>>>>>>>>>>>>", dets)
                # dets = self._processor.merge_nearby_detections(dets)
                duration_ms = (time.perf_counter() - t0) * 1000
                results.append(DetectionResult(
                    task=item.task,
                    detections=dets,
                    duration_ms=round(duration_ms, 1),
                    error="",
                ))
                self._detected += 1
            except Exception as exc:
                duration_ms = (time.perf_counter() - t0) * 1000
                logger.warning(f"[Detect] Failed {item.task.img_filename}: {exc}")
                results.append(DetectionResult(
                    task=item.task, detections=[],
                    duration_ms=round(duration_ms, 1),
                    error=str(exc),
                ))
                self._errors += 1

        total_ms = (time.perf_counter() - t_batch) * 1000
        logger.info(
            f"[Detect] Batch {len(batch)} images in {total_ms:.0f}ms "
            f"({total_ms / max(len(batch), 1):.0f} ms/image) "
            f"[total detected={self._detected}]"
        )
        return results


# ── Stage 3: ParquetWriterStage ───────────────────────────────────────────────

def _build_parquet_row(
    w_id: str,
    i_id: str,
    result: DetectionResult,
) -> Dict[str, Any]:
    """Convert a DetectionResult into one parquet row with per-class JSON columns."""
    by_class: Dict[str, List[Dict[str, Any]]] = {
        "header": [], "footer": [], "footnote": [], "text-area": [],
    }
    for d in result.detections:
        cn = d.get("class_name", "")
        if cn in by_class:
            by_class[cn].append({
                "bbox": [round(v, 1) for v in d["bbox"]],
                "confidence": round(float(d.get("confidence", 1.0)), 4),
            })

    return {
        "w_id":           w_id,
        "i_id":           i_id,
        "filename":       result.task.img_filename,
        "header_boxes":   json.dumps(by_class["header"]),
        "footer_boxes":   json.dumps(by_class["footer"]),
        "footnote_boxes": json.dumps(by_class["footnote"]),
        "body_boxes":     json.dumps(by_class["text-area"]),
        "n_header":       len(by_class["header"]),
        "n_footer":       len(by_class["footer"]),
        "n_footnote":     len(by_class["footnote"]),
        "n_body":         len(by_class["text-area"]),
        "has_header":     len(by_class["header"]) > 0,
        "has_footer":     len(by_class["footer"]) > 0,
        "has_body":       len(by_class["text-area"]) > 0,
        "duration_ms":    result.duration_ms,
        "error":          result.error,
    }


class ParquetWriterStage:
    """Writes one JSONL row per detected image (durable after every image),
    then converts to parquet when the volume finishes.

    Local output per volume:
        detections.jsonl    — appended after every image (crash-safe, readable live)
        detections.parquet  — written once at end of volume (complete, compact)

    S3 output: parquet uploaded once at end of volume.

    The JSONL file is kept alongside the parquet so partial results are
    available immediately even if the volume crashes mid-way.
    """

    def __init__(
        self,
        w_id: str,
        i_id: str,
        output_dir: Optional[Path],
        output_s3_prefix: Optional[str],
        s3_client: Any,
        q_in: asyncio.Queue,
    ) -> None:
        self.w_id = w_id
        self.i_id = i_id
        self.output_dir = output_dir
        self.output_s3_prefix = output_s3_prefix
        self.s3_client = s3_client
        self.q_in = q_in
        self._count = 0
        self._jsonl_fh: Optional[Any] = None   # file handle for incremental writes

    def _vol_dir(self) -> Path:
        """Return (and create) the per-volume output directory."""
        d = self.output_dir / self.w_id / self.i_id  # type: ignore[operator]
        d.mkdir(parents=True, exist_ok=True)
        return d

    async def run(self) -> int:
        """Drain q_in, append JSONL after every image, write parquet at EOS."""

        # Open JSONL file if writing locally
        if self.output_dir is not None:
            vol_dir = await asyncio.to_thread(self._vol_dir)
            jsonl_path = vol_dir / "detections.jsonl"
            self._jsonl_fh = open(jsonl_path, "a", encoding="utf-8")  # noqa: SIM115

        try:
            while True:
                msg = await self.q_in.get()

                if isinstance(msg, _EOS):
                    if self._jsonl_fh:
                        self._jsonl_fh.close()
                        self._jsonl_fh = None
                    await asyncio.to_thread(self._finalise)
                    logger.info(
                        f"[Writer] Done — {self._count} rows for {self.w_id}/{self.i_id}"
                    )
                    return self._count

                assert isinstance(msg, DetectionResult)
                row = _build_parquet_row(self.w_id, self.i_id, msg)
                self._count += 1

                # Append to JSONL immediately so each image is durable on disk
                if self._jsonl_fh is not None:
                    await asyncio.to_thread(self._append_jsonl, row)

                if self._count % 100 == 0:
                    logger.info(f"[Writer] {self._count} images written")

        except Exception:
            if self._jsonl_fh:
                self._jsonl_fh.close()
                self._jsonl_fh = None
            raise

    def _append_jsonl(self, row: Dict[str, Any]) -> None:
        """Append one JSON line and flush to the OS buffer immediately."""
        self._jsonl_fh.write(json.dumps(row) + "\n")
        self._jsonl_fh.flush()

    def _finalise(self) -> None:
        """Read detections.jsonl → write detections.parquet → upload to S3."""
        import pandas as pd

        rows: List[Dict[str, Any]] = []

        # Read rows back from JSONL (handles partial volumes on resume too)
        if self.output_dir is not None:
            vol_dir = self._vol_dir()
            jsonl_path = vol_dir / "detections.jsonl"

            if jsonl_path.exists():
                with open(jsonl_path, encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            rows.append(json.loads(line))

            if rows:
                df = pd.DataFrame(rows)
                out_path = vol_dir / "detections.parquet"
                tmp_path = out_path.with_suffix(".tmp.parquet")
                df.to_parquet(tmp_path, index=False)
                tmp_path.rename(out_path)
                logger.info(f"[Writer] → {out_path} ({len(df)} rows)")

        if self.output_s3_prefix and self.s3_client:
            # If we didn't already load rows from local JSONL, they came from memory
            if not rows:
                logger.warning(f"[Writer] No rows to upload to S3 for {self.w_id}/{self.i_id}")
                return
            df = pd.DataFrame(rows)
            buf = io.BytesIO()
            df.to_parquet(buf, index=False)
            parsed = urlparse(self.output_s3_prefix.rstrip("/"))
            bucket = parsed.netloc
            key_prefix = parsed.path.lstrip("/")
            key = f"{key_prefix}/{self.w_id}/{self.i_id}/detections.parquet"
            self.s3_client.put_object(
                Bucket=bucket,
                Key=key,
                Body=buf.getvalue(),
                ContentType="application/octet-stream",
            )
            logger.info(f"[Writer] → s3://{bucket}/{key} ({len(df)} rows)")


# ── Pipeline ──────────────────────────────────────────────────────────────────

class HFFDetectPipeline:
    """Wires all four stages and runs them concurrently under a timeout.

    Queue layout (all bounded):
        q_fetch   : FetchedBytes | PipelineError | EndOfStream   (maxsize=200)
        q_decoded : DecodedImage | DetectionResult | _EOS         (maxsize=64)
        q_results : DetectionResult | _EOS                        (maxsize=500)

    q_decoded maxsize must be ≥ detect_batch_size so the detect stage can
    always fill a full batch without deadlocking the decoder.
    """

    def __init__(
        self,
        cfg: DetectConfig,
        volume_task: Any,           # VolumeTask
        w_id: str,
        i_id: str,
        detector: Any,              # pre-loaded SuryaLayoutDetector
        processor: Any,             # pre-loaded HFFProcessor
        output_dir: Optional[Path],
        output_s3_prefix: Optional[str],
        s3ctx: Any,                 # S3Context
        s3_client_plain: Any,       # plain boto3 client for parquet upload
    ) -> None:
        from bec_orch.jobs.shared.prefetch import S3Prefetcher

        q_decoded_size = max(64, cfg.detect_batch_size * 4)

        self.q_fetch:   asyncio.Queue = asyncio.Queue(maxsize=200)
        self.q_decoded: asyncio.Queue = asyncio.Queue(maxsize=q_decoded_size)
        self.q_results: asyncio.Queue = asyncio.Queue(maxsize=500)

        self.prefetcher = S3Prefetcher(cfg, s3ctx, volume_task, self.q_fetch)
        self.decoder    = DecodeStage(self.q_fetch, self.q_decoded)
        self.detector   = DetectStage(cfg, detector, processor, self.q_decoded, self.q_results)
        self.writer     = ParquetWriterStage(
            w_id=w_id, i_id=i_id,
            output_dir=output_dir,
            output_s3_prefix=output_s3_prefix,
            s3_client=s3_client_plain,
            q_in=self.q_results,
        )
        self.cfg = cfg
        self.w_id = w_id
        self.i_id = i_id
        self._tasks: List[asyncio.Task] = []

    async def __aenter__(self) -> "HFFDetectPipeline":
        return self

    async def __aexit__(self, exc_type, exc, tb) -> bool:
        await self.aclose()
        return False

    async def aclose(self) -> None:
        """Cancel all running stage tasks (best-effort cleanup)."""
        for t in self._tasks:
            if not t.done():
                t.cancel()
        with contextlib.suppress(Exception):
            await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks = []

    async def run(self) -> int:
        """Run the full pipeline. Returns number of rows written."""
        try:
            return await asyncio.wait_for(
                self._run_stages(),
                timeout=self.cfg.volume_timeout_s,
            )
        except asyncio.TimeoutError:
            logger.error(
                f"[Pipeline] Timeout after {self.cfg.volume_timeout_s}s "
                f"for {self.w_id}/{self.i_id}"
            )
            await self.aclose()
            raise

    async def _run_stages(self) -> int:
        n = len(self.prefetcher.image_tasks)
        logger.info(f"[Pipeline] {self.w_id}/{self.i_id} — {n} images")
        t0 = time.perf_counter()

        self._tasks = [
            asyncio.create_task(self.prefetcher.run(), name="prefetcher"),
            asyncio.create_task(self.decoder.run(),    name="decoder"),
            asyncio.create_task(self.detector.run(),   name="detector"),
            asyncio.create_task(self.writer.run(),     name="writer"),
        ]

        results = await asyncio.gather(*self._tasks, return_exceptions=True)

        elapsed = time.perf_counter() - t0
        stage_names = ["prefetcher", "decoder", "detector", "writer"]
        rows_written = 0
        failed = []

        for name, result in zip(stage_names, results):
            if isinstance(result, Exception) and not isinstance(result, asyncio.CancelledError):
                logger.error(f"[Pipeline] Stage '{name}' failed: {result}", exc_info=result)
                failed.append((name, result))
            elif name == "writer" and isinstance(result, int):
                rows_written = result

        logger.info(
            f"[Pipeline] {self.w_id}/{self.i_id} finished in {elapsed:.1f}s "
            f"— {rows_written} rows"
        )
        await self.aclose()

        if failed:
            name, exc = failed[0]
            raise RuntimeError(f"Stage '{name}' failed: {exc}") from exc

        return rows_written


# ── Per-volume runner ─────────────────────────────────────────────────────────

async def detect_volume(
    *,
    cfg: DetectConfig,
    w_id: str,
    i_id: str,
    detector: Any,
    processor: Any,
    output_dir: Optional[Path],
    output_s3_prefix: Optional[str],
    limit: int = 0,
    start: int = 0,
) -> int:
    """Stream one BDRC volume from S3, detect HFF regions, write parquet.

    Returns number of rows written (= number of images processed).
    """
    import boto3
    from bec_orch.jobs.ldv1.types_common import ImageTask, VolumeTask
    from bec_orch.jobs.shared.s3ctx import S3Context

    s3_plain = boto3.client("s3", region_name=cfg.s3_region)
    prefix = _build_s3_prefix(w_id, i_id)
    filenames = _load_manifest(s3_plain, cfg.s3_bucket, prefix)

    if start:
        filenames = filenames[start:]
    if limit:
        filenames = filenames[:limit]

    if not filenames:
        logger.warning(f"[detect_volume] No images for {w_id}/{i_id}")
        return 0

    image_tasks = [
        ImageTask(
            source_uri=f"s3://{cfg.s3_bucket}/{prefix}{fn}",
            img_filename=fn,
        )
        for fn in filenames
    ]

    volume_task = VolumeTask(
        io_mode="s3",
        debug_folder_path="/tmp/bec_hff_debug",
        output_parquet_uri="",
        output_jsonl_uri="",
        image_tasks=image_tasks,
    )

    global_sem = asyncio.Semaphore(cfg.bulk_prefetch_concurrency)
    s3ctx = S3Context(cfg=cfg, global_sem=global_sem)

    try:
        async with HFFDetectPipeline(
            cfg=cfg,
            volume_task=volume_task,
            w_id=w_id,
            i_id=i_id,
            detector=detector,
            processor=processor,
            output_dir=output_dir,
            output_s3_prefix=output_s3_prefix,
            s3ctx=s3ctx,
            s3_client_plain=s3_plain,
        ) as pipeline:
            return await pipeline.run()
    finally:
        await s3ctx.close()


# ── Model loader ──────────────────────────────────────────────────────────────

def load_model(cfg: DetectConfig) -> Tuple[Any, Any]:
    """Load Surya layout detector and HFF processor (call ONCE before volume loop).

    Returns (detector, processor).
    """
    try:
        from hff_remover.detector import SuryaLayoutDetector
        from hff_remover.processor import HFFProcessor
    except ImportError as exc:
        raise SystemExit(
            f"hff_remover not installed: {exc}\n"
            "Install: pip install git+https://github.com/OpenPecha/HFF-Remover.git"
        ) from exc

    logger.info("[Model] Loading Surya layout model (this may take ~30s) …")
    t0 = time.perf_counter()
    detector = SuryaLayoutDetector(confidence_threshold=cfg.confidence_threshold)
    processor = HFFProcessor(margin=cfg.margin)
    logger.info(f"[Model] Ready in {time.perf_counter() - t0:.1f}s")
    return detector, processor


# ── CLI ───────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Stream BDRC volume images from S3, detect header/footer/footnote/body "
            "using Surya, and write per-volume parquet files."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    src = p.add_argument_group("Source (provide one)")
    src.add_argument("--w-id",  help="Work ID, e.g. W22084")
    src.add_argument("--i-id",  help="Image group ID, e.g. I0886 or I1ER1")
    src.add_argument("--csv",   help="CSV file with w_id / i_id columns")
    src.add_argument("--w-col", default="w_id", help="CSV column for work ID")
    src.add_argument("--i-col", default="i_id", help="CSV column for image group ID")

    out = p.add_argument_group("Output (provide at least one)")
    out.add_argument("--output-dir", help="Local directory: writes w_id/i_id/detections.parquet")
    out.add_argument("--output-s3",  help="S3 URI prefix, e.g. s3://my-bucket/hff-detections/")

    s3g = p.add_argument_group("S3 source settings")
    s3g.add_argument("--bucket",      default="archive.tbrc.org")
    s3g.add_argument("--region",      default="us-east-1")
    s3g.add_argument("--concurrency", type=int, default=32, help="S3 fetch concurrency")
    s3g.add_argument("--streaming",   action="store_true",  help="Use streaming S3 mode instead of bulk")

    det = p.add_argument_group("Detection settings")
    det.add_argument("--confidence",  type=float, default=0.5, help="Min detection confidence (0–1)")
    det.add_argument("--batch-size",  type=int,   default=8,   help="Images per detection batch")
    det.add_argument("--margin",      type=int,   default=0,   help="Merge nearby detections gap (px)")

    misc = p.add_argument_group("Misc")
    misc.add_argument("--start",    type=int,   default=0, help="Skip first N images per volume")
    misc.add_argument("--limit",    type=int,   default=0, help="Max images per volume (0 = all)")
    misc.add_argument("--dry-run",  action="store_true",   help="Print what would run, no model or S3 writes")
    misc.add_argument("-v", "--verbose", action="store_true")

    return p.parse_args()


def main() -> None:
    args = _parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    # Validate output
    if not args.output_dir and not args.output_s3:
        raise SystemExit("Provide at least one of --output-dir or --output-s3")

    # Validate source
    if args.csv:
        if args.w_id or args.i_id:
            raise SystemExit("Use --csv OR --w-id/--i-id, not both")
    else:
        if not (args.w_id and args.i_id):
            raise SystemExit("Provide both --w-id and --i-id, or use --csv")

    cfg = DetectConfig(
        s3_region=args.region,
        s3_bucket=args.bucket,
        bulk_prefetch=not args.streaming,
        bulk_prefetch_concurrency=args.concurrency,
        confidence_threshold=args.confidence,
        detect_batch_size=args.batch_size,
        margin=args.margin,
    )

    output_dir = Path(args.output_dir) if args.output_dir else None
    output_s3 = args.output_s3 or None

    # Build volume list
    if args.csv:
        pairs: List[Tuple[str, str]] = []
        with open(args.csv, newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                w = (row.get(args.w_col) or "").strip()
                i = (row.get(args.i_col) or "").strip()
                if w and i:
                    pairs.append((w, i))
        if not pairs:
            raise SystemExit("No valid rows found in CSV")
    else:
        pairs = [(args.w_id, args.i_id)]

    # Dry run: print volumes and exit
    if args.dry_run:
        print(f"DRY RUN — {len(pairs)} volume(s), limit={args.limit or 'all'}")
        for w_id, i_id in pairs[:10]:
            print(f"  {w_id} / {i_id}")
        if len(pairs) > 10:
            print(f"  ... and {len(pairs) - 10} more")
        return

    # Load model ONCE before the volume loop
    detector, processor = load_model(cfg)

    total_rows = 0
    failed_volumes: List[str] = []

    print(f"\nProcessing {len(pairs)} volume(s) …")
    for idx, (w_id, i_id) in enumerate(pairs, start=1):
        print(f"\n=== [{idx}/{len(pairs)}] {w_id}/{i_id} ===")
        try:
            rows = asyncio.run(
                detect_volume(
                    cfg=cfg,
                    w_id=w_id,
                    i_id=i_id,
                    detector=detector,
                    processor=processor,
                    output_dir=output_dir,
                    output_s3_prefix=output_s3,
                    limit=args.limit,
                    start=args.start,
                )
            )
            total_rows += rows
        except Exception as exc:
            logger.error(f"Volume {w_id}/{i_id} failed: {exc}", exc_info=True)
            failed_volumes.append(f"{w_id}/{i_id}")

    print(f"\n{'=' * 60}")
    print(f"Done — {len(pairs) - len(failed_volumes)}/{len(pairs)} volumes succeeded")
    print(f"Total rows written: {total_rows:,}")
    if failed_volumes:
        print(f"Failed volumes ({len(failed_volumes)}):")
        for v in failed_volumes:
            print(f"  {v}")


if __name__ == "__main__":
    main()
