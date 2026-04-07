"""S3 image download pipeline.

Async two-stage pipeline modeled after LDVolumeWorker (bec_orch/jobs/ldv1/worker.py):

  S3Prefetcher  →  ImageSaverStage
  (fetches bytes    (writes each image
   from S3 in        to output_dir/
   parallel)         filename)

Usage:
  # single volume
  python scripts/s3_download_pipeline.py \\
      --w-id W22084 --i-id I0886 \\
      --output-dir ./downloads

  # batch from CSV (columns: w_id, i_id)
  python scripts/s3_download_pipeline.py \\
      --csv portrait_volumes.csv \\
      --output-dir ./downloads

  # with explicit credentials
  AWS_ACCESS_KEY_ID=... AWS_SECRET_ACCESS_KEY=... \\
  python scripts/s3_download_pipeline.py --w-id W22084 --i-id I0886 --output-dir ./downloads
"""

from __future__ import annotations

import sys
import os
# Make `bec_orch` importable when running the script directly (no install needed)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import asyncio
import contextlib
import csv
import gzip
import hashlib
import json
import logging
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class DownloadConfig:
    """Configuration for the S3 download pipeline.

    Mirrors the fields that S3Context and S3Prefetcher read off cfg.
    """
    s3_region: str = "us-east-1"
    s3_bucket: str = "archive.tbrc.org"

    # Prefetch tuning — bulk mode fires all fetches at once (high throughput)
    bulk_prefetch: bool = True
    bulk_prefetch_concurrency: int = 32   # lower if archive.tbrc.org rate-limits you

    # Streaming mode fallback (used when bulk_prefetch=False)
    inflight_per_worker: int = 8

    # Overall per-volume timeout
    volume_timeout_s: float = 600.0

    # aws_profile: read by S3Context when constructing boto3.Session
    aws_profile: str = "default"


# ---------------------------------------------------------------------------
# S3 / manifest helpers
# ---------------------------------------------------------------------------

def _build_s3_prefix(w_id: str, i_id: str) -> str:
    """Return the S3 key prefix for a BDRC volume's images.

    Pattern: Works/{md5_2}/{w_id}/images/{w_id}-{i_id_m}/
    """
    md5_2 = hashlib.md5(w_id.encode("utf-8")).hexdigest()[:2]
    # I#### → strip the leading "I"; all other formats stay as-is
    i_id_m = i_id[1:] if re.fullmatch(r"I\d{4}", i_id) else i_id
    return f"Works/{md5_2}/{w_id}/images/{w_id}-{i_id_m}/"


def _load_manifest(s3_client, bucket: str, prefix: str) -> list[str]:
    """Fetch dimensions.json from S3 and return ordered list of image filenames."""
    key = f"{prefix}dimensions.json"
    logger.info(f"[Manifest] Fetching s3://{bucket}/{key}")

    resp = s3_client.get_object(Bucket=bucket, Key=key)
    payload = resp["Body"].read()

    try:
        data = json.loads(gzip.decompress(payload).decode("utf-8"))
    except OSError:
        # Not gzip — try plain JSON
        data = json.loads(payload.decode("utf-8"))

    if not isinstance(data, list):
        raise RuntimeError(f"dimensions.json must be a JSON array, got {type(data)}")

    # Keep image files only, sorted numerically (same logic as bdrc_download_images.py)
    image_re = re.compile(r".*\.(jpg|jpeg|tif|tiff|png)$", re.IGNORECASE)
    kept: list[tuple[int | None, str]] = []
    for item in data:
        fn = item.get("filename") if isinstance(item, dict) else None
        if not fn or not image_re.fullmatch(fn):
            continue
        stem = Path(fn).stem
        m = re.search(r"(\d+)$", stem) or re.search(r"(\d+)", stem)
        num = int(m.group(1)) if m else None
        kept.append((num, fn))

    kept.sort(key=lambda row: (row[0] is None, row[0] if row[0] is not None else 0, row[1]))
    filenames = [fn for _, fn in kept]
    logger.info(f"[Manifest] {len(filenames)} image(s) found")
    return filenames


# ---------------------------------------------------------------------------
# Stage: ImageSaverStage
# ---------------------------------------------------------------------------

class ImageSaverStage:
    """Reads FetchedBytes from q_in and writes each image to output_dir.

    Mirrors the pattern of ParquetWriter / OutputWriterStage — runs as a
    single async coroutine consuming from a bounded queue until EndOfStream.
    """

    def __init__(
        self,
        output_dir: Path,
        q_in: asyncio.Queue,
    ) -> None:
        self.output_dir = output_dir
        self.q_in = q_in
        self._saved = 0
        self._errors = 0

    async def run(self) -> None:
        from bec_orch.jobs.ldv1.types_common import EndOfStream, FetchedBytes, PipelineError

        start = time.perf_counter()

        while True:
            msg = await self.q_in.get()

            if isinstance(msg, EndOfStream):
                elapsed = time.perf_counter() - start
                logger.info(
                    f"[ImageSaver] Done — saved={self._saved} errors={self._errors} "
                    f"elapsed={elapsed:.2f}s"
                )
                return

            if isinstance(msg, PipelineError):
                self._errors += 1
                logger.warning(
                    f"[ImageSaver] Fetch error for {msg.task.img_filename}: "
                    f"{msg.error_type}: {msg.message}"
                )
                continue

            # FetchedBytes — write to disk in a thread so we don't block the event loop
            assert isinstance(msg, FetchedBytes)
            dest = self.output_dir / msg.task.img_filename
            await asyncio.to_thread(dest.write_bytes, msg.file_bytes)
            self._saved += 1
            logger.debug(f"[ImageSaver] Saved {dest} ({len(msg.file_bytes)} bytes)")

            if self._saved % 50 == 0:
                logger.info(f"[ImageSaver] Progress: {self._saved} saved, {self._errors} errors")


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

class S3DownloadPipeline:
    """Async two-stage pipeline: S3Prefetcher → ImageSaverStage.

    Modeled after LDVolumeWorker:
    - Bounded queue enforces backpressure between stages.
    - All stages run as asyncio.Tasks gathered together.
    - Timeout via asyncio.wait_for.
    - aclose() cancels lingering tasks on error/timeout.
    """

    def __init__(
        self,
        cfg: DownloadConfig,
        volume_task,           # VolumeTask
        output_dir: Path,
        s3ctx,                 # S3Context
    ) -> None:
        from bec_orch.jobs.shared.prefetch import S3Prefetcher

        self.cfg = cfg
        self.volume_task = volume_task
        self.output_dir = output_dir
        self.s3ctx = s3ctx

        # Bounded queue: prefetcher pushes, saver pulls.
        # 200 slots → ~200 images buffered at most (backpressure on prefetcher).
        self.q_prefetcher_to_saver: asyncio.Queue = asyncio.Queue(maxsize=200)

        # Stage instances
        self.prefetcher = S3Prefetcher(cfg, s3ctx, volume_task, self.q_prefetcher_to_saver)
        self.saver = ImageSaverStage(output_dir, self.q_prefetcher_to_saver)

        self._tasks: list[asyncio.Task] = []

    # -- context manager support (mirrors LDVolumeWorker.__aenter__/__aexit__)

    async def __aenter__(self) -> "S3DownloadPipeline":
        return self

    async def __aexit__(self, exc_type, exc, tb) -> bool:
        await self.aclose()
        return False

    async def aclose(self) -> None:
        """Cancel all running stage tasks (best-effort)."""
        if not self._tasks:
            return
        for t in self._tasks:
            if not t.done():
                t.cancel()
        with contextlib.suppress(Exception):
            await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks = []

    # -- public entry point

    async def run(self) -> None:
        """Run the pipeline with timeout protection."""
        try:
            await asyncio.wait_for(
                self._run_pipeline(),
                timeout=self.cfg.volume_timeout_s,
            )
        except asyncio.TimeoutError:
            logger.error(f"[S3DownloadPipeline] Timeout after {self.cfg.volume_timeout_s}s")
            await self.aclose()
            raise

    async def _run_pipeline(self) -> None:
        """Run all stages concurrently and wait for completion."""
        n = len(self.volume_task.image_tasks)
        logger.info(f"[S3DownloadPipeline] Starting — {n} image(s) → {self.output_dir}")
        start = time.perf_counter()

        self._tasks = [
            asyncio.create_task(self.prefetcher.run(), name="prefetcher"),
            asyncio.create_task(self.saver.run(), name="saver"),
        ]

        results = await asyncio.gather(*self._tasks, return_exceptions=True)

        stage_names = ["prefetcher", "saver"]
        failed = []
        for name, result in zip(stage_names, results):
            if isinstance(result, Exception):
                logger.error(f"[S3DownloadPipeline] Stage '{name}' failed: {result}", exc_info=result)
                failed.append((name, result))

        elapsed = time.perf_counter() - start
        logger.info(f"[S3DownloadPipeline] Finished in {elapsed:.2f}s")

        await self.aclose()

        if failed:
            name, exc = failed[0]
            raise RuntimeError(f"Stage '{name}' failed: {exc}") from exc


# ---------------------------------------------------------------------------
# Per-volume runner
# ---------------------------------------------------------------------------

async def download_volume(
    *,
    cfg: DownloadConfig,
    w_id: str,
    i_id: str,
    output_dir: Path,
    limit: int = 0,
    start: int = 0,
    dry_run: bool = False,
) -> None:
    """Fetch one BDRC volume's images from S3 into output_dir."""
    import boto3
    from bec_orch.jobs.ldv1.types_common import ImageTask, VolumeTask
    from bec_orch.jobs.shared.s3ctx import S3Context

    output_dir.mkdir(parents=True, exist_ok=True)

    # Resolve the S3 prefix and load the manifest using a plain boto3 client
    s3_plain = boto3.client("s3", region_name=cfg.s3_region)
    prefix = _build_s3_prefix(w_id, i_id)
    filenames = _load_manifest(s3_plain, cfg.s3_bucket, prefix)

    # Slice by start / limit
    if start:
        filenames = filenames[start:]
    if limit:
        filenames = filenames[:limit]

    if not filenames:
        logger.warning(f"[download_volume] No images to download for {w_id}/{i_id}")
        return

    if dry_run:
        print(f"DRY RUN — would download {len(filenames)} image(s) to {output_dir}")
        for fn in filenames[:10]:
            print(f"  s3://{cfg.s3_bucket}/{prefix}{fn}")
        if len(filenames) > 10:
            print(f"  ... and {len(filenames) - 10} more")
        return

    # Build ImageTask list (source_uri = s3://bucket/key)
    image_tasks = [
        ImageTask(
            source_uri=f"s3://{cfg.s3_bucket}/{prefix}{fn}",
            img_filename=fn,
        )
        for fn in filenames
    ]

    volume_task = VolumeTask(
        io_mode="s3",
        debug_folder_path="/tmp/bec_debug",
        output_parquet_uri="",   # not used by this pipeline
        output_jsonl_uri="",     # not used by this pipeline
        image_tasks=image_tasks,
    )

    # S3Context owns the thread pool + boto3 client used by S3Prefetcher
    global_sem = asyncio.Semaphore(cfg.bulk_prefetch_concurrency)
    s3ctx = S3Context(cfg=cfg, global_sem=global_sem)

    try:
        async with S3DownloadPipeline(cfg, volume_task, output_dir, s3ctx) as pipeline:
            await pipeline.run()
    finally:
        await s3ctx.close()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download BDRC volume images from S3 using the async pipeline"
    )
    parser.add_argument("--w-id", help="Work ID, e.g. W22084")
    parser.add_argument("--i-id", help="Image group ID, e.g. I0886 or I1ER1")
    parser.add_argument("--csv", help="CSV file with w_id/i_id columns")
    parser.add_argument("--w-col", default="w_id", help="CSV column for work ID (default: w_id)")
    parser.add_argument("--i-col", default="i_id", help="CSV column for image group ID (default: i_id)")
    parser.add_argument("--output-dir", required=True, help="Local directory to save images")
    parser.add_argument("--bucket", default="archive.tbrc.org", help="S3 bucket (default: archive.tbrc.org)")
    parser.add_argument("--region", default="us-east-1", help="AWS region (default: us-east-1)")
    parser.add_argument("--concurrency", type=int, default=32, help="S3 fetch concurrency (default: 32)")
    parser.add_argument("--start", type=int, default=0, help="Skip first N images (default: 0)")
    parser.add_argument("--limit", type=int, default=0, help="Max images to download, 0 = all (default: 0)")
    parser.add_argument("--dry-run", action="store_true", help="Print what would be downloaded, no actual fetch")
    parser.add_argument("--streaming", action="store_true", help="Use streaming mode instead of bulk prefetch")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable DEBUG logging")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    if args.csv:
        if args.w_id or args.i_id:
            raise SystemExit("Use either --csv OR --w-id/--i-id, not both")
    else:
        if not (args.w_id and args.i_id):
            raise SystemExit("Provide both --w-id and --i-id, or use --csv")

    cfg = DownloadConfig(
        s3_region=args.region,
        s3_bucket=args.bucket,
        bulk_prefetch=not args.streaming,
        bulk_prefetch_concurrency=args.concurrency,
    )

    output_root = Path(args.output_dir)

    if args.csv:
        pairs: list[tuple[str, str]] = []
        with open(args.csv, newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                w = (row.get(args.w_col) or "").strip()
                i = (row.get(args.i_col) or "").strip()
                if w and i:
                    pairs.append((w, i))
        if not pairs:
            raise SystemExit("No valid rows found in CSV")

        print(f"Processing {len(pairs)} volume(s) from {args.csv}")
        for idx, (w_id, i_id) in enumerate(pairs, start=1):
            vol_dir = output_root / f"{w_id}-{i_id}"
            print(f"\n=== [{idx}/{len(pairs)}] {w_id}/{i_id} → {vol_dir} ===")
            try:
                asyncio.run(
                    download_volume(
                        cfg=cfg,
                        w_id=w_id,
                        i_id=i_id,
                        output_dir=vol_dir,
                        limit=args.limit,
                        start=args.start,
                        dry_run=args.dry_run,
                    )
                )
            except Exception as exc:
                logger.error(f"Volume {w_id}/{i_id} failed: {exc}", exc_info=True)
    else:
        vol_dir = output_root / f"{args.w_id}-{args.i_id}"
        asyncio.run(
            download_volume(
                cfg=cfg,
                w_id=args.w_id,
                i_id=args.i_id,
                output_dir=vol_dir,
                limit=args.limit,
                start=args.start,
                dry_run=args.dry_run,
            )
        )


if __name__ == "__main__":
    main()
