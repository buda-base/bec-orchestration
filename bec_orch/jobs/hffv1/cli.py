"""HFFv1 local CLI — run HFF removal on a local folder of images and write
a parquet file summarising the detections.

Usage
-----
    python -m bec_orch.jobs.hffv1.cli --input-folder /path/to/images

    # custom output path
    python -m bec_orch.jobs.hffv1.cli --input-folder /path/to/images \\
        --output /path/to/results.parquet

    # tune model settings
    python -m bec_orch.jobs.hffv1.cli --input-folder /path/to/images \\
        --confidence 0.4 --margin 5
"""
from __future__ import annotations

import argparse
import io
import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from PIL import Image
from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
)

console = Console(stderr=True)
logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tiff", ".tif", ".bmp", ".webp"}


# ──────────────────────────────────────────────────────────────────────────────
# Image helpers
# ──────────────────────────────────────────────────────────────────────────────

def _load_bgr(path: Path) -> np.ndarray:
    """Load an image from disk → BGR numpy array."""
    pil = Image.open(path).convert("RGB")
    return np.array(pil)[:, :, ::-1]


def _collect_images(folder: Path) -> List[Path]:
    """Return all supported image files in *folder* (non-recursive, sorted)."""
    images = sorted(
        p for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
    )
    return images


# ──────────────────────────────────────────────────────────────────────────────
# Core processing
# ──────────────────────────────────────────────────────────────────────────────

def _process_folder(
    input_folder: Path,
    output_path: Path,
    confidence: float,
    margin: int,
) -> None:
    """Run the Surya HFF detector on every image in *input_folder* and write
    a parquet file to *output_path*.

    Parquet schema (one row per image)
    -----------------------------------
    filename         : str   — image filename (basename)
    filepath         : str   — absolute path to source image
    n_detections     : int   — number of HFF detections after merging
    classes_detected : str   — comma-separated class names (e.g. "header,footer")
    bboxes_json      : str   — JSON array of detection dicts
                               [{bbox:[x1,y1,x2,y2], class_name, confidence}, ...]
    duration_ms      : float — inference + merge time in milliseconds
    error            : str   — error message, or empty string on success
    """
    try:
        import pandas as pd
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:
        console.print(f"[red]Error:[/red] {exc}. Install pandas and pyarrow.")
        raise SystemExit(1) from exc

    try:
        from hff_remover.detector import SuryaLayoutDetector
        from hff_remover.processor import HFFProcessor
    except ImportError as exc:
        console.print(
            f"[red]Error:[/red] {exc}. "
            "Install hff-remover: pip install git+https://github.com/OpenPecha/HFF-Remover.git"
        )
        raise SystemExit(1) from exc

    images = _collect_images(input_folder)
    if not images:
        console.print(f"[yellow]No supported images found in {input_folder}[/yellow]")
        return

    console.print(f"Found [bold]{len(images)}[/bold] image(s) in {input_folder}")
    console.print("Loading Surya layout model …")

    detector = SuryaLayoutDetector(confidence_threshold=confidence)
    merger = HFFProcessor(margin=margin)

    console.print("[green]Model loaded.[/green]")

    rows: List[Dict[str, Any]] = []

    with Progress(
        TextColumn("[bold]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Processing images", total=len(images))

        for img_path in images:
            t0 = time.time()
            error_msg = ""
            dets: List[Dict[str, Any]] = []

            try:
                bgr = _load_bgr(img_path)
                dets = detector.detect(bgr, filter_to_hff_only=True)
                dets = merger.merge_nearby_detections(dets)
            except Exception as exc:
                error_msg = str(exc)
                logger.warning("Failed to process %s: %s", img_path.name, exc)

            duration_ms = (time.time() - t0) * 1000

            classes = sorted({d["class_name"] for d in dets})
            bboxes_json = json.dumps(
                [
                    {
                        "bbox": [round(v, 1) for v in d["bbox"]],
                        "class_name": d["class_name"],
                        "confidence": round(float(d.get("confidence", 1.0)), 4),
                    }
                    for d in dets
                ]
            )

            rows.append(
                {
                    "filename": img_path.name,
                    "filepath": str(img_path.resolve()),
                    "n_detections": len(dets),
                    "classes_detected": ",".join(classes),
                    "bboxes_json": bboxes_json,
                    "duration_ms": round(duration_ms, 1),
                    "error": error_msg,
                }
            )

            progress.advance(task)

    # ── write parquet ──────────────────────────────────────────────────────────
    df = pd.DataFrame(rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)

    n_ok = (df["error"] == "").sum()
    n_err = (df["error"] != "").sum()
    n_with_dets = (df["n_detections"] > 0).sum()

    console.print(
        f"\n[green]✓[/green] Done — "
        f"{n_ok} ok / {n_err} errors / "
        f"{n_with_dets} images with detections"
    )
    console.print(f"Parquet written → [bold]{output_path}[/bold]")


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(
        description="Run HFF removal on a local image folder and write a parquet report.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    p.add_argument(
        "--input-folder",
        required=True,
        metavar="DIR",
        help="Folder containing images to process.",
    )
    p.add_argument(
        "--output",
        metavar="FILE",
        help=(
            "Output parquet file path. "
            "Defaults to <input-folder>/hff_detections.parquet"
        ),
    )
    p.add_argument(
        "--confidence",
        type=float,
        default=0.5,
        metavar="FLOAT",
        help="Minimum detection confidence threshold (0–1).",
    )
    p.add_argument(
        "--margin",
        type=int,
        default=0,
        metavar="PX",
        help="Extra pixels added around merged detection boxes.",
    )
    p.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="WARNING",
        help="Python logging level.",
    )

    args = p.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    input_folder = Path(args.input_folder).expanduser().resolve()
    if not input_folder.is_dir():
        p.error(f"--input-folder does not exist or is not a directory: {input_folder}")

    output_path = (
        Path(args.output).expanduser().resolve()
        if args.output
        else input_folder / "hff_detections.parquet"
    )

    _process_folder(
        input_folder=input_folder,
        output_path=output_path,
        confidence=args.confidence,
        margin=args.margin,
    )


if __name__ == "__main__":
    main()
