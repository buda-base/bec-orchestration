"""Sync a PaddleOCR-VL checkpoint from S3 to local disk before inference.

The fine-tuned checkpoint is self-contained (weights + processor + tokenizer)
but also holds training-only artifacts (``optimizer.pt``, ``scheduler.pt``,
``trainer_state.json``) worth ~3.6 GB that inference does not need — those are
skipped via ``PaddleOCRConfig.sync_exclude_*``.

Downloads are idempotent: a file already present with the same size is left
untouched, so a worker restart (or a re-used AMI) does not re-download.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
from pathlib import Path
from typing import TYPE_CHECKING

import boto3
from botocore.config import Config as BotoConfig

if TYPE_CHECKING:
    from .config import PaddleOCRConfig

logger = logging.getLogger(__name__)

# Files that MUST be present locally for inference to work.
_REQUIRED_FILES = ("config.json", "model.safetensors")

# The image placeholder token the native PaddleOCRVLProcessor reads (already in
# the warmed vocab; only the top-level ``image_token`` attribute is missing).
_IMAGE_TOKEN = "<|IMAGE_PLACEHOLDER|>"


def _patch_checkpoint(local_dir: Path, cfg: PaddleOCRConfig) -> None:
    """Make a warmed checkpoint self-contained for the *native* vLLM path.

    Mirrors ``bec-ocr-training/deploy/fast_inference/patch_checkpoint.py``:

    1. ``tokenizer_config.json`` gets the top-level ``image_token`` the native
       ``PaddleOCRVLProcessor`` requires (else it raises ``TokenizersBackend has
       no attribute image_token``).
    2. ``processor_config.json``'s image-processor ``size`` is pinned to the 1x
       budget the checkpoint was evaluated at
       (``{shortest_edge, longest_edge}``), so native transformers AND vLLM
       reproduce the headline CER with ``trust_remote_code=False`` and no
       per-call kwargs.

    Idempotent (only writes when a value actually changes) and backs up each
    file once as ``*.orig``.
    """

    def _save(path: Path, obj: dict) -> None:
        orig = path.with_suffix(path.suffix + ".orig")
        if not orig.exists():
            shutil.copy(path, orig)
        with path.open("w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=2)

    tcp = local_dir / "tokenizer_config.json"
    if tcp.exists():
        tc = json.loads(tcp.read_text(encoding="utf-8"))
        if tc.get("image_token") != _IMAGE_TOKEN:
            tc["image_token"] = _IMAGE_TOKEN
            _save(tcp, tc)
            logger.info(f"[paddleocr.sync] patched tokenizer_config.json image_token={_IMAGE_TOKEN}")
    else:
        logger.warning(f"[paddleocr.sync] {tcp} missing; cannot set image_token")

    pcp = local_dir / "processor_config.json"
    if pcp.exists():
        pc = json.loads(pcp.read_text(encoding="utf-8"))
        want = {
            "shortest_edge": cfg.processor_shortest_edge,
            "longest_edge": cfg.processor_longest_edge(),
        }
        ip = pc.get("image_processor", {})
        if ip.get("size") != want:
            ip["size"] = want
            pc["image_processor"] = ip
            _save(pcp, pc)
            logger.info(f"[paddleocr.sync] patched processor_config.json size={want}")
    else:
        logger.warning(f"[paddleocr.sync] {pcp} missing; cannot pin processor size")


def _parse_s3_uri(uri: str) -> tuple[str, str]:
    body = uri[len("s3://"):] if uri.startswith("s3://") else uri
    bucket, _, key = body.partition("/")
    return bucket, key.strip("/")


def sync_checkpoint(cfg: PaddleOCRConfig) -> str:
    """Ensure ``cfg.checkpoint_s3_uri`` is available locally; return its dir."""
    local_dir = Path(cfg.resolved_model_dir())
    local_dir.mkdir(parents=True, exist_ok=True)

    bucket, prefix = _parse_s3_uri(cfg.checkpoint_s3_uri)
    if prefix and not prefix.endswith("/"):
        prefix += "/"

    region = os.environ.get("BEC_REGION", "us-east-1")
    s3 = boto3.client(
        "s3",
        config=BotoConfig(region_name=region, retries={"max_attempts": 5, "mode": "standard"}),
    )

    logger.info(
        f"[paddleocr.sync] syncing s3://{bucket}/{prefix} -> {local_dir} "
        f"(exclude suffixes={cfg.sync_exclude_suffixes}, names={cfg.sync_exclude_names})"
    )

    paginator = s3.get_paginator("list_objects_v2")
    n_downloaded = 0
    n_skipped = 0
    n_excluded = 0
    seen_names: set[str] = set()

    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            rel = key[len(prefix):] if prefix else key
            if not rel or rel.endswith("/"):
                continue  # directory marker
            name = os.path.basename(rel)
            seen_names.add(name)

            if name in cfg.sync_exclude_names or any(
                name.endswith(suf) for suf in cfg.sync_exclude_suffixes
            ):
                n_excluded += 1
                continue

            dst = local_dir / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            remote_size = int(obj["Size"])
            if dst.exists() and dst.stat().st_size == remote_size:
                n_skipped += 1
                continue

            logger.info(f"[paddleocr.sync] downloading {rel} ({remote_size / 1e6:.1f} MB)")
            s3.download_file(bucket, key, str(dst))
            n_downloaded += 1

    missing = [f for f in _REQUIRED_FILES if not (local_dir / f).exists()]
    if missing:
        raise FileNotFoundError(
            f"checkpoint sync incomplete: missing {missing} under {local_dir} "
            f"(from s3://{bucket}/{prefix}); files seen: {sorted(seen_names)}"
        )

    # Make the checkpoint self-contained for the native vLLM path.
    _patch_checkpoint(local_dir, cfg)

    logger.info(
        f"[paddleocr.sync] ready: {local_dir} "
        f"(downloaded={n_downloaded}, skipped={n_skipped}, excluded={n_excluded})"
    )
    return str(local_dir)
