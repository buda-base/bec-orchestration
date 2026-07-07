import logging
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download
from PIL import Image

from . import config, transforms

logger = logging.getLogger(__name__)

# Blank-page pre-filter disabled: untested, no promising results. Every
# image proceeds unconditionally to orientation + 6-class scoring. Kept
# importable (commented) so re-enabling is a one-line change if upstream's
# blank filter is revisited.
# from .blank import is_blank
from .models import Classifier

# `flip_applied` is dropped from the row (see below): it's a 1:1 boolean
# encoding of `rotation_applied` (0/180) already carried in the same row.
_NULL_FIELDS = dict(
    exif_orientation_tag=None,
    orientation_pred=None,
    orientation_prob=None,
    orientation_probs=None,
    rotation_applied=None,
    sixclass_label=None,
    sixclass_probs=None,
    final_label=None,
)


def _short_sha(resolved_path: str) -> str:
    # default hf_hub_download cache layout: .../snapshots/<commit_sha>/<filename>
    return Path(resolved_path).parent.name[:8]


def _row(**overrides) -> dict:
    row = {"status": "ok", "error": None, **_NULL_FIELDS}
    row.update(overrides)
    return row


class Pipeline:
    def __init__(self, cfg=None):
        # Duck-typed rather than importing ScriptClassificationConfig, so
        # vendor/ (kept diffable against upstream) doesn't gain a dependency
        # on the job-level config package. Deviation from upstream: upstream
        # is CPU-only. `use_gpu` defaults to True (GPU used whenever CUDA is
        # available) with a warn-and-fallback to CPU, never a hard failure
        # -- matches ldv1/worker.py's device-selection convention.
        device = "cpu"
        if getattr(cfg, "use_gpu", True):
            if torch.cuda.is_available():
                device = "cuda"
                logger.info(f"[script_classification] using GPU: {torch.cuda.get_device_name(0)}")
            else:
                logger.warning(
                    "[script_classification] use_gpu=True but CUDA not available, using CPU"
                )
        self._device = device

        # Thread pool for parallel per-image decode+resize+crop inside
        # run_batch() (CPU-bound work; libvips releases the GIL, so this
        # genuinely parallelizes). Owned for the process lifetime -- Pipeline
        # itself is already a process-lifetime singleton via loader.py's
        # get_pipeline(), so there's no per-volume teardown/recreation.
        self._decode_pool = ThreadPoolExecutor(
            max_workers=max(1, getattr(cfg, "decode_workers", 1)),
            thread_name_prefix="scriptcls-decode",
        )

        self._processor = transforms.get_processor()

        orient_path = hf_hub_download(
            repo_id=config.ORIENTATION_REPO_ID, filename=config.CHECKPOINT_FILENAME
        )
        six_path = hf_hub_download(
            repo_id=config.SIXCLASS_REPO_ID, filename=config.CHECKPOINT_FILENAME
        )

        self._orientation = Classifier.from_checkpoint(
            orient_path, config.BACKBONE_ID, device=device
        )
        self._sixclass = Classifier.from_checkpoint(six_path, config.BACKBONE_ID, device=device)

        self.model_version = (
            f"orientation:{_short_sha(orient_path)};sixclass:{_short_sha(six_path)}"
        )

    @staticmethod
    def _ordered_labels(idx_to_label: dict) -> list[str]:
        # Class name per logit index (0..C-1), so a bare probability vector
        # (orientation_probs / sixclass_probs) is self-describing downstream.
        return [idx_to_label[i] for i in range(len(idx_to_label))]

    @property
    def orientation_labels(self) -> list[str]:
        return self._ordered_labels(self._orientation.idx_to_label)

    @property
    def sixclass_labels(self) -> list[str]:
        return self._ordered_labels(self._sixclass.idx_to_label)

    def run(self, image_bytes: bytes) -> dict:
        try:
            return self._run(image_bytes)
        except Exception as e:
            return _row(status="error", error=str(e), model_version=self.model_version)

    def _run(self, image_bytes: bytes) -> dict:
        # decode_and_resize decodes AND resizes-short-edge in a single pass
        # (libvips fast-path, PIL fallback) -- see transforms.py. img here
        # is already small (~config.CROP_SIZE short edge), so the 180°
        # rotation below (when needed) operates on the small result instead
        # of a full-resolution re-decode/re-resize.
        img, exif_tag = transforms.decode_and_resize(image_bytes)

        # if is_blank(img):
        #     return _row(
        #         blank=True,
        #         exif_orientation_tag=exif_tag,
        #         model_version=self.model_version,
        #     )
        # NOTE: if re-enabled, is_blank(img) would now see the short-edge-
        # resized image (post decode_and_resize), not the full-resolution
        # decode as before this restructure -- re-validate any blank-
        # detection thresholds against the resized input before re-enabling.

        tensor_orig = transforms.preprocess(img)
        orient_label, orient_probs = self._orientation.predict(tensor_orig)
        orient_prob = max(orient_probs)

        if orient_label == config.ORIENTATION_FLIPPED_LABEL:
            rotation_applied = 180
            # img is already short-edge-resized, so this rotation is cheap
            # regardless of decode library -- resize-short-edge happens
            # exactly once (above), never redone at full resolution. This is
            # mathematically exact: for a uniform-scale aspect-preserving
            # resize, 180° rotation commutes with resize, so this is pixel-
            # identical to the old rotate-full-res-then-resize order.
            img_rot = img.transpose(Image.Transpose.ROTATE_180)  # lossless, not .rotate(180)
            tensor_final = transforms.preprocess(img_rot)
        else:
            rotation_applied = 0
            tensor_final = tensor_orig  # upright page: reuse, skip second backbone pass

        six_label, six_probs = self._sixclass.predict(tensor_final)

        return _row(
            exif_orientation_tag=exif_tag,
            orientation_pred=orient_label,
            orientation_prob=orient_prob,
            orientation_probs=orient_probs,
            rotation_applied=rotation_applied,
            sixclass_label=six_label,
            sixclass_probs=six_probs,
            final_label=six_label,
            model_version=self.model_version,
        )

    def run_batch(self, images_bytes: list[bytes]) -> list[dict]:
        """Batched counterpart to ``run``: same contract (never raises, one
        row per input, same order), but decodes+preprocesses the whole batch
        in parallel and runs each model exactly once per call instead of
        once per image.

        Failure isolation (mirrors ``ldv1``'s tile-batching pattern): a
        per-image decode failure is dropped before batching (never padded
        into the tensor) and immediately becomes a ``status="error"`` row
        for that image only. A failure in a whole-batch step (the shared
        normalize call or either model's forward pass) marks every
        surviving image in the batch as ``status="error"`` rather than
        raising and losing the whole batch.

        Known narrow residual risk, checked empirically: ``_center_crop``'s
        rare white-padding fallback (only hit when resize rounding leaves a
        crop 1px short -- see ``transforms.py``) computes its paste offset
        via floor division, biasing the extra padding pixel to the
        bottom/right. Rotating the source image before cropping (the old
        per-image path) vs. cropping then flipping the padded, normalized
        tensor (this method) do NOT produce pixel-identical output in this
        branch -- confirmed via synthetic 1px-short test images (max
        normalized-tensor diff ~4.5, ~19% of pixels affected). However, on
        those same test images neither the orientation nor six-class
        predicted label changed. This is a narrow, rare (rounding-only) edge
        case, not exhaustively re-tested against every possible crop-short
        scenario -- flagged here rather than silently assumed safe.
        """
        n = len(images_bytes)
        if n == 0:
            return []
        results: list[dict | None] = [None] * n

        # Step 1: decode+resize+crop each image in parallel. Futures are
        # submitted (and collected) in input order -- not as_completed() --
        # so results[] indices line up without a re-sort step; the threads
        # themselves still run concurrently.
        futures = [self._decode_pool.submit(transforms.decode_resize_crop, b) for b in images_bytes]
        imgs: list[Image.Image] = []
        exif_tags: list[int | None] = []
        idxs: list[int] = []
        for i, fut in enumerate(futures):
            try:
                img, tag = fut.result()
                imgs.append(img)
                exif_tags.append(tag)
                idxs.append(i)
            except Exception as e:
                results[i] = _row(status="error", error=str(e), model_version=self.model_version)

        if not imgs:  # every image in the batch failed to decode
            return results

        # Step 2: one batched normalize call -> [M,3,CROP_SIZE,CROP_SIZE].
        # Move to the model device exactly ONCE here: both forward passes and
        # the in-place flip below then operate on-device, so the batch crosses
        # the host->device boundary a single time (predict_batch's own
        # ``.to(device)`` becomes a no-op for an already-on-device tensor).
        try:
            tensor_orig = transforms.preprocess_batch(imgs).to(self._device)
        except Exception as e:
            for i in idxs:
                results[i] = _row(status="error", error=str(e), model_version=self.model_version)
            return results

        # Step 3: one orientation forward pass for the whole batch.
        try:
            orient_results = self._orientation.predict_batch(tensor_orig)
        except Exception as e:
            for i in idxs:
                results[i] = _row(status="error", error=str(e), model_version=self.model_version)
            return results

        # Step 4: build the sixclass input in place. Same [M,...] shape as
        # tensor_orig -- the batch does NOT grow. Rows the orientation model
        # called "flipped" are overwritten with their 180°-rotated version
        # at the same row index; every image still contributes exactly one
        # row, mirroring _run's tensor_final = tensor_orig (reuse) vs.
        # tensor_final = preprocess(img_rot) (replace) either/or.
        try:
            flip_mask = [label == config.ORIENTATION_FLIPPED_LABEL for label, _ in orient_results]
            tensor_final = tensor_orig.clone()
            flip_pos = [j for j, f in enumerate(flip_mask) if f]
            if flip_pos:
                # Index tensor on the same device as the batch so the gather/
                # flip/scatter stays entirely on-device (no host round-trip).
                pos = torch.tensor(flip_pos, dtype=torch.long, device=tensor_orig.device)
                tensor_final[pos] = torch.flip(tensor_orig[pos], dims=(-2, -1))
        except Exception as e:
            for i in idxs:
                results[i] = _row(status="error", error=str(e), model_version=self.model_version)
            return results

        # Step 5: one sixclass forward pass for the whole (corrected) batch.
        try:
            six_results = self._sixclass.predict_batch(tensor_final)
        except Exception as e:
            for i in idxs:
                results[i] = _row(status="error", error=str(e), model_version=self.model_version)
            return results

        # Step 6: assemble successful rows back into their original positions.
        try:
            for j, i in enumerate(idxs):
                orient_label, orient_probs = orient_results[j]
                six_label, six_probs = six_results[j]
                results[i] = _row(
                    exif_orientation_tag=exif_tags[j],
                    orientation_pred=orient_label,
                    orientation_prob=max(orient_probs),
                    orientation_probs=orient_probs,
                    rotation_applied=180 if flip_mask[j] else 0,
                    sixclass_label=six_label,
                    sixclass_probs=six_probs,
                    final_label=six_label,
                    model_version=self.model_version,
                )
            assert all(r is not None for r in results)
        except Exception as e:
            for i in idxs:
                results[i] = _row(status="error", error=str(e), model_version=self.model_version)
            return results

        return results
