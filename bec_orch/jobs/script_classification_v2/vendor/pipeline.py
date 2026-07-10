"""Single-model classification pipeline for ``script_classification_v2``.

Loads one fine-tuned DINOv3 classifier and runs it over each page. Public
surface mirrors ``script_classification``'s pipeline (``run`` / ``run_batch``,
never raise, one row per input, input order preserved) so the worker plumbing
is nearly identical — but each row carries a single ``label`` / ``prob`` /
``probs`` instead of orientation + six-class fields.
"""

import logging
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download

from . import transforms
from .models import Classifier

logger = logging.getLogger(__name__)

_NULL_FIELDS = dict(
    exif_orientation_tag=None,
    label=None,
    prob=None,
    probs=None,
)


def _short_sha(resolved_path: str) -> str:
    # hf_hub_download cache layout: .../snapshots/<commit_sha>/<filename>
    return Path(resolved_path).parent.name[:8]


def _row(**overrides) -> dict:
    row = {"status": "ok", "error": None, **_NULL_FIELDS}
    row.update(overrides)
    return row


class Pipeline:
    def __init__(self, cfg=None):
        # Duck-typed against the job config so vendor/ has no hard dependency
        # on the job-level config package.
        self._backbone_id = getattr(cfg, "backbone_id", "facebook/dinov3-vits16-pretrain-lvd1689m")
        self._crop_size = int(getattr(cfg, "crop_size", 224))
        model_repo_id = getattr(cfg, "model_repo_id", "BDRC/8-class-tibetan-page-classifier")
        checkpoint_filename = getattr(cfg, "checkpoint_filename", "final_model.pt")

        device = "cpu"
        if getattr(cfg, "use_gpu", True):
            if torch.cuda.is_available():
                device = "cuda"
                logger.info(
                    f"[script_classification_v2] using GPU: {torch.cuda.get_device_name(0)}"
                )
            else:
                logger.warning(
                    "[script_classification_v2] use_gpu=True but CUDA not available, using CPU"
                )
        self._device = device

        self._decode_pool = ThreadPoolExecutor(
            max_workers=max(1, getattr(cfg, "decode_workers", 1)),
            thread_name_prefix="scriptclsv2-decode",
        )

        ckpt_path = hf_hub_download(repo_id=model_repo_id, filename=checkpoint_filename)
        self._clf = Classifier.from_checkpoint(ckpt_path, self._backbone_id, device=device)
        self.model_version = f"{Path(model_repo_id).name}:{_short_sha(ckpt_path)}"

        logger.info(
            f"[script_classification_v2] loaded {model_repo_id} "
            f"({len(self._clf.idx_to_label)} classes, crop={self._crop_size}, "
            f"pooling={self._clf.model.pooling}) on {device}"
        )

    @property
    def labels(self) -> list[str]:
        # Class name per logit index (0..C-1), so a bare probability vector is
        # self-describing downstream.
        idx_to_label = self._clf.idx_to_label
        return [idx_to_label[i] for i in range(len(idx_to_label))]

    def run(self, image_bytes: bytes) -> dict:
        return self.run_batch([image_bytes])[0]

    def run_batch(self, images_bytes: list[bytes]) -> list[dict]:
        """Decode + preprocess the whole batch in parallel, then run the model
        once. Never raises: a per-image decode failure becomes a single
        ``status="error"`` row; a whole-batch step failure marks every
        surviving image in the batch as ``status="error"``.
        """
        n = len(images_bytes)
        if n == 0:
            return []
        results: list[dict | None] = [None] * n

        # Step 1: decode+resize+crop each image in parallel, in input order.
        futures = [
            self._decode_pool.submit(transforms.decode_resize_crop, b, self._crop_size)
            for b in images_bytes
        ]
        imgs = []
        exif_tags: list[int | None] = []
        idxs: list[int] = []
        for i, fut in enumerate(futures):
            try:
                img, tag = fut.result()
                imgs.append(img)
                exif_tags.append(tag)
                idxs.append(i)
            except Exception as e:  # noqa: BLE001
                results[i] = _row(status="error", error=str(e), model_version=self.model_version)

        if not imgs:  # every image in the batch failed to decode
            return results

        # Step 2: one batched normalize call.
        try:
            tensor = transforms.preprocess_batch(imgs, self._backbone_id, self._crop_size).to(
                self._device
            )
        except Exception as e:  # noqa: BLE001
            for i in idxs:
                results[i] = _row(status="error", error=str(e), model_version=self.model_version)
            return results

        # Step 3: one forward pass for the whole batch.
        try:
            preds = self._clf.predict_batch(tensor)
        except Exception as e:  # noqa: BLE001
            for i in idxs:
                results[i] = _row(status="error", error=str(e), model_version=self.model_version)
            return results

        # Step 4: assemble successful rows back into their original positions.
        try:
            for j, i in enumerate(idxs):
                label, probs = preds[j]
                results[i] = _row(
                    exif_orientation_tag=exif_tags[j],
                    label=label,
                    prob=max(probs),
                    probs=probs,
                    model_version=self.model_version,
                )
            assert all(r is not None for r in results)
        except Exception as e:  # noqa: BLE001
            for i in idxs:
                results[i] = _row(status="error", error=str(e), model_version=self.model_version)
            return results

        return results
