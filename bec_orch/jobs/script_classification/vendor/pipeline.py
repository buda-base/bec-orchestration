import io
from pathlib import Path

from huggingface_hub import hf_hub_download
from PIL import Image

from . import config, transforms

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
    def __init__(self):
        self._processor = transforms.get_processor()

        orient_path = hf_hub_download(
            repo_id=config.ORIENTATION_REPO_ID, filename=config.CHECKPOINT_FILENAME
        )
        six_path = hf_hub_download(
            repo_id=config.SIXCLASS_REPO_ID, filename=config.CHECKPOINT_FILENAME
        )

        self._orientation = Classifier.from_checkpoint(orient_path, config.BACKBONE_ID)
        self._sixclass = Classifier.from_checkpoint(six_path, config.BACKBONE_ID)

        self.model_version = (
            f"orientation:{_short_sha(orient_path)};sixclass:{_short_sha(six_path)}"
        )

    def run(self, image_bytes: bytes) -> dict:
        try:
            return self._run(image_bytes)
        except Exception as e:
            return _row(status="error", error=str(e), model_version=self.model_version)

    def _run(self, image_bytes: bytes) -> dict:
        img = Image.open(io.BytesIO(image_bytes))
        exif_tag = img.getexif().get(0x0112)  # raw tag only, never applied to pixels
        img = img.convert("RGB")

        # if is_blank(img):
        #     return _row(
        #         blank=True,
        #         exif_orientation_tag=exif_tag,
        #         model_version=self.model_version,
        #     )

        tensor_orig = transforms.preprocess(img)
        orient_label, orient_probs = self._orientation.predict(tensor_orig)
        orient_prob = max(orient_probs)

        if orient_label == config.ORIENTATION_FLIPPED_LABEL:
            rotation_applied = 180
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
            rotation_applied=rotation_applied,
            sixclass_label=six_label,
            sixclass_probs=six_probs,
            final_label=six_label,
            model_version=self.model_version,
        )
