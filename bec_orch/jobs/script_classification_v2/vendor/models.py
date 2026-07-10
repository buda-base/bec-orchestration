"""DINOv3 classifier model + checkpoint loader.

Copied verbatim from ``script_classification/vendor/models.py`` — this loader
is model-agnostic (reads num_classes / pooling / register-token count from the
checkpoint), so it loads the 8-class page classifier unchanged.
"""

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel


def _build_backbone(model_id: str, num_register_tokens: int | None = None):
    """Instantiate the DINOv3 backbone *architecture* (random init; the real
    fine-tuned weights are loaded over it by ``Classifier.from_checkpoint``).

    Prefer the canonical Hub config when reachable, but fall back to
    transformers' built-in ``DINOv3ViTConfig`` so we never depend on gated
    access to the backbone repo. ``num_register_tokens`` (4 for the BDRC
    models, vs the library default 0) is taken from the checkpoint so the
    fallback backbone matches the weights exactly.
    """
    try:
        config = AutoConfig.from_pretrained(model_id)
        return AutoModel.from_config(config)
    except Exception:
        from transformers import DINOv3ViTConfig, DINOv3ViTModel

        kwargs = {}
        if num_register_tokens is not None:
            kwargs["num_register_tokens"] = num_register_tokens
        return DINOv3ViTModel(DINOv3ViTConfig(**kwargs))


class DINOv3Classifier(nn.Module):
    def __init__(
        self,
        model_id: str,
        num_classes: int,
        dropout: float = 0.1,
        pooling: str = "cls",
        num_register_tokens: int | None = None,
    ):
        super().__init__()
        self.backbone = _build_backbone(model_id, num_register_tokens)
        self.pooling = pooling
        hidden = self.backbone.config.hidden_size
        # "cls_mean_std" concatenates CLS with mean/std of patch tokens ->
        # 3x hidden; "cls" is CLS only.
        feat_dim = hidden * 3 if pooling == "cls_mean_std" else hidden
        self.head = nn.Sequential(
            nn.LayerNorm(feat_dim),
            nn.Dropout(dropout),
            nn.Linear(feat_dim, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, pixel_values):
        out = self.backbone(pixel_values=pixel_values)
        seq = out.last_hidden_state
        cls = seq[:, 0, :]  # CLS token (index 0)
        if self.pooling == "cls_mean_std":
            num_register = self.backbone.config.num_register_tokens
            patches = seq[:, 1 + num_register:, :]
            feat = torch.cat([cls, patches.mean(dim=1), patches.std(dim=1)], dim=-1)
        else:
            feat = cls
        return self.head(feat)


class Classifier:
    def __init__(self, model: DINOv3Classifier, idx_to_label: dict, device: str = "cpu"):
        self.model = model.eval().to(device)
        self.idx_to_label = idx_to_label
        self.device = device

    @classmethod
    def from_checkpoint(cls, path: str, model_id: str, device: str = "cpu") -> "Classifier":
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        # checkpoints store either idx_to_label (str keys) or label_to_idx;
        # normalize to int-keyed idx_to_label.
        if "idx_to_label" in ckpt:
            idx_to_label = {int(k): v for k, v in ckpt["idx_to_label"].items()}
        else:
            idx_to_label = {v: k for k, v in ckpt["label_to_idx"].items()}
        pooling = ckpt.get("pooling", "cls")
        reg = ckpt["model_state_dict"].get("backbone.embeddings.register_tokens")
        num_register_tokens = int(reg.shape[1]) if reg is not None else None
        model = DINOv3Classifier(
            model_id,
            num_classes=len(idx_to_label),
            pooling=pooling,
            num_register_tokens=num_register_tokens,
        )
        model.load_state_dict(ckpt["model_state_dict"])
        return cls(model, idx_to_label, device=device)

    def predict(self, pixel_values):
        return self.predict_batch(pixel_values)[0]

    @torch.no_grad()
    def predict_batch(self, pixel_values):
        # Batched inference. Safe for batch size 1: the head is LayerNorm
        # (per-sample) + Dropout (disabled by .eval()), and the ViT backbone
        # has no BatchNorm, so there is no cross-row coupling.
        logits = self.model(pixel_values.to(self.device))
        probs = torch.softmax(logits, dim=-1).tolist()
        out = []
        for row in probs:
            idx = max(range(len(row)), key=row.__getitem__)
            out.append((self.idx_to_label[idx], row))
        return out
