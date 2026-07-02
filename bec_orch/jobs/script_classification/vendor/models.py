import torch
import torch.nn as nn
from transformers import AutoModel


class DINOv3Classifier(nn.Module):
    def __init__(self, model_id: str, num_classes: int, dropout: float = 0.1, pooling: str = "cls"):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(model_id)
        self.pooling = pooling
        hidden = self.backbone.config.hidden_size
        # "cls_mean_std" (used by the 6-class checkpoint) concatenates CLS with
        # mean/std of patch tokens -> 3x hidden; "cls" (orientation checkpoint) is CLS only
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
            num_register = self.backbone.config.num_register_tokens  # prefix tokens before patch tokens
            patches = seq[:, 1 + num_register:, :]
            feat = torch.cat([cls, patches.mean(dim=1), patches.std(dim=1)], dim=-1)
        else:
            feat = cls
        return self.head(feat)


class Classifier:
    def __init__(self, model: DINOv3Classifier, idx_to_label: dict):
        self.model = model.eval()
        self.idx_to_label = idx_to_label

    @classmethod
    def from_checkpoint(cls, path: str, model_id: str) -> "Classifier":
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        # checkpoints store either idx_to_label (str keys) or label_to_idx; normalize to int-keyed idx_to_label
        if "idx_to_label" in ckpt:
            idx_to_label = {int(k): v for k, v in ckpt["idx_to_label"].items()}
        else:
            idx_to_label = {v: k for k, v in ckpt["label_to_idx"].items()}
        pooling = ckpt.get("pooling", "cls")
        model = DINOv3Classifier(model_id, num_classes=len(idx_to_label), pooling=pooling)
        model.load_state_dict(ckpt["model_state_dict"])
        return cls(model, idx_to_label)

    @torch.no_grad()
    def predict(self, pixel_values):
        logits = self.model(pixel_values)
        probs = torch.softmax(logits, dim=-1)[0].tolist()
        idx = max(range(len(probs)), key=probs.__getitem__)
        return self.idx_to_label[idx], probs
