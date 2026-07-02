import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel


class DINOv3Classifier(nn.Module):
    def __init__(self, model_id: str, num_classes: int, dropout: float = 0.1, pooling: str = "cls"):
        super().__init__()
        # Deviation from upstream: upstream calls AutoModel.from_pretrained(model_id),
        # which downloads the full gated backbone checkpoint just to have every
        # one of its weights immediately overwritten by from_checkpoint()'s
        # load_state_dict(ckpt["model_state_dict"]) below (confirmed empirically:
        # the checkpoint's state dict contains all 211 backbone.* keys, and
        # load_state_dict defaults to strict=True). Building from config only
        # fetches the small config.json (still needs gated-repo access, but no
        # full weight download/materialization) since the real fine-tuned
        # weights get loaded right after regardless of initial values.
        config = AutoConfig.from_pretrained(model_id)
        self.backbone = AutoModel.from_config(config)
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
    # Deviation from upstream: adds device handling (upstream is CPU-only,
    # no .to(device) anywhere). `device` defaults to "cpu" so any external/
    # test caller that constructs `Classifier(model, idx_to_label)` directly
    # keeps working unchanged.
    def __init__(self, model: DINOv3Classifier, idx_to_label: dict, device: str = "cpu"):
        self.model = model.eval().to(device)
        self.idx_to_label = idx_to_label
        self.device = device

    @classmethod
    def from_checkpoint(cls, path: str, model_id: str, device: str = "cpu") -> "Classifier":
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        # checkpoints store either idx_to_label (str keys) or label_to_idx; normalize to int-keyed idx_to_label
        if "idx_to_label" in ckpt:
            idx_to_label = {int(k): v for k, v in ckpt["idx_to_label"].items()}
        else:
            idx_to_label = {v: k for k, v in ckpt["label_to_idx"].items()}
        pooling = ckpt.get("pooling", "cls")
        model = DINOv3Classifier(model_id, num_classes=len(idx_to_label), pooling=pooling)
        model.load_state_dict(ckpt["model_state_dict"])
        return cls(model, idx_to_label, device=device)

    @torch.no_grad()
    def predict(self, pixel_values):
        logits = self.model(pixel_values.to(self.device))
        probs = torch.softmax(logits, dim=-1)[0].tolist()
        idx = max(range(len(probs)), key=probs.__getitem__)
        return self.idx_to_label[idx], probs
