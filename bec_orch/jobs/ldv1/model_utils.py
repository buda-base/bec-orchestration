"""Model loading utilities for the line detection pipeline.

This variant adds optional inference-time performance knobs:
- FP16 (recommended on NVIDIA T4 / CC 7.5) or BF16 (Ampere+ only)
- channels_last
- torch.compile

Notes:
- On T4, BF16 is not hardware-supported; prefer FP16.
- For FP16, we keep BatchNorm layers in FP32 for numerical stability.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Literal, Optional

import torch


Precision = Literal["fp32", "fp16", "bf16", "auto"]


def _supports_bf16_cuda() -> bool:
    # PyTorch exposes a helper; fall back to compute capability check.
    try:
        return bool(torch.cuda.is_bf16_supported())  # type: ignore[attr-defined]
    except Exception:
        try:
            major, _minor = torch.cuda.get_device_capability()
            return major >= 8
        except Exception:
            return False


def _convert_batchnorm_to_fp32(model: torch.nn.Module) -> None:
    import torch.nn as nn

    for m in model.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            m.float()


def load_model(
    checkpoint_path: str | Path,
    *,
    classes: int = 1,
    device: Optional[torch.device | str] = None,
    precision: Precision = "auto",
    use_channels_last: bool = True,
    compile_model: bool = False,
    compile_mode: str = "reduce-overhead",
) -> torch.nn.Module:
    """
    Load the segmentation model from a checkpoint.

    Assumptions (as per existing training artifacts):
      - checkpoint is a dict with a "state_dict" key.
      - model architecture is DeepLabV3Plus from segmentation_models_pytorch.

    Performance defaults:
      - load weights on CPU (fast, avoids GPU RAM spikes)
      - eval + disable grads
      - optionally move to CUDA, set FP16/BF16, channels_last, and torch.compile

    Args:
      device: e.g. "cuda" or "cuda:0" or "cpu". If None, keeps model on CPU.
      precision:
        - "auto": BF16 on Ampere+ else FP16 on CUDA else FP32
        - "fp16": force FP16 (good default on T4)
        - "bf16": force BF16 (Ampere+ only; raises if unsupported on CUDA)
        - "fp32": keep FP32
      compile_model: if True, calls torch.compile(model, mode=compile_mode)
    """
    ckpt_path = Path(checkpoint_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {ckpt_path}")

    checkpoint = torch.load(ckpt_path, map_location="cpu")
    if not isinstance(checkpoint, dict) or "state_dict" not in checkpoint:
        raise ValueError("Expected checkpoint dict with a 'state_dict' key")

    state_dict: Dict[str, torch.Tensor] = checkpoint["state_dict"]
    # Common training wrappers (DataParallel, Lightning, etc.)
    if any(k.startswith("module.") for k in state_dict.keys()):
        state_dict = {k[len("module."):]: v for k, v in state_dict.items()}

    import segmentation_models_pytorch as sm

    # Keep this fully local/offline by default (no encoder pretrained weights download).
    model = sm.DeepLabV3Plus(encoder_name="resnet34", encoder_weights=None, classes=classes)
    model.load_state_dict(state_dict, strict=True)

    model.eval()
    model.requires_grad_(False)

    # ---- Optional inference optimizations
    if device is not None:
        device = torch.device(device)

        # cuDNN benchmarking: when True, benchmarks different algorithms for each
        # input shape and caches the fastest. HOWEVER, this causes multi-second
        # delays when batch size varies (e.g., last batch of a volume).
        # Set to False for consistent performance with variable batch sizes.
        if device.type == "cuda":
            torch.backends.cudnn.benchmark = False

            # TF32 is Ampere+ only; harmless to set on T4.
            try:
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
            except Exception:
                pass

        # Move weights first, then apply memory format / dtype.
        model = model.to(device)

        if use_channels_last and device.type == "cuda":
            model = model.to(memory_format=torch.channels_last)

        chosen: Precision = precision
        if precision == "auto":
            if device.type == "cuda" and _supports_bf16_cuda():
                chosen = "bf16"
            elif device.type == "cuda":
                chosen = "fp16"
            else:
                chosen = "fp32"

        if chosen == "bf16":
            if device.type != "cuda":
                raise ValueError("bf16 requested but device is not CUDA")
            if not _supports_bf16_cuda():
                raise ValueError("bf16 requested but CUDA device does not support bf16 (need Ampere+/CC>=8.0)")
            model = model.to(dtype=torch.bfloat16)
            _convert_batchnorm_to_fp32(model)

        elif chosen == "fp16":
            if device.type == "cuda":
                model = model.to(dtype=torch.float16)
                _convert_batchnorm_to_fp32(model)

        # Stash for callers (optional use)
        model._ld_precision = chosen  # type: ignore[attr-defined]

        if compile_model:
            # torch.compile works best after the model is on its final device/dtype.
            model = torch.compile(model, mode=compile_mode)  # type: ignore[assignment]

    return model

