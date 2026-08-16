"""vLLM general plugin: force PaddleOCR-VL into the *sequential* image-token M-RoPE regime.

Vendored from ``bec-ocr-training/deploy/vllm_paddleocr_seqpos`` so the production serving
recipe is reproduced exactly. Install into the SAME env that runs vLLM (e.g. the DLAMI
``/opt/pytorch`` venv):

    /opt/pytorch/bin/pip install bec_orch/jobs/paddleocr/vllm_paddleocr_seqpos

Why this exists
---------------
PaddleOCR-VL uses M-RoPE. The ``mm_token_type_ids`` mask selects the image-token position
regime: **sequential** (mask zeroed -> image tokens get plain 1D positions, like text) vs
**grid** (mask marked -> native 2D image-grid M-RoPE). vLLM's in-tree PaddleOCR-VL model
(``vllm/model_executor/models/paddleocr_vl.py``) does not read ``mm_token_type_ids`` at all;
its ``get_mrope_input_positions`` *unconditionally* builds the 2D grid from ``image_grid_thw``.
So stock vLLM can only serve the ``grid`` regime, and there is no processor/kwarg knob to
change it.

Checkpoints trained/validated in the ``sequential`` regime (e.g. ``elie_v6_coarse_grow26_ep2``)
therefore suffer a train/serve mismatch under vLLM: line skips/merges on structured book/list
pages and a large CER regression. This plugin closes that gap by overriding the one method so
image tokens get 1D sequential positions -- verified on vLLM 0.27.1 to reproduce the HF
``--image-token-positions sequential`` oracle exactly (TSAM-CHOE page 2: 479 generated tokens,
vs 816 under grid, with the catalog numbering restored).

How it loads
------------
vLLM calls ``vllm.general_plugins`` entry points with no args in process0, the EngineCore
process, and every worker process (``vllm/plugins/__init__.py``), which is exactly where
``get_mrope_input_positions`` runs. The override is a class-attribute swap, applied once
(idempotent -- plugins may be loaded multiple times per process).

Gating
------
The plugin is a **no-op** unless the environment variable ``OCR_VLLM_IMAGE_TOKEN_POSITIONS``
equals ``sequential`` (case-insensitive). The serving process sets this from the checkpoint's
resolved regime before constructing the engine; the child processes inherit it. Anything else
(``grid``/unset) leaves vLLM's native behaviour untouched.
"""

from __future__ import annotations

import logging
import os
from typing import Any

logger = logging.getLogger(__name__)

ENV_FLAG = "OCR_VLLM_IMAGE_TOKEN_POSITIONS"
_state = {"patched": False}


def _sequential_mrope_positions(self, input_tokens, mm_features) -> tuple[Any, int]:  # noqa: ANN001, ARG001
    """1D sequential positions for the whole sequence (image tokens included).

    Equivalent to zeroing ``mm_token_type_ids``: every token (text and image) gets the plain
    ``0..n-1`` position broadcast across the 3 M-RoPE sections, so M-RoPE degenerates to
    standard 1D RoPE. ``mrope_position_delta`` is 0 so decode continues at ``n, n+1, ...``.
    """
    import numpy as np
    import torch

    n = len(input_tokens)
    positions = np.broadcast_to(np.arange(n, dtype=np.int64), (3, n))
    return torch.from_numpy(np.ascontiguousarray(positions)), 0


def apply() -> None:
    """Entry point invoked by vLLM in every process. Idempotent; gated by env."""
    if _state["patched"]:
        return
    regime = os.environ.get(ENV_FLAG, "").strip().lower()
    if regime != "sequential":
        logger.debug("paddleocr-seqpos: %s=%r != 'sequential'; not patching", ENV_FLAG, regime)
        return
    try:
        from vllm.model_executor.models import paddleocr_vl as mod
    except Exception:
        logger.exception("paddleocr-seqpos: could not import PaddleOCR-VL model; not patching")
        return

    mod.PaddleOCRVLForConditionalGeneration.get_mrope_input_positions = _sequential_mrope_positions
    _state["patched"] = True
    logger.warning(
        "paddleocr-seqpos: forcing SEQUENTIAL (1D) image-token M-RoPE for PaddleOCR-VL "
        "(mm_token_type_ids-zeroed regime). Grid-trained checkpoints must NOT set %s=sequential.",
        ENV_FLAG,
    )
