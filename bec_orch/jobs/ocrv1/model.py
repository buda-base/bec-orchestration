"""OCR model wrapper for ONNX inference - GPU only."""

import logging
from typing import Union

import numpy as np
import numpy.typing as npt
import onnxruntime as ort
from scipy.special import log_softmax as scipy_log_softmax

from .utils import get_execution_providers

logger = logging.getLogger(__name__)

# Check if PyTorch with CUDA is available for GPU operations
_TORCH_CUDA_AVAILABLE = False
_torch = None
_F = None
try:
    import torch as _torch
    import torch.nn.functional as _F
    if _torch.cuda.is_available():
        _TORCH_CUDA_AVAILABLE = True
        logger.info("[OCRModel] PyTorch CUDA available - will use GPU for log_softmax and vocab pruning")
    else:
        logger.info("[OCRModel] PyTorch available but CUDA not available - using CPU operations")
except ImportError:
    logger.info("[OCRModel] PyTorch not available - using scipy CPU operations")


class OCRModel:
    def __init__(
        self,
        model_file: str,
        input_layer: str,
        output_layer: str,
        squeeze_channel: bool,
        swap_hw: bool,
        apply_log_softmax: bool = True,
        use_gpu_operations: bool = True,
        vocab_prune_threshold: float | None = -10.0,
    ) -> None:
        self._input_layer = input_layer
        self._output_layer = output_layer
        self._squeeze_channel_dim = squeeze_channel
        self._swap_hw = swap_hw
        self._apply_log_softmax = apply_log_softmax
        self._use_gpu = use_gpu_operations and _TORCH_CUDA_AVAILABLE
        self._vocab_prune_threshold = vocab_prune_threshold

        execution_providers = get_execution_providers()
        self.session = ort.InferenceSession(model_file, providers=execution_providers)
        
        if self._use_gpu:
            logger.info(
                f"[OCRModel] Using PyTorch GPU for log_softmax + vocab pruning "
                f"(threshold={vocab_prune_threshold})"
            )
        else:
            logger.info(
                f"[OCRModel] Using scipy/numpy CPU for log_softmax + vocab pruning "
                f"(threshold={vocab_prune_threshold})"
            )

    def _process_logits_gpu(
        self, logits: npt.NDArray, vocab_axis: int
    ) -> tuple[npt.NDArray, npt.NDArray | None]:
        """Apply log_softmax and vocab pruning on GPU using PyTorch.
        
        Returns:
            Tuple of (log_probs, keep_indices) where keep_indices is None if no pruning.
        """
        # Convert to PyTorch tensor on GPU
        logits_tensor = _torch.from_numpy(logits).cuda()
        
        # Apply log_softmax on vocab axis
        log_probs_tensor = _F.log_softmax(logits_tensor, dim=vocab_axis)
        
        # Apply vocabulary pruning on GPU if threshold is set
        keep_indices = None
        if self._vocab_prune_threshold is not None:
            # Find max log prob per vocab token across time
            # For (vocab, time), max over time is dim=1 for 2D, or last dim for batch
            time_axis = 1 if logits.ndim == 2 else 2
            max_per_token = log_probs_tensor.max(dim=time_axis).values
            
            # For batch, take max across batch too
            if logits.ndim == 3:
                max_per_token = max_per_token.max(dim=0).values
            
            # Create mask for tokens above threshold (always keep blank at index 0)
            keep_mask = max_per_token > self._vocab_prune_threshold
            keep_mask[0] = True  # Always keep blank token
            
            # Get indices of kept tokens
            keep_indices = _torch.where(keep_mask)[0].cpu().numpy()
            
            # Prune vocabulary dimension
            if logits.ndim == 2:
                # (vocab, time) -> (pruned_vocab, time)
                log_probs_tensor = log_probs_tensor[keep_mask, :]
            else:
                # (batch, vocab, time) -> (batch, pruned_vocab, time)
                log_probs_tensor = log_probs_tensor[:, keep_mask, :]
        
        # Copy back to CPU numpy array
        log_probs = log_probs_tensor.cpu().numpy().astype(np.float32)
        
        return log_probs, keep_indices

    def _process_logits_cpu(
        self, logits: npt.NDArray, vocab_axis: int
    ) -> tuple[npt.NDArray, npt.NDArray | None]:
        """Apply log_softmax and vocab pruning on CPU using scipy/numpy.
        
        Returns:
            Tuple of (log_probs, keep_indices) where keep_indices is None if no pruning.
        """
        log_probs = scipy_log_softmax(logits, axis=vocab_axis).astype(np.float32)
        
        # Apply vocabulary pruning on CPU if threshold is set
        keep_indices = None
        if self._vocab_prune_threshold is not None:
            # Find max log prob per vocab token across time
            # For (vocab, time), max over time is axis=1 for 2D
            time_axis = 1 if logits.ndim == 2 else 2
            max_per_token = log_probs.max(axis=time_axis)
            
            # For batch, take max across batch too
            if logits.ndim == 3:
                max_per_token = max_per_token.max(axis=0)
            
            # Create mask for tokens above threshold (always keep blank at index 0)
            keep_mask = max_per_token > self._vocab_prune_threshold
            keep_mask[0] = True  # Always keep blank token
            
            # Get indices of kept tokens
            keep_indices = np.where(keep_mask)[0]
            
            # Prune vocabulary dimension
            if logits.ndim == 2:
                # (vocab, time) -> (pruned_vocab, time)
                log_probs = log_probs[keep_mask, :]
            else:
                # (batch, vocab, time) -> (batch, pruned_vocab, time)
                log_probs = log_probs[:, keep_mask, :]
        
        return log_probs, keep_indices

    def predict(
        self, tensor: npt.NDArray
    ) -> tuple[npt.NDArray, npt.NDArray | None]:
        """Run ONNX inference on preprocessed tensor, return log probabilities.
        
        Applies log_softmax immediately after inference.
        Also applies vocabulary pruning to reduce IPC bandwidth.
        
        Returns:
            Tuple of (log_probs, keep_indices) where keep_indices contains the
            original vocabulary indices that were kept after pruning, or None
            if pruning is disabled.
        """
        tensor = tensor.astype(np.float32)

        if self._swap_hw:
            tensor = np.transpose(tensor, axes=[0, 2, 1])

        if not self._squeeze_channel_dim:
            tensor = np.expand_dims(tensor, axis=1)

        ort_value = ort.OrtValue.ortvalue_from_numpy(tensor)
        results = self.session.run_with_ort_values([self._output_layer], {self._input_layer: ort_value})
        logits = np.squeeze(results[0].numpy())
        
        # Apply log_softmax (and optionally vocab pruning) immediately after inference
        if self._apply_log_softmax:
            # The model outputs (vocab, time) for single items, (batch, vocab, time) for batches
            # Vocab dimension is axis 0 for 2D, axis 1 for 3D
            vocab_axis = 0 if logits.ndim == 2 else 1
            
            if self._use_gpu:
                return self._process_logits_gpu(logits, vocab_axis)
            else:
                return self._process_logits_cpu(logits, vocab_axis)
        
        return logits, None
