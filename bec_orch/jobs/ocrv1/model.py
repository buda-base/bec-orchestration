"""OCR model wrapper for ONNX inference - GPU only."""

import logging

import numpy as np
import numpy.typing as npt
import onnxruntime as ort
from scipy.special import log_softmax as scipy_log_softmax

from .utils import get_execution_providers

logger = logging.getLogger(__name__)

# Check if PyTorch with CUDA is available for GPU log_softmax
_TORCH_CUDA_AVAILABLE = False
try:
    import torch
    import torch.nn.functional as F
    if torch.cuda.is_available():
        _TORCH_CUDA_AVAILABLE = True
        logger.info("[OCRModel] PyTorch CUDA available - will use GPU for log_softmax")
    else:
        logger.info("[OCRModel] PyTorch available but CUDA not available - using CPU log_softmax")
except ImportError:
    logger.info("[OCRModel] PyTorch not available - using scipy CPU log_softmax")


class OCRModel:
    def __init__(
        self,
        model_file: str,
        input_layer: str,
        output_layer: str,
        squeeze_channel: bool,
        swap_hw: bool,
        apply_log_softmax: bool = True,
        use_gpu_log_softmax: bool = True,
    ) -> None:
        self._input_layer = input_layer
        self._output_layer = output_layer
        self._squeeze_channel_dim = squeeze_channel
        self._swap_hw = swap_hw
        self._apply_log_softmax = apply_log_softmax
        self._use_gpu_log_softmax = use_gpu_log_softmax and _TORCH_CUDA_AVAILABLE

        execution_providers = get_execution_providers()
        self.session = ort.InferenceSession(model_file, providers=execution_providers)
        
        if self._apply_log_softmax:
            if self._use_gpu_log_softmax:
                logger.info("[OCRModel] Using PyTorch GPU log_softmax")
            else:
                logger.info("[OCRModel] Using scipy CPU log_softmax")

    def _log_softmax_gpu(self, logits: npt.NDArray, axis: int) -> npt.NDArray:
        """Apply log_softmax on GPU using PyTorch."""
        # Convert to PyTorch tensor on GPU
        logits_tensor = torch.from_numpy(logits).cuda()
        # Apply log_softmax on specified axis
        log_probs_tensor = F.log_softmax(logits_tensor, dim=axis)
        # Copy back to CPU numpy array
        return log_probs_tensor.cpu().numpy().astype(np.float32)

    def _log_softmax_cpu(self, logits: npt.NDArray, axis: int) -> npt.NDArray:
        """Apply log_softmax on CPU using scipy."""
        return scipy_log_softmax(logits, axis=axis).astype(np.float32)

    def predict(self, tensor: npt.NDArray) -> npt.NDArray:
        """Run ONNX inference on preprocessed tensor, return log probabilities.
        
        Applies log_softmax immediately after inference.
        Uses GPU (PyTorch) if available, otherwise falls back to CPU (scipy).
        """
        tensor = tensor.astype(np.float32)

        if self._swap_hw:
            tensor = np.transpose(tensor, axes=[0, 2, 1])

        if not self._squeeze_channel_dim:
            tensor = np.expand_dims(tensor, axis=1)

        ort_value = ort.OrtValue.ortvalue_from_numpy(tensor)
        results = self.session.run_with_ort_values([self._output_layer], {self._input_layer: ort_value})
        logits = np.squeeze(results[0].numpy())
        
        # Apply log_softmax immediately after inference
        # This converts raw logits to log probabilities batch-wise
        if self._apply_log_softmax:
            # The model outputs (vocab, time) for single items, (batch, vocab, time) for batches
            # Vocab dimension is typically ~10000, time is typically ~800
            # We need to apply log_softmax along the VOCAB axis, not time
            log_softmax_fn = self._log_softmax_gpu if self._use_gpu_log_softmax else self._log_softmax_cpu
            
            if logits.ndim == 2:
                # Single item: shape (vocab, time) - vocab is axis 0
                logits = log_softmax_fn(logits, axis=0)
            else:
                # Batch: shape (batch, vocab, time) - vocab is axis 1
                logits = log_softmax_fn(logits, axis=1)
        
        return logits
