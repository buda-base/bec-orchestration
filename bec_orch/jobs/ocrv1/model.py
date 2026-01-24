"""OCR model wrapper for ONNX inference - GPU only."""

import numpy as np
import numpy.typing as npt
import onnxruntime as ort
from scipy.special import log_softmax

from .utils import get_execution_providers


class OCRModel:
    def __init__(
        self,
        model_file: str,
        input_layer: str,
        output_layer: str,
        squeeze_channel: bool,
        swap_hw: bool,
        apply_log_softmax: bool = True,
    ) -> None:
        self._input_layer = input_layer
        self._output_layer = output_layer
        self._squeeze_channel_dim = squeeze_channel
        self._swap_hw = swap_hw
        self._apply_log_softmax = apply_log_softmax

        execution_providers = get_execution_providers()
        self.session = ort.InferenceSession(model_file, providers=execution_providers)

    def predict(self, tensor: npt.NDArray) -> npt.NDArray:
        """Run ONNX inference on preprocessed tensor, return log probabilities.
        
        Applies log_softmax immediately after inference while data is hot in cache.
        This is more efficient than applying it later per-line in worker processes.
        """
        tensor = tensor.astype(np.float32)

        if self._swap_hw:
            tensor = np.transpose(tensor, axes=[0, 2, 1])

        if not self._squeeze_channel_dim:
            tensor = np.expand_dims(tensor, axis=1)

        ort_value = ort.OrtValue.ortvalue_from_numpy(tensor)
        results = self.session.run_with_ort_values([self._output_layer], {self._input_layer: ort_value})
        logits = np.squeeze(results[0].numpy())
        
        # Apply log_softmax immediately while data is hot in cache
        # This converts raw logits to log probabilities batch-wise
        if self._apply_log_softmax:
            # The model outputs (vocab, time) for single items, (batch, vocab, time) for batches
            # Vocab dimension is typically ~10000, time is typically ~800
            # We need to apply log_softmax along the VOCAB axis, not time
            if logits.ndim == 2:
                # Single item: shape (vocab, time) - vocab is axis 0
                logits = log_softmax(logits, axis=0).astype(np.float32)
            else:
                # Batch: shape (batch, vocab, time) - vocab is axis 1
                logits = log_softmax(logits, axis=1).astype(np.float32)
        
        return logits
