"""OCR model wrapper for ONNX inference."""

import numpy as np
import numpy.typing as npt
import onnxruntime as ort

from .utils import get_execution_providers
from pyctcdecode.decoder import build_ctcdecoder


class OCRModel:
    def __init__(
        self,
        model_file: str,
        input_layer: str,
        output_layer: str,
        charset: str | list[str],
        squeeze_channel: bool,
        swap_hw: bool,
        add_blank: bool,
    ) -> None:
        self._input_layer = input_layer
        self._output_layer = output_layer
        self._squeeze_channel_dim = squeeze_channel
        self._swap_hw = swap_hw

        execution_providers = get_execution_providers()
        self.session = ort.InferenceSession(model_file, providers=execution_providers)

        if isinstance(charset, str):
            self.charset = list(charset)
        else:
            self.charset = charset

        self.ctc_vocab = self.charset.copy()
        if add_blank and " " not in self.ctc_vocab:
            self.ctc_vocab.insert(0, " ")

        self.ctc_decoder = build_ctcdecoder(self.ctc_vocab)

    def predict(self, tensor: npt.NDArray) -> npt.NDArray:
        """Run ONNX inference on preprocessed tensor, return logits."""
        tensor = tensor.astype(np.float32)

        if self._swap_hw:
            tensor = np.transpose(tensor, axes=[0, 2, 1])

        if not self._squeeze_channel_dim:
            tensor = np.expand_dims(tensor, axis=1)

        ort_value = ort.OrtValue.ortvalue_from_numpy(tensor)
        results = self.session.run_with_ort_values(
            [self._output_layer], {self._input_layer: ort_value}
        )
        return np.squeeze(results[0].numpy())

    def decode(self, logits: npt.NDArray) -> str:
        """CTC decode logits to text."""
        if logits.shape[0] == len(self.ctc_vocab):
            logits = np.transpose(logits, axes=[1, 0])
        return self.ctc_decoder.decode(logits).replace(" ", "")
