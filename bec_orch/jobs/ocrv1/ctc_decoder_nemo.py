"""
GPU-accelerated CTC decoder using NVIDIA NeMo.

This module provides a drop-in replacement for ctc_decoder.py using NeMo's
GPU-accelerated CTC decoding with beam search support.

Requirements:
    pip install nemo_toolkit[asr]

Usage:
    from ctc_decoder_nemo import CTCDecoderNemo

    decoder = CTCDecoderNemo(charset, add_blank=True, device="cuda")
    text = decoder.decode(logits)
"""

import logging

import numpy as np
import numpy.typing as npt
import torch

logger = logging.getLogger(__name__)

try:
    from nemo.collections.asr.parts.submodules.ctc_beam_decoding import BeamCTCInfer

    NEMO_AVAILABLE = True
except ImportError:
    NEMO_AVAILABLE = False
    logger.warning("NeMo not available. Install with: pip install nemo_toolkit[asr]")

BLANK_IDX = 0
DEFAULT_BEAM_WIDTH = 10


def _check_nemo_available():
    if not NEMO_AVAILABLE:
        raise ImportError("NeMo is not installed. Install with: pip install nemo_toolkit[asr]")


def _logits_to_torch(logits: npt.NDArray, device: str = "cuda") -> torch.Tensor:
    """Convert numpy logits to torch tensor on specified device."""
    if isinstance(logits, torch.Tensor):
        return logits.to(device)
    return torch.from_numpy(logits.copy()).to(device, dtype=torch.float32)


class CTCDecoderNemo:
    """
    GPU-accelerated CTC decoder using NVIDIA NeMo.

    Supports both greedy and beam search decoding on GPU.
    """

    def __init__(
        self,
        charset: str | list[str],
        add_blank: bool,
        device: str = "cuda",
        beam_width: int = DEFAULT_BEAM_WIDTH,
    ):
        _check_nemo_available()

        self.blank_sign = "<blk>"
        self.device = device
        self.beam_width = beam_width

        if isinstance(charset, str):
            self.charset = list(charset)
        else:
            self.charset = charset

        self.ctc_vocab = self.charset.copy()
        if add_blank:
            self.ctc_vocab.insert(0, self.blank_sign)
            self.blank_idx = 0
        else:
            self.blank_idx = -1

        self.vocab_size = len(self.ctc_vocab)

        # Initialize NeMo beam search decoder
        self._decoder = BeamCTCInfer(
            blank_id=self.blank_idx,
            beam_size=beam_width,
            search_type="default",
            return_best_hypothesis=True,
        )

        # Set vocabulary and decoding type - required by BeamCTCInfer
        # decoding_type must be 'char' or 'subword', not 'beam'
        self._decoder.set_vocabulary(self.ctc_vocab)
        self._decoder.set_decoding_type("char")

        logger.info(
            f"[CTCDecoderNemo] Initialized: vocab_size={self.vocab_size}, device={device}, beam_width={beam_width}"
        )

    def _indices_to_text(self, indices: list[int]) -> str:
        """Convert token indices to text string."""
        chars = []
        for idx in indices:
            if idx >= 0 and idx < len(self.ctc_vocab) and idx != self.blank_idx:
                chars.append(self.ctc_vocab[idx])
        return "".join(chars)

    def decode(self, logits: npt.NDArray) -> str:
        """
        Decode logits to text using NeMo GPU decoding.

        Args:
            logits: shape (time, vocab) or (vocab, time) - log probabilities

        Returns:
            Decoded text string
        """
        # Ensure shape is (time, vocab)
        if logits.shape[0] == self.vocab_size:
            logits = np.transpose(logits, axes=[1, 0])

        # Convert to torch on GPU: NeMo expects (batch, time, vocab)
        log_probs = _logits_to_torch(logits, self.device).unsqueeze(0)  # (1, T, V)

        # Create length tensor
        lengths = torch.tensor([log_probs.shape[1]], dtype=torch.long, device=self.device)

        # Decode using BeamCTCInfer
        with torch.no_grad():
            hypotheses = self._decoder(
                decoder_output=log_probs,
                decoder_lengths=lengths,
            )
            if hypotheses and len(hypotheses) > 0:
                hyp = hypotheses[0]
                if hasattr(hyp, "y_sequence"):
                    indices = hyp.y_sequence
                elif hasattr(hyp, "text"):
                    # Some versions return text directly
                    return hyp.text.replace("§", " ")
                else:
                    indices = []
            else:
                indices = []

        # Convert indices to text
        if isinstance(indices, torch.Tensor):
            indices = indices.cpu().tolist()

        text = self._indices_to_text(indices)
        return text.replace(self.blank_sign, "").replace("§", " ")

    def decode_batch(self, batch_logits: list[npt.NDArray]) -> list[str]:
        """
        Decode a batch of logits efficiently on GPU.

        Args:
            batch_logits: list of logits arrays, each shape (time, vocab) or (vocab, time)

        Returns:
            List of decoded text strings
        """
        if not batch_logits:
            return []

        # Transpose if needed and collect lengths
        processed = []
        lengths = []
        for logits in batch_logits:
            if logits.shape[0] == self.vocab_size:
                logits = np.transpose(logits, axes=[1, 0])
            processed.append(logits)
            lengths.append(logits.shape[0])

        batch_size = len(processed)
        max_len = max(lengths)

        # Pad and stack on GPU
        # Use very negative value for padding
        padded = torch.full(
            (batch_size, max_len, self.vocab_size),
            fill_value=-1000.0,
            device=self.device,
            dtype=torch.float32,
        )

        for i, logits in enumerate(processed):
            t = logits.shape[0]
            padded[i, :t, :] = _logits_to_torch(logits, self.device)

        # Create lengths tensor
        lengths_tensor = torch.tensor(lengths, dtype=torch.long, device=self.device)

        # Batch decode using BeamCTCInfer
        with torch.no_grad():
            hypotheses = self._decoder(
                decoder_output=padded,
                decoder_lengths=lengths_tensor,
            )

        # Extract texts
        texts = []
        for i in range(batch_size):
            if hypotheses and i < len(hypotheses):
                hyp = hypotheses[i]
                if hasattr(hyp, "y_sequence"):
                    indices = hyp.y_sequence
                elif isinstance(hyp, (list, tuple)):
                    indices = hyp[0] if hyp else []
                else:
                    indices = hyp if hyp is not None else []

                if isinstance(indices, torch.Tensor):
                    indices = indices.cpu().tolist()

                text = self._indices_to_text(indices)
            else:
                text = ""

            texts.append(text.replace(self.blank_sign, "").replace("§", " "))

        return texts
