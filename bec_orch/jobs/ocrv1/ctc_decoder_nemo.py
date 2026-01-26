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
        kenlm_path: str | None = None,
    ):
        _check_nemo_available()

        self.blank_sign = "<pad>"
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

        # Initialize NeMo beam search decoder using flashlight backend
        # flashlight-text must be installed: pip install flashlight-text
        from nemo.collections.asr.parts.submodules.ctc_beam_decoding import FlashlightConfig

        flashlight_cfg = FlashlightConfig(
            beam_size_token=beam_width,
            beam_threshold=25.0,
        )

        if kenlm_path:
            # Try pyctcdecode backend first (faster, less overhead)
            try:
                from nemo.collections.asr.parts.submodules.ctc_beam_decoding import PyCTCDecodeConfig
                
                pyctcdecode_cfg = PyCTCDecodeConfig(
                    beam_width=beam_width,
                )
                
                self._decoder = BeamCTCInfer(
                    blank_id=self.blank_idx,
                    beam_size=beam_width,
                    search_type="pyctcdecode",
                    return_best_hypothesis=True,
                    pyctcdecode_cfg=pyctcdecode_cfg,
                )
                logger.info(
                    f"[CTCDecoderNemo] Initialized with pyctcdecode: vocab_size={self.vocab_size}, "
                    f"device={device}, beam_width={beam_width}"
                )
            except ImportError:
                # Fall back to flashlight backend
                flashlight_cfg.lexicon_path = None  # Lexicon-free decoding
                flashlight_cfg.beam_size_token = beam_width
                self._decoder = BeamCTCInfer(
                    blank_id=self.blank_idx,
                    beam_size=beam_width,
                    search_type="flashlight",
                    return_best_hypothesis=True,
                    ngram_lm_model=kenlm_path,
                    ngram_lm_alpha=0.5,
                    flashlight_cfg=flashlight_cfg,
                )
                logger.info(
                    f"[CTCDecoderNemo] Initialized with flashlight+KenLM: vocab_size={self.vocab_size}, "
                    f"device={device}, beam_width={beam_width}, kenlm={kenlm_path}"
                )
        else:
            self._decoder = BeamCTCInfer(
                blank_id=self.blank_idx,
                beam_size=beam_width,
                search_type="flashlight",
                return_best_hypothesis=True,
                flashlight_cfg=flashlight_cfg,
            )
            logger.info(
                f"[CTCDecoderNemo] Initialized with flashlight (no LM): vocab_size={self.vocab_size}, "
                f"device={device}, beam_width={beam_width}"
            )

        # Set vocabulary and decoding type - required by BeamCTCInfer
        self._decoder.set_vocabulary(self.ctc_vocab)
        self._decoder.set_decoding_type("char")

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
        Decode a batch of logits using NeMo's BeamCTCInfer.

        Args:
            batch_logits: List of logits arrays, each shape (time, vocab) or (vocab, time)

        Returns:
            List of decoded text strings
        """
        import time
        start_time = time.perf_counter()
        
        if not batch_logits:
            return []

        # Transpose if needed and collect lengths
        processed = []
        lengths = []
        vocab_sizes = []
        for logits in batch_logits:
            # Check if we need to transpose (vocab_size should be first dimension)
            if logits.shape[1] == self.vocab_size or logits.shape[0] != self.vocab_size:
                # Shape is (time, vocab) or (vocab, time) where vocab != self.vocab_size
                # We want (time, vocab)
                if logits.shape[0] < logits.shape[1]:
                    # Likely (vocab, time) - transpose
                    logits = np.transpose(logits, axes=[1, 0])
            processed.append(logits)
            lengths.append(logits.shape[0])
            vocab_sizes.append(logits.shape[1])

        batch_size = len(processed)
        max_len = max(lengths)
        max_vocab = max(vocab_sizes)

        # Convert all logits to torch tensors at once (more efficient)
        torch_tensors = []
        for logits in processed:
            t = logits.shape[0]
            v = logits.shape[1]
            # Pad vocab dimension if needed
            if v < max_vocab:
                logits_padded = np.pad(logits, ((0, 0), (0, max_vocab - v)), constant_values=-1000.0)
            else:
                logits_padded = logits
            torch_tensors.append(_logits_to_torch(logits_padded, self.device))
        
        # Pad and stack on GPU
        # Use very negative value for padding
        padded = torch.full(
            (batch_size, max_len, max_vocab),
            fill_value=-1000.0,
            device=self.device,
            dtype=torch.float32,
        )

        for i, tensor in enumerate(torch_tensors):
            t = processed[i].shape[0]
            padded[i, :t, :] = tensor

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
        # When return_best_hypothesis=True, hypotheses is a list of Hypothesis objects
        # one for each item in the batch
        if hypotheses:
            for i in range(batch_size):
                if i < len(hypotheses):
                    hyp = hypotheses[i]
                    # Extract indices from hypothesis
                    if hasattr(hyp, "y_sequence"):
                        indices = hyp.y_sequence
                    elif hasattr(hyp, "tokens"):
                        indices = hyp.tokens
                    elif hasattr(hyp, "text"):
                        # Some backends return text directly
                        text = hyp.text
                        texts.append(text.replace(self.blank_sign, "").replace("§", " "))
                        continue
                    else:
                        # For flashlight backend, try to get the text directly
                        if hasattr(hyp, "text"):
                            text = hyp.text
                            texts.append(text.replace(self.blank_sign, "").replace("§", " "))
                            continue
                        else:
                            # Last resort - try to convert to string
                            text = str(hyp)
                            texts.append(text.replace(self.blank_sign, "").replace("§", " "))
                            continue

                    if isinstance(indices, torch.Tensor):
                        indices = indices.cpu().tolist()

                    text = self._indices_to_text(indices)
                    texts.append(text.replace(self.blank_sign, "").replace("§", " "))
                else:
                    texts.append("")
        else:
            texts = [""] * batch_size

        elapsed = (time.perf_counter() - start_time) * 1000
        import logging
        logger = logging.getLogger(__name__)
        logger.info(f"[CTCDecoderNemo] decode_batch: {len(batch_logits)} lines in {elapsed:.1f}ms ({elapsed/len(batch_logits):.1f}ms/line)")
        
        return texts
