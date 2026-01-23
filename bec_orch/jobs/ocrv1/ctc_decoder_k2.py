"""
GPU-accelerated CTC decoder using k2.

This module provides a drop-in replacement for ctc_decoder.py using k2's
GPU-accelerated CTC decoding. k2 uses FST-based decoding which is much
faster than pyctcdecode's beam search, especially on GPU.

Requirements:
    pip install k2

Usage:
    from ctc_decoder_k2 import CTCDecoderK2, decode_logits_k2

    decoder = CTCDecoderK2(charset, add_blank=True, device="cuda")
    text = decoder.decode(logits)
"""

import logging

import numpy as np
import numpy.typing as npt
import torch

logger = logging.getLogger(__name__)

try:
    import k2

    K2_AVAILABLE = True
except ImportError:
    K2_AVAILABLE = False
    logger.warning("k2 not available. Install with: pip install k2")

BLANK_IDX = 0
BEAM_WIDTH = 10  # k2 uses different beam semantics, 10 is usually enough


def _check_k2_available():
    if not K2_AVAILABLE:
        raise ImportError(
            "k2 is not installed. Install with: pip install k2\n"
            "Note: k2 requires CUDA. See https://k2-fsa.github.io/k2/installation/index.html"
        )


def _logits_to_torch(logits: npt.NDArray, device: str = "cuda") -> torch.Tensor:
    """Convert numpy logits to torch tensor on specified device."""
    if isinstance(logits, torch.Tensor):
        return logits.to(device)
    return torch.from_numpy(logits).to(device)


def _build_ctc_topo(vocab_size: int, device: str = "cuda") -> "k2.Fsa":
    """
    Build a CTC topology FSA for decoding.

    The CTC topology allows:
    - Self-loops on blank
    - Transitions from blank to any token
    - Self-loops on tokens (for repeated characters)
    - Transitions from any token back to blank
    """
    _check_k2_available()

    # Build standard CTC topology
    # This creates an FSA that models CTC's blank-insertion rules
    ctc_topo = k2.ctc_topo(vocab_size, modified=False, device=device)
    return ctc_topo


def decode_logits_k2(
    logits: npt.NDArray,
    vocab: list[str],
    device: str = "cuda",
    beam_width: int = BEAM_WIDTH,
) -> str:
    """
    Decode logits using k2's GPU-accelerated CTC decoding.

    Args:
        logits: shape (time, vocab) - log probabilities
        vocab: vocabulary list (index 0 should be blank)
        device: "cuda" or "cpu"
        beam_width: beam size for decoding

    Returns:
        Decoded text string
    """
    _check_k2_available()

    vocab_size = len(vocab)

    # Convert to torch tensor
    log_probs = _logits_to_torch(logits, device)  # (T, V)

    # k2 expects (N, T, V) where N is batch size
    log_probs = log_probs.unsqueeze(0)  # (1, T, V)

    # Get sequence length
    T = log_probs.shape[1]
    supervision_segments = torch.tensor([[0, 0, T]], dtype=torch.int32, device=device)

    # Create dense fsa vec from log probs
    dense_fsa_vec = k2.DenseFsaVec(log_probs, supervision_segments)

    # Build CTC topology
    ctc_topo = _build_ctc_topo(vocab_size, device)

    # Intersect with topology and find best path
    lattice = k2.intersect_dense(ctc_topo, dense_fsa_vec, output_beam=beam_width)

    # Get best path
    best_path = k2.shortest_path(lattice, use_double_scores=True)

    # Extract labels (excluding -1 which marks final state)
    labels = best_path.labels.tolist()
    labels = [label for label in labels if label > 0]  # Remove blanks (0) and final (-1)

    # Convert to text
    text = "".join(vocab[idx] for idx in labels if idx < len(vocab))

    return text


def decode_batch_k2(
    batch_logits: list[npt.NDArray],
    vocab: list[str],
    device: str = "cuda",
    beam_width: int = BEAM_WIDTH,
) -> list[str]:
    """
    Decode a batch of logits using k2's GPU-accelerated CTC decoding.

    This is more efficient than decoding one at a time because it
    batches the GPU operations.

    Args:
        batch_logits: list of logits arrays, each shape (time, vocab)
        vocab: vocabulary list (index 0 should be blank)
        device: "cuda" or "cpu"
        beam_width: beam size for decoding

    Returns:
        List of decoded text strings
    """
    _check_k2_available()

    if not batch_logits:
        return []

    vocab_size = len(vocab)
    batch_size = len(batch_logits)

    # Pad sequences to same length
    max_len = max(logit.shape[0] for logit in batch_logits)

    # Create padded tensor
    padded = torch.zeros(batch_size, max_len, vocab_size, device=device)
    lengths = []

    for i, logits in enumerate(batch_logits):
        T = logits.shape[0]
        lengths.append(T)
        padded[i, :T, :] = _logits_to_torch(logits, device)

    # Create supervision segments: (sequence_idx, start_frame, num_frames)
    supervision_segments = torch.tensor(
        [[i, 0, lengths[i]] for i in range(batch_size)],
        dtype=torch.int32,
        device=device,
    )

    # Create dense fsa vec
    dense_fsa_vec = k2.DenseFsaVec(padded, supervision_segments)

    # Build CTC topology
    ctc_topo = _build_ctc_topo(vocab_size, device)

    # Intersect and find best paths
    lattice = k2.intersect_dense(ctc_topo, dense_fsa_vec, output_beam=beam_width)
    best_paths = k2.shortest_path(lattice, use_double_scores=True)

    # Extract texts
    texts = []
    for i in range(batch_size):
        # Get labels for this sequence
        fsa = best_paths[i]
        labels = fsa.labels.tolist()
        labels = [label for label in labels if label > 0]  # Remove blanks and final
        text = "".join(vocab[idx] for idx in labels if idx < len(vocab))
        texts.append(text)

    return texts


class CTCDecoderK2:
    """GPU-accelerated CTC decoder using k2."""

    def __init__(
        self,
        charset: str | list[str],
        add_blank: bool,
        device: str = "cuda",
        beam_width: int = BEAM_WIDTH,
    ):
        _check_k2_available()

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

        # Pre-build CTC topology for reuse
        self._ctc_topo = _build_ctc_topo(self.vocab_size, device)

        logger.info(f"CTCDecoderK2 initialized: vocab_size={self.vocab_size}, device={device}")

    def decode(self, logits: npt.NDArray) -> str:
        """
        Decode logits to text using k2 GPU decoding.

        Args:
            logits: shape (time, vocab) or (vocab, time) - log probabilities

        Returns:
            Decoded text string
        """
        # Ensure shape is (time, vocab)
        if logits.shape[0] == self.vocab_size:
            logits = np.transpose(logits, axes=[1, 0])

        # Convert to torch
        log_probs = _logits_to_torch(logits, self.device).unsqueeze(0)  # (1, T, V)

        T = log_probs.shape[1]
        supervision_segments = torch.tensor([[0, 0, T]], dtype=torch.int32, device=self.device)

        # Create dense fsa vec
        dense_fsa_vec = k2.DenseFsaVec(log_probs, supervision_segments)

        # Intersect and find best path
        lattice = k2.intersect_dense(self._ctc_topo, dense_fsa_vec, output_beam=self.beam_width)
        best_path = k2.shortest_path(lattice, use_double_scores=True)

        # Extract labels
        labels = best_path.labels.tolist()
        labels = [label for label in labels if label > 0]

        # Convert to text
        text = "".join(self.ctc_vocab[idx] for idx in labels if idx < self.vocab_size)
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

        # Transpose if needed
        processed = []
        for logits in batch_logits:
            if logits.shape[0] == self.vocab_size:
                logits = np.transpose(logits, axes=[1, 0])
            processed.append(logits)

        batch_size = len(processed)
        max_len = max(logit.shape[0] for logit in processed)

        # Pad and stack
        padded = torch.zeros(batch_size, max_len, self.vocab_size, device=self.device)
        lengths = []

        for i, logits in enumerate(processed):
            T = logits.shape[0]
            lengths.append(T)
            padded[i, :T, :] = _logits_to_torch(logits, self.device)

        # Supervision segments
        supervision_segments = torch.tensor(
            [[i, 0, lengths[i]] for i in range(batch_size)],
            dtype=torch.int32,
            device=self.device,
        )

        # Decode
        dense_fsa_vec = k2.DenseFsaVec(padded, supervision_segments)
        lattice = k2.intersect_dense(self._ctc_topo, dense_fsa_vec, output_beam=self.beam_width)
        best_paths = k2.shortest_path(lattice, use_double_scores=True)

        # Extract texts
        texts = []
        for i in range(batch_size):
            fsa = best_paths[i]
            labels = fsa.labels.tolist()
            labels = [label for label in labels if label > 0]
            text = "".join(self.ctc_vocab[idx] for idx in labels if idx < self.vocab_size)
            texts.append(text.replace(self.blank_sign, "").replace("§", " "))

        return texts
