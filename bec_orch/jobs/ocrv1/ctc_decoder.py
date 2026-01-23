import logging

import numpy as np
import numpy.typing as npt
from pyctcdecode.decoder import build_ctcdecoder

# Suppress noisy pyctcdecode warnings
logging.getLogger("pyctcdecode.alphabet").setLevel(logging.ERROR)

# Global decoder instance for multiprocessing (avoids pickling issues)
_GLOBAL_DECODER = None
_GLOBAL_VOCAB_LEN = None  # Use length for fast comparison
_GLOBAL_BLANK_SIGN = "<blk>"

# Beam width for CTC decoding
BEAM_WIDTH = 80

# Token pruning - skip tokens with log probability below this threshold
# Default is -5.0, more negative = less pruning, less negative = more pruning
TOKEN_MIN_LOGP = -5.0

# Blank index in vocabulary (blank is always first token)
BLANK_IDX = 0

# Vocabulary pruning threshold - tokens with max log prob below this are removed
# Set to None to disable vocabulary pruning
VOCAB_PRUNE_THRESHOLD = None  # Disabled - doesn't help performance and hurts accuracy

# Pruning mode: "line" (per-line pruning) or "page" (shared across page)
# Per-line gives smaller vocab but rebuilds decoder per line
# Per-page shares one decoder across all lines in a page
VOCAB_PRUNE_MODE = "line"


def _prune_vocabulary(
    logits: npt.NDArray,
    vocab: list[str],
    threshold: float = -10.0,
) -> tuple[npt.NDArray, list[str], npt.NDArray]:
    """
    Prune vocabulary to only tokens that appear with meaningful probability.

    This reduces the effective vocabulary size, speeding up beam search which is
    O(T × beam × vocab). For a line with only 20 active characters out of 200,
    this gives ~10x speedup.

    Args:
        logits: shape (time, vocab) - log probabilities
        vocab: full vocabulary list (index 0 should be blank)
        threshold: minimum max log probability to keep a token

    Returns:
        pruned_logits: shape (time, reduced_vocab)
        pruned_vocab: reduced vocabulary list
        keep_indices: indices of kept tokens (for debugging/mapping back)
    """
    # Find tokens that exceed threshold at ANY timestep
    max_per_token = logits.max(axis=0)  # shape: (vocab,)
    keep_mask = max_per_token > threshold

    # Always keep blank (index 0)
    keep_mask[0] = True

    # Get indices of tokens to keep
    keep_indices = np.where(keep_mask)[0]

    # Extract only those columns
    pruned_logits = logits[:, keep_mask]

    # Build reduced vocabulary
    pruned_vocab = [vocab[i] for i in keep_indices]

    return pruned_logits, pruned_vocab, keep_indices


def compute_page_keep_mask(
    page_logits: list[npt.NDArray],
    vocab_size: int,
    threshold: float = -10.0,
) -> npt.NDArray:
    """
    Compute a shared keep mask for all lines in a page.

    Use this for per-page vocabulary pruning mode.

    Args:
        page_logits: list of logits arrays, one per line, each shape (time, vocab)
        vocab_size: size of the full vocabulary
        threshold: minimum max log probability to keep a token

    Returns:
        keep_mask: boolean array of shape (vocab,) - True for tokens to keep
    """
    keep_mask = np.zeros(vocab_size, dtype=bool)
    keep_mask[0] = True  # Always keep blank

    for logits in page_logits:
        max_per_token = logits.max(axis=0)
        keep_mask |= max_per_token > threshold

    return keep_mask


def decode_logits_greedy(logits: npt.NDArray, vocab: list[str]) -> str:
    """
    Simple greedy CTC decode - take argmax at each timestep, collapse repeats.

    Much faster than beam search (~0.6ms vs ~11ms) but may be slightly less accurate.

    Args:
        logits: shape (time, vocab) - log probabilities
        vocab: vocabulary list (index 0 should be blank)

    Returns:
        Decoded text string
    """
    # Get best token at each timestep
    best_path = np.argmax(logits, axis=1)

    # Collapse repeats and remove blanks
    decoded = []
    prev = -1
    for idx in best_path:
        if idx != prev and idx != 0:  # 0 is blank
            decoded.append(vocab[idx])
        prev = idx
    return "".join(decoded)


def _collapse_blank_frames(logits: npt.NDArray) -> npt.NDArray:
    """
    Collapse consecutive frames where blank has the highest probability.

    This reduces sequence length T, speeding up beam search which is O(T × beam × vocab).
    We keep one frame per run of consecutive blank-dominant frames.
    """
    if len(logits) <= 1:
        return logits

    # logits shape: (time, vocab)
    # Find frames where blank (index 0) has the highest probability
    best_tokens = np.argmax(logits, axis=1)
    is_blank = best_tokens == BLANK_IDX

    # Keep frames where current is not blank OR previous was not blank
    # This keeps the first frame of each blank run and all non-blank frames
    prev_is_blank = np.concatenate([[False], is_blank[:-1]])
    keep_mask = ~(is_blank & prev_is_blank)

    return logits[keep_mask]


def _init_global_decoder(vocab: list[str]) -> None:
    """Initialize the global decoder for use in worker processes."""
    global _GLOBAL_DECODER, _GLOBAL_VOCAB_LEN
    if _GLOBAL_DECODER is None or _GLOBAL_VOCAB_LEN != len(vocab):
        _GLOBAL_DECODER = build_ctcdecoder(vocab)
        _GLOBAL_VOCAB_LEN = len(vocab)


def init_worker_process(vocab: list[str]) -> None:
    """
    Initializer function for ProcessPoolExecutor workers.

    Called once when each worker process starts, avoiding repeated
    decoder initialization overhead.
    """
    _init_global_decoder(vocab)


def decode_logits_beam_search(
    logits: npt.NDArray,
    vocab: list[str],
    keep_mask: npt.NDArray | None = None,
    beam_width: int | None = None,
    token_min_logp: float | None = None,
    vocab_prune_threshold: float | None = None,
    vocab_prune_mode: str | None = None,
) -> str:
    """
    Module-level beam search decode function for use with ProcessPoolExecutor.

    This function can be pickled and sent to worker processes.
    Expects logits in (time, vocab) shape - caller must transpose if needed.

    Args:
        logits: shape (time, vocab) - log probabilities
        vocab: full vocabulary list
        keep_mask: optional pre-computed mask for per-page pruning mode.
                   If None and vocab_prune_mode is "line", computes per-line mask.
        beam_width: beam width for decoding (defaults to module BEAM_WIDTH)
        token_min_logp: token min log prob (defaults to module TOKEN_MIN_LOGP)
        vocab_prune_threshold: vocabulary pruning threshold (defaults to module VOCAB_PRUNE_THRESHOLD)
        vocab_prune_mode: "line" or "page" (defaults to module VOCAB_PRUNE_MODE)
    """
    # Use passed values or fall back to module defaults
    if beam_width is None:
        beam_width = BEAM_WIDTH
    if token_min_logp is None:
        token_min_logp = TOKEN_MIN_LOGP
    if vocab_prune_threshold is None:
        vocab_prune_threshold = VOCAB_PRUNE_THRESHOLD
    if vocab_prune_mode is None:
        vocab_prune_mode = VOCAB_PRUNE_MODE
    # Collapse consecutive blank-dominant frames to reduce sequence length
    logits = _collapse_blank_frames(logits)

    # Apply vocabulary pruning if enabled
    if vocab_prune_threshold is not None:
        if vocab_prune_mode == "line" and keep_mask is None:
            # Per-line pruning: compute mask for this line only
            pruned_logits, pruned_vocab, _ = _prune_vocabulary(logits, vocab, vocab_prune_threshold)
        elif keep_mask is not None:
            # Per-page pruning: use pre-computed mask
            pruned_logits = logits[:, keep_mask]
            pruned_vocab = [vocab[i] for i in np.where(keep_mask)[0]]
        else:
            # Pruning disabled or invalid mode
            pruned_logits = logits
            pruned_vocab = vocab

        # Build a decoder for the pruned vocabulary and decode
        decoder = build_ctcdecoder(pruned_vocab)
        result = decoder.decode(pruned_logits, beam_width=beam_width, token_min_logp=token_min_logp)
        return result.replace(_GLOBAL_BLANK_SIGN, "")

    # No pruning - use global decoder
    global _GLOBAL_DECODER, _GLOBAL_VOCAB_LEN

    if _GLOBAL_DECODER is None or _GLOBAL_VOCAB_LEN != len(vocab):
        _init_global_decoder(vocab)

    if _GLOBAL_DECODER is None:
        raise RuntimeError("CTC decoder not initialized")

    return _GLOBAL_DECODER.decode(logits, beam_width=beam_width, token_min_logp=token_min_logp).replace(
        _GLOBAL_BLANK_SIGN, ""
    )


class CTCDecoder:
    """CTC decoder with beam search."""

    def __init__(self, charset: str | list[str], add_blank: bool):
        self.blank_sign = "<blk>"

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

        # Build beam search decoder
        self._beam_decoder = build_ctcdecoder(self.ctc_vocab)

    def decode(self, logits: npt.NDArray, keep_mask: npt.NDArray | None = None) -> str:
        """Decode logits to text using beam search.

        Args:
            logits: shape (time, vocab) or (vocab, time) - log probabilities
            keep_mask: optional pre-computed mask for per-page pruning mode
        """
        # Ensure shape is (time, vocab)
        if logits.shape[0] == len(self.ctc_vocab):
            logits = np.transpose(logits, axes=[1, 0])

        # Collapse consecutive blank-dominant frames to reduce sequence length
        logits = _collapse_blank_frames(logits)

        # Apply vocabulary pruning if enabled
        if VOCAB_PRUNE_THRESHOLD is not None:
            if VOCAB_PRUNE_MODE == "line" and keep_mask is None:
                pruned_logits, pruned_vocab, _ = _prune_vocabulary(logits, self.ctc_vocab, VOCAB_PRUNE_THRESHOLD)
            elif keep_mask is not None:
                pruned_logits = logits[:, keep_mask]
                pruned_vocab = [self.ctc_vocab[i] for i in np.where(keep_mask)[0]]
            else:
                pruned_logits = logits
                pruned_vocab = self.ctc_vocab

            decoder = build_ctcdecoder(pruned_vocab)
            return decoder.decode(pruned_logits, beam_width=BEAM_WIDTH, token_min_logp=TOKEN_MIN_LOGP).replace(
                self.blank_sign, ""
            )

        return self._beam_decoder.decode(logits, beam_width=BEAM_WIDTH, token_min_logp=TOKEN_MIN_LOGP).replace(
            self.blank_sign, ""
        )
