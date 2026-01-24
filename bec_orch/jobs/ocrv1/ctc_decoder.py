import logging

import numpy as np
import numpy.typing as npt
from scipy.special import log_softmax as scipy_log_softmax

from pyctcdecode.decoder import build_ctcdecoder

# Suppress noisy pyctcdecode warnings
logging.getLogger("pyctcdecode.alphabet").setLevel(logging.ERROR)

# Global decoder instance for multiprocessing (avoids pickling issues)
_GLOBAL_DECODER = None
_GLOBAL_VOCAB_LEN = None  # Use length for fast comparison
_GLOBAL_BLANK_SIGN = "<pad>"

# Beam width for CTC decoding
BEAM_WIDTH = 64

# Token pruning - skip tokens with log probability below this threshold
# Default is -3.0, more negative = less pruning, less negative = more pruning
TOKEN_MIN_LOGP = -3.0

# Greedy confidence threshold for hybrid decoding
# If greedy decode confidence is above this, skip beam search
GREEDY_CONFIDENCE_THRESHOLD = -0.5

# Blank index in vocabulary (blank is always first token)
BLANK_IDX = 0

# Vocabulary pruning threshold - tokens with max log prob below this are removed
# Tokens that never exceed this threshold at any timestep are pruned from vocabulary
# Set to None to disable vocabulary pruning
VOCAB_PRUNE_THRESHOLD = -10.0  # Prune tokens with max log prob below -10

# Pruning mode: "line" (per-line pruning) or "page" (shared across page)
# Per-line gives smaller vocab but rebuilds decoder per line
# Per-page shares one decoder across all lines in a page
VOCAB_PRUNE_MODE = "line"


def _apply_log_softmax(logits: npt.NDArray) -> npt.NDArray:
    """
    Apply log_softmax to convert raw logits to log probabilities.

    This is done once here to avoid pyctcdecode doing it internally.
    Uses scipy which is faster than numpy implementation.

    Args:
        logits: shape (time, vocab) - raw logits from OCR model

    Returns:
        log_probs: shape (time, vocab) - log probabilities (sum to 0 in log space per row)
    """
    return scipy_log_softmax(logits, axis=1).astype(np.float32)


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
    Note: argmax works the same on raw logits or log probabilities.

    Args:
        logits: shape (time, vocab) - raw logits or log probabilities
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


def _decode_logits_greedy_with_confidence_internal(
    log_probs: npt.NDArray, vocab: list[str]
) -> tuple[str, float]:
    """
    Greedy CTC decode with confidence score (internal, expects log probabilities).

    Args:
        log_probs: shape (time, vocab) - log probabilities (already softmaxed)
        vocab: vocabulary list (index 0 should be blank)

    Returns:
        Tuple of (decoded text, confidence score as mean log prob)
    """
    # Get best token at each timestep
    best_path = np.argmax(log_probs, axis=1)
    best_log_probs = log_probs[np.arange(len(log_probs)), best_path]

    # Collapse repeats and remove blanks, accumulate log probs
    decoded = []
    scores = []
    prev = -1
    for i, idx in enumerate(best_path):
        if idx != prev and idx != 0:  # 0 is blank
            decoded.append(vocab[idx])
            scores.append(best_log_probs[i])
        prev = idx

    text = "".join(decoded)

    # Confidence is mean log probability of decoded tokens
    if scores:
        confidence = float(np.mean(scores))
    else:
        # Empty decode - low confidence
        confidence = -float("inf")

    return text, confidence


def decode_logits_greedy_with_confidence(
    logits: npt.NDArray, vocab: list[str], logits_are_log_probs: bool = True
) -> tuple[str, float]:
    """
    Greedy CTC decode with confidence score.

    The confidence is the mean log probability of the best token at each
    non-blank timestep, normalized by the number of decoded characters.

    Args:
        logits: shape (time, vocab) - log probabilities (or raw logits if logits_are_log_probs=False)
        vocab: vocabulary list (index 0 should be blank)
        logits_are_log_probs: if True (default), logits are already log probabilities

    Returns:
        Tuple of (decoded text, confidence score as mean log prob)
    """
    if logits_are_log_probs:
        log_probs = logits
    else:
        log_probs = _apply_log_softmax(logits)
    return _decode_logits_greedy_with_confidence_internal(log_probs, vocab)


def decode_logits_hybrid_global(
    logits: npt.NDArray,
    vocab: list[str],
    confidence_threshold: float | None = None,
    beam_width: int | None = None,
    token_min_logp: float | None = None,
    logits_are_log_probs: bool = True,
) -> str:
    """
    Hybrid decode: try greedy first, fall back to beam search if confidence is low.

    This provides a good speed/accuracy tradeoff - greedy is ~17x faster but
    may be less accurate on difficult lines. By checking confidence, we can
    use beam search only when needed.

    Args:
        logits: shape (time, vocab) - log probabilities (or raw logits if logits_are_log_probs=False)
        vocab: vocabulary list (index 0 should be blank)
        confidence_threshold: if greedy confidence is above this, use greedy result.
                              Defaults to GREEDY_CONFIDENCE_THRESHOLD.
        beam_width: beam width for fallback beam search (defaults to BEAM_WIDTH)
        token_min_logp: token min log prob for beam search (defaults to TOKEN_MIN_LOGP)
        logits_are_log_probs: if True (default), logits are already log probabilities

    Returns:
        Decoded text string
    """
    if confidence_threshold is None:
        confidence_threshold = GREEDY_CONFIDENCE_THRESHOLD

    # Use log probs directly if already converted, otherwise apply log_softmax
    if logits_are_log_probs:
        log_probs = logits
    else:
        log_probs = _apply_log_softmax(logits)

    # Try greedy first (using already-converted log probs)
    text, confidence = _decode_logits_greedy_with_confidence_internal(log_probs, vocab)

    # If confidence is high enough, use greedy result
    if confidence >= confidence_threshold:
        return text

    # Fall back to beam search for low-confidence lines
    # Pass log_probs directly - decode_logits_beam_search will NOT recompute softmax
    # since we pass log_probs as logits and it will apply log_softmax again (which is idempotent-ish)
    # Actually, we need to skip the log_softmax in beam_search since we already did it
    return _decode_logits_beam_search_internal(
        log_probs, vocab, beam_width=beam_width, token_min_logp=token_min_logp
    )


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


def _decode_logits_beam_search_internal(
    log_probs: npt.NDArray,
    vocab: list[str],
    keep_mask: npt.NDArray | None = None,
    beam_width: int | None = None,
    token_min_logp: float | None = None,
    vocab_prune_threshold: float | None = None,
    vocab_prune_mode: str | None = None,
) -> str:
    """
    Internal beam search decode function - expects log probabilities (already softmaxed).

    Args:
        log_probs: shape (time, vocab) - log probabilities (already converted via log_softmax)
        vocab: full vocabulary list
        keep_mask: optional pre-computed mask for per-page pruning mode
        beam_width: beam width for decoding (defaults to module BEAM_WIDTH)
        token_min_logp: token min log prob (defaults to module TOKEN_MIN_LOGP)
        vocab_prune_threshold: vocabulary pruning threshold (defaults to module VOCAB_PRUNE_THRESHOLD)
        vocab_prune_mode: "line" or "page" (defaults to module VOCAB_PRUNE_MODE)
    """
    import time

    # Use passed values or fall back to module defaults
    if beam_width is None:
        beam_width = BEAM_WIDTH
    if token_min_logp is None:
        token_min_logp = TOKEN_MIN_LOGP
    if vocab_prune_threshold is None:
        vocab_prune_threshold = VOCAB_PRUNE_THRESHOLD
    if vocab_prune_mode is None:
        vocab_prune_mode = VOCAB_PRUNE_MODE

    t0 = time.perf_counter()
    orig_shape = log_probs.shape

    # Skip collapse - vocab pruning already provides big gains, and collapse
    # adds ~40-60ms overhead for modest T reduction (800->450)
    # log_probs = _collapse_blank_frames(log_probs)
    collapsed_shape = log_probs.shape  # Same as orig since we skip collapse

    t1 = time.perf_counter()

    # Apply vocabulary pruning if enabled (on log-probabilities)
    if vocab_prune_threshold is not None:
        if vocab_prune_mode == "line" and keep_mask is None:
            # Per-line pruning: compute mask for this line only
            pruned_logits, pruned_vocab, _ = _prune_vocabulary(log_probs, vocab, vocab_prune_threshold)
        elif keep_mask is not None:
            # Per-page pruning: use pre-computed mask
            pruned_logits = log_probs[:, keep_mask]
            pruned_vocab = [vocab[i] for i in np.where(keep_mask)[0]]
        else:
            # Pruning disabled or invalid mode
            pruned_logits = log_probs
            pruned_vocab = vocab

        t2 = time.perf_counter()

        # Build a decoder for the pruned vocabulary and decode
        # Pass logits_are_log_probs=True to skip pyctcdecode's internal log_softmax
        decoder = build_ctcdecoder(pruned_vocab)
        t3 = time.perf_counter()

        result = decoder.decode(
            pruned_logits, beam_width=beam_width, token_min_logp=token_min_logp,
            logits_are_log_probs=True
        )
        t4 = time.perf_counter()

        logging.info(
            f"[CTC timing] orig={orig_shape}, collapsed={collapsed_shape}, pruned_vocab={len(pruned_vocab)}, "
            f"collapse={1000*(t1-t0):.1f}ms, prune={1000*(t2-t1):.1f}ms, "
            f"build_decoder={1000*(t3-t2):.1f}ms, decode={1000*(t4-t3):.1f}ms"
        )
        return result.replace(_GLOBAL_BLANK_SIGN, "")

    t2 = time.perf_counter()

    # No pruning - use global decoder
    global _GLOBAL_DECODER, _GLOBAL_VOCAB_LEN

    if _GLOBAL_DECODER is None or _GLOBAL_VOCAB_LEN != len(vocab):
        _init_global_decoder(vocab)

    if _GLOBAL_DECODER is None:
        raise RuntimeError("CTC decoder not initialized")

    t3 = time.perf_counter()

    # Pass logits_are_log_probs=True to skip pyctcdecode's internal log_softmax
    result = _GLOBAL_DECODER.decode(
        log_probs, beam_width=beam_width, token_min_logp=token_min_logp,
        logits_are_log_probs=True
    )

    t4 = time.perf_counter()

    logging.info(
        f"[CTC timing] orig={orig_shape}, collapsed={collapsed_shape}, "
        f"collapse={1000*(t1-t0):.1f}ms, init={1000*(t3-t2):.1f}ms, "
        f"decode={1000*(t4-t3):.1f}ms"
    )

    return result.replace(_GLOBAL_BLANK_SIGN, "")


def decode_logits_beam_search(
    logits: npt.NDArray,
    vocab: list[str],
    keep_mask: npt.NDArray | None = None,
    beam_width: int | None = None,
    token_min_logp: float | None = None,
    vocab_prune_threshold: float | None = None,
    vocab_prune_mode: str | None = None,
    logits_are_log_probs: bool = True,
) -> str:
    """
    Module-level beam search decode function for use with ProcessPoolExecutor.

    This function can be pickled and sent to worker processes.
    Expects logits in (time, vocab) shape - caller must transpose if needed.

    Args:
        logits: shape (time, vocab) - log probabilities (or raw logits if logits_are_log_probs=False)
        vocab: full vocabulary list
        keep_mask: optional pre-computed mask for per-page pruning mode.
                   If None and vocab_prune_mode is "line", computes per-line mask.
        beam_width: beam width for decoding (defaults to module BEAM_WIDTH)
        token_min_logp: token min log prob (defaults to module TOKEN_MIN_LOGP)
        vocab_prune_threshold: vocabulary pruning threshold (defaults to module VOCAB_PRUNE_THRESHOLD)
        vocab_prune_mode: "line" or "page" (defaults to module VOCAB_PRUNE_MODE)
        logits_are_log_probs: if True (default), logits are already log probabilities from model.
                              If False, apply log_softmax here.
    """
    import time

    t0 = time.perf_counter()

    # Apply log_softmax only if not already done (e.g., model already applied it)
    if logits_are_log_probs:
        log_probs = logits
    else:
        log_probs = _apply_log_softmax(logits)
        t_softmax = time.perf_counter()
        logging.info(f"[CTC timing] softmax={1000*(t_softmax-t0):.1f}ms for shape {logits.shape}")

    return _decode_logits_beam_search_internal(
        log_probs, vocab, keep_mask, beam_width, token_min_logp,
        vocab_prune_threshold, vocab_prune_mode
    )


class CTCDecoder:
    """CTC decoder with beam search."""

    def __init__(self, charset: str | list[str], add_blank: bool):
        self.blank_sign = "<pad>"

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

    def decode(
        self, logits: npt.NDArray, keep_mask: npt.NDArray | None = None,
        logits_are_log_probs: bool = True
    ) -> str:
        """Decode logits to text using beam search.

        Args:
            logits: shape (time, vocab) or (vocab, time) - log probabilities (or raw logits)
            keep_mask: optional pre-computed mask for per-page pruning mode
            logits_are_log_probs: if True (default), logits are already log probabilities
        """
        # Ensure shape is (time, vocab)
        if logits.shape[0] == len(self.ctc_vocab):
            logits = np.transpose(logits, axes=[1, 0])

        # Apply log_softmax only if not already done
        if logits_are_log_probs:
            log_probs = logits
        else:
            log_probs = _apply_log_softmax(logits)

        # Collapse consecutive blank-dominant frames to reduce sequence length
        log_probs = _collapse_blank_frames(log_probs)

        # Apply vocabulary pruning if enabled
        if VOCAB_PRUNE_THRESHOLD is not None:
            if VOCAB_PRUNE_MODE == "line" and keep_mask is None:
                pruned_logits, pruned_vocab, _ = _prune_vocabulary(log_probs, self.ctc_vocab, VOCAB_PRUNE_THRESHOLD)
            elif keep_mask is not None:
                pruned_logits = log_probs[:, keep_mask]
                pruned_vocab = [self.ctc_vocab[i] for i in np.where(keep_mask)[0]]
            else:
                pruned_logits = log_probs
                pruned_vocab = self.ctc_vocab

            decoder = build_ctcdecoder(pruned_vocab)
            return decoder.decode(
                pruned_logits, beam_width=BEAM_WIDTH, token_min_logp=TOKEN_MIN_LOGP,
                logits_are_log_probs=True
            ).replace(self.blank_sign, "")

        return self._beam_decoder.decode(
            log_probs, beam_width=BEAM_WIDTH, token_min_logp=TOKEN_MIN_LOGP,
            logits_are_log_probs=True
        ).replace(self.blank_sign, "")
