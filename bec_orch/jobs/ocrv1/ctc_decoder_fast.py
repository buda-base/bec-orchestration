"""
Fast CTC decoder using fast-ctc-decode library.

This module provides a drop-in replacement for ctc_decoder.py but uses
the fast-ctc-decode library (Rust-based) instead of pyctcdecode.

fast-ctc-decode is significantly faster than pyctcdecode for beam search
decoding, making it suitable for high-throughput OCR pipelines.

Install: pip install fast-ctc-decode
"""

import numpy as np
import numpy.typing as npt

try:
    from fast_ctc_decode import beam_search, viterbi_search
except ImportError as e:
    raise ImportError(
        f"fast-ctc-decode library not found. Install with: pip install fast-ctc-decode\n"
        f"Original error: {e}"
    )

# Global state for multiprocessing (mirrors ctc_decoder.py interface)
_GLOBAL_VOCAB: list[str] | None = None
_GLOBAL_VOCAB_LEN: int | None = None
_GLOBAL_ALPHABET: str | None = None  # fast-ctc-decode uses alphabet string
_GLOBAL_BLANK_PLACEHOLDER: str = "\uE000"  # Placeholder used in alphabet
_GLOBAL_BLANK_SIGN = "<pad>"

# Debug mode - set to True to enable verbose logging
DEBUG_FAST_CTC = False

# Maximum vocabulary size supported by fast-ctc-decode
# Larger vocabularies cause crashes/memory issues in the Rust library
MAX_VOCAB_SIZE = 1000

# Vocabulary pruning threshold (log probability)
# Tokens with max log prob below this are removed before decoding
# This dramatically reduces vocabulary for large character sets (e.g., Tibetan)
# Set to None for adaptive threshold (keeps top tokens by percentile)
VOCAB_PRUNE_THRESHOLD: float | None = None  # Use adaptive

# When using adaptive threshold, keep tokens in the top N percentile of max log probs
VOCAB_PRUNE_PERCENTILE = 99.0  # Keep top 1% of tokens (by max log prob)

# Whether to automatically enable pruning for large vocabularies
AUTO_PRUNE_LARGE_VOCAB = True

# Beam width for CTC decoding
BEAM_WIDTH = 64

# Beam cut threshold - prune beams with probability below this fraction of best beam
# Set to None to let fast-ctc-decode compute it automatically from the posteriors
# (required because threshold must be <= min probability in the matrix)
BEAM_CUT_THRESHOLD: float | None = None

# Blank index in vocabulary (blank is always first token)
BLANK_IDX = 0


def _prune_vocabulary_for_fast_ctc(
    logits: npt.NDArray,
    alphabet: str,
    threshold: float | None = None,
    percentile: float = 99.0,
    min_vocab_size: int = 50,
    max_vocab_size: int = 500,
) -> tuple[npt.NDArray, str, npt.NDArray]:
    """
    Prune vocabulary to only tokens that appear with meaningful probability.

    This is essential for fast-ctc-decode with large vocabularies (e.g., Tibetan with ~10k chars).
    Reduces effective vocabulary from ~10,000 to ~100-500 active characters per line.

    Args:
        logits: shape (time, vocab) - log probabilities (NOT posteriors yet)
        alphabet: full alphabet string for fast-ctc-decode
        threshold: fixed threshold, or None for adaptive (percentile-based)
        percentile: when threshold is None, keep tokens above this percentile
        min_vocab_size: minimum tokens to keep (prevents over-pruning)
        max_vocab_size: maximum tokens to keep (for fast-ctc-decode safety)

    Returns:
        pruned_logits: shape (time, reduced_vocab)
        pruned_alphabet: reduced alphabet string
        keep_indices: indices of kept tokens
    """
    # Find max log probability for each token across all timesteps
    max_per_token = logits.max(axis=0)  # shape: (vocab,)

    if threshold is None:
        # Adaptive threshold: use percentile of max log probs
        threshold = float(np.percentile(max_per_token, percentile))

        if DEBUG_FAST_CTC:
            import sys
            print(
                f"[fast-ctc] Adaptive threshold: percentile={percentile}, "
                f"threshold={threshold:.2f}, max_logit={max_per_token.max():.2f}, "
                f"min_logit={max_per_token.min():.2f}",
                file=sys.stderr,
                flush=True,
            )

    keep_mask = max_per_token >= threshold

    # Always keep blank (index 0)
    keep_mask[0] = True

    # Ensure we keep at least min_vocab_size tokens
    num_kept = keep_mask.sum()
    if num_kept < min_vocab_size:
        # Keep top min_vocab_size tokens by max log prob
        top_indices = np.argsort(max_per_token)[-min_vocab_size:]
        keep_mask[:] = False
        keep_mask[top_indices] = True
        keep_mask[0] = True  # Always keep blank

    # Ensure we don't exceed max_vocab_size tokens
    num_kept = keep_mask.sum()
    if num_kept > max_vocab_size:
        # Keep only top max_vocab_size tokens
        indices_sorted = np.argsort(max_per_token)[::-1]  # descending
        keep_mask[:] = False
        keep_mask[indices_sorted[:max_vocab_size]] = True
        keep_mask[0] = True  # Always keep blank

    # Get indices of tokens to keep
    keep_indices = np.where(keep_mask)[0]

    # Extract only those columns
    pruned_logits = logits[:, keep_mask]

    # Build reduced alphabet (each character corresponds to a vocab index)
    pruned_alphabet = "".join(alphabet[i] for i in keep_indices)

    return pruned_logits, pruned_alphabet, keep_indices


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


def _logits_to_posteriors(logits: npt.NDArray) -> npt.NDArray:
    """
    Convert log probabilities to probabilities using softmax.

    fast-ctc-decode expects probabilities, not log probabilities.
    We apply softmax to ensure proper probability distribution.

    Returns a contiguous float32 array as required by fast-ctc-decode.
    """
    # Ensure float32 dtype (fast-ctc-decode requires this)
    if logits.dtype != np.float32:
        logits = logits.astype(np.float32)

    # Handle edge cases
    if logits.size == 0:
        return logits

    # Check for NaN/Inf and replace with safe values
    if not np.isfinite(logits).all():
        logits = np.nan_to_num(logits, nan=0.0, posinf=0.0, neginf=-100.0)

    # Numerical stability: subtract max before exp
    logits_shifted = logits - np.max(logits, axis=-1, keepdims=True)
    exp_logits = np.exp(logits_shifted)
    posteriors = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)

    # Ensure contiguous array (required by fast-ctc-decode)
    if not posteriors.flags['C_CONTIGUOUS']:
        posteriors = np.ascontiguousarray(posteriors)

    return posteriors


def _vocab_to_alphabet(vocab: list[str]) -> tuple[str, str]:
    """
    Convert vocabulary list to alphabet string for fast-ctc-decode.

    fast-ctc-decode expects a string where each character is a token.
    The blank token should be first (index 0).

    For multi-character tokens (like <pad>), we use a placeholder approach.

    Returns:
        Tuple of (alphabet_string, blank_placeholder_char)
    """
    # fast-ctc-decode works with single characters
    # For the blank token, we use a Unicode private use area character
    # that won't appear in actual text
    BLANK_PLACEHOLDER = "\uE000"  # Unicode Private Use Area

    alphabet_chars = []
    for token in vocab:
        if token == _GLOBAL_BLANK_SIGN or token == "<pad>":
            alphabet_chars.append(BLANK_PLACEHOLDER)
        elif len(token) == 1:
            alphabet_chars.append(token)
        else:
            # For multi-character tokens, use the first character as placeholder
            # This is a limitation - fast-ctc-decode works best with single-char alphabets
            alphabet_chars.append(token[0] if token else BLANK_PLACEHOLDER)

    return "".join(alphabet_chars), BLANK_PLACEHOLDER


def decode_logits_greedy(logits: npt.NDArray, vocab: list[str]) -> str:
    """
    Greedy CTC decode using fast-ctc-decode's viterbi_search.

    Args:
        logits: shape (time, vocab) - log probabilities
        vocab: vocabulary list (index 0 should be blank)

    Returns:
        Decoded text string
    """
    try:
        # Collapse blank frames for efficiency
        logits = _collapse_blank_frames(logits)

        # Convert to probabilities
        posteriors = _logits_to_posteriors(logits)

        # Convert vocab to alphabet string
        alphabet, blank_char = _vocab_to_alphabet(vocab)

        # Decode using viterbi (greedy)
        decoded, _path = viterbi_search(posteriors, alphabet)

        # Remove blank placeholders
        return decoded.replace(blank_char, "")
    except Exception as e:
        # Log error and return empty string rather than crashing the worker
        import logging
        logging.getLogger(__name__).error(f"fast-ctc-decode greedy failed: {e}")
        return ""


def init_worker_process(vocab: list[str]) -> None:
    """
    Initializer function for ProcessPoolExecutor workers.

    Called once when each worker process starts, avoiding repeated
    initialization overhead.
    """
    import os
    import sys

    global _GLOBAL_VOCAB, _GLOBAL_VOCAB_LEN, _GLOBAL_ALPHABET, _GLOBAL_BLANK_PLACEHOLDER
    _GLOBAL_VOCAB = vocab
    _GLOBAL_VOCAB_LEN = len(vocab)
    _GLOBAL_ALPHABET, _GLOBAL_BLANK_PLACEHOLDER = _vocab_to_alphabet(vocab)

    if len(vocab) > MAX_VOCAB_SIZE:
        if AUTO_PRUNE_LARGE_VOCAB:
            print(
                f"[fast-ctc] Large vocabulary ({len(vocab)} chars) detected. "
                f"Auto-pruning enabled (threshold={VOCAB_PRUNE_THRESHOLD}).",
                file=sys.stderr,
                flush=True,
            )
        else:
            print(
                f"[fast-ctc] WARNING: Vocabulary size {len(vocab)} exceeds recommended maximum "
                f"of {MAX_VOCAB_SIZE}. fast-ctc-decode may crash or perform poorly. "
                f"Consider using pyctcdecode instead (remove --fast-ctc flag).",
                file=sys.stderr,
                flush=True,
            )

    if DEBUG_FAST_CTC:
        print(
            f"[fast-ctc] Worker {os.getpid()} initialized: vocab_len={len(vocab)}, "
            f"alphabet_len={len(_GLOBAL_ALPHABET)}",
            file=sys.stderr,
            flush=True,
        )


def decode_logits_beam_search_global(
    logits: npt.NDArray,
    keep_mask: npt.NDArray | None = None,
    beam_width: int | None = None,
    token_min_logp: float | None = None,  # Not used by fast-ctc-decode, kept for API compat
    vocab_prune_threshold: float | None = None,  # Not used, kept for API compat
    vocab_prune_mode: str | None = None,  # Not used, kept for API compat
) -> str:
    """
    Beam search decode using the process-global decoder/vocab.

    This avoids passing (and pickling) the full vocabulary on every task.
    `init_worker_process()` must have been called in the worker.
    """
    global _GLOBAL_VOCAB, _GLOBAL_ALPHABET
    if _GLOBAL_VOCAB is None or _GLOBAL_ALPHABET is None:
        raise RuntimeError("Global CTC vocab not initialized. Did you forget init_worker_process()?")

    return decode_logits_beam_search(
        logits,
        _GLOBAL_VOCAB,
        keep_mask,
        beam_width,
        token_min_logp,
        vocab_prune_threshold,
        vocab_prune_mode,
    )


def decode_logits_beam_search(
    logits: npt.NDArray,
    vocab: list[str],
    keep_mask: npt.NDArray | None = None,  # Not used by fast-ctc-decode
    beam_width: int | None = None,
    token_min_logp: float | None = None,  # Not used by fast-ctc-decode
    vocab_prune_threshold: float | None = None,  # Not used by fast-ctc-decode
    vocab_prune_mode: str | None = None,  # Not used by fast-ctc-decode
) -> str:
    """
    Beam search decode using fast-ctc-decode.

    This function provides the same interface as ctc_decoder.py but uses
    the faster Rust-based fast-ctc-decode library.

    Args:
        logits: shape (time, vocab) - log probabilities
        vocab: vocabulary list (index 0 should be blank)
        keep_mask: NOT USED - kept for API compatibility with ctc_decoder.py
        beam_width: beam width for decoding (defaults to module BEAM_WIDTH)
        token_min_logp: NOT USED - fast-ctc-decode uses beam_cut_threshold instead
        vocab_prune_threshold: NOT USED - fast-ctc-decode handles pruning internally
        vocab_prune_mode: NOT USED - kept for API compatibility

    Returns:
        Decoded text string
    """
    try:
        if beam_width is None:
            beam_width = BEAM_WIDTH

        # Collapse consecutive blank-dominant frames to reduce sequence length
        logits = _collapse_blank_frames(logits)

        if logits.size == 0:
            return ""

        # Get base alphabet
        global _GLOBAL_ALPHABET, _GLOBAL_VOCAB_LEN, _GLOBAL_BLANK_PLACEHOLDER
        if _GLOBAL_ALPHABET is not None and _GLOBAL_VOCAB_LEN == len(vocab):
            full_alphabet = _GLOBAL_ALPHABET
            blank_char = _GLOBAL_BLANK_PLACEHOLDER
        else:
            full_alphabet, blank_char = _vocab_to_alphabet(vocab)

        # Apply vocabulary pruning for large vocabularies
        # This is essential for fast-ctc-decode to work with large character sets
        vocab_size = len(vocab)
        if AUTO_PRUNE_LARGE_VOCAB and vocab_size > MAX_VOCAB_SIZE:
            # Prune vocabulary based on log probabilities BEFORE converting to posteriors
            pruned_logits, pruned_alphabet, keep_indices = _prune_vocabulary_for_fast_ctc(
                logits,
                full_alphabet,
                threshold=VOCAB_PRUNE_THRESHOLD,
                percentile=VOCAB_PRUNE_PERCENTILE,
            )
            alphabet = pruned_alphabet
            logits_for_decode = pruned_logits

            if DEBUG_FAST_CTC:
                import sys
                print(
                    f"[fast-ctc] Pruned vocabulary: {vocab_size} -> {len(pruned_alphabet)}",
                    file=sys.stderr,
                    flush=True,
                )
        else:
            alphabet = full_alphabet
            logits_for_decode = logits

        # Convert log probabilities to probabilities
        posteriors = _logits_to_posteriors(logits_for_decode)

        # Validate posteriors
        if posteriors.size == 0:
            return ""

        # Validate alphabet matches posteriors
        if len(alphabet) != posteriors.shape[1]:
            import logging
            logging.getLogger(__name__).error(
                f"Alphabet length {len(alphabet)} != posteriors vocab size {posteriors.shape[1]}"
            )
            return ""

        # Decode using beam search
        # Note: fast-ctc-decode's beam_size parameter corresponds to our beam_width
        # beam_cut_threshold must be <= min probability in posteriors, so we compute it dynamically
        min_prob = float(np.min(posteriors))

        # CRITICAL: Ensure min_prob is not zero (causes crash in fast-ctc-decode)
        # Replace zeros with tiny positive values
        if min_prob == 0.0:
            # Find smallest non-zero value or use float32 minimum
            non_zero_min = posteriors[posteriors > 0].min() if (posteriors > 0).any() else 1e-38
            # Replace zeros with a tiny value
            posteriors = np.where(posteriors == 0, np.float32(1e-38), posteriors)
            # Renormalize rows
            posteriors = posteriors / posteriors.sum(axis=-1, keepdims=True)
            posteriors = np.ascontiguousarray(posteriors.astype(np.float32))
            min_prob = float(np.min(posteriors))

            if DEBUG_FAST_CTC:
                import sys
                print(
                    f"[fast-ctc] Fixed zero posteriors: new min={min_prob:.2e}",
                    file=sys.stderr,
                    flush=True,
                )

        # Use configured threshold if valid, otherwise use 90% of min probability
        if BEAM_CUT_THRESHOLD is not None and BEAM_CUT_THRESHOLD < min_prob:
            cut_threshold = BEAM_CUT_THRESHOLD
        else:
            cut_threshold = max(min_prob * 0.9, 1e-38)  # Never let it be zero

        if DEBUG_FAST_CTC:
            import sys
            print(
                f"[fast-ctc] posteriors shape={posteriors.shape}, dtype={posteriors.dtype}, "
                f"min={min_prob:.2e}, cut={cut_threshold:.2e}, beam={beam_width}, "
                f"alphabet_len={len(alphabet)}",
                file=sys.stderr,
                flush=True,
            )

        # Save input for debugging if env var is set (helps reproduce crashes)
        import os
        if os.environ.get("FAST_CTC_SAVE_INPUT"):
            debug_dir = os.environ.get("FAST_CTC_SAVE_INPUT", "/tmp/fast_ctc_debug")
            os.makedirs(debug_dir, exist_ok=True)
            debug_file = os.path.join(debug_dir, f"input_{os.getpid()}_{id(posteriors)}.npz")
            np.savez(debug_file, posteriors=posteriors, alphabet=alphabet, beam_width=beam_width, cut_threshold=cut_threshold)
            print(f"[fast-ctc] Saved input to {debug_file}", file=sys.stderr, flush=True)

        decoded, _path = beam_search(
            posteriors,
            alphabet,
            beam_size=beam_width,
            beam_cut_threshold=cut_threshold,
        )

        # Remove blank placeholders
        return decoded.replace(blank_char, "")
    except Exception as e:
        # Log error and return empty string rather than crashing the worker
        import logging
        logging.getLogger(__name__).error(f"fast-ctc-decode beam_search failed: {e}")
        return ""


class CTCDecoderFast:
    """
    Fast CTC decoder using fast-ctc-decode library.

    Drop-in replacement for CTCDecoder from ctc_decoder.py.
    """

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

        # Pre-compute alphabet string for fast-ctc-decode
        self._alphabet, self._blank_char = _vocab_to_alphabet(self.ctc_vocab)

    def decode(self, logits: npt.NDArray, keep_mask: npt.NDArray | None = None) -> str:
        """
        Decode logits to text using beam search.

        Args:
            logits: shape (time, vocab) or (vocab, time) - log probabilities
            keep_mask: NOT USED - kept for API compatibility
        """
        try:
            # Ensure shape is (time, vocab)
            if logits.shape[0] == len(self.ctc_vocab):
                logits = np.transpose(logits, axes=[1, 0])

            # Collapse consecutive blank-dominant frames to reduce sequence length
            logits = _collapse_blank_frames(logits)

            if logits.size == 0:
                return ""

            # Apply vocabulary pruning for large vocabularies
            vocab_size = len(self.ctc_vocab)
            if AUTO_PRUNE_LARGE_VOCAB and vocab_size > MAX_VOCAB_SIZE:
                pruned_logits, pruned_alphabet, _ = _prune_vocabulary_for_fast_ctc(
                    logits,
                    self._alphabet,
                    threshold=VOCAB_PRUNE_THRESHOLD,
                    percentile=VOCAB_PRUNE_PERCENTILE,
                )
                alphabet = pruned_alphabet
                logits_for_decode = pruned_logits
            else:
                alphabet = self._alphabet
                logits_for_decode = logits

            # Convert to probabilities
            posteriors = _logits_to_posteriors(logits_for_decode)

            if posteriors.size == 0:
                return ""

            # Compute dynamic beam_cut_threshold (must be <= min probability)
            min_prob = float(np.min(posteriors))

            # CRITICAL: Ensure min_prob is not zero (causes crash in fast-ctc-decode)
            if min_prob == 0.0:
                posteriors = np.where(posteriors == 0, np.float32(1e-38), posteriors)
                posteriors = posteriors / posteriors.sum(axis=-1, keepdims=True)
                posteriors = np.ascontiguousarray(posteriors.astype(np.float32))
                min_prob = float(np.min(posteriors))

            if BEAM_CUT_THRESHOLD is not None and BEAM_CUT_THRESHOLD < min_prob:
                cut_threshold = BEAM_CUT_THRESHOLD
            else:
                cut_threshold = max(min_prob * 0.9, 1e-38)  # Never let it be zero

            # Decode using beam search
            decoded, _path = beam_search(
                posteriors,
                alphabet,
                beam_size=BEAM_WIDTH,
                beam_cut_threshold=cut_threshold,
            )

            # Remove blank placeholders
            return decoded.replace(self._blank_char, "")
        except Exception as e:
            import logging
            logging.getLogger(__name__).error(f"fast-ctc-decode decode failed: {e}")
            return ""

    def decode_greedy(self, logits: npt.NDArray) -> str:
        """
        Decode logits to text using greedy (viterbi) decoding.

        Faster than beam search but potentially less accurate.

        Args:
            logits: shape (time, vocab) or (vocab, time) - log probabilities
        """
        try:
            # Ensure shape is (time, vocab)
            if logits.shape[0] == len(self.ctc_vocab):
                logits = np.transpose(logits, axes=[1, 0])

            # Collapse consecutive blank-dominant frames
            logits = _collapse_blank_frames(logits)

            # Convert to probabilities
            posteriors = _logits_to_posteriors(logits)

            # Decode using viterbi (greedy)
            decoded, _path = viterbi_search(posteriors, self._alphabet)

            # Remove blank placeholders
            return decoded.replace(self._blank_char, "")
        except Exception as e:
            import logging
            logging.getLogger(__name__).error(f"fast-ctc-decode decode_greedy failed: {e}")
            return ""


# Alias for drop-in replacement
CTCDecoder = CTCDecoderFast
