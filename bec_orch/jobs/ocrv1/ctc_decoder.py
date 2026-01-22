import numpy as np
import numpy.typing as npt

# Global decoder instance for multiprocessing (avoids pickling issues)
_GLOBAL_DECODER = None
_GLOBAL_VOCAB_LEN = None  # Use length for fast comparison
_GLOBAL_BLANK_SIGN = "<blk>"


# Beam width for CTC decoding (default pyctcdecode is 100, but 20 is usually sufficient)
BEAM_WIDTH = 50


def _init_global_decoder(vocab: list[str]) -> None:
    """Initialize the global decoder for use in worker processes."""
    global _GLOBAL_DECODER, _GLOBAL_VOCAB_LEN
    if _GLOBAL_DECODER is None or _GLOBAL_VOCAB_LEN != len(vocab):
        from pyctcdecode.decoder import build_ctcdecoder

        _GLOBAL_DECODER = build_ctcdecoder(vocab)
        _GLOBAL_VOCAB_LEN = len(vocab)


def init_worker_process(vocab: list[str]) -> None:
    """
    Initializer function for ProcessPoolExecutor workers.

    Called once when each worker process starts, avoiding repeated
    decoder initialization overhead.
    """
    _init_global_decoder(vocab)


def decode_logits_beam_search(logits: npt.NDArray, vocab: list[str]) -> str:
    """
    Module-level beam search decode function for use with ProcessPoolExecutor.

    This function can be pickled and sent to worker processes.
    """
    global _GLOBAL_DECODER, _GLOBAL_VOCAB_LEN, _GLOBAL_BLANK_SIGN

    # Initialize decoder in worker process if needed (should already be done by initializer)
    if _GLOBAL_DECODER is None or _GLOBAL_VOCAB_LEN != len(vocab):
        _init_global_decoder(vocab)

    # Ensure shape is (time, vocab)
    if logits.shape[0] == len(vocab):
        logits = np.transpose(logits, axes=[1, 0])

    if _GLOBAL_DECODER is None:
        raise RuntimeError("CTC decoder not initialized")

    return _GLOBAL_DECODER.decode(logits, beam_width=BEAM_WIDTH).replace(_GLOBAL_BLANK_SIGN, "")


class CTCDecoder:
    """CTC decoder with greedy (fast) and beam search (slow) modes."""

    def __init__(self, charset: str | list[str], add_blank: bool, use_beam_search: bool = True):
        self.blank_sign = "<blk>"
        self.use_beam_search = use_beam_search

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

        # Build beam search decoder if needed (for in-process use)
        self._beam_decoder = None
        if use_beam_search:
            from pyctcdecode.decoder import build_ctcdecoder

            self._beam_decoder = build_ctcdecoder(self.ctc_vocab)

    def decode(self, logits: npt.NDArray) -> str:
        """Decode logits to text using greedy or beam search."""
        # Ensure shape is (time, vocab)
        if logits.shape[0] == len(self.ctc_vocab):
            logits = np.transpose(logits, axes=[1, 0])

        if self.use_beam_search and self._beam_decoder is not None:
            return self._beam_decoder.decode(logits, beam_width=BEAM_WIDTH).replace(self.blank_sign, "")

        return self._greedy_decode(logits)

    def _greedy_decode(self, logits: npt.NDArray) -> str:
        """Fast greedy CTC decoding."""
        # Handle different shapes - ensure 2D (time, vocab)
        if logits.ndim == 1:
            # Single timestep - just return best char
            best_idx = int(np.argmax(logits))
            if best_idx < len(self.ctc_vocab) and best_idx != self.blank_idx:
                char = self.ctc_vocab[best_idx]
                if char != self.blank_sign:
                    return char
            return ""

        # Get best class at each timestep (axis=1 for vocab dimension)
        best_indices = np.argmax(logits, axis=1)

        # Collapse repeated characters and remove blanks
        chars = []
        prev_idx = -1
        for idx in best_indices:
            if idx != prev_idx and idx != self.blank_idx:
                if idx < len(self.ctc_vocab):
                    char = self.ctc_vocab[idx]
                    if char != self.blank_sign:
                        chars.append(char)
            prev_idx = idx

        return "".join(chars)
