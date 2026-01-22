import numpy as np
import numpy.typing as npt

# Global decoder instance for multiprocessing (avoids pickling issues)
_GLOBAL_DECODER = None
_GLOBAL_VOCAB_LEN = None  # Use length for fast comparison
_GLOBAL_BLANK_SIGN = "<blk>"

# Beam width for CTC decoding
BEAM_WIDTH = 50

# Token pruning - skip tokens with log probability below this threshold
# Default is -5.0, more negative = less pruning, less negative = more pruning
TOKEN_MIN_LOGP = -5.0

# Blank index in vocabulary (blank is always first token)
BLANK_IDX = 0


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
    Expects logits in (time, vocab) shape - caller must transpose if needed.
    """
    global _GLOBAL_DECODER, _GLOBAL_VOCAB_LEN

    # Initialize decoder in worker process if needed (should already be done by initializer)
    if _GLOBAL_DECODER is None or _GLOBAL_VOCAB_LEN != len(vocab):
        _init_global_decoder(vocab)

    # Collapse consecutive blank-dominant frames to reduce sequence length
    logits = _collapse_blank_frames(logits)

    if _GLOBAL_DECODER is None:
        raise RuntimeError("CTC decoder not initialized")

    return _GLOBAL_DECODER.decode(logits, beam_width=BEAM_WIDTH, token_min_logp=TOKEN_MIN_LOGP).replace(
        _GLOBAL_BLANK_SIGN, ""
    )


class CTCDecoder:
    """CTC decoder with beam search."""

    def __init__(self, charset: str | list[str], add_blank: bool, use_beam_search: bool = True):
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
        from pyctcdecode.decoder import build_ctcdecoder

        self._beam_decoder = build_ctcdecoder(self.ctc_vocab)

    def decode(self, logits: npt.NDArray) -> str:
        """Decode logits to text using beam search."""
        # Ensure shape is (time, vocab)
        if logits.shape[0] == len(self.ctc_vocab):
            logits = np.transpose(logits, axes=[1, 0])

        # Collapse consecutive blank-dominant frames to reduce sequence length
        logits = _collapse_blank_frames(logits)

        return self._beam_decoder.decode(logits, beam_width=BEAM_WIDTH, token_min_logp=TOKEN_MIN_LOGP).replace(
            self.blank_sign, ""
        )
