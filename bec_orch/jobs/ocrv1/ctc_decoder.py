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

# Disable decode_batch with pool - it hangs because pool workers don't have decoder initialized.
# The ProcessPoolExecutor approach works well on both Linux and macOS.
CAN_USE_MP_POOL = False


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
    global _GLOBAL_DECODER, _GLOBAL_VOCAB_LEN

    # Initialize decoder in worker process if needed (should already be done by initializer)
    if _GLOBAL_DECODER is None or _GLOBAL_VOCAB_LEN != len(vocab):
        _init_global_decoder(vocab)

    # Ensure shape is (time, vocab)
    if logits.shape[0] == len(vocab):
        logits = np.transpose(logits, axes=[1, 0])

    if _GLOBAL_DECODER is None:
        raise RuntimeError("CTC decoder not initialized")

    return _GLOBAL_DECODER.decode(logits, beam_width=BEAM_WIDTH, token_min_logp=TOKEN_MIN_LOGP).replace(
        _GLOBAL_BLANK_SIGN, ""
    )


def decode_logits_batch(logits_list: list[npt.NDArray], vocab: list[str], pool=None) -> list[str]:
    """
    Batch decode multiple logit arrays using pyctcdecode's decode_batch.

    Args:
        logits_list: List of logit arrays to decode
        vocab: Vocabulary list
        pool: Optional multiprocessing pool (only works on Linux with 'fork' context)
    """
    global _GLOBAL_DECODER, _GLOBAL_VOCAB_LEN

    if _GLOBAL_DECODER is None or _GLOBAL_VOCAB_LEN != len(vocab):
        _init_global_decoder(vocab)

    if _GLOBAL_DECODER is None:
        raise RuntimeError("CTC decoder not initialized")

    # Ensure all logits are (time, vocab) shape
    vocab_size = len(vocab)
    prepared = []
    for logits in logits_list:
        if logits.shape[0] == vocab_size:
            logits = np.transpose(logits, axes=[1, 0])
        prepared.append(logits)

    # Use decode_batch - pool enables parallel decoding on Linux
    results = _GLOBAL_DECODER.decode_batch(
        pool=pool,
        logits_list=prepared,
        beam_width=BEAM_WIDTH,
        token_min_logp=TOKEN_MIN_LOGP,
    )

    return [text.replace(_GLOBAL_BLANK_SIGN, "") for text in results]


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

        return self._beam_decoder.decode(logits, beam_width=BEAM_WIDTH, token_min_logp=TOKEN_MIN_LOGP).replace(
            self.blank_sign, ""
        )
