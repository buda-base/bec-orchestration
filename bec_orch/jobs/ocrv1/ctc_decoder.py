import numpy as np
import numpy.typing as npt
from pyctcdecode.decoder import build_ctcdecoder


class CTCDecoder:
    def __init__(self, charset: str | list[str], add_blank: bool):
        self.blank_sign = "<blk>"

        if isinstance(charset, str):
            self.charset = list(charset)
        else:
            self.charset = charset

        self.ctc_vocab = self.charset.copy()
        if add_blank:
            self.ctc_vocab.insert(0, self.blank_sign)

        self.ctc_decoder = build_ctcdecoder(self.ctc_vocab)

    def decode(self, logits: npt.NDArray) -> str:
        if logits.shape[0] == len(self.ctc_vocab):
            logits = np.transpose(logits, axes=[1, 0])
        return self.ctc_decoder.decode(logits).replace(self.blank_sign, "")
