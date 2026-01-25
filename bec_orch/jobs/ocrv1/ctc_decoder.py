import logging
from dataclasses import dataclass
from functools import lru_cache
from typing import Collection

import numpy as np
import numpy.typing as npt
from scipy.special import log_softmax as scipy_log_softmax

from pyctcdecode.decoder import build_ctcdecoder


@dataclass
class SyllableSegment:
    """A decoded syllable/word segment with position and confidence.
    
    Attributes:
        start_pixel: Starting pixel position (x-coordinate) in the line image
        end_pixel: Ending pixel position (x-coordinate) in the line image
        text: The clean decoded text (syllable without trailing delimiters)
        trailing_delimiters: Delimiter(s) that follow this syllable (e.g., "་" or "།")
        confidence: Confidence score (normalized log probability, typically -inf to 0)
    """
    start_pixel: int
    end_pixel: int
    text: str
    trailing_delimiters: str
    confidence: float
    
    @property
    def full_text(self) -> str:
        """Return text with trailing delimiters included."""
        return self.text + self.trailing_delimiters
    
    def as_tuple(self) -> tuple[int, int, str, float]:
        """Return (start_pixel, end_pixel, full_text, confidence) tuple."""
        return (self.start_pixel, self.end_pixel, self.full_text, self.confidence)


@dataclass
class LineDecodeResult:
    """Complete decode result for a line including text and segment details.
    
    Attributes:
        text: Full decoded text for the line
        segments: List of syllable/word segments with positions and confidence
        line_confidence: Overall confidence for the entire line
        logit_score: Raw cumulative logit score from beam search
    """
    text: str
    segments: list[SyllableSegment]
    line_confidence: float
    logit_score: float

# Suppress noisy pyctcdecode warnings
logging.getLogger("pyctcdecode.alphabet").setLevel(logging.ERROR)

# Global decoder instance for multiprocessing (avoids pickling issues)
_GLOBAL_DECODER = None
_GLOBAL_VOCAB_LEN = None  # Use length for fast comparison
_GLOBAL_BLANK_SIGN = "<pad>"

# Tibetan word/syllable delimiters
# These characters trigger word boundaries for language model scoring and frame tracking.
# Includes: tsheg (་), tsheg-like (༌), space, shad (།), sbrul-shad (༴), visarga (ཿ),
# head mark (࿒), brackets (༼༽), and other punctuation (࿙࿚༔)
TIBETAN_WORD_DELIMITERS: frozenset[str] = frozenset({
    "་",  # tsheg (most common syllable separator)
    "༌",  # tsheg-like / non-breaking tsheg
    " ",  # space
    "།",  # shad (sentence/phrase delimiter)
    "༴",  # sbrul-shad (repetition mark)
    "ཿ",  # visarga / rnam-bcad
    "࿒",  # head mark
    "༼",  # opening bracket
    "༽",  # closing bracket
    "࿙",  # leading ornament
    "࿚",  # trailing ornament
    "༔",  # ter-tsheg / gter-tsheg
})

# Space-only delimiters - original behavior for backward compatibility
# Use this for stable reference outputs that match previous decoder behavior
SPACE_ONLY_DELIMITERS: frozenset[str] = frozenset({" "})

# Default word delimiters - use Tibetan delimiters for syllable-level decoding
# This enables proper syllable boundaries for:
# - Per-syllable timing (text_frames)
# - Per-syllable LM scoring (when using KenLM trained on syllables)
# - Per-syllable confidence scores
# Note: This changes beam search dynamics compared to space-only delimiters
DEFAULT_WORD_DELIMITERS: frozenset[str] = TIBETAN_WORD_DELIMITERS


@lru_cache(maxsize=32)
def _get_cached_decoder(
    vocab_tuple: tuple[str, ...],
    word_delimiters: frozenset[str] | None = None,
):
    """Get or create a cached CTC decoder for a given vocabulary.
    
    Caches decoders by vocabulary to avoid rebuilding for lines with the same
    pruned vocabulary (common in batch processing with model-side pruning).
    
    Args:
        vocab_tuple: vocabulary as a tuple (hashable for caching)
        word_delimiters: characters that trigger word boundaries.
                        Default is space-only for backward compatibility.
                        Use TIBETAN_WORD_DELIMITERS for syllable-level decoding.
    """
    if word_delimiters is None:
        word_delimiters = DEFAULT_WORD_DELIMITERS
    return build_ctcdecoder(list(vocab_tuple), word_delimiters=word_delimiters)


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
    return _decode_logits_beam_search_internal(
        log_probs, vocab, beam_width=beam_width, token_min_logp=token_min_logp
    )


def _init_global_decoder(
    vocab: list[str],
    word_delimiters: Collection[str] | None = None,
) -> None:
    """Initialize the global decoder for use in worker processes.
    
    Args:
        vocab: vocabulary list
        word_delimiters: characters that trigger word boundaries.
                        Default is space-only for backward compatibility.
                        Use TIBETAN_WORD_DELIMITERS for syllable-level decoding.
    """
    global _GLOBAL_DECODER, _GLOBAL_VOCAB_LEN
    if _GLOBAL_DECODER is None or _GLOBAL_VOCAB_LEN != len(vocab):
        if word_delimiters is None:
            word_delimiters = DEFAULT_WORD_DELIMITERS
        _GLOBAL_DECODER = build_ctcdecoder(vocab, word_delimiters=word_delimiters)
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
    beam_width: int | None = None,
    token_min_logp: float | None = None,
) -> str:
    """
    Internal beam search decode function - expects log probabilities (already softmaxed).

    Vocabulary pruning is now handled in the model before IPC, so this function
    receives pre-pruned log_probs and vocab.

    Args:
        log_probs: shape (time, vocab) - log probabilities (already pruned by model)
        vocab: vocabulary list (already pruned by model)
        beam_width: beam width for decoding (defaults to module BEAM_WIDTH)
        token_min_logp: token min log prob (defaults to module TOKEN_MIN_LOGP)
    """
    import time

    # Use passed values or fall back to module defaults
    if beam_width is None:
        beam_width = BEAM_WIDTH
    if token_min_logp is None:
        token_min_logp = TOKEN_MIN_LOGP

    t0 = time.perf_counter()
    orig_shape = log_probs.shape

    # Get cached decoder for vocabulary (avoids rebuild for same vocab)
    vocab_tuple = tuple(vocab)
    decoder = _get_cached_decoder(vocab_tuple)
    t1 = time.perf_counter()

    result = decoder.decode(
        log_probs, beam_width=beam_width, token_min_logp=token_min_logp,
        logits_are_log_probs=True
    )
    t2 = time.perf_counter()

    logging.info(
        f"[CTC timing] shape={orig_shape}, vocab={len(vocab)}, "
        f"build_decoder={1000*(t1-t0):.1f}ms, decode={1000*(t2-t1):.1f}ms"
    )
    return result.replace(_GLOBAL_BLANK_SIGN, "")


def decode_logits_beam_search(
    logits: npt.NDArray,
    vocab: list[str],
    beam_width: int | None = None,
    token_min_logp: float | None = None,
    logits_are_log_probs: bool = True,
) -> str:
    """
    Module-level beam search decode function for use with ProcessPoolExecutor.

    This function can be pickled and sent to worker processes.
    Expects logits in (time, vocab) shape - caller must transpose if needed.

    Vocabulary pruning is now handled in the model before IPC, so this function
    receives pre-pruned logits and vocab.

    Args:
        logits: shape (time, vocab) - log probabilities (or raw logits if logits_are_log_probs=False)
        vocab: vocabulary list (already pruned by model)
        beam_width: beam width for decoding (defaults to module BEAM_WIDTH)
        token_min_logp: token min log prob (defaults to module TOKEN_MIN_LOGP)
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
        log_probs, vocab, beam_width, token_min_logp
    )


def _split_into_syllables(
    text: str,
    delimiters: frozenset[str],
) -> list[tuple[str, str]]:
    """
    Split text into syllables, returning (clean_syllable, trailing_delimiters) pairs.
    
    This handles Tibetan text where delimiters follow syllables:
    "བཀྲ་ཤིས།" -> [("བཀྲ", "་"), ("ཤིས", "།")]
    
    Args:
        text: Full decoded text
        delimiters: Set of delimiter characters
    
    Returns:
        List of (syllable, trailing_delimiters) tuples
    """
    if not text:
        return []
    
    result = []
    current_syllable = ""
    current_delimiters = ""
    
    for char in text:
        if char in delimiters:
            # This is a delimiter
            if current_syllable:
                # Accumulate delimiter after the syllable
                current_delimiters += char
            else:
                # Delimiter at start or consecutive - add to previous or skip
                if result:
                    # Add to previous syllable's delimiters
                    prev_syl, prev_delim = result[-1]
                    result[-1] = (prev_syl, prev_delim + char)
                else:
                    # Leading delimiter - skip or treat as part of first syllable
                    current_delimiters += char
        else:
            # This is a regular character
            if current_syllable and current_delimiters:
                # We have a complete syllable with delimiters, save it
                result.append((current_syllable, current_delimiters))
                current_syllable = char
                current_delimiters = ""
            else:
                # Continue building current syllable
                current_syllable += char
    
    # Don't forget the last syllable
    if current_syllable or current_delimiters:
        result.append((current_syllable, current_delimiters))
    
    return result


def _compute_segment_confidences(
    log_probs: npt.NDArray,
    text_frames: list[tuple[str, tuple[int, int]]],
) -> list[float]:
    """
    Compute per-segment confidence from frame log probabilities.
    
    For each segment, computes the mean of the maximum log probability
    at each frame. This represents "how confident was the model about
    its best choice at each timestep within this segment?"
    
    Args:
        log_probs: shape (num_frames, vocab_size) - log probabilities
        text_frames: list of (word_text, (start_frame, end_frame)) from decoder
    
    Returns:
        List of confidence scores (one per segment), each is mean log prob
    """
    confidences = []
    for _word_text, (start_frame, end_frame) in text_frames:
        if end_frame <= start_frame:
            confidences.append(float("-inf"))
            continue
        
        # Get the log probs for frames in this segment
        segment_log_probs = log_probs[start_frame:end_frame]
        
        # Max log prob at each frame (confidence in best token)
        max_per_frame = segment_log_probs.max(axis=1)
        
        # Average log probability for this segment
        segment_confidence = float(max_per_frame.mean())
        confidences.append(segment_confidence)
    
    return confidences


def decode_logits_with_segments(
    logits: npt.NDArray,
    vocab: list[str],
    original_width: int,
    beam_width: int | None = None,
    token_min_logp: float | None = None,
    logits_are_log_probs: bool = True,
    word_delimiters: frozenset[str] | None = None,
) -> LineDecodeResult:
    """
    Module-level beam search decode with syllable/word segments.

    Returns structured output with each segment containing:
    - start_pixel, end_pixel: position in the original line image
    - text: the decoded text for this segment
    - confidence: per-segment confidence score

    Args:
        logits: shape (time, vocab) - log probabilities (or raw logits)
        vocab: vocabulary list (already pruned by model)
        original_width: width of original line image in pixels
        beam_width: beam width for decoding (defaults to module BEAM_WIDTH)
        token_min_logp: token min log prob (defaults to module TOKEN_MIN_LOGP)
        logits_are_log_probs: if True (default), logits are already log probabilities
        word_delimiters: characters that trigger word/syllable boundaries

    Returns:
        LineDecodeResult with text, segments, and confidence scores
    """
    if beam_width is None:
        beam_width = BEAM_WIDTH
    if token_min_logp is None:
        token_min_logp = TOKEN_MIN_LOGP
    if word_delimiters is None:
        word_delimiters = DEFAULT_WORD_DELIMITERS

    # Apply log_softmax only if not already done
    if logits_are_log_probs:
        log_probs = logits
    else:
        log_probs = _apply_log_softmax(logits)

    num_frames = log_probs.shape[0]

    # Get cached decoder
    vocab_tuple = tuple(vocab)
    decoder = _get_cached_decoder(vocab_tuple, word_delimiters)

    # Use decode_beams to get frame information
    output_beams = decoder.decode_beams(
        log_probs,
        beam_width=beam_width,
        token_min_logp=token_min_logp,
        logits_are_log_probs=True,
    )

    if not output_beams:
        return LineDecodeResult(
            text="",
            segments=[],
            line_confidence=float("-inf"),
            logit_score=0.0,
        )

    # Take the best beam
    best_beam = output_beams[0]
    full_text = best_beam.text.replace(_GLOBAL_BLANK_SIGN, "")
    logit_score = best_beam.logit_score

    # Calculate overall line confidence (normalized by number of frames)
    line_confidence = logit_score / num_frames if num_frames > 0 else float("-inf")

    # Parse syllables from the full text using our delimiters
    # This handles Tibetan properly (pyctcdecode's whitespace split doesn't work)
    syllables = _split_into_syllables(full_text, word_delimiters)
    
    # Get frame ranges - pyctcdecode stores these without the word text for non-space delimiters
    # text_frames is List[(word, (start, end))] but the words may be wrong for non-space delimited text
    frame_ranges = [frames for _, frames in best_beam.text_frames]
    
    # If frame count doesn't match syllable count, fall back to distributing evenly
    if len(frame_ranges) != len(syllables) and syllables:
        # Distribute frames proportionally across syllables
        total_frames = num_frames
        syllable_lengths = [len(syl) + len(delim) for syl, delim in syllables]
        total_chars = sum(syllable_lengths)
        if total_chars > 0:
            frame_ranges = []
            current_frame = 0
            for length in syllable_lengths:
                frames_for_syllable = max(1, int(total_frames * length / total_chars))
                frame_ranges.append((current_frame, current_frame + frames_for_syllable))
                current_frame += frames_for_syllable
    
    # Compute per-segment confidence from frame probabilities
    # Create fake text_frames for confidence computation
    fake_text_frames = [(syl, frames) for (syl, _), frames in zip(syllables, frame_ranges)]
    segment_confidences = _compute_segment_confidences(log_probs, fake_text_frames)

    # Convert frames to pixels
    pixels_per_frame = original_width / num_frames if num_frames > 0 else 1.0

    segments: list[SyllableSegment] = []
    for (syllable, trailing), (start_frame, end_frame), seg_conf in zip(
        syllables, frame_ranges, segment_confidences
    ):
        if not syllable and not trailing:
            continue

        # Convert frames to pixels
        start_pixel = int(start_frame * pixels_per_frame)
        end_pixel = int(end_frame * pixels_per_frame)

        segments.append(SyllableSegment(
            start_pixel=start_pixel,
            end_pixel=end_pixel,
            text=syllable,
            trailing_delimiters=trailing,
            confidence=seg_conf,
        ))

    return LineDecodeResult(
        text=full_text,
        segments=segments,
        line_confidence=line_confidence,
        logit_score=logit_score,
    )


class CTCDecoder:
    """CTC decoder with beam search."""

    def __init__(
        self,
        charset: str | list[str],
        add_blank: bool,
        word_delimiters: Collection[str] | None = None,
    ):
        """Initialize CTC decoder.
        
        Args:
            charset: character set as string or list
            add_blank: whether to add blank token at index 0
            word_delimiters: characters that trigger word boundaries for LM scoring
                           and frame tracking. Default is space-only for backward
                           compatibility. Use TIBETAN_WORD_DELIMITERS for syllable-level.
        """
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

        # Store word delimiters for reference
        if word_delimiters is None:
            self.word_delimiters = DEFAULT_WORD_DELIMITERS
        else:
            self.word_delimiters = frozenset(word_delimiters)

        # Build beam search decoder with word delimiters
        self._beam_decoder = build_ctcdecoder(
            self.ctc_vocab, word_delimiters=self.word_delimiters
        )

    def decode(
        self, logits: npt.NDArray, logits_are_log_probs: bool = True
    ) -> str:
        """Decode logits to text using beam search.

        Vocabulary pruning is now handled in the model before IPC.

        Args:
            logits: shape (time, vocab) or (vocab, time) - log probabilities (or raw logits)
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

        return self._beam_decoder.decode(
            log_probs, beam_width=BEAM_WIDTH, token_min_logp=TOKEN_MIN_LOGP,
            logits_are_log_probs=True
        ).replace(self.blank_sign, "")

    def decode_with_segments(
        self,
        logits: npt.NDArray,
        original_width: int,
        logits_are_log_probs: bool = True,
        beam_width: int | None = None,
        token_min_logp: float | None = None,
    ) -> LineDecodeResult:
        """Decode logits to text with syllable/word segments including positions and confidence.

        Returns structured output with each syllable/word segment containing:
        - start_pixel, end_pixel: position in the original line image
        - text: the decoded text for this segment
        - confidence: per-segment confidence score

        Args:
            logits: shape (time, vocab) or (vocab, time) - log probabilities (or raw logits)
            original_width: width of the original line image in pixels (for frame-to-pixel conversion)
            logits_are_log_probs: if True (default), logits are already log probabilities
            beam_width: beam width for decoding (defaults to module BEAM_WIDTH)
            token_min_logp: minimum token log probability (defaults to module TOKEN_MIN_LOGP)

        Returns:
            LineDecodeResult with text, segments, and confidence scores
        """
        if beam_width is None:
            beam_width = BEAM_WIDTH
        if token_min_logp is None:
            token_min_logp = TOKEN_MIN_LOGP

        # Ensure shape is (time, vocab)
        if logits.shape[0] == len(self.ctc_vocab):
            logits = np.transpose(logits, axes=[1, 0])

        # Apply log_softmax only if not already done
        if logits_are_log_probs:
            log_probs = logits
        else:
            log_probs = _apply_log_softmax(logits)

        num_frames = log_probs.shape[0]

        # Use decode_beams to get frame information
        output_beams = self._beam_decoder.decode_beams(
            log_probs,
            beam_width=beam_width,
            token_min_logp=token_min_logp,
            logits_are_log_probs=True,
        )

        if not output_beams:
            return LineDecodeResult(
                text="",
                segments=[],
                line_confidence=float("-inf"),
                logit_score=0.0,
            )

        # Take the best beam
        best_beam = output_beams[0]
        full_text = best_beam.text.replace(self.blank_sign, "")
        logit_score = best_beam.logit_score

        # Calculate overall line confidence (normalized by number of frames)
        # logit_score is cumulative log probability; normalize by frames for comparability
        line_confidence = logit_score / num_frames if num_frames > 0 else float("-inf")

        # Parse syllables from the full text using our delimiters
        syllables = _split_into_syllables(full_text, self.word_delimiters)
        
        # Get frame ranges from pyctcdecode output
        frame_ranges = [frames for _, frames in best_beam.text_frames]
        
        # If frame count doesn't match syllable count, distribute evenly
        if len(frame_ranges) != len(syllables) and syllables:
            total_frames = num_frames
            syllable_lengths = [len(syl) + len(delim) for syl, delim in syllables]
            total_chars = sum(syllable_lengths)
            if total_chars > 0:
                frame_ranges = []
                current_frame = 0
                for length in syllable_lengths:
                    frames_for_syllable = max(1, int(total_frames * length / total_chars))
                    frame_ranges.append((current_frame, current_frame + frames_for_syllable))
                    current_frame += frames_for_syllable
        
        # Compute per-segment confidence from frame probabilities
        fake_text_frames = [(syl, frames) for (syl, _), frames in zip(syllables, frame_ranges)]
        segment_confidences = _compute_segment_confidences(log_probs, fake_text_frames)

        # Convert frames to pixels
        pixels_per_frame = original_width / num_frames if num_frames > 0 else 1.0

        segments: list[SyllableSegment] = []
        for (syllable, trailing), (start_frame, end_frame), seg_conf in zip(
            syllables, frame_ranges, segment_confidences
        ):
            if not syllable and not trailing:
                continue

            # Convert frames to pixels
            start_pixel = int(start_frame * pixels_per_frame)
            end_pixel = int(end_frame * pixels_per_frame)

            segments.append(SyllableSegment(
                start_pixel=start_pixel,
                end_pixel=end_pixel,
                text=syllable,
                trailing_delimiters=trailing,
                confidence=seg_conf,
            ))

        return LineDecodeResult(
            text=full_text,
            segments=segments,
            line_confidence=line_confidence,
            logit_score=logit_score,
        )
