"""DRY ("Don't Repeat Yourself") repetition penalty for vLLM, with fire telemetry.

Vendored (behaviour) from
``bec-ocr-training/deploy/fast_inference/dry_logits_processor.py`` so the
production serving recipe is reproduced exactly.

Why DRY (and not ``repetition_penalty`` / ``no_repeat_ngram_size``)
------------------------------------------------------------------
The loop-hallucination study (``docs/experiments/loop_hallucination.md``, E7)
found that on Tibetan OCR ``repetition_penalty`` / LZ penalties wreck normal
pages, and ``no_repeat_ngram_size=20`` is a hard ban that corrupts
legitimately-repetitive text (mantras / litanies). **DRY is the one soft guard
surgical enough**: with ``multiplier=0.8, base=1.75, allowed_length=12`` and
**no** sequence breakers it zeroes hard loops with minimal clean-page tax. Shad
(``།``) breakers backfire (Tibetan loops are shad-delimited), so use a larger
``allowed_length`` instead of breakers.

DRY penalises, for each candidate next token, the length ``L`` of the longest
repeated suffix choosing it would extend::

    logits[t] -= multiplier * base ** (L - allowed_length)   # only if L >= allowed_length

Fire telemetry
--------------
DRY fires on ~57% of Tibetan OCR pages, but almost always as a mild single-token
nip (docs/eval_in_production.md). Only the ~2% of pages where it fires *hard*
(``fires>=100``) are worth re-decoding. The processor records per-request
severity (``fires``, ``max_L``, ``max_penalty``, ``sum_penalty``, first/last
position) and — because it runs in the vLLM EngineCore *worker* process, not the
driver — writes it to a ``<stats_path>/<id>.json`` file side-channel the driver
reads back via :func:`load_dry_stats_dir` after ``generate``.
"""

from __future__ import annotations

import json
import os

import torch

DEFAULT_MULTIPLIER = 0.8
DEFAULT_BASE = 1.75
DEFAULT_ALLOWED_LENGTH = 12
DEFAULT_WINDOW = 512
DEFAULT_MAX_MATCH = 50


def dry_match_lengths(
    gen: list[int],
    sequence_breakers: set[int] | None = None,
    max_match: int = DEFAULT_MAX_MATCH,
) -> dict[int, int]:
    """Longest repeated-suffix match length per continuation-candidate token."""
    breakers = sequence_breakers or ()
    n = len(gen)
    if n < 2:
        return {}
    last = gen[-1]
    if last in breakers:
        return {}
    s = "".join(map(chr, gen))
    last_ch = s[-1]
    ml: dict[int, int] = {}
    limit = n - 1  # anchors must be strictly before the final position
    start = 0
    while start < limit:
        i = s.find(last_ch, start, limit)
        if i == -1:
            break
        cont = gen[i + 1]
        if cont not in breakers:
            match_length = 1
            m = 1
            while (i - m) >= 0 and (n - 1 - m) >= 0 and m <= max_match:
                a = gen[i - m]
                if a != gen[n - 1 - m] or a in breakers:
                    break
                match_length += 1
                m += 1
            if match_length > ml.get(cont, 0):
                ml[cont] = match_length
        start = i + 1
    return ml


def _dry_hits(
    gen: list[int],
    multiplier: float,
    base: float,
    allowed_length: int,
    sequence_breakers: set[int] | None,
    window: int,
    max_match: int,
) -> list[tuple[int, int, float]]:
    """Tokens DRY would penalise: ``(token_id, match_length, penalty)``."""
    if multiplier <= 0.0 or len(gen) < 2:
        return []
    if len(gen) > window:
        gen = gen[-window:]
    ml = dry_match_lengths(gen, sequence_breakers, max_match)
    if not ml:
        return []
    hits: list[tuple[int, int, float]] = []
    for tok, length in ml.items():
        if length >= allowed_length:
            hits.append((tok, length, -multiplier * (base ** (length - allowed_length))))
    return hits


def _apply_dry(
    gen: list[int],
    logits: torch.Tensor,
    multiplier: float,
    base: float,
    allowed_length: int,
    sequence_breakers: set[int] | None,
    window: int,
    max_match: int,
    stats: dict | None = None,
    gen_len: int | None = None,
) -> torch.Tensor:
    """Apply the DRY penalty in-place to a 1-D ``logits`` row for one request.

    When ``stats`` is provided, a fire is recorded as ``fires += 1`` plus running
    ``max_L`` / ``max_penalty`` / ``first_pos`` / ``sum_penalty``. ``gen_len`` is
    the un-windowed generated-token count (used as the fire position).
    """
    hits = _dry_hits(gen, multiplier, base, allowed_length, sequence_breakers,
                     window, max_match)
    if not hits:
        return logits
    if stats is not None:
        stats["fires"] = int(stats.get("fires", 0)) + 1
        max_L = max(h[1] for h in hits)
        max_pen = max(abs(h[2]) for h in hits)
        stats["max_L"] = max(int(stats.get("max_L", 0)), int(max_L))
        stats["max_penalty"] = max(float(stats.get("max_penalty", 0.0)), float(max_pen))
        stats["sum_penalty"] = float(stats.get("sum_penalty", 0.0)) + float(max_pen)
        pos = int(gen_len if gen_len is not None else len(gen))
        if "first_pos" not in stats:
            stats["first_pos"] = pos
        stats["last_pos"] = pos
    toks = [h[0] for h in hits]
    vals = [h[2] for h in hits]
    idx = torch.tensor(toks, device=logits.device, dtype=torch.long)
    pen = torch.tensor(vals, device=logits.device, dtype=logits.dtype)
    logits.index_add_(0, idx, pen)
    return logits


def _safe_stats_id(sid: str) -> str:
    return "".join(c if c.isalnum() or c in "._-" else "_" for c in str(sid))[:200] or "page"


def flush_dry_stats(path: str, sid: str, payload: dict) -> None:
    """Atomically write one request's DRY summary into ``path/<id>.json``.

    The logits processor runs in the vLLM EngineCore worker, so this file
    side-channel is how the driver reads fire/severity after ``generate``.
    """
    if not path or not sid:
        return
    try:
        os.makedirs(path, exist_ok=True)
        safe = _safe_stats_id(sid)
        dst = os.path.join(path, f"{safe}.json")
        tmp = dst + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False)
        os.replace(tmp, dst)
    except Exception:
        pass


def load_dry_stats_dir(path: str) -> dict[str, dict]:
    """Read worker-written ``<id>.json`` summaries. Missing dir -> empty dict."""
    out: dict[str, dict] = {}
    if not path or not os.path.isdir(path):
        return out
    for name in os.listdir(path):
        if not name.endswith(".json"):
            continue
        fp = os.path.join(path, name)
        try:
            with open(fp, encoding="utf-8") as f:
                rec = json.load(f)
            sid = str(rec.get("id") or name[:-5])
            out[sid] = rec
        except Exception:
            continue
    return out


class DRYRequest:
    """Per-request vLLM logits processor callable ``(output_ids, logits)``.

    ``output_ids`` is the live list of tokens decoded so far for this request
    (prompt excluded, exactly what DRY needs); ``logits`` is the 1-D next-token
    logits row for this request. When ``stats_id`` / ``stats_path`` are set, the
    per-request fire summary is flushed to a JSON file side-channel.
    """

    def __init__(
        self,
        multiplier: float,
        base: float,
        allowed_length: int,
        sequence_breakers: set[int] | None,
        window: int,
        max_match: int,
        stats_id: str | None = None,
        stats_path: str | None = None,
    ) -> None:
        self.multiplier = float(multiplier)
        self.base = float(base)
        self.allowed_length = int(allowed_length)
        self.sequence_breakers = set(sequence_breakers or ())
        self.window = int(window)
        self.max_match = int(max_match)
        self.stats_id = stats_id
        self.stats_path = stats_path
        self.stats: dict = {"id": stats_id, "fires": 0, "max_L": 0,
                            "max_penalty": 0.0, "sum_penalty": 0.0}

    def _maybe_flush(self) -> None:
        if not self.stats_id or not self.stats_path:
            return
        if int(self.stats.get("fires", 0)) <= 0:
            return
        flush_dry_stats(self.stats_path, self.stats_id, self.stats)

    def __call__(self, output_ids: list[int], logits: torch.Tensor) -> torch.Tensor:
        gen = output_ids
        if hasattr(output_ids, "tolist"):
            gen = output_ids.tolist()
        out = _apply_dry(
            gen, logits, self.multiplier, self.base, self.allowed_length,
            self.sequence_breakers, self.window, self.max_match,
            stats=self.stats, gen_len=len(gen),
        )
        self._maybe_flush()
        return out


def _import_adapter():
    # vLLM V1 per-request logits processor base class.
    from vllm.v1.sample.logits_processor import AdapterLogitsProcessor
    return AdapterLogitsProcessor


def _make_dry_logits_processor_cls():
    AdapterLogitsProcessor = _import_adapter()

    class DRYLogitsProcessor(AdapterLogitsProcessor):
        """vLLM adapter: reads DRY config from ``SamplingParams.extra_args``.

        Register once at engine init::

            LLM(model=..., logits_processors=[DRYLogitsProcessor])

        then enable per request::

            SamplingParams(temperature=0, extra_args={
                "dry_multiplier": 0.8, "dry_base": 1.75, "dry_allowed_length": 12,
                "dry_stats_id": "000123", "dry_stats_path": "/tmp/dry_stats",
            })

        Requests without ``dry_multiplier > 0`` are untouched (returns None).
        """

        def is_argmax_invariant(self) -> bool:
            # DRY lowers loop-continuation logits and can flip the greedy
            # argmax, so it must run for greedy too.
            return False

        def new_req_logits_processor(self, params):
            ea = getattr(params, "extra_args", None) or {}
            mult = float(ea.get("dry_multiplier", 0.0) or 0.0)
            if mult <= 0.0:
                return None
            breakers = ea.get("dry_sequence_breakers") or None
            if breakers is not None:
                breakers = set(int(x) for x in breakers)
            sid = ea.get("dry_stats_id")
            path = ea.get("dry_stats_path")
            return DRYRequest(
                multiplier=mult,
                base=float(ea.get("dry_base", DEFAULT_BASE)),
                allowed_length=int(ea.get("dry_allowed_length", DEFAULT_ALLOWED_LENGTH)),
                sequence_breakers=breakers,
                window=int(ea.get("dry_window", DEFAULT_WINDOW)),
                max_match=int(ea.get("dry_max_match", DEFAULT_MAX_MATCH)),
                stats_id=str(sid) if sid is not None else None,
                stats_path=str(path) if path else None,
            )

    return DRYLogitsProcessor


# Lazily built so importing this module doesn't require vLLM (e.g. a GPU-less
# host importing the registry).
try:  # pragma: no cover - depends on runtime env
    DRYLogitsProcessor = _make_dry_logits_processor_cls()
except Exception:  # vLLM not installed / different version
    DRYLogitsProcessor = None  # type: ignore[assignment]
