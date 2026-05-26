"""Blind (no-reference) quality evaluation for ``ocr_qwen_v1``.

Walks a folder of images (non-recursively), runs the OCR Qwen model on each
image under several vLLM measurement configurations, and emits:

  <out_dir>/
    by_image/<sweep_id>/<image>.json   raw per-image metrics (resumable)
    <sweep_id>.jsonl                   flat per-sweep rows (one line per image)
    results.csv                        final CSV (one row per image, from the
                                       ``full`` sweep)
    report.md                          wall-time comparison of sweeps showing
                                       how each vLLM measurement (logprobs,
                                       n=2, second temperature) impacts cost

CSV columns (per user spec):
    image_file_name        exact basename of the image
    itl                    inter-token latency in ms/token  (mean for t=0 run)
    generation_throughput  tokens/second                    (mean for t=0 run)
    num_generated_tokens   number of decoded tokens         (t=0 run)
    mean_sequence_entropy  mean per-step Shannon entropy from top-5 logprobs
                           (t=0 run; nats)
    edit_dist              Levenshtein character edit distance between the
                           t=0 transcription and the t=1 transcription
                           (raw, not normalised)

The script can be re-run safely; sweeps that already wrote a per-image json
are skipped. ``--sweeps`` lets you select a subset of measurement configs.

Usage:
    python -m bec_orch.jobs.ocr_qwen_v1.blind_eval \
        --images /path/to/images \
        --out    /path/to/eval_out \
        --sweeps minimal,entropy,consistency,full

Or with the venv's python directly:
    /opt/pytorch/bin/python -m bec_orch.jobs.ocr_qwen_v1.blind_eval ...
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import math
import os
import statistics
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .config import OCRQwenV1Config
from .preprocess import bytes_to_rgb

logger = logging.getLogger("ocr_qwen_v1.blind_eval")

# Image extensions we accept in the input folder. Everything else is ignored.
_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp", ".bmp"}


# ---------------------------------------------------------------------------
# Sweep configurations
# ---------------------------------------------------------------------------
#
# Each sweep is a (name, description, builder) triplet. The builder takes the
# job config and returns a list of ``SamplingParams`` to call ``llm.chat``
# with — one SamplingParams per OCR run we want for the image. We pass them
# all together in a single ``llm.chat([msg]*N, sampling_params=[sp,...])``
# call so vLLM batches the runs concurrently.
#
# The runs returned in each sweep are interpreted positionally:
#   index 0 → "primary" run (t=0). Drives the timing / token / entropy
#             columns of the final CSV.
#   index 1 (if present) → "secondary" run. Used for edit-distance against
#             the primary run.


@dataclass
class SweepDef:
    name: str
    description: str
    needs_logprobs: bool
    n_runs: int  # number of LLM.chat outputs per image (>= 1)

    def build_sampling_params(self, cfg: OCRQwenV1Config) -> list[Any]:
        # Imported lazily so this module is importable without vLLM.
        from vllm import SamplingParams

        common = {
            "max_tokens": cfg.max_tokens,
            "repetition_penalty": cfg.repetition_penalty,
            "frequency_penalty": cfg.frequency_penalty,
            "presence_penalty": cfg.presence_penalty,
        }
        logprobs_kw = {"logprobs": 5} if self.needs_logprobs else {}

        if self.name == "minimal":
            # Greedy, no measurement overhead beyond what vLLM always tracks.
            return [SamplingParams(temperature=0.0, **common)]
        if self.name == "entropy":
            # Greedy + top-5 logprobs so we can compute per-step entropy.
            return [SamplingParams(temperature=0.0, **logprobs_kw, **common)]
        if self.name == "consistency":
            # Two greedy samples at moderate temperature — the Gemini doc's
            # recommended ``n=2`` consistency trick. Cheaper than two separate
            # SamplingParams because vLLM packs them under one request.
            return [
                SamplingParams(
                    n=2, best_of=2, temperature=0.5, **logprobs_kw, **common
                )
            ]
        if self.name == "full":
            # The user's primary CSV target: t=0 with logprobs (drives ITL /
            # throughput / entropy columns) AND t=1 without logprobs (drives
            # the edit-distance column).
            return [
                SamplingParams(temperature=0.0, **logprobs_kw, **common),
                SamplingParams(temperature=1.0, **common),
            ]
        raise ValueError(f"unknown sweep: {self.name}")


SWEEPS: dict[str, SweepDef] = {
    "minimal": SweepDef(
        name="minimal",
        description="greedy t=0, no logprobs, 1 run/image — baseline cost",
        needs_logprobs=False,
        n_runs=1,
    ),
    "entropy": SweepDef(
        name="entropy",
        description="greedy t=0 with logprobs=5 — adds Shannon entropy",
        needs_logprobs=True,
        n_runs=1,
    ),
    "consistency": SweepDef(
        name="consistency",
        description="single request with n=2 at t=0.5 — Gemini consistency trick",
        needs_logprobs=False,
        n_runs=1,  # one chat call returns 2 completions internally
    ),
    "full": SweepDef(
        name="full",
        description="t=0 (with logprobs) + t=1 — fills every CSV column",
        needs_logprobs=True,
        n_runs=2,
    ),
}


# ---------------------------------------------------------------------------
# Per-image / per-run metric extraction
# ---------------------------------------------------------------------------


@dataclass
class RunMetrics:
    """Metrics extracted from one ``CompletionOutput`` of one chat call."""

    text: str
    num_generated_tokens: int
    finish_reason: str
    itl_ms: float | None  # mean inter-token latency in ms (None if unavailable)
    generation_throughput: float | None  # tokens / s (None if unavailable)
    e2e_latency_s: float | None
    prefill_s: float | None
    decode_s: float | None
    mean_sequence_entropy: float | None  # nats (None if logprobs absent)


@dataclass
class ImageResult:
    image_file_name: str
    sweep: str
    elapsed_s: float  # wall-time of the chat call for this image (excl. decode)
    decode_s: float  # CPU time spent decoding the image to RGB
    runs: list[RunMetrics] = field(default_factory=list)
    edit_dist: int | None = None  # filled when n_runs >= 2 or sweep yields 2 outputs


def _step_entropy_nats(top_logprobs: dict[int, Any]) -> float:
    """Shannon entropy over a normalised top-K logprob distribution (nats).

    Renormalises the K candidates to sum to 1 before computing entropy, so
    the result is the local entropy among those K candidates rather than the
    full-vocabulary entropy. Matches the Gemini-share formula.
    """
    if not top_logprobs:
        return 0.0
    raw_lps = [item.logprob for item in top_logprobs.values()]
    max_lp = max(raw_lps)
    exps = [math.exp(lp - max_lp) for lp in raw_lps]
    z = sum(exps)
    if z <= 0:
        return 0.0
    probs = [e / z for e in exps]
    return -sum(p * math.log(p) for p in probs if p > 0)


def _extract_run_metrics(completion: Any, request_output: Any) -> RunMetrics:
    """Turn one ``CompletionOutput`` (+ its parent ``RequestOutput``) into a
    ``RunMetrics``. Falls back to ``None`` for any field whose underlying
    vLLM stat is unavailable in the running engine version.
    """
    n_tokens = len(getattr(completion, "token_ids", []) or [])
    text = getattr(completion, "text", "") or ""
    finish_reason = getattr(completion, "finish_reason", "") or ""

    # Mean per-step entropy from top-5 logprobs (if collected).
    mean_entropy: float | None = None
    step_logprobs = getattr(completion, "logprobs", None)
    if step_logprobs:
        entropies = [
            _step_entropy_nats(d) for d in step_logprobs if d  # skip None entries
        ]
        if entropies:
            mean_entropy = statistics.fmean(entropies)

    # Lifecycle timestamps (vLLM V1 ``RequestStateStats``).
    metrics = getattr(request_output, "metrics", None)
    itl_ms: float | None = None
    gen_tps: float | None = None
    e2e_s: float | None = None
    prefill_s: float | None = None
    decode_s: float | None = None
    if metrics is not None:
        arrival = getattr(metrics, "arrival_time", None)
        scheduled = getattr(metrics, "scheduled_ts", None)
        first_tok = getattr(metrics, "first_token_ts", None)
        last_tok = getattr(metrics, "last_token_ts", None)
        if scheduled is not None and first_tok is not None:
            prefill_s = max(0.0, first_tok - scheduled)
        if first_tok is not None and last_tok is not None:
            decode_s = max(0.0, last_tok - first_tok)
            if n_tokens > 0 and decode_s > 0:
                itl_ms = (decode_s / n_tokens) * 1000.0
                gen_tps = n_tokens / decode_s
        if arrival is not None and last_tok is not None:
            e2e_s = max(0.0, last_tok - arrival)

    return RunMetrics(
        text=text,
        num_generated_tokens=n_tokens,
        finish_reason=finish_reason,
        itl_ms=itl_ms,
        generation_throughput=gen_tps,
        e2e_latency_s=e2e_s,
        prefill_s=prefill_s,
        decode_s=decode_s,
        mean_sequence_entropy=mean_entropy,
    )


# ---------------------------------------------------------------------------
# Image discovery / loading
# ---------------------------------------------------------------------------


def list_images(image_dir: Path) -> list[Path]:
    if not image_dir.is_dir():
        raise SystemExit(f"--images must be an existing directory: {image_dir}")
    return sorted(
        p
        for p in image_dir.iterdir()
        if p.is_file() and p.suffix.lower() in _IMAGE_EXTS
    )


def decode_image(path: Path, cfg: OCRQwenV1Config):
    data = path.read_bytes()
    return bytes_to_rgb(data, cfg)


# ---------------------------------------------------------------------------
# Sweep runner
# ---------------------------------------------------------------------------


def _build_messages(image, prompt: str) -> list[dict]:
    """One chat message in the format vLLM ``LLM.chat`` expects."""
    content: list[dict] = [{"type": "image_pil", "image_pil": image}]
    if prompt:
        content.append({"type": "text", "text": prompt})
    return [{"role": "user", "content": content}]


def run_sweep(
    llm: Any,
    sweep: SweepDef,
    cfg: OCRQwenV1Config,
    images: list[Path],
    out_dir: Path,
    overwrite: bool,
) -> tuple[float, int]:
    """Run one sweep over all images. Returns ``(wall_seconds, n_processed)``.

    Per-image state is written to ``out_dir/by_image/<sweep>/<image>.json`` so
    re-runs skip already-completed images unless ``overwrite=True``.
    """
    sampling = sweep.build_sampling_params(cfg)

    per_image_dir = out_dir / "by_image" / sweep.name
    per_image_dir.mkdir(parents=True, exist_ok=True)
    sweep_jsonl = out_dir / f"{sweep.name}.jsonl"

    already_done: set[str] = set()
    if not overwrite:
        already_done = {p.stem for p in per_image_dir.glob("*.json")}
        if already_done:
            logger.info(
                "[%s] skipping %d image(s) already done in %s",
                sweep.name,
                len(already_done),
                per_image_dir,
            )

    todo = [p for p in images if p.name not in already_done and p.stem not in already_done]
    logger.info(
        "[%s] %s — processing %d/%d image(s)",
        sweep.name,
        sweep.description,
        len(todo),
        len(images),
    )

    t_sweep_start = time.perf_counter()
    n_processed = 0

    # Open the JSONL in append mode so the report can be rebuilt without
    # re-running everything.
    with sweep_jsonl.open("a", encoding="utf-8") as jsonl_f:
        for img_path in todo:
            t_dec = time.perf_counter()
            try:
                img = decode_image(img_path, cfg)
            except Exception as e:  # noqa: BLE001
                logger.warning("[%s] decode failed for %s: %s", sweep.name, img_path.name, e)
                _write_per_image(
                    per_image_dir,
                    img_path,
                    {"error": f"decode failed: {e}", "sweep": sweep.name},
                    jsonl_f,
                )
                continue
            decode_s = time.perf_counter() - t_dec

            # Build one chat message per sampling-params variant. With
            # ``sampling=[sp_t0, sp_t1]`` and ``messages=[m, m]``, vLLM
            # treats them as two independent requests batched concurrently.
            messages = [_build_messages(img, cfg.prompt) for _ in sampling]
            t_call = time.perf_counter()
            try:
                outputs = llm.chat(messages, sampling_params=sampling, use_tqdm=False)
            except Exception as e:  # noqa: BLE001
                logger.exception("[%s] llm.chat failed for %s", sweep.name, img_path.name)
                _write_per_image(
                    per_image_dir,
                    img_path,
                    {"error": f"llm.chat failed: {e}", "sweep": sweep.name},
                    jsonl_f,
                )
                continue
            elapsed_s = time.perf_counter() - t_call

            # Each ``RequestOutput`` may have one OR multiple
            # ``CompletionOutput``s (n>1 via the ``consistency`` sweep).
            # Flatten everything in declaration order.
            runs: list[RunMetrics] = []
            for ro in outputs:
                comps = list(getattr(ro, "outputs", []) or [])
                if not comps:
                    runs.append(
                        RunMetrics(
                            text="",
                            num_generated_tokens=0,
                            finish_reason="",
                            itl_ms=None,
                            generation_throughput=None,
                            e2e_latency_s=None,
                            prefill_s=None,
                            decode_s=None,
                            mean_sequence_entropy=None,
                        )
                    )
                    continue
                for comp in comps:
                    runs.append(_extract_run_metrics(comp, ro))

            edit_dist = _edit_distance(runs[0].text, runs[1].text) if len(runs) >= 2 else None

            result = ImageResult(
                image_file_name=img_path.name,
                sweep=sweep.name,
                elapsed_s=elapsed_s,
                decode_s=decode_s,
                runs=runs,
                edit_dist=edit_dist,
            )
            _write_per_image(per_image_dir, img_path, asdict(result), jsonl_f)
            n_processed += 1

            # Free image buffer asap.
            try:
                img.close()
            except Exception:  # noqa: BLE001
                pass
            del img, outputs
            if n_processed % 16 == 0:
                gc.collect()

    sweep_wall = time.perf_counter() - t_sweep_start
    logger.info(
        "[%s] done in %.1fs (%d processed this run)",
        sweep.name,
        sweep_wall,
        n_processed,
    )
    return sweep_wall, n_processed


def _edit_distance(a: str, b: str) -> int:
    # rapidfuzz is faster than python-Levenshtein and is already a transitive
    # dep of the project venv. Falls back to a naïve O(n*m) if unavailable.
    try:
        from rapidfuzz.distance import Levenshtein

        return int(Levenshtein.distance(a, b))
    except ImportError:
        return _naive_edit_distance(a, b)


def _naive_edit_distance(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    # Two-row DP — keep it simple, we only hit this if rapidfuzz is missing.
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        curr = [i] + [0] * len(b)
        for j, cb in enumerate(b, 1):
            curr[j] = min(
                curr[j - 1] + 1,
                prev[j] + 1,
                prev[j - 1] + (0 if ca == cb else 1),
            )
        prev = curr
    return prev[-1]


def _write_per_image(per_image_dir: Path, img_path: Path, payload: dict, jsonl_f) -> None:
    """Atomic per-image write + append to the sweep JSONL."""
    per_image_dir.mkdir(parents=True, exist_ok=True)
    out = per_image_dir / f"{img_path.stem}.json"
    tmp = out.with_suffix(out.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, out)
    jsonl_f.write(json.dumps(payload, ensure_ascii=False))
    jsonl_f.write("\n")
    jsonl_f.flush()


# ---------------------------------------------------------------------------
# Final aggregation: CSV + report
# ---------------------------------------------------------------------------


def write_final_csv(out_dir: Path, images: list[Path]) -> Path:
    """Build the user-requested CSV from the ``full`` sweep's per-image jsons.

    Falls back row-by-row to ``entropy`` / ``minimal`` sweeps for any column
    that wasn't collected by ``full`` (shouldn't happen in normal use — but
    keeps the CSV well-formed if some sweeps were skipped).
    """
    csv_path = out_dir / "results.csv"
    cols = [
        "image_file_name",
        "itl",
        "generation_throughput",
        "num_generated_tokens",
        "mean_sequence_entropy",
        "edit_dist",
    ]

    def _load(sweep: str, stem: str) -> dict | None:
        p = out_dir / "by_image" / sweep / f"{stem}.json"
        if not p.exists():
            return None
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            return None

    import csv as _csv

    with csv_path.open("w", encoding="utf-8", newline="") as f:
        w = _csv.writer(f)
        w.writerow(cols)
        for img_path in images:
            stem = img_path.stem
            data = _load("full", stem) or _load("entropy", stem) or _load("minimal", stem)
            if data is None or "runs" not in data or not data["runs"]:
                w.writerow([img_path.name, "", "", "", "", ""])
                continue
            primary = data["runs"][0]
            edit_dist = data.get("edit_dist")
            w.writerow(
                [
                    img_path.name,
                    _fmt(primary.get("itl_ms")),
                    _fmt(primary.get("generation_throughput")),
                    primary.get("num_generated_tokens") or "",
                    _fmt(primary.get("mean_sequence_entropy")),
                    "" if edit_dist is None else int(edit_dist),
                ]
            )
    return csv_path


def _fmt(x: Any) -> str:
    if x is None:
        return ""
    if isinstance(x, float):
        return f"{x:.4f}"
    return str(x)


def write_report(
    out_dir: Path,
    sweep_walls: dict[str, float],
    sweep_counts: dict[str, int],
    n_images_total: int,
) -> Path:
    """Markdown report comparing the wall-time cost of each sweep."""
    report_path = out_dir / "report.md"

    # Recompute per-sweep aggregates from the on-disk jsonl so the report
    # picks up earlier (resumed) runs too.
    def _agg(sweep: str) -> dict[str, float]:
        rows = _load_jsonl(out_dir / f"{sweep}.jsonl")
        if not rows:
            return {}
        elapsed = [r.get("elapsed_s", 0.0) for r in rows if r.get("elapsed_s")]
        decode = [r.get("decode_s", 0.0) for r in rows if r.get("decode_s")]
        n_tokens = [
            r["runs"][0].get("num_generated_tokens", 0)
            for r in rows
            if r.get("runs")
        ]
        itl = [
            r["runs"][0].get("itl_ms")
            for r in rows
            if r.get("runs") and r["runs"][0].get("itl_ms") is not None
        ]
        ent = [
            r["runs"][0].get("mean_sequence_entropy")
            for r in rows
            if r.get("runs") and r["runs"][0].get("mean_sequence_entropy") is not None
        ]
        return {
            "n": len(rows),
            "wall_s_sum": float(sum(elapsed)),
            "decode_s_sum": float(sum(decode)),
            "tokens_sum": int(sum(n_tokens)) if n_tokens else 0,
            "itl_mean_ms": statistics.fmean(itl) if itl else float("nan"),
            "entropy_mean_nats": statistics.fmean(ent) if ent else float("nan"),
        }

    aggs = {name: _agg(name) for name in SWEEPS}
    base = aggs.get("minimal", {})
    base_wall = base.get("wall_s_sum") or 0.0

    lines: list[str] = []
    lines.append("# ocr_qwen_v1 — blind eval report\n")
    lines.append(
        f"Input folder had **{n_images_total}** image(s). Sweeps executed in this "
        f"invocation: {', '.join(f'`{k}`' for k, v in sweep_counts.items() if v > 0) or '(none — all sweeps were already complete)'}.\n"
    )
    lines.append("## Wall-time breakdown\n")
    lines.append(
        "Wall-time below excludes image decode (which is sweep-independent — it's a "
        "constant CPU cost paid once per image regardless of vLLM settings).\n"
    )
    lines.append(
        "| sweep | n | description | wall (s) | per page (s) | vs minimal | mean ITL (ms/tok) | tokens/sweep | mean entropy (nats) |\n"
        "|---|---:|---|---:|---:|---:|---:|---:|---:|\n"
    )
    for name, sw in SWEEPS.items():
        a = aggs[name]
        if not a or a["n"] == 0:
            lines.append(f"| `{name}` | 0 | {sw.description} | – | – | – | – | – | – |\n")
            continue
        rel = ""
        if base_wall > 0 and name != "minimal":
            rel = f"+{(a['wall_s_sum'] - base_wall) / base_wall * 100:.0f} %"
        elif name == "minimal":
            rel = "baseline"
        lines.append(
            f"| `{name}` | {a['n']} | {sw.description} | "
            f"{a['wall_s_sum']:.1f} | {a['wall_s_sum'] / a['n']:.2f} | {rel} | "
            f"{_fmt(a['itl_mean_ms'])} | {a['tokens_sum']} | "
            f"{_fmt(a['entropy_mean_nats']) if not math.isnan(a['entropy_mean_nats']) else '–'} |\n"
        )

    lines.append("\n## Interpretation\n")
    lines.append(
        "- `minimal` is the baseline. Any extra cost in the other rows is purely "
        "the overhead of the additional vLLM measurement (logprobs collection, "
        "extra completion via `n=2`, or a second-temperature run).\n"
        "- `entropy` − `minimal` ≈ cost of capturing top-5 logprobs at every "
        "decode step. Usually a few %.\n"
        "- `consistency` − `minimal` ≈ cost of one extra completion per request "
        "(vLLM continues-batches the two, so it's well below 2x).\n"
        "- `full` − `entropy` ≈ cost of running the t=1 second pass for "
        "edit-distance computation. This is roughly a second pass at full price, "
        "but reuses prefix cache + vision tower outputs, so usually ~1.4× of the "
        "`entropy` row.\n"
    )
    if sweep_walls:
        lines.append("\n## This invocation\n")
        for name, w in sweep_walls.items():
            n = sweep_counts.get(name, 0)
            lines.append(f"- `{name}`: processed {n} image(s) in {w:.1f}s\n")

    report_path.write_text("".join(lines), encoding="utf-8")
    return report_path


def _load_jsonl(p: Path) -> list[dict]:
    if not p.exists():
        return []
    rows: list[dict] = []
    with p.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _build_llm(cfg: OCRQwenV1Config) -> Any:
    """Construct the vLLM engine with the same defaults as production."""
    from vllm import LLM

    logger.info(
        "loading vLLM: model=%s dtype=%s gpu_mem_util=%.2f max_model_len=%d",
        cfg.model_id,
        cfg.dtype,
        cfg.gpu_memory_utilization,
        cfg.max_model_len,
    )
    t0 = time.perf_counter()
    base_kwargs = dict(
        model=cfg.model_id,
        dtype=cfg.dtype,
        tensor_parallel_size=cfg.tensor_parallel_size,
        gpu_memory_utilization=cfg.gpu_memory_utilization,
        max_model_len=cfg.max_model_len,
        trust_remote_code=cfg.trust_remote_code,
        enforce_eager=cfg.enforce_eager,
    )
    # Try to keep internal stats so RequestOutput.metrics is populated (the
    # Gemini-share writeup recommends this). The kwarg name has wobbled
    # across vLLM versions, so fall back silently if rejected.
    try:
        llm = LLM(**base_kwargs, disable_log_stats=False)
    except TypeError:
        logger.info("LLM(disable_log_stats=False) not accepted by this vLLM; falling back")
        llm = LLM(**base_kwargs)
    logger.info("vLLM ready in %.1fs", time.perf_counter() - t0)
    return llm


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="Blind quality evaluation for ocr_qwen_v1.",
    )
    p.add_argument("--images", required=True, type=Path, help="folder of images (non-recursive)")
    p.add_argument("--out", required=True, type=Path, help="output directory")
    p.add_argument(
        "--sweeps",
        default="full",
        help=(
            "comma-separated list of sweeps to run, in order; choose from: "
            + ", ".join(SWEEPS) + ". 'all' is a shortcut for everything."
        ),
    )
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="re-run images that already have a per-image json (default: skip them)",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=None,
        help="process at most N images (after sorting). Useful for quick test runs.",
    )
    p.add_argument(
        "--gpu-mem-util",
        type=float,
        default=None,
        help="override OCRQwenV1Config.gpu_memory_utilization (e.g. 0.55 on a shared GPU)",
    )
    p.add_argument(
        "--no-vllm",
        action="store_true",
        help="skip vLLM execution and only rebuild CSV+report from on-disk per-image jsons",
    )
    p.add_argument("--log-level", default="INFO")
    args = p.parse_args(argv)

    logging.basicConfig(
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        level=args.log_level.upper(),
    )

    # Resolve sweep list.
    if args.sweeps.strip().lower() == "all":
        sweep_names = list(SWEEPS.keys())
    else:
        sweep_names = [s.strip() for s in args.sweeps.split(",") if s.strip()]
        bad = [s for s in sweep_names if s not in SWEEPS]
        if bad:
            raise SystemExit(f"unknown sweep(s): {bad}; choose from {list(SWEEPS)}")

    cfg = OCRQwenV1Config()
    if args.gpu_mem_util is not None:
        cfg = OCRQwenV1Config(gpu_memory_utilization=args.gpu_mem_util)

    args.out.mkdir(parents=True, exist_ok=True)

    images = list_images(args.images)
    if args.limit:
        images = images[: args.limit]
    logger.info("found %d image(s) in %s", len(images), args.images)
    if not images:
        raise SystemExit("no images to process")

    sweep_walls: dict[str, float] = {}
    sweep_counts: dict[str, int] = {}

    if not args.no_vllm:
        llm = _build_llm(cfg)
        try:
            for name in sweep_names:
                sweep = SWEEPS[name]
                wall, n = run_sweep(
                    llm,
                    sweep,
                    cfg,
                    images,
                    args.out,
                    overwrite=args.overwrite,
                )
                sweep_walls[name] = wall
                sweep_counts[name] = n
        finally:
            # vLLM holds the GPU; release it explicitly before exit so the
            # process doesn't block subsequent runs / next user.
            try:
                del llm
            except Exception:  # noqa: BLE001
                pass
            gc.collect()

    csv_path = write_final_csv(args.out, images)
    report_path = write_report(args.out, sweep_walls, sweep_counts, len(images))

    logger.info("wrote CSV  → %s", csv_path)
    logger.info("wrote report → %s", report_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
