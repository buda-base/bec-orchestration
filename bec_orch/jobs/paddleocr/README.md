# `paddleocr` — Tibetan OCR with PaddleOCR-VL

Full-page Tibetan OCR using a fine-tuned **PaddleOCR-VL** checkpoint, served
with the in-process **vLLM** engine (native `PaddleOCRVLForConditionalGeneration`
support). The `grow26-ep2` checkpoint uses a warmed Tibetan **unicode-stack
tokenizer** and emits **Tibetan Unicode directly** — there is no Wylie/EWTS step;
post-processing is canonical Unicode normalization only.

Serving recipe (from `bec-ocr-training/docs/eval_in_production.md`):
**vLLM + sequential image-token M-RoPE + DRY-a12 loop guard + adaptive per-page
resolution**, greedy decoding, with a high-severity temperature retry.

> **Sequential regime (required).** `grow26-ep2` was trained with
> `mm_token_type_ids` zeroed (the `sequential` regime). Stock vLLM *always* builds
> 2D grid M-RoPE and never reads `mm_token_type_ids`, so serving it as-is is a
> train/serve mismatch (line skips/merges, large CER regression). The vendored
> **`vllm_paddleocr_seqpos`** vLLM general plugin fixes this; it must be
> pip-installed into the same venv as vLLM and `OCR_VLLM_IMAGE_TOKEN_POSITIONS=sequential`
> set (the worker sets the env and refuses to start `sequential` without the
> plugin). Set `image_token_positions: "grid"` to opt out.

This package is **version-agnostic**: everything model-specific lives in
`PaddleOCRConfig`. The registered job `paddleocr_v1` uses the defaults verbatim.
A future `paddleocr_v2` (same code, different checkpoint) is a one-line registry
entry — see the comment in `core/registry.py`.

## At a glance

| | |
|---|---|
| Model | PaddleOCR-VL-1.6 fine-tune, depth-upscaled (`grow26-ep2`), bf16 |
| Engine | **vLLM** offline `LLM.generate` (greedy + DRY loop guard) |
| Checkpoint (v1) | `s3://bec.bdrc.io/checkpoints/PaddleOCR/elie_v6_coarse_grow26_ep2/final/` |
| Output | Tibetan **Unicode** (normalized), one string per page |
| GPU | A10G / L40S / Blackwell, bf16 |
| Per-volume timeout | 1 h (`volume_timeout_s`, configurable) |

## Pipeline

1. **Sync + patch** the checkpoint from S3 to local disk (`model_sync.py`),
   skipping training-only files (`*.pt`, `trainer_state.json`). The sync then
   patches the checkpoint for the native vLLM path (adds the top-level
   `image_token` and pins the processor `size` to `{shortest_edge: 1024,
   longest_edge: 1280*28*28}`), so it loads with `trust_remote_code=False`.
2. **Preprocess + resolution-route** each page (`preprocess.py`): decode to RGB
   (longest side capped at `max_longest_side`, 3500 px), then downsize to a
   per-page pixel budget. In `adaptive` mode the router estimates the p75
   connected-component (glyph-body) height and picks the cheapest `res_scales`
   budget (of `{0.6, 0.75, 1.0}` × 1x) that keeps that height `>= res_tfloor`
   (24 px). The vision prefill dominates GPU time, so this is a near-free-lunch
   speedup at CER ≤ 1x. Falls back to the full budget when scale can't be
   estimated. The chosen fraction is recorded per page (`res_scale`).
3. **Prompt** (single user turn, image + text) via
   `processor.apply_chat_template(..., add_generation_prompt=True)`:
   `"Extract all Tibetan text. Preserve line breaks."`
4. **Generate** with vLLM (greedy: `temperature=0`, `max_tokens=2048`). The
   **DRY** repetition guard (`multiplier=0.8, base=1.75, allowed_length=12`, no
   sequence breakers) is registered on the engine and enabled per request — it
   is the only loop guard that doesn't wreck clean/repetitive Tibetan (do **not**
   use `repetition_penalty` / `no_repeat_ngram_size`). vLLM does continuous
   batching up to `max_num_seqs`. Per-page **DRY fire telemetry** (`dry_fires`,
   `dry_max_L`, …) is routed from the engine worker through a temp-dir
   side-channel and read back after the batch.
5. **Temperature retry (high-severity only).** DRY fires on ~57 % of pages but
   almost always as a mild single-token nip; only the ~2 % that fire *hard* are
   worth re-decoding. Pages with `dry_fires >= dry_retry_min_fires` (default
   **100**, the measured knee) — or a leftover hard loop (`rep_score >=
   dry_retry_min_rep`) — are re-decoded at `dry_retry_temp` (**0.4**, `n=3`, DRY
   still on) and the lowest-`rep_score` (then shortest) sample is kept
   (`retried=true`). Retrying *every* fire instead regresses CER, so the gate
   matters. Set `dry_retry_temp: 0` to disable.
6. **Post-process** (`postprocess.py`): canonical Unicode normalization
   (`normalize_unicode_text`: NFD reorder + graphical fold, matching the
   training/eval scorer exactly), hardened to never raise. Also compute
   `rep_score = 1 - unique(20-grams)/total` on the raw prediction (syllable
   tokens split on tsheg/shad/whitespace); `>= 0.5` sets `likely_loop=true` for
   review (belt-and-suspenders — DRY already removes hard loops).

> OCR should run only after upstream layout/language/orientation pre-filters
> (skip illustrations, tables, non-Tibetan, rotated pages).

## Pre-filter (`script_classification_v2`)

Before OCR, the worker loads the sibling `script_classification_v2` parquet for
the **same volume + version** (auto-located next to this job's output — only the
job-name path segment differs) and skips pages the classifier flagged. Skipped
pages are still written (`skipped=true`, `skip_reason=<label>`), never touch the
GPU, and don't count toward the failure rate.

| config key | default | meaning |
|---|---|---|
| `filter_enabled` | `true` | turn the pre-filter on/off |
| `filter_job_name` | `script_classification_v2` | sibling job whose output gates OCR |
| `filter_skip_labels` | `("blank","non_tibetan","nonplaintext")` | labels to skip |
| `filter_min_prob` | `0.30` | also skip pages with predicted `prob` below this |
| `filter_required` | `false` | if `true`, a missing classifier artifact fails the volume; else warn + OCR all |

`filter_min_prob=0.30` catches blank/pure-white scans the classifier mislabels:
they normalize to a near-uniform softmax (argmax `prob` ~0.13) so the `blank`
label doesn't reliably win, but they always sit below the wide empty gap
(~0.40–0.84) that separates them from real content (~0.3% of real pages fall
below 0.30). Set `filter_min_prob: 0.0` to disable and skip on labels only.

## Output

    s3://<dest-bucket>/paddleocr_v1/<W>/<I>/<version>/<W>-<I>-<version>.parquet
    s3://<dest-bucket>/paddleocr_v1/<W>/<I>/<version>/<W>-<I>-<version>-errors.jsonl.gz  (optional)

Parquet schema (one row per page):

| column | type | meaning |
|---|---|---|
| `img_file_name` | string | base name, matches manifest entry |
| `source_etag` | string | S3 ETag of the source image |
| `ok` | bool | OCR produced text (even if truncated) |
| `truncated` | bool | hit `max_new_tokens` — `page_text` is partial |
| `finish_reason` | string | `"stop"` or `"length"` |
| `page_text` | string | Unicode transcription (normalized, canonical form) |
| `raw_text` | string | raw model output (Unicode, pre-normalization) |
| `rep_score` | float64 | repeated-20-gram fraction on the raw prediction |
| `likely_loop` | bool | `rep_score >= rep_score_threshold` |
| `output_tokens` | int32 | generated tokens |
| `res_scale` | float64 | adaptive resolution budget fraction used (1.0 = 1x) |
| `dry_fires` | int32 | times the DRY guard fired while decoding this page |
| `dry_max_L` | int32 | longest repeated-suffix match DRY penalised (severity) |
| `retried` | bool | page re-decoded at temperature (`dry_fires` ≥ threshold); `page_text` is the retry pick |
| `skipped` | bool | page skipped by the pre-filter (no OCR run) |
| `skip_reason` | string | classifier label / rule that caused the skip |
| `error_stage` | string | `""` / `"fetch"` / `"decode"` / `"ocr"` / `"postprocess"` |
| `error_message` | string | short error (max 512 chars) |
| `model_id` | string | checkpoint identifier that produced the row |

## Failure semantics

- Per-volume failure rate above `max_page_failure_rate` (default 5 %) →
  task is **retryable** (likely transient S3 / GPU issue).
- At or below the threshold → success; bad pages recorded with `ok=false`.
- A page that hits `max_new_tokens` is `truncated=true` and **not** an error by
  default (toggle with `treat_truncation_as_failure: true`).

## Dependencies / venv

vLLM is pip-installed into the DLAMI venv (`/opt/pytorch`); it pins its own
torch, so let pip resolve torch. **The `vllm_paddleocr_seqpos` plugin must be
installed into that same venv** (`pip install ./vllm_paddleocr_seqpos`) so vLLM
discovers its `vllm.general_plugins` entry point — without it, `sequential`
serving refuses to start. `opencv-python-headless` + `numpy` power the
adaptive-resolution router. No `pyewts` / `botok` (Unicode output, vendored
normalization). On boxes without the CUDA toolkit,
`VLLM_USE_FLASHINFER_SAMPLER=0` is set (the worker also sets it defensively).

## Running locally

```bash
export PATH=/opt/pytorch/bin:$PATH
export BEC_DEST_S3_BUCKET=bdrc-artifacts
export BEC_PADDLEOCR_MODEL_CACHE=/var/cache/bec-paddleocr/models  # writable dir
export BEC_SQL_HOST=...   # + other BEC_* env vars from CLI_GUIDE.md

bec worker --job-name paddleocr_v1 --visibility-timeout 3600
```

Create the job (defaults match the serving recipe, so an empty config is valid):

```bash
bec jobs create --name paddleocr_v1 --config-text '{}'
```
