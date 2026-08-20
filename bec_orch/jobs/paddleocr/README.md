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
`paddleocr_v2` reuses the same checkpoint and turns on header/footer
background fill from `layout_detection_v1` (see below).

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

## Header/footer background fill (`paddleocr_v2`)

When `layout_detection_v1` output exists for the same volume + version, v2
paints detected **header** and **footer** boxes with an estimate of the page
background colour before OCR, so running titles and folio numbers don't leak
into the transcription. The `removed` column records what was blanked:
`none`, `h` (header only), `f` (footer only), or `hf` (both). Overlap with
`text-area` / `footnote` is subtracted from the H/F mask so an oversized
header box cannot wipe body text or notes.

**Footnotes** are cropped and OCR'd as separate requests
(`layout_isolate_footnotes`). A footnote fully inside another — or ≥95% of
its area covered by another (so a few-pixel overhang still counts) — is
dropped so the same note isn't OCR'd twice. Their text is **not** in `page_text`; it is merged (top-to-bottom,
`\n\n`) into `footnote_text`. Each crop is the detection box on synthetic
page-background padding. A footer (or header) that sits inside the note is
painted out on that crop.

**Text-areas** are always extracted the same way when column split is on:
one crop per kept box (two-column pair if the overlap heuristic matches,
otherwise **every** non-nested text-area), synthetic margin, and anything
outside the box is background. Overlapping footnotes are blanked on the
body crop so they only appear in `footnote_text`.

When a page has three or more text-areas, the crops are OCR'd in **reading
order** computed by a recursive **XY-cut** (`reading_order_xycut`): the page
is split at the widest whitespace valley (horizontal or vertical), recursing
until each region is a single box. This reads a spanning title band first,
then the columns beneath it top-to-bottom (column-major), which handles
modern book pages such as a top two-column band, a centred title, then a
second two-column band. The transcriptions are joined with `\n\n` in that
order.

If the layout artifact is missing, the worker logs a warning and OCRs the
unmodified pages (`layout_mask_required=false`). v1 leaves this path off.

| config key | v1 | v2 | meaning |
|---|---|---|---|
| `layout_mask_enabled` | `false` | `true` | load sibling layout parquet and paint |
| `layout_mask_job_name` | `layout_detection_v1` | same | sibling job |
| `layout_mask_labels` | `("header","footer")` | same | classes to blank |
| `layout_mask_protect_labels` | `("text-area","footnote")` | same | never blank these |
| `layout_mask_pad_px` | `2` | same | extra pixels around each painted box |
| `layout_mask_required` | `false` | same | missing layout fails the volume if true |

## Two-column split (`paddleocr_v2`)

When two `text-area` boxes overlap **≥ 60% vertically** and **< 5%
horizontally**, v2 crops each column. Otherwise each remaining text-area
(after dropping boxes fully inside another) is cropped the same way. Crops
sit on synthetic background margin, including past the page edge, and are
queued as **separate** OCR requests. Column transcriptions are concatenated
**left-to-right** with two line breaks (`\n\n`). A full-width text-area
sitting on top of two columns will not pair with either (horizontal overlap
is high); the pair is kept and the envelope is ignored.

| config key | v1 | v2 | meaning |
|---|---|---|---|
| `layout_split_columns` | `false` | `true` | crop + OCR columns separately |
| `layout_column_label` | `text-area` | same | class used as a column candidate |
| `layout_column_min_vert_overlap` | `0.60` | same | shorter-box vertical overlap |
| `layout_column_max_horiz_overlap` | `0.05` | same | narrower-box horizontal overlap |
| `layout_column_margin_frac` | `0.02` | same | margin as a fraction of min(w,h) |
| `layout_column_join` | `"\\n\\n"` | same | separator when reassembling |
| `layout_isolate_footnotes` | `false` | `true` | OCR footnotes into `footnote_text` |
| `layout_footnote_label` | `footnote` | same | class cropped as a footnote |
| `layout_footnote_margin_frac` | `0.05` | same | footnote crop margin vs min(page_w, page_h) |
| `layout_footnote_margin_min_px` | `32` | same | floor on footnote crop margin |
| `layout_footnote_box_margin_frac` | `0.5` | same | also at least this fraction of the box height |

Create the job (empty config is valid; masking, column split, and footnote isolation are on via the registry default):

```bash
bec jobs create --name paddleocr_v2 --config-text '{}'
```

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
| `layout_masked` | bool | header/footer boxes were painted with page background |
| `n_masked_boxes` | int32 | number of header+footer boxes painted (0 if none) |
| `n_columns` | int32 | number of text-area crops OCR'd (`1` if none / full page) |
| `footnote_text` | string | isolated footnote transcription (merged with `\n\n`, or empty) |
| `n_footnotes` | int32 | number of footnote boxes OCR'd |
| `removed` | string | header/footer left out of `page_text`: `none` / `h` / `f` / `hf` |
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
