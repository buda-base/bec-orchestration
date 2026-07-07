# `script_classification` — BDRC Tibetan page classifier

Per-page classification of Tibetan manuscript scans using a vendored port
of [`tibetan-manuscript-classifier`](https://github.com/OpenPecha/tibetan-manuscript-classifier):
orientation detection (upright / flipped, with lossless 180° correction)
followed by 6-class script classification. Both models are DINOv3 ViT-S/16
fine-tunes (`facebook/dinov3-vits16-pretrain-lvd1689m` backbone) published
by BDRC on the HuggingFace Hub.

## At a glance

| | |
|---|---|
| Models | `BDRC/tibetan-page-orientation-classifier`, `BDRC/6-class-tibetan-script-classifier` |
| Engine | vendored PyTorch pipeline (GPU by default with CPU fallback, batched — one model call per S3-fetch batch) |
| Integration | code vendored into `vendor/` — see "Known limitations" below |

## Output

For each volume the worker writes one parquet file:

    s3://<dest-bucket>/script_classification/<W>/<I>/<version>/<W>-<I>-<version>.parquet

…plus an optional gzipped jsonl with per-page failure details:

    s3://<dest-bucket>/script_classification/<W>/<I>/<version>/<W>-<I>-<version>-errors.jsonl.gz

Parquet schema:

| column | type | meaning |
|---|---|---|
| `img_file_name` | string | base name, matches manifest entry |
| `source_etag` | string | S3 ETag of the source image |
| `status` | string | `"ok"` \| `"error"` |
| `error` | string | message when `status="error"` (else `""`) |
| `exif_orientation_tag` | int32 (nullable) | raw EXIF tag, read but never applied to pixels |
| `orientation_pred` | string | `"non_flipped"` \| `"flipped"` |
| `orientation_prob` | float32 | probability of the predicted orientation class |
| `orientation_probs` | list\<float32\> | full orientation softmax vector, model label order |
| `rotation_applied` | int32 | degrees actually applied before 6-class scoring: `0` or `180` |
| `sixclass_label` | string | argmax script class |
| `sixclass_probs` | list\<float32\> | full probability vector, ordered by the checkpoint's label mapping |
| `final_label` | string | equal to `sixclass_label` |
| `model_version` | string | short checkpoint hashes for both models |
| `error_stage` | string | `""` / `"fetch"` / `"classify"` |
| `error_message` | string | short error (max 512 chars) |

Both probability vectors (`orientation_probs`, `sixclass_probs`) are the
models' complete softmax output. Their per-class label ordering (`probs[i]`
⇄ `labels[i]`) is stored **once** in the parquet file's schema metadata
under `orientation_labels` / `sixclass_labels` (JSON arrays), plus
`model_version` — read e.g. via
`pyarrow.parquet.read_schema(path).metadata`.

## Known limitations / deviations from upstream

- **Blank-page pre-filter is disabled.** Upstream's `is_blank()` check was
  found untested and showed no promising results, so its call site in
  `vendor/pipeline.py` is commented out (not deleted — re-enabling is a
  one-line change). Every image is scored by both models unconditionally.
  Consequently there is **no `blank` column** in the output — upstream's
  `blank` field isn't kept as a permanently-`False` column, since that
  would misleadingly look like a real filter result to downstream
  consumers.
- **No `flip_applied` column.** Upstream returns both `flip_applied` (bool)
  and `rotation_applied` (int, 0/180) — these are a 1:1 encoding of the
  same fact. Only `rotation_applied` is kept.
- **GPU by default, CPU fallback.** Deviation from upstream (which is
  CPU-only, no `.to(device)` calls anywhere). `ScriptClassificationConfig.use_gpu`
  defaults to `True`: both classifiers run on CUDA when available. If CUDA
  isn't available, `vendor/pipeline.py::Pipeline.__init__` logs a warning
  and falls back to CPU — never a hard failure, so this also works
  unchanged in a CPU-only dev/smoke-test environment. Set `use_gpu = False`
  to force CPU even when a GPU is present.
- **Batched inference.** Deviation from upstream (which contracts one image
  per `pipe.run()` call). `vendor/pipeline.py::Pipeline.run_batch(list[bytes])`
  is the worker's actual entry point: it decode+resize+crops every image in
  the batch in parallel (`ScriptClassificationConfig.decode_workers`
  thread pool), normalizes the whole batch in one HF-processor call, moves
  it to the model device once, runs the orientation model **once** for the
  whole batch, corrects the
  six-class model's input in place — `torch.flip`ping just the rows the
  orientation model called "flipped" (mathematically identical to the old
  per-image rotate-then-renormalize for the common case, since
  normalization commutes with a spatial flip) — then runs the six-class
  model **once** more. A per-image decode failure only marks that one image
  `status="error"`; a failure in a shared batch step (normalize, or either
  model's forward pass) marks every surviving image in that batch as
  `status="error"` rather than crashing the worker. The single-image
  `pipe.run()` still exists unchanged (used by the smoke-test snippet
  below) but is no longer what the worker calls in production.
  **Known narrow exception**: `_center_crop`'s rare white-padding fallback
  (only hit when resize rounding leaves a crop 1px short) does *not*
  commute exactly under this restructure — confirmed via synthetic
  1px-short test images (see `vendor/pipeline.py::Pipeline.run_batch`'s
  docstring), though the predicted label never changed in that check. This
  branch is rare enough (a rounding artifact) that it's shipped as-is with
  this caveat rather than adding a slower per-image fallback path for it.
- **S3 fetch overlaps compute.** A background prefetch thread fetches the
  next batch(es) of raw bytes from S3 (`ScriptClassificationConfig.prefetch_batches`
  deep, bounded queue) while the main thread runs the current batch's CPU
  decode + GPU forwards, so neither the network nor the GPU idles waiting for
  the other. The worker logs `cum_fetch_wait` — near-zero means fetch is
  fully hidden; a large value means S3 is the bottleneck (raise
  `s3_fetch_concurrency` and/or `prefetch_batches`). Costs ~`prefetch_batches`
  extra batches of resident raw bytes.
- **Decode/resize restructured (behavior-preserving).** Upstream decodes
  with PIL and redundantly re-resizes from full resolution for both the
  upright and 180°-rotated passes. This vendor decodes via `libvips` when
  available (fused decode+resize in one pass, falling back to PIL
  otherwise — see `vendor/transforms.py::decode_and_resize`) and resizes
  short-edge exactly once per page; the rotated pass rotates that
  already-small result instead of re-resizing from full resolution. Output
  is mathematically identical (a uniform-scale resize commutes with 180°
  rotation) at lower cost, and was validated to produce 0 label
  disagreements against the old PIL-only path across 180 real sample pages
  spanning all 6 script classes.

## Vendored code

`vendor/` is a 1:1 port of upstream's `tibetan_manuscript_classifier/`
package, kept diffable against
https://github.com/OpenPecha/tibetan-manuscript-classifier for future
syncs. It's vendored (not an external pip/git dependency) because:
- upstream pins `requires-python>=3.14`, but the fleet's DLAMI base is
  Python 3.12.12 — pip cannot install it as-is today, even though nothing
  in the code is actually 3.14-specific;
- disabling the blank filter (above) requires editing the package's own
  `pipeline.py`.

## Failure semantics

- A per-volume failure rate **above `max_page_failure_rate` (default 5%)**
  marks the task **retryable** (likely a transient S3 issue, since neither
  `pipe.run()` nor `pipe.run_batch()` ever raises — classification errors
  are also captured as `status="error"` rows, not exceptions).
- A failure rate **at or below the threshold** is considered a success —
  bad pages are still recorded in the parquet with `status="error"`.

## HF authentication

All three HF repos (both classifiers plus the shared DINOv3 backbone) are
**gated**. Set `HF_TOKEN` in `/etc/bec/worker.env` (or run
`huggingface-cli login` in the deployment environment) before starting the
worker — `huggingface_hub`/`transformers` read it automatically, no code
changes needed. Missing this causes model download to fail in
`ScriptClassificationJobWorker.__init__`, so the worker process never
becomes ready.

## Running locally

```bash
export PATH=/opt/pytorch/bin:$PATH
export HF_TOKEN=...
export BEC_DEST_S3_BUCKET=bdrc-artifacts
export BEC_SQL_HOST=...
# ...other BEC_* env vars from CLI_GUIDE.md

bec worker --job-name script_classification
```

Or bypass SQS entirely against one volume:

```bash
bec run-volume --job-name script_classification --w <W...> --i <I...>
```

Before either, smoke-test the vendored pipeline directly against a few
local images (no S3/SQS/DB needed) to confirm `HF_TOKEN` auth works and to
measure per-image CPU latency:

```bash
python -c "
from bec_orch.jobs.script_classification.vendor.loader import get_pipeline
import sys, time
pipe = get_pipeline()
for path in sys.argv[1:]:
    t0 = time.time()
    row = pipe.run(open(path, 'rb').read())
    print(path, row['final_label'], row['status'], f'{time.time()-t0:.2f}s')
" path/to/image1.jpg path/to/image2.jpg
```
