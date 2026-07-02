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
| Engine | vendored PyTorch pipeline (CPU-only for v1, synchronous, one image per call) |
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
| `rotation_applied` | int32 | degrees actually applied before 6-class scoring: `0` or `180` |
| `sixclass_label` | string | argmax script class |
| `sixclass_probs` | list\<float32\> | full probability vector, ordered by the checkpoint's label mapping |
| `final_label` | string | equal to `sixclass_label` |
| `model_version` | string | short checkpoint hashes for both models |
| `error_stage` | string | `""` / `"fetch"` / `"classify"` |
| `error_message` | string | short error (max 512 chars) |

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
- **CPU-only.** The vendored `Classifier`/`DINOv3Classifier` code has no
  `.to(device)` calls, matching upstream as-is. Revisit if per-page latency
  on the target CPU instance proves too slow (see "Running locally" below
  for how to measure this).
- **Sequential inference.** `pipe.run()` is one image per call by upstream
  contract — there is no batched tensor inference in the vendored code.
  `ScriptClassificationConfig.inference_workers` defaults to `1`
  (strictly sequential); raising it only overlaps S3 fetch with inference
  of previous pages, it does not create batched model calls.

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
  marks the task **retryable** (likely a transient S3 issue, since
  `pipe.run()` itself never raises — classification errors are also
  captured as `status="error"` rows, not exceptions).
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
