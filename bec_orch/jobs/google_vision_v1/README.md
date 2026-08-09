# google_vision_v1

Google Cloud Vision OCR (`DOCUMENT_TEXT_DETECTION`, async batch) as a BEC
orchestration job. **No GPU.** Each volume is processed end-to-end inside a
single `JobWorker.run(ctx)` call, with all batch/operation state kept **in
memory** (no external Vision-state database — unlike the standalone
`buda-scripts` pipeline that used PostgreSQL).

## What one volume does

1. **Transfer** — stream-copy every page image from
   `s3://{source_bucket}/{image_prefix}{filename}` to a GCS staging blob at
   `gs://{staging_bucket}/{staging_prefix}{w}/{i}/{version}/{filename}`.
   Images are **kept** on GCS (no cleanup). Existing staging blobs are skipped.
   Pages are split into **lanes** (mirrors the standalone pipeline):
   - `images` — `jpg` / `jpeg` / `png` / `jp2`
   - `files`  — `tif` / `tiff` (multi-page documents live here)
2. **Submit** — per lane, chunk pages into batches of `batch_size` (≤ 2000) and
   submit one async `async_batch_annotate_images` operation per batch, up to
   `max_concurrent_ops` in flight, backing off on `429`/`RESOURCE_EXHAUSTED`.
   Vision writes its JSON to
   `gs://{output_bucket}/{output_prefix}{w}/{i}/{version}/{lane}/{batch_id}/`.
3. **Wait** — poll the long-running operations every `poll_interval_s` until all
   batches finish (or `volume_timeout_s` is hit → retryable failure). The
   runtime's SQS visibility extender keeps the message in flight meanwhile.
4. **Export** — download the Vision JSON from GCS, parse it, and write to the
   destination S3 bucket under `s3_artifact_prefix` (default `gv/`, matching the
   other Google Vision runs) at `gv/{w}/{i}/{version}/`:
   - `{w}-{i}-{version}-gv.parquet` — one row per page (schema below)
   - `{w}-{i}-{version}-gv.jsonl.zst` — raw Vision responses (one JSON per line)

   The runtime still writes its own `success.json` idempotency marker under the
   `{job_name}/{w}/{i}/{version}/` location. Set `s3_artifact_prefix` to empty
   to write the data artifacts under that same `{job_name}/...` location instead.

## Parquet schema

Matches `buda-scripts/.../export_volume_ocr.py` so downstream tooling is
unchanged:

| column | type | notes |
|---|---|---|
| `img_file_name` | string | base filename from the manifest |
| `img_idx` | int16 | 0-based page index (manifest order) |
| `source_etag` | string | S3 ETag of the source image |
| `nb_pages` | int16 | pages in the Vision response (multi-page TIFF) |
| `languages` | list<string> | detected language codes, best-confidence first |
| `confidence` | float16 | median word confidence (first page) |
| `text_len` | int32 | length of the full text in characters |
| `nb_lines_tib` | int16 | lines with ≥1 Tibetan char (U+0F00–U+0FFF) |
| `text` | string | full page text |

## Configuration

All fields have defaults (see `config.py`); override via the DB `jobs.config`
JSON at job-creation time. Notable keys:

- `google_credentials_path` — path to the service-account JSON used for **both**
  Vision and GCS. If omitted, Application Default Credentials are used.
- `staging_bucket` / `staging_prefix` — GCS staging area (default
  `archive-mirror.tbrc.org` + `google_vision_v1_staging/`).
- `output_bucket` / `output_prefix` — GCS Vision-output area (default
  `bec.bdrc.io` + `google_vision_v1_vision-json/`).
- `s3_artifact_prefix` — top-level S3 prefix for the parquet + jsonl.zst
  artifacts in the dest bucket (default `gv/`; empty → `{job_name}/...`).
- `batch_size` (default 500), `max_concurrent_ops` (8), `poll_interval_s` (30),
  `volume_timeout_s` (10800), `max_page_failure_rate` (0.05).

## Dependencies

See `requirements.txt`. Unlike the ML jobs, **no torch/GPU** is required:

```bash
pip install -r bec_orch/jobs/google_vision_v1/requirements.txt
```

## Run

```bash
# Create the job + its SQS queues (put your GCP key path in the config):
bec jobs create --name google_vision_v1 \
  --config-text '{"google_credentials_path": "/etc/bec/gcp-vision.json"}'

# Enqueue volumes and start a worker (any non-GPU host):
bec queue enqueue --job-name google_vision_v1 --file lists/volumes.txt
bec worker --job-name google_vision_v1

# Or process a single volume directly (no SQS), for testing:
bec run-volume --job-name google_vision_v1 --w W3KG609 --i I3KG1563
```

The generic templated systemd unit (`bec-worker.service`, `%i =
google_vision_v1`) works as-is; because a volume may wait several minutes on
Vision, a `SIGTERM` mid-volume simply releases the message for redelivery after
the visibility timeout (processing is idempotent via `success.json`).
