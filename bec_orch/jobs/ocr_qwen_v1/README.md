# `ocr_qwen_v1` — Tibetan OCR with Qwen3.5-VL 0.8B

Full-page Tibetan OCR using the
[`buddhist-nlp/bdrc-mitra-ocr-qwen35-0.8b`](https://huggingface.co/buddhist-nlp/bdrc-mitra-ocr-qwen35-0.8b)
fine-tune of Qwen3.5-VL, served via vLLM's offline Python API.

See `scratch/findings.md` (repo root) for the full benchmark report that
drove the defaults baked into `config.py`.

## At a glance

| | |
|---|---|
| Model | `buddhist-nlp/bdrc-mitra-ocr-qwen35-0.8b` (Qwen3.5-VL, 853 M params, BF16, 1.7 GB) |
| Engine | `vllm.LLM` (synchronous, offline) |
| Throughput on A10G | **2.2 pages / s** end-to-end (S3 + decode + OCR) |
| Bottleneck | GPU decode of ~2 000 output tokens/page |
| Per-volume RAM peak | ~2.5 GB (decoded RGB images held until `LLM.chat`) |
| Per-volume timeout | 30 min (configurable) |

## Output

For each volume the worker writes one parquet file:

    s3://<dest-bucket>/ocr_qwen_v1/<W>/<I>/<version>/<W>-<I>-<version>.parquet

…plus an optional gzipped jsonl with per-page failure details:

    s3://<dest-bucket>/ocr_qwen_v1/<W>/<I>/<version>/<W>-<I>-<version>-errors.jsonl.gz

Parquet schema:

| column | type | meaning |
|---|---|---|
| `img_file_name` | string | base name, matches manifest entry |
| `source_etag` | string | S3 ETag of the source image |
| `ok` | bool | OCR produced text (even if truncated) |
| `truncated` | bool | hit `max_tokens` — `page_text` is partial |
| `finish_reason` | string | `"stop"` or `"length"` |
| `page_text` | string | full transcription (empty on error) |
| `output_tokens` | int32 | tokens emitted by the model |
| `error_stage` | string | `""` / `"fetch"` / `"decode"` / `"ocr"` |
| `error_message` | string | short error (max 512 chars) |
| `model_id` | string | HF id of the model that produced the row |

## Failure semantics

- A per-volume failure rate **above `max_page_failure_rate` (default 5 %)**
  marks the task **retryable** (likely a transient S3 / GPU issue).
- A failure rate **at or below the threshold** is considered a success — the
  bad pages are still recorded in the parquet with `ok=false`.
- A page that hits `max_tokens` is recorded with `truncated=true` and is
  **not** counted as an error by default (toggle with
  `treat_truncation_as_failure: true`).

## Tuning per GPU

- `BEC_OCR_QWEN_MEM_UTIL` env var overrides `gpu_memory_utilization` at
  runtime so the same job config can run on different GPUs.
- Default is `0.85` (dedicated GPU); drop to ~`0.55` if something else
  shares the GPU on the same box.

## Running locally

```bash
export PATH=/opt/pytorch/bin:$PATH
# BEC_OCR_QWEN_MEM_UTIL is only needed on shared GPUs (default is 0.85)
export BEC_DEST_S3_BUCKET=bdrc-artifacts
export BEC_SQL_HOST=...
# ...other BEC_* env vars from CLI_GUIDE.md

bec worker --job-name ocr_qwen_v1
```
