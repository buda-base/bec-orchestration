# `detect_hff_pipeline.py`

Streams BDRC volume images from S3, detects **header / footer / footnote / body** regions using the [Surya](https://github.com/VikParuchuri/surya) layout model, and writes the results to disk or S3.

---

## What it does

1. Reads a `dimensions.json` manifest from S3 to get the ordered list of images in a volume.
2. Downloads images concurrently from S3 (no images are saved to disk).
3. Runs the Surya layout detector on each image (GPU-accelerated, in batches).
4. Writes one `detections.jsonl` row per image immediately (crash-safe), then a `detections.parquet` at the end of each volume.

The model is loaded **once** before the volume loop, even when processing thousands of volumes in batch.

---

## Output — one row per image

| field | type | description |
|---|---|---|
| `w_id` | string | Work ID |
| `i_id` | string | Image group ID |
| `filename` | string | Image filename |
| `header_boxes` | JSON | List of polygon bboxes `[[[x,y], …], …]` |
| `footer_boxes` | JSON | Same format |
| `footnote_boxes` | JSON | Same format |
| `body_boxes` | JSON | Same format |

Each bbox is four `[x, y]` corner points (top-left → top-right → bottom-right → bottom-left).

Files are written to `<output-dir>/<w_id>/<i_id>/detections.jsonl` and `detections.parquet`.

---

## How to run

**Single volume → local output**
```bash
python scripts/detect_hff_pipeline.py \
    --w-id W22084 --i-id I0886 \
    --output-dir ./output
```

**Batch from CSV** (columns must be `w_id`, `i_id`)
```bash
python scripts/detect_hff_pipeline.py \
    --csv volumes.csv \
    --output-dir ./output
```

**Write parquet directly to S3**
```bash
python scripts/detect_hff_pipeline.py \
    --w-id W22084 --i-id I0886 \
    --output-s3 s3://my-bucket/hff-detections/
```

**Dry run** (no model load, no S3 writes)
```bash
python scripts/detect_hff_pipeline.py \
    --w-id W22084 --i-id I0886 --output-dir ./output --dry-run
```

---

## Key options

| flag | default | description |
|---|---|---|
| `--confidence` | `0.5` | Minimum detection confidence (0–1) |
| `--batch-size` | `8` | Images per GPU batch |
| `--limit N` | `0` (all) | Process only first N images per volume |
| `--start N` | `0` | Skip first N images per volume |
| `--concurrency` | `32` | Parallel S3 download workers |
| `--streaming` | off | Use streaming S3 mode instead of bulk prefetch |
| `-v` | off | Verbose logging |

---

## Dependency

Requires `hff_remover` (the Surya-based detector):
```bash
pip install git+https://github.com/OpenPecha/HFF-Remover.git
```
