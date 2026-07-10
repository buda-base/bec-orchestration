# script_classification_v2

Single-head DINOv3 page classifier running
[`BDRC/8-class-tibetan-page-classifier`](https://huggingface.co/BDRC/8-class-tibetan-page-classifier).

Unlike [`script_classification`](../script_classification/README.md) — which
chains an orientation classifier and a 6-class script classifier — this job
runs **one** fine-tuned DINOv3 image classifier over each page and records the
full softmax vector for that single model.

## Model

| | |
|---|---|
| Repo | `BDRC/8-class-tibetan-page-classifier` (gated — needs `HF_TOKEN`) |
| Checkpoint | `final_model.pt` (`model_state_dict` + `idx_to_label` + `pooling`=`cls_mean_std`) |
| Backbone | `facebook/dinov3-vitb16-pretrain-lvd1689m` (ViT-B/16, hidden 768; architecture only, weights from checkpoint) |
| Classes (8) | `danyig_pedri`, `druma`, `gyuyig_tsugdri`, `multiscript`, `non_tibetan`, `uchen`, `blank`, `nonplaintext` |
| Engine | plain PyTorch + transformers (no vLLM) |
| Input | short-edge resize + center-crop to `crop_size`² (default **448**), ImageNet normalize |

The checkpoint loader (`vendor/models.py::Classifier.from_checkpoint`) is
model-agnostic: it reads the class count, pooling mode and DINOv3
register-token count from the checkpoint, so it loads any BDRC DINOv3
`final_model.pt` classifier unchanged.

### crop_size

`crop_size` (config, default **448**) MUST match the size the checkpoint was
trained at, or accuracy degrades. The 8-class checkpoint trains at 448
(`results.json`: `preprocess.size = 448`, short-edge resize then center crop).
Override via job config only for a differently-trained checkpoint:

```json
{"crop_size": 224}
```

## Output schema

One parquet row per page (`s3://<dest-bucket>/script_classification_v2/<W>/<I>/<version>/<W>-<I>-<version>.parquet`):

| Column | Type | Notes |
|--------|------|-------|
| `img_file_name` | string | manifest filename |
| `source_etag` | string | S3 ETag |
| `status` | string | `"ok"` \| `"error"` |
| `error` | string | message when error |
| `exif_orientation_tag` | int32 (nullable) | raw EXIF tag, read but never applied |
| `label` | string | argmax class label |
| `prob` | float32 | probability of the predicted class |
| `probs` | list\<float32\> | full softmax, checkpoint `idx_to_label` order |
| `model_version` | string | `<repo-name>:<short-checkpoint-sha>` |
| `error_stage` | string | `""` / `"fetch"` / `"classify"` |
| `error_message` | string | ≤512 chars |

Schema footer metadata (self-describing `probs`): `labels` (JSON array,
`probs[i]` ↔ `labels[i]`), `model_version`, `model_repo`.

## HF auth

`BDRC/8-class-tibetan-page-classifier` is gated. Provide `HF_TOKEN` (env var or
`huggingface-cli login`) with access before the worker starts — it's loaded
from `/etc/bec/worker.env` by the systemd unit.

## Running

Production (SQS-driven volumes, standard BDRC `Works/{md5}/…` layout):

```bash
bec jobs create --name script_classification_v2   # creates SQS queues
bec queue enqueue --job-name script_classification_v2 --file volumes.txt
bec worker --job-name script_classification_v2 --visibility-timeout 1800
```

Single volume (S3 → S3 parquet):

```bash
bec run-volume --job-name script_classification_v2 --w W21809 --i I1831
```

Arbitrary flat S3 image folder → **local** parquet (no DB / SQS; benchmark
tool):

```bash
/opt/pytorch/bin/python scripts/run_script_classification_v2_s3_folder.py \
    --s3-uri s3://bec.bdrc.io/ocr_benchmark/images/W1RAS1/I1RAS1/202603/ \
    --out /tmp/out.parquet
```

## Deploy on an EC2 worker

```bash
sudo cp bec_orch/jobs/script_classification_v2/bec-script-classification-v2.service \
    /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now bec-script-classification-v2
journalctl -u bec-script-classification-v2 -f
```

Requires `/etc/bec/worker.env` with `HF_TOKEN`, `BEC_SQL_*`,
`BEC_DEST_S3_BUCKET`, `BEC_REGION`, AWS creds.
