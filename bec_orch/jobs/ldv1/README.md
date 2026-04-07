# Volume Pipeline Overview

This directory contains the runtime pieces that make a volume worker efficient, safe, and easy to reason about.

## Dependencies / runtime assumptions

This pipeline is designed and tested against an **AWS Deep Learning AMI (DLAMI) for PyTorch (GPU)** (Ubuntu 22.04), where **CUDA + PyTorch are preinstalled and known-compatible**.

- **`requirements.txt` intentionally does not install `torch`** to avoid accidentally replacing the DLAMI-provided CUDA/PyTorch stack (a common source of “CUDA not available” / binary mismatch issues).
- **If you run outside DLAMI** (or in a custom container), install a matching CUDA-enabled PyTorch build first (per the official PyTorch install instructions), then install `requirements.txt`.

## Components

```
S3 → Prefetcher (async) → Decoder (thread pool) → LD GPU Batcher (async; two lanes)
       │                                │                 └→ first-pass summary → LD TransformController
       │                                │                                                     │
       └────────────────────────────────┴─────────────────────────────────────────────────────┘
                                                                                 ↘ (if needed) Reprocess lane
                                                                                                       │
                                                                                                       ↓
                                                                                               S3ParquetWriter
```

- Each arrow between stages is a **bounded asyncio.Queue** that provides **backpressure**.
- Stages are **coroutines** (async tasks) except `Decoder`, which uses a **ThreadPoolExecutor** for CPU-bound OpenCV work that releases the GIL.
- The **GPU batcher** has **two lanes**:
  - the **initial lane** with normal priority (first pass on raw frames)
  - the **reprocess lane** with higher priority (second pass on transformed frames)
  A simple **weighted scheduler** favors the reprocess lane to keep second‑pass latency low without starving first‑pass work.

### Prefetcher

- **input:** ImageTask(s3_key, img_filename)
- **output:** FetchedBytes(img_filename, source_etag, bytes)

Fetches bytes on s3 asynchronously, keeps the ETag returned in the s3 GET.

### Decoder

- **input:** FetchedBytes(img_filename, source_etag, bytes)
- **output:** DecodedFrame(img_filename, source_etag, frame, is_binary, first_pass=true, rotation=0, tps_points=null)

where:
- `frame` is a grayscale uint8 cv2 image resized to max_width, max_height coming from the pipeline configuration
- `is_binary` indicates if the original image was binary. In that case the values in the frame are {0,255}

Decodes the bytes from s3 into a grayscale image in a thread pool.

### LDGpuBatcher

- **input:** DecodedFrame(img_filename, source_etag, frame, is_binary, first_pass=true, rotation_angle=null, tps_points=null)
- **output:** InferredFrame(img_filename, source_etag, frame, is_binary, line_mask, first_pass, rotation_angle, tps_points)

where line_mask is a binarized ({0,255}) uint8 frame of the same size as frame.

The GpuBatches uses pytorch to do data transformation:
- binarization on {0,255} using algorithm similar to ADAPTIVE_THRESH_GAUSSIAN_C
- tiling on 3,512,512,FP32
- inference on tiles
- merging inferred tiles into 3, H, W, FP32 (or maybe H, W, FP16? directly)
- reshaping in H, W, uint8

It builds micro‑batches of N (or less) 512×512 patches on GPU and runs the model on these.

### LDPostProcessor

- **input:** InferredFrame(img_filename, source_etag, frame, is_binary, line_mask, first_pass, rotation_angle, tps_points)
- **output:** 2 options:
   * LDRecord(img_filename, source_etag, frame_w, frame_h, contours, nb_contours, contours_bboxes, rotation_angle=0, tps_points=null)
   * DecodedFrame(img_filename, source_etag, frame, is_binary, first_pass=false, rotation_angle, tps_points)

depending if it sees the image requires another inference or not.

In the first case the LDRecord is queued to the S3ParquetWriter.

In the second, the DecodedFrame is queued to a priority lane of the LDGpuBatcher.

For InferredFrames with first_pass = true, it:
- calculates contours on the line_mask
- checks if a rotation is needed based on the contours, if so set rotation_angle and rotates the contours
- checks if a tps operation is needed, if so sets tps_points
- if rotation or tps is needed, apply it on frame, without re-binarization for originally binary images, and send it back as a DecodedFrame. It makes sure the transformed frame is of exactly the same H, W.
- else it creates a LDRecord and queues it

For InferredFrames with first_pass = false, it:
- detects the contours of the line_mask
- creates a LDRecord and queues it 

### S3ParquetWriter

- **input:** LDRecord(img_filename, source_etag, frame_w, frame_h, contours, nb_contours, contours_bboxes, rotation_angle=0, tps_points=null)
- **output:** none

writes the LDRecord on s3.

### LDVolumeWorker

Wires all the components together.

## Error handling and end-of-stream signaling

This pipeline uses **explicit messages** to handle errors and termination in a composable and backpressure-friendly way.

### Sentinel (end-of-stream)

Each stage communicates completion by sending a **sentinel** value through its output queue.

* The sentinel is a unique object (not `None`) that cannot be confused with valid data.
* It means: **no more messages will ever be produced on this queue**.
* Each consumer coroutine exits its processing loop when it receives the sentinel, then forwards it downstream.

This approach avoids relying on queue emptiness (which is transient) and ensures clean shutdown without extra coordination primitives.

### Error messages

Errors are represented as **first-class messages** that flow through the same queues as data.

Instead of raising exceptions across async boundaries, a stage emits a `PipelineError` message containing:

* the pipeline stage where the error occurred (e.g. `"prefetch"`, `"decode"`),
* the associated task (when available),
* error type and message,
* optional traceback,
* retry metadata (attempt count, retryable flag).

Downstream stages typically:

* **pass error messages through unchanged**, and
* continue processing other items unless a fail-fast policy is explicitly desired.

This design allows the pipeline to:

* continue processing valid inputs when possible,
* collect and report partial failures,
* make error handling explicit, observable, and testable.

### Summary

At each stage boundary, queue messages are one of:

* a **data message** (stage output),
* a **`PipelineError`** (non-fatal error signal),
* the **sentinel** (end-of-stream).

This pattern is widely used in async pipelines and stream processors, and provides clear semantics for shutdown, backpressure, and error propagation.


## Why Bounded Queues? (Backpressure 101)

A bounded queue has a **maximum size**. When a downstream stage is slow, the upstream `await q.put(...)` **blocks**, naturally slowing the pipeline. This keeps memory stable and avoids flooding any stage.

## Coroutines vs Threads

- **Coroutines** (`async def ...`) are used for I/O‑bound stages (S3 networking, GPU scheduling). They are cheap and cooperative.
- **Threads** are used for CPU‑bound decoding with OpenCV (which releases the GIL during native calls). We keep a small, fixed pool.

## Two‑Lane GPU Batching (non‑obvious but important)

- First pass produces a **summary**; only **some** frames need transform + second pass.
- To keep completion time tight for those frames, the reprocess lane is served with a higher **weight** (e.g., 3:1). The batcher pulls up to `re_weight` items from the reprocess lane before it takes an initial‑lane item.
- Batching (e.g., 8/16 patches per forward) happens **across frames** inside each lane, so the GPU runs efficiently even when a single image has only a few patches.

## Quick Example

```python
import asyncio
from pipeline.config import PipelineConfig
from pipeline.types import ImageTask
from pipeline.s3ctx import S3Context
from pipeline.worker import VolumeWorker

async def main():
    cfg = PipelineConfig(
        s3_bucket="my-bucket",
        out_prefix="s3://my-bucket/out",
        staging_prefix="s3://my-bucket/staging",
    )
    tasks = [ImageTask(key=f"vol_A/img_{i:05d}.jpg", etag="etagA", size=None, volume_id="vol_A") for i in range(600)]
    global_sem = asyncio.Semaphore(cfg.s3_max_inflight_global)
    s3 = S3Context(cfg, global_sem)
    worker = LDVolumeWorker(cfg, s3, "vol_A", tasks)
    await worker.run()

if __name__ == "__main__":
    asyncio.run(main())
```
