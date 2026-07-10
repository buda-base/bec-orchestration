"""``script_classification_v2`` job package.

Single-head DINOv3 page classifier running ``BDRC/8-class-tibetan-page-classifier``.

Unlike ``script_classification`` (orientation + 6-class script, two chained
heads), this job runs exactly one classification model over each page and
emits one probability vector per image. It reuses the same S3 fetch / batched
inference / streaming-parquet plumbing.
"""
