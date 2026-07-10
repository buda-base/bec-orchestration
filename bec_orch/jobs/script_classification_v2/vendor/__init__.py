"""Vendored single-head DINOv3 classifier pipeline for ``script_classification_v2``.

Adapted from ``bec_orch/jobs/script_classification/vendor`` but reduced to a
single classification model (no orientation + 6-class chaining). The model
loader (``models.Classifier.from_checkpoint``) is unchanged and generic: it
reads ``idx_to_label``, ``pooling`` and the DINOv3 register-token count from
the checkpoint, so it loads any BDRC DINOv3 ``final_model.pt`` classifier.
"""
