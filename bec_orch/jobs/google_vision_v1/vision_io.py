"""I/O clients and helpers for ``google_vision_v1``.

Groups everything that talks to an external service so the worker stays
readable:

- ``lane_and_media_type`` -- filename -> (lane, mime) exactly like the
  standalone ``generate_volume_manifests.get_lane_and_media_type``.
- ``build_s3_client`` -- boto3 S3 client (source images + dest artifacts).
- ``VisionClient`` -- submit an async batch / poll one operation.
- ``GcsIO`` -- service-account GCS client with the few operations we need
  (stream-copy from S3, existence check, list + download Vision output blobs).

Google libraries are imported lazily so that importing this module (and hence
registering the job) never fails when ``google-cloud-*`` is not installed.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Optional

import boto3
from botocore.config import Config as BotoConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Lane / media-type routing (mirrors generate_volume_manifests.py)
# ---------------------------------------------------------------------------

def lane_and_media_type(filename: str) -> tuple[str, str]:
    """Return ``(lane, media_type)`` for a page filename.

    - tif/tiff -> ("files",  "image/tiff")   [multi-page docs live here]
    - jpg/jpeg -> ("images", "image/jpeg")
    - png      -> ("images", "image/png")
    - jp2      -> ("images", "image/jp2")
    - other    -> ("images", "image/<ext>" | "application/octet-stream")
    """
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    if ext in ("tif", "tiff"):
        return "files", "image/tiff"
    if ext in ("jpg", "jpeg"):
        return "images", "image/jpeg"
    if ext == "png":
        return "images", "image/png"
    if ext == "jp2":
        return "images", "image/jp2"
    return "images", (f"image/{ext}" if ext else "application/octet-stream")


# ---------------------------------------------------------------------------
# S3
# ---------------------------------------------------------------------------

def build_s3_client(cfg: Any) -> Any:
    import os

    region = os.environ.get("BEC_REGION", "us-east-1")
    boto_cfg = BotoConfig(
        region_name=region,
        retries={"max_attempts": cfg.s3_max_attempts, "mode": "standard"},
        connect_timeout=cfg.s3_get_timeout_s,
        read_timeout=cfg.s3_get_timeout_s,
        max_pool_connections=max(cfg.transfer_concurrency * 2, 32),
    )
    return boto3.client("s3", config=boto_cfg)


# ---------------------------------------------------------------------------
# Google Vision
# ---------------------------------------------------------------------------

class VisionClient:
    """Thin wrapper over ``vision_v1.ImageAnnotatorClient`` (built lazily)."""

    def __init__(self, credentials_path: Optional[str], feature_type: str, model: str) -> None:
        self._credentials_path = credentials_path
        self._feature_type = feature_type
        self._model = model
        self._client: Any = None

    def _get(self) -> Any:
        if self._client is None:
            from google.cloud import vision_v1

            if self._credentials_path:
                from google.oauth2 import service_account

                creds = service_account.Credentials.from_service_account_file(
                    self._credentials_path,
                    scopes=["https://www.googleapis.com/auth/cloud-platform"],
                )
                self._client = vision_v1.ImageAnnotatorClient(credentials=creds)
            else:
                self._client = vision_v1.ImageAnnotatorClient()
        return self._client

    def submit_batch(self, image_uris: list[str], output_prefix: str, output_shard_size: int) -> str:
        """Submit one async batch; return the long-running operation name."""
        from google.cloud import vision_v1

        client = self._get()

        feature = vision_v1.types.Feature(type_=vision_v1.types.Feature.Type[self._feature_type])
        if self._model:
            feature.model = self._model

        requests = [
            vision_v1.types.AnnotateImageRequest(
                image=vision_v1.types.Image(source=vision_v1.types.ImageSource(image_uri=uri)),
                features=[feature],
            )
            for uri in image_uris
        ]
        output_config = vision_v1.types.OutputConfig(
            gcs_destination=vision_v1.types.GcsDestination(uri=output_prefix),
            batch_size=output_shard_size,
        )
        operation = client.async_batch_annotate_images(requests=requests, output_config=output_config)
        return operation.operation.name

    def check_operation(self, operation_name: str) -> tuple[bool, Optional[str]]:
        """Return ``(is_done, error_message)`` for a long-running operation."""
        client = self._get()
        op = client.transport.operations_client.get_operation(operation_name)
        if op.done:
            if op.error.code != 0:
                return True, f"Error {op.error.code}: {op.error.message}"
            return True, None
        return False, None


# ---------------------------------------------------------------------------
# Google Cloud Storage
# ---------------------------------------------------------------------------

_GS_URI_RE = re.compile(r"^gs://([^/]+)/(.+)$")


def parse_gs_uri(gs_uri: str) -> tuple[str, str]:
    m = _GS_URI_RE.match(gs_uri)
    if not m:
        raise ValueError(f"not a gs:// URI: {gs_uri}")
    return m.group(1), m.group(2)


class GcsIO:
    """The handful of GCS operations the worker needs (client built lazily)."""

    def __init__(self, credentials_path: Optional[str], project: Optional[str]) -> None:
        self._credentials_path = credentials_path
        self._project = project
        self._client: Any = None

    def _get(self) -> Any:
        if self._client is None:
            from google.cloud import storage

            kwargs: dict[str, Any] = {}
            if self._project:
                kwargs["project"] = self._project
            if self._credentials_path:
                from google.oauth2 import service_account

                kwargs["credentials"] = service_account.Credentials.from_service_account_file(
                    self._credentials_path
                )
            self._client = storage.Client(**kwargs)
        return self._client

    def blob_exists(self, bucket: str, blob_name: str) -> bool:
        client = self._get()
        return client.bucket(bucket).blob(blob_name).exists(client=client)

    def gs_uri_exists(self, gs_uri: str) -> bool:
        try:
            bucket, blob_name = parse_gs_uri(gs_uri)
        except ValueError:
            return False
        return self.blob_exists(bucket, blob_name)

    def upload_from_s3(
        self,
        s3_client: Any,
        s3_bucket: str,
        s3_key: str,
        gcs_bucket: str,
        gcs_blob_name: str,
        content_type: Optional[str] = None,
    ) -> str:
        """Stream one S3 object into a GCS blob; return its S3 ETag."""
        obj = s3_client.get_object(Bucket=s3_bucket, Key=s3_key)
        etag = obj.get("ETag", "").strip('"')
        body = obj["Body"]  # botocore StreamingBody -> uploaded without full buffering
        blob = self._get().bucket(gcs_bucket).blob(gcs_blob_name)
        blob.upload_from_file(body, rewind=False, content_type=content_type)
        return etag

    def list_output_blob_names(self, gcs_bucket: str, prefix: str) -> list[str]:
        """List Vision ``output-N-to-M.json`` blobs under a prefix, sorted."""
        client = self._get()
        bucket = client.bucket(gcs_bucket)
        return sorted(
            b.name
            for b in bucket.list_blobs(prefix=prefix)
            if re.search(r"output-\d+-to-\d+\.json$", b.name)
        )

    def download_blob_bytes(self, gcs_bucket: str, blob_name: str, timeout: int) -> bytes:
        client = self._get()
        return client.bucket(gcs_bucket).blob(blob_name).download_as_bytes(timeout=timeout)
