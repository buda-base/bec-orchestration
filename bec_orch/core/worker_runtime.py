from __future__ import annotations
from typing import Any, Dict, Optional, Tuple

from bec_orch.config import OrchestrationConfig
from bec_orch.core.models import ArtifactLocation, VolumeManifest, SqsTaskMessage, TaskResult, VolumeRef
from bec_orch.io.db import DBClient
from bec_orch.io.sqs import SQSClient

# TODO: get last modification timestamp in addition to etag, force etag to convert to md5 format

SESSION = boto3.Session() if boto3 is not None else None
S3 = SESSION.client("s3") if SESSION is not None else None

# Common image file extensions (case-insensitive)
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.tif', '.tiff'}


def get_s3_folder_prefix(w_id, i_id):
    """
    gives the s3 prefix (~folder) in which the volume will be present.
    inpire from https://github.com/buda-base/buda-iiif-presentation/blob/master/src/main/java/
    io/bdrc/iiif/presentation/ImageInfoListService.java#L73
    Example:
       - w_id=W22084, i_id=I0886
       - result = "Works/60/W22084/images/W22084-0886/
    where:
       - 60 is the first two characters of the md5 of the string W22084
       - 0886 is:
          * the image group ID without the initial "I" if the image group ID is in the form I\\d\\d\\d\\d
          * or else the full image group ID (incuding the "I")
    """
    md5 = hashlib.md5(str.encode(w_id))
    two = md5.hexdigest()[:2]

    pre, rest = i_id[0], i_id[1:]
    if pre == 'I' and rest.isdigit() and len(rest) == 4:
        suffix = rest
    else:
        suffix = i_id

    return 'Works/{two}/{RID}/images/{RID}-{suffix}/'.format(two=two, RID=w_id, suffix=suffix)


def gets3blob(s3Key: str) -> Tuple[Optional[io.BytesIO], Optional[str]]:
    """
    Downloads an S3 object and returns (BytesIO buffer, etag).
    Returns (None, None) if object not found.
    """
    if S3 is None or botocore is None:
        raise RuntimeError(
            "S3 mode requires boto3+botocore. Install them (see requirements.txt) "
            "or use --input-folder / --output-folder file:///... for local mode."
        )
    try:
        # Single request: get_object provides both Body and ETag.
        obj = S3.get_object(Bucket="archive.tbrc.org", Key=s3Key)
        etag = obj.get("ETag", None)
        body_bytes: bytes = obj["Body"].read()
        return io.BytesIO(body_bytes), etag
    except botocore.exceptions.ClientError as e:
        if e.response['Error']['Code'] == '404':
            return None, None
        else:
            raise


def get_volume_version(w_id, i_id, s3_etag):
    return s3_etag.replace('"', "")[:6]


def get_image_list_and_version_s3(w_id: str, i_id: str) -> Tuple[Optional[List[ImageTask]], Optional[str]]:
    """
    Gets manifest of files in a volume and returns list of ImageTasks and version.
    Returns (None, None) if manifest not found.
    """
    vol_s3_prefix = get_s3_folder_prefix(w_id, i_id)
    vol_manifest_s3_key = vol_s3_prefix + "dimensions.json"
    blob, etag = gets3blob(vol_manifest_s3_key)
    if blob is None:
        return None, None
    
    i_version = get_volume_version(w_id, i_id, etag or "")
    blob.seek(0)
    b = blob.read()
    ub = gzip.decompress(b)
    s = ub.decode('utf8')
    data = json.loads(s)
    # data is in the form: [ { "filename": "I123.jpg", ... }, ... ]
    
    # Convert to ImageTask list
    image_tasks = []
    for item in data:
        filename = item.get("filename")
        if not filename:
            continue

        # Filter files by extension (images only)
        ext = Path(str(filename)).suffix.lower()
        if ext not in IMAGE_EXTENSIONS:
            continue

        if filename:
            # Build full S3 key by prefixing with volume prefix
            s3_key = vol_s3_prefix + filename
            source_uri = f"s3://archive.tbrc.org/{s3_key}"
            image_tasks.append(ImageTask(
                source_uri=source_uri,
                img_filename=filename
            ))
    
    return image_tasks, i_version



class BECWorkerRuntime:
    def __init__(
        self,
        cfg: OrchestrationConfig,
        db: DBClient,
        sqs: SQSClient,
        s3: S3Client,
    ): ...

    def initialize(self) -> None:
        """
        - open DB connection
        - ensure worker is registered in sql
        - fetch job record in sql and parse config (json if applicable)
        - load JobWorker implementation via registry
        """
        ...

    def run_forever(self) -> None:
        """Main loop: poll SQS, process message, exit after N empty polls."""
        ...

    def shutdown(self) -> None:
        """Mark worker stopped and close DB connection."""
        ...

    # ---- internals ----
    def _process_message(self, msg: SqsTaskMessage) -> None:
        """
        High-level flow:
        - resolve manifest info + etag
        - resolve volume_id
        - claim task in DB (idempotent)
        - compute artifact location
        - check success marker (success.json, optional secondary guard)
        - run job worker
        - write success.json
        - update DB
        - delete SQS message when safe
        """
        ...

    def _get_volume_manifest(self, volume: VolumeRef) -> VolumeManifest:
        """
        Wrapper around your get_manifest(w_id, i_id).
        """
        ...

    def _get_artifact_location(self, volume: VolumeRef, s3_etag: str) -> ArtifactLocation:
        """
        Wrapper around your get_s3_artefacts_prefix(job_name, w_id, i_id, etag).
        """
        ...

    def _write_success_marker(
        self,
        artifacts: ArtifactLocation,
        payload: Dict[str, Any],
    ) -> None: ...

    def _classify_exception(self, exc: Exception) -> Tuple[bool, str]:
        """
        Returns (retryable, reason).
        Default: retryable=True unless it's a TerminalTaskError.
        """
        ...
