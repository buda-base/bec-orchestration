from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

@dataclass(frozen=True)
class VolumeRef:
    w_id: str
    i_id: str

@dataclass(frozen=True)
class VolumeManifest:
    manifest: List[Dict[str, Any]] # [ {filename: "I2KG0123.jpg" }, ... ]
    s3_etag: str               # as returned by S3 head (string form)
    last_modified_iso: str     # ISO timestamp string

@dataclass(frozen=True)
class ArtifactLocation:
    bucket: str
    prefix: str                # no leading '/', no trailing required
    basename: str              # base name for artifacts (e.g., volume identifier)

    @property
    def success_key(self) -> str:
        p = self.prefix.rstrip("/")
        return f"{p}/success.json"

@dataclass(frozen=True)
class SqsTaskMessage:
    message_id: str
    receipt_handle: str
    body: str # no expectation of a body for now, the volume is enough
    volume: VolumeRef

@dataclass(frozen=True)
class TaskClaim:
    task_execution_id: int
    volume_id: int
    s3_etag_bytes: bytes       # representation stored in DB

@dataclass(frozen=True)
class TaskResult:
    total_images: int
    nb_errors: int
    total_duration_ms: float
    avg_duration_per_page_ms: float

@dataclass(frozen=True)
class JobRecord:
    id: int
    name: str
    config_text: str           # json most of the time; free-form
    queue_url: str             # SQS queue URL for tasks
    dlq_url: Optional[str]     # optional dead-letter queue URL

@dataclass(frozen=True)
class WorkerRecord:
    worker_id: int
    instance_id: str
    hostname: str
