from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, Protocol

from bec_orch.core.models import ArtifactLocation, VolumeManifest, TaskResult, VolumeRef

@dataclass(frozen=True)
class JobContext:
    job_id: int
    volume: VolumeRef
    job_name: str
    job_config: Dict[str, Any]
    config_str: str # raw string from SQL jobs.config
    volume_manifest: VolumeManifest
    artifacts_location: ArtifactLocation
    # Resolved image source routing (see bec_orch.core.sources). Workers should
    # fetch page images from f"s3://{source_bucket}/{image_prefix}{filename}"
    # instead of recomputing the BDRC path, so alternate sources work.
    source: str = "bdrc"
    source_bucket: str = ""
    image_prefix: str = ""

class JobWorker(Protocol):
    """
    Pure business logic: no SQS, no DB.
    Should write artifacts under ctx.artifacts and return metrics.
    """

    def run(self, ctx: JobContext) -> TaskResult: ...
