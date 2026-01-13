from dataclasses import dataclass
from typing import Optional

@dataclass(frozen=True)
class OrchestrationConfig:
    db_dsn: str
    aws_region: str

    job_id: int
    job_name: str

    queue_url: str
    dlq_url: Optional[str] = None

    poll_wait_seconds: int = 20
    max_messages: int = 1                 # sequential worker
    visibility_timeout_seconds: int = 300
    visibility_extend_every_seconds: int = 60

    shutdown_after_empty_polls: int = 6   # e.g. 6 * 20s = 2 minutes
