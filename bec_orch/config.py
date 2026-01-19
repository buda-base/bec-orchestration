from dataclasses import dataclass
from typing import Optional

@dataclass(frozen=True)
class OrchestrationConfig:
    db_dsn: str
    aws_region: str

    job_name: str  # Job name (unique identifier, more user-friendly than ID)

    poll_wait_seconds: int = 20
    max_messages: int = 1                 # sequential worker
    visibility_timeout_seconds: int = 450  # 7.5 minutes - gives buffer for long volumes
    visibility_extend_every_seconds: int = 60  # Extend every 60s to prevent expiration

    # Shutdown behavior:
    # > 0: Exit after N empty polls (e.g., 6 * 20s = 2 minutes) - for batch jobs
    # <= 0: Run indefinitely (daemon mode) - for systemd services
    shutdown_after_empty_polls: int = 6
