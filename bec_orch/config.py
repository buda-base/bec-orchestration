from dataclasses import dataclass


@dataclass(frozen=True)
class OrchestrationConfig:
    db_dsn: str
    aws_region: str

    job_name: str  # Job name (unique identifier, more user-friendly than ID)

    poll_wait_seconds: int = 20
    max_messages: int = 1  # sequential worker
    visibility_timeout_seconds: int = 450  # 7.5 minutes - gives buffer for long volumes
    visibility_extend_every_seconds: int = 60  # Re-arm every 60s while processing (0 disables)
    # Upper bound on how long one message may stay in flight. Past this the
    # worker stops extending, so a wedged worker releases the volume for
    # redelivery instead of holding it until it is restarted.
    visibility_max_total_seconds: int = 14400  # 4 hours

    # Shutdown behavior:
    # > 0: Exit after N empty polls (e.g., 6 * 20s = 2 minutes) - for batch jobs
    # <= 0: Run indefinitely (daemon mode) - for systemd services
    shutdown_after_empty_polls: int = 6
