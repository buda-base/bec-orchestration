from __future__ import annotations
from typing import Optional
from bec_orch.core.models import SqsTaskMessage

class SQSClient:
    def __init__(self, region: str): ...

    def receive_one(
        self,
        queue_url: str,
        wait_seconds: int,
        visibility_timeout: int,
    ) -> Optional[SqsTaskMessage]:
        """Long-poll and return a single message or None."""
        ...

    def delete(self, queue_url: str, receipt_handle: str) -> None: ...

    def change_visibility(
        self,
        queue_url: str,
        receipt_handle: str,
        timeout_seconds: int,
    ) -> None: ...

    def send_raw(self, queue_url: str, body: str) -> None: ...
