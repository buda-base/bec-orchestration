from __future__ import annotations
from typing import Iterable, List, Optional
from bec_orch.core.models import VolumeRef
from bec_orch.io.sqs import SQSClient

def enqueue_volumes(
    sqs: SQSClient,
    queue_url: str,
    volumes: Iterable[VolumeRef],
) -> int:
    """Send SQS messages {'w_id':..., 'i_id':...}. Returns count enqueued."""
    ...
