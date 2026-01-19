from __future__ import annotations
import json
import logging
from typing import Iterable, List, Optional, Callable

from bec_orch.core.models import VolumeRef
from bec_orch.io.sqs import SQSClient

logger = logging.getLogger(__name__)


def enqueue_volumes(
    sqs: SQSClient,
    queue_url: str,
    volumes: Iterable[VolumeRef],
) -> int:
    """
    Send SQS messages for volumes using batch operations.
    
    Each message has format: {'w_id': '...', 'i_id': '...'}
    with message attributes for easy filtering.
    
    Uses AWS SQS batch send (up to 10 messages per API call) for better performance.
    
    Args:
        sqs: SQS client
        queue_url: Queue URL
        volumes: Iterable of VolumeRef
        
    Returns:
        Count of volumes enqueued
    """
    # Prepare all messages first
    messages = []
    for volume in volumes:
        body = json.dumps({
            'w_id': volume.w_id,
            'i_id': volume.i_id,
        })
        messages.append((body, volume.w_id, volume.i_id))
    
    # Send in batches (much faster than one at a time)
    if messages:
        total = len(messages)
        logger.info(f"Enqueueing {total} volumes in batches...")
        count = sqs.send_batch(queue_url, messages)
        logger.info(f"Enqueued {count} volumes total")
        return count
    
    return 0


def enqueue_volume_list_from_file(
    sqs: SQSClient,
    queue_url: str,
    file_path: str,
    filter_func: Optional[Callable[[VolumeRef], bool]] = None,
    limit: Optional[int] = None,
) -> tuple[int, int]:
    """
    Enqueue volumes from a file.
    
    File format: one volume per line, format: "W12345,I0123" or "W12345 I0123"
    Lines starting with # are comments.
    
    Args:
        sqs: SQS client
        queue_url: Queue URL
        file_path: Path to file
        filter_func: Optional function to filter volumes (return True to include, False to skip)
        limit: Optional limit on number of volumes to enqueue (after filtering)
        
    Returns:
        Tuple of (count of volumes enqueued, count of volumes skipped by filter)
    """
    volumes = []
    skipped_count = 0
    
    with open(file_path, 'r') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            
            # Skip empty lines and comments
            if not line or line.startswith('#'):
                continue
            
            # Parse line: "W12345,I0123" or "W12345 I0123"
            parts = line.replace(',', ' ').split()
            if len(parts) != 2:
                logger.warning(f"Line {line_num}: Invalid format, expected 'W12345 I0123', got: {line}")
                continue
            
            w_id, i_id = parts
            volume = VolumeRef(w_id=w_id, i_id=i_id)
            
            # Apply filter if provided
            if filter_func is not None:
                if not filter_func(volume):
                    skipped_count += 1
                    logger.debug(f"Skipped volume {w_id}/{i_id} (already done on latest version)")
                    continue
            
            volumes.append(volume)
            
            # Apply limit if provided
            if limit is not None and len(volumes) >= limit:
                logger.info(f"Reached limit of {limit} volumes")
                break
    
    enqueued_count = enqueue_volumes(sqs, queue_url, volumes)
    return enqueued_count, skipped_count


def get_queue_stats(sqs: SQSClient, queue_url: str) -> dict:
    """
    Get queue statistics.
    
    Args:
        sqs: SQS client
        queue_url: Queue URL
        
    Returns:
        Dict with queue attributes
    """
    response = sqs.client.get_queue_attributes(
        QueueUrl=queue_url,
        AttributeNames=['All']
    )
    
    attrs = response.get('Attributes', {})
    
    return {
        'approximate_messages': int(attrs.get('ApproximateNumberOfMessages', 0)),
        'approximate_messages_not_visible': int(attrs.get('ApproximateNumberOfMessagesNotVisible', 0)),
        'approximate_messages_delayed': int(attrs.get('ApproximateNumberOfMessagesDelayed', 0)),
        'created_timestamp': int(attrs.get('CreatedTimestamp', 0)),
        'last_modified_timestamp': int(attrs.get('LastModifiedTimestamp', 0)),
    }


def purge_queue(sqs: SQSClient, queue_url: str) -> None:
    """
    Purge all messages from a queue.
    
    WARNING: This is irreversible!
    
    Args:
        sqs: SQS client
        queue_url: Queue URL
    """
    sqs.client.purge_queue(QueueUrl=queue_url)
    logger.info(f"Purged queue: {queue_url}")
