from __future__ import annotations
import json
from typing import Optional

import boto3
from botocore.exceptions import ClientError

from bec_orch.core.models import SqsTaskMessage, VolumeRef


class SQSClient:
    """AWS SQS client for task queue management."""

    def __init__(self, region: str):
        """
        Initialize SQS client.
        
        Args:
            region: AWS region (e.g., "us-east-1")
        """
        self.region = region
        self.client = boto3.client('sqs', region_name=region)

    def receive_one(
        self,
        queue_url: str,
        wait_seconds: int,
        visibility_timeout: int,
    ) -> Optional[SqsTaskMessage]:
        """
        Long-poll and return a single message or None.
        
        Args:
            queue_url: SQS queue URL
            wait_seconds: Long polling wait time (0-20 seconds)
            visibility_timeout: How long the message should be hidden from other consumers
            
        Returns:
            SqsTaskMessage or None if no messages available
        """
        try:
            response = self.client.receive_message(
                QueueUrl=queue_url,
                MaxNumberOfMessages=1,
                WaitTimeSeconds=wait_seconds,
                VisibilityTimeout=visibility_timeout,
                MessageAttributeNames=['All']
            )
            
            messages = response.get('Messages', [])
            if not messages:
                return None
            
            msg = messages[0]
            message_id = msg['MessageId']
            receipt_handle = msg['ReceiptHandle']
            body = msg.get('Body', '')
            
            # Parse message attributes for w_id and i_id
            attrs = msg.get('MessageAttributes', {})
            
            # Try to parse w_id and i_id from message attributes
            w_id = None
            i_id = None
            
            if 'w_id' in attrs:
                w_id = attrs['w_id'].get('StringValue')
            if 'i_id' in attrs:
                i_id = attrs['i_id'].get('StringValue')
            
            # If not in attributes, try to parse from body as JSON
            if not w_id or not i_id:
                try:
                    body_data = json.loads(body) if body else {}
                    w_id = w_id or body_data.get('w_id')
                    i_id = i_id or body_data.get('i_id')
                except (json.JSONDecodeError, ValueError):
                    pass
            
            # If still not found, raise error
            if not w_id or not i_id:
                raise ValueError(f"Message missing w_id or i_id: {message_id}")
            
            volume = VolumeRef(w_id=w_id, i_id=i_id)
            
            return SqsTaskMessage(
                message_id=message_id,
                receipt_handle=receipt_handle,
                body=body,
                volume=volume
            )
            
        except ClientError as e:
            # Log and re-raise
            raise RuntimeError(f"Failed to receive message from SQS: {e}") from e

    def delete(self, queue_url: str, receipt_handle: str) -> None:
        """
        Delete a message from the queue.
        
        Args:
            queue_url: SQS queue URL
            receipt_handle: Receipt handle from received message
        """
        try:
            self.client.delete_message(
                QueueUrl=queue_url,
                ReceiptHandle=receipt_handle
            )
        except ClientError as e:
            raise RuntimeError(f"Failed to delete SQS message: {e}") from e

    def change_visibility(
        self,
        queue_url: str,
        receipt_handle: str,
        timeout_seconds: int,
    ) -> None:
        """
        Change the visibility timeout of a message.
        
        This is useful for extending the processing time of a long-running task.
        
        Args:
            queue_url: SQS queue URL
            receipt_handle: Receipt handle from received message
            timeout_seconds: New visibility timeout in seconds (0-43200)
        """
        try:
            self.client.change_message_visibility(
                QueueUrl=queue_url,
                ReceiptHandle=receipt_handle,
                VisibilityTimeout=timeout_seconds
            )
        except ClientError as e:
            raise RuntimeError(f"Failed to change visibility: {e}") from e

    def send_raw(self, queue_url: str, body: str, w_id: Optional[str] = None, i_id: Optional[str] = None) -> None:
        """
        Send a raw message to the queue.
        
        Args:
            queue_url: SQS queue URL
            body: Message body (typically JSON string)
            w_id: Optional work ID (will be added as message attribute)
            i_id: Optional image group ID (will be added as message attribute)
        """
        try:
            message_attributes = {}
            
            if w_id:
                message_attributes['w_id'] = {
                    'StringValue': w_id,
                    'DataType': 'String'
                }
            if i_id:
                message_attributes['i_id'] = {
                    'StringValue': i_id,
                    'DataType': 'String'
                }
            
            kwargs = {
                'QueueUrl': queue_url,
                'MessageBody': body
            }
            
            if message_attributes:
                kwargs['MessageAttributes'] = message_attributes
            
            self.client.send_message(**kwargs)
            
        except ClientError as e:
            raise RuntimeError(f"Failed to send SQS message: {e}") from e
