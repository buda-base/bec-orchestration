from __future__ import annotations

import json

import boto3
from botocore.exceptions import ClientError

from bec_orch.core.models import SqsTaskMessage, VolumeRef

# A message ready to be batched: (body, w_id, i_id, source, i_version).
# Trailing elements are optional; only ``body`` is strictly required.
VolumeMessage = tuple

_DEFAULT_SOURCE = "bdrc"


def _volume_message_attributes(
    w_id: str | None,
    i_id: str | None,
    source: str | None = None,
    i_version: str | None = None,
) -> dict:
    """Build SQS MessageAttributes for a volume, skipping empty/default fields."""
    attrs: dict = {}
    if w_id:
        attrs["w_id"] = {"StringValue": w_id, "DataType": "String"}
    if i_id:
        attrs["i_id"] = {"StringValue": i_id, "DataType": "String"}
    # Only carry source when it's non-default, to keep BDRC messages unchanged.
    if source and source != _DEFAULT_SOURCE:
        attrs["source"] = {"StringValue": source, "DataType": "String"}
    if i_version:
        attrs["i_version"] = {"StringValue": i_version, "DataType": "String"}
    return attrs

# SQS error codes meaning the receipt handle can no longer be acted upon: the
# message was deleted, its visibility already expired, or it was redelivered to
# another consumer (which invalidates our handle).
_NOT_INFLIGHT_ERROR_CODES = {
    "AWS.SimpleQueueService.MessageNotInflight",
    "InvalidParameterValue",
    "ReceiptHandleIsInvalid",
}

# ChangeMessageVisibility rejects anything above 12 h.
MAX_VISIBILITY_TIMEOUT_SECONDS = 43200


class MessageNotInflightError(RuntimeError):
    """Raised when a receipt handle is stale: the message is no longer in flight."""


class SQSClient:
    """AWS SQS client for task queue management."""

    def __init__(self, region: str) -> None:
        """
        Initialize SQS client.

        Args:
            region: AWS region (e.g., "us-east-1")
        """
        self.region = region
        self.client = boto3.client("sqs", region_name=region)

    def receive_one(
        self,
        queue_url: str,
        wait_seconds: int,
        visibility_timeout: int,
    ) -> SqsTaskMessage | None:
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
                MessageAttributeNames=["All"],
            )

            messages = response.get("Messages", [])
            if not messages:
                return None

            msg = messages[0]
            message_id = msg["MessageId"]
            receipt_handle = msg["ReceiptHandle"]
            body = msg.get("Body", "")

            # Parse message attributes for volume fields
            attrs = msg.get("MessageAttributes", {})

            def _attr(name: str) -> str | None:
                return attrs[name].get("StringValue") if name in attrs else None

            w_id = _attr("w_id")
            i_id = _attr("i_id")
            source = _attr("source")
            i_version = _attr("i_version")

            # The body is the canonical carrier; fall back to it for anything
            # not present as a message attribute.
            if not (w_id and i_id and source and i_version):
                try:
                    body_data = json.loads(body) if body else {}
                except (json.JSONDecodeError, ValueError):
                    body_data = {}
                w_id = w_id or body_data.get("w_id")
                i_id = i_id or body_data.get("i_id")
                source = source or body_data.get("source")
                i_version = i_version or body_data.get("i_version")

            # If still not found, raise error
            if not w_id or not i_id:
                raise ValueError(f"Message missing w_id or i_id: {message_id}")

            volume = VolumeRef(
                w_id=w_id,
                i_id=i_id,
                source=(source or "bdrc").strip().lower(),
                i_version=i_version or None,
            )

            return SqsTaskMessage(message_id=message_id, receipt_handle=receipt_handle, body=body, volume=volume)

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
            self.client.delete_message(QueueUrl=queue_url, ReceiptHandle=receipt_handle)
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

        Raises:
            MessageNotInflightError: The message is no longer in flight (deleted,
                already visible again, or handed to another consumer).
            RuntimeError: Any other SQS failure (throttling, network, IAM, ...).
        """
        try:
            self.client.change_message_visibility(
                QueueUrl=queue_url, ReceiptHandle=receipt_handle, VisibilityTimeout=timeout_seconds
            )
        except ClientError as e:
            code = e.response.get("Error", {}).get("Code", "")
            if code in _NOT_INFLIGHT_ERROR_CODES:
                raise MessageNotInflightError(f"Message no longer in flight ({code}): {e}") from e
            raise RuntimeError(f"Failed to change visibility: {e}") from e

    def send_raw(
        self,
        queue_url: str,
        body: str,
        w_id: str | None = None,
        i_id: str | None = None,
        source: str | None = None,
        i_version: str | None = None,
    ) -> None:
        """
        Send a raw message to the queue.

        Args:
            queue_url: SQS queue URL
            body: Message body (typically JSON string)
            w_id: Optional work ID (will be added as message attribute)
            i_id: Optional image group ID (will be added as message attribute)
            source: Optional image source (added as message attribute if set)
            i_version: Optional image version (added as message attribute if set)
        """
        try:
            message_attributes = _volume_message_attributes(w_id, i_id, source, i_version)

            kwargs = {"QueueUrl": queue_url, "MessageBody": body}

            if message_attributes:
                kwargs["MessageAttributes"] = message_attributes

            self.client.send_message(**kwargs)

        except ClientError as e:
            raise RuntimeError(f"Failed to send SQS message: {e}") from e

    def send_batch(self, queue_url: str, messages: "list[VolumeMessage]") -> int:
        """
        Send multiple messages to the queue in batches.

        AWS SQS supports up to 10 messages per batch request.
        This method will automatically split larger lists into multiple batch requests.

        Args:
            queue_url: SQS queue URL
            messages: List of VolumeMessage tuples: (body, w_id, i_id, source, i_version)
                Only ``body`` is required; the rest become message attributes when set.
                Legacy 3-tuples (body, w_id, i_id) are still accepted.

        Returns:
            Total count of messages successfully sent
        """
        if not messages:
            return 0

        total_sent = 0
        batch_size = 10  # AWS SQS limit

        try:
            # Process messages in batches of 10
            for i in range(0, len(messages), batch_size):
                batch = messages[i : i + batch_size]
                entries = []

                for idx, msg in enumerate(batch):
                    body = msg[0]
                    w_id = msg[1] if len(msg) > 1 else None
                    i_id = msg[2] if len(msg) > 2 else None
                    source = msg[3] if len(msg) > 3 else None
                    i_version = msg[4] if len(msg) > 4 else None

                    entry = {
                        "Id": str(idx),  # Must be unique within this batch
                        "MessageBody": body,
                    }

                    message_attributes = _volume_message_attributes(w_id, i_id, source, i_version)
                    if message_attributes:
                        entry["MessageAttributes"] = message_attributes

                    entries.append(entry)

                # Send batch
                response = self.client.send_message_batch(QueueUrl=queue_url, Entries=entries)

                # Check for failures
                if response.get("Failed"):
                    failed_ids = [f["Id"] for f in response["Failed"]]
                    raise RuntimeError(f"Failed to send {len(failed_ids)} messages in batch: {response['Failed']}")

                total_sent += len(entries)

            return total_sent

        except ClientError as e:
            raise RuntimeError(f"Failed to send batch messages to SQS: {e}") from e
