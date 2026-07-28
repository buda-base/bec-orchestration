from __future__ import annotations

import logging
import threading
import time

from bec_orch.io.sqs import MAX_VISIBILITY_TIMEOUT_SECONDS, MessageNotInflightError, SQSClient

logger = logging.getLogger("bec.core.visibility")

# Give up after this many consecutive failures that are not "message not in
# flight" (throttling, transient network errors, a revoked IAM permission).
_MAX_CONSECUTIVE_FAILURES = 3


class VisibilityExtender:
    """
    Keep an in-flight SQS message hidden while the job worker runs.

    ``receive_one`` only buys ``timeout_seconds``. A volume that takes longer
    becomes visible again mid-processing and a second worker picks it up; the
    duplicate is eventually caught by the success.json / task-claim checks, but
    it burns GPU time and inflates ``ApproximateNumberOfMessagesNotVisible``
    beyond the fleet size. So re-arm the timeout from a daemon thread instead.

    Extension stops after ``max_total_seconds`` so a wedged worker releases its
    message for redelivery (and ultimately the DLQ) rather than holding it
    forever.
    """

    def __init__(
        self,
        sqs: SQSClient,
        queue_url: str,
        receipt_handle: str,
        *,
        timeout_seconds: int,
        extend_every_seconds: int,
        max_total_seconds: int,
        message_id: str = "",
    ) -> None:
        self._sqs = sqs
        self._queue_url = queue_url
        self._receipt_handle = receipt_handle
        self._timeout_seconds = min(timeout_seconds, MAX_VISIBILITY_TIMEOUT_SECONDS)
        self._extend_every_seconds = extend_every_seconds
        self._max_total_seconds = min(max_total_seconds, MAX_VISIBILITY_TIMEOUT_SECONDS)
        self._message_id = message_id

        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    @property
    def enabled(self) -> bool:
        return self._extend_every_seconds > 0 and bool(self._receipt_handle)

    def start(self) -> None:
        if not self.enabled or self._thread is not None:
            return

        self._thread = threading.Thread(
            target=self._run,
            name="sqs-visibility-extender",
            daemon=True,
        )
        self._thread.start()
        logger.info(
            f"Extending visibility of message {self._message_id} every {self._extend_every_seconds}s "
            f"(+{self._timeout_seconds}s per extension, up to {self._max_total_seconds}s total)"
        )
        if self._extend_every_seconds >= self._timeout_seconds:
            logger.warning(
                f"Visibility extension interval ({self._extend_every_seconds}s) is not shorter than the "
                f"visibility timeout ({self._timeout_seconds}s); the message can be redelivered "
                "before the first extension lands"
            )

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            # The thread only ever waits on the event or on a single SQS call.
            self._thread.join(timeout=30.0)
            self._thread = None

    def _run(self) -> None:
        deadline = time.monotonic() + self._max_total_seconds
        failures = 0

        while not self._stop.wait(self._extend_every_seconds):
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                logger.warning(
                    f"Message {self._message_id} has been in flight for {self._max_total_seconds}s; "
                    "no longer extending visibility. Another worker may pick it up — "
                    "this worker is probably stuck."
                )
                return

            # Never hide the message past the deadline, so redelivery happens
            # on time even if this thread's last extension is the one granted.
            timeout = max(1, int(min(self._timeout_seconds, remaining)))

            try:
                self._sqs.change_visibility(self._queue_url, self._receipt_handle, timeout)
                failures = 0
                logger.debug(f"Extended visibility of message {self._message_id} by {timeout}s")
            except MessageNotInflightError as e:
                logger.warning(f"Stopped extending visibility of message {self._message_id}: {e}")
                return
            except Exception as e:
                failures += 1
                logger.warning(
                    f"Failed to extend visibility of message {self._message_id} "
                    f"({failures}/{_MAX_CONSECUTIVE_FAILURES}): {e}"
                )
                if failures >= _MAX_CONSECUTIVE_FAILURES:
                    logger.exception(
                        f"Giving up on extending visibility of message {self._message_id}; "
                        "it may be redelivered to another worker while this one keeps processing it."
                    )
                    return
