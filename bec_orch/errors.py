class RetryableTaskError(Exception):
    """Task failed but should be retried (transient failure)."""

class TerminalTaskError(Exception):
    """Task failed and should not be retried (bad input, invariant broken)."""
