import json
import logging
import os
import sys
import time
import warnings
from typing import Any, Dict

# CloudWatch-ready logs (to stdout/stderr, appended by systemd to /var/log/bec/worker.log)
# cloudwatch config is in cloudwatch.json
# start cloudwatch with 
# sudo amazon-cloudwatch-agent-ctl -a fetch-config -m ec2 -c file:/.../cloudwatch.json -s
# everything is usually WARNING, except logging.getLogger("bec") (and sub loggers) which is INFO for CW
# Log Group: /bec/workers
# Log Stream: something like {worker_name}/bec-worker

class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: Dict[str, Any] = {
            "ts": time.time(),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }

        for k, v in record.__dict__.items():
            if k in (
                "args", "msg", "levelname", "name", "pathname", "filename",
                "module", "exc_info", "exc_text", "stack_info", "lineno",
                "funcName", "created", "msecs", "relativeCreated", "thread",
                "threadName", "processName", "process"
            ):
                continue
            if k.startswith("_"):
                continue
            payload.setdefault(k, v)

        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)

        return json.dumps(payload, ensure_ascii=False)

def setup_logging(verbose: bool = False) -> None:
    """
    Setup logging with JSON formatting for CloudWatch.
    
    Args:
        verbose: If True, sets bec logger to DEBUG and root logger to INFO.
                 If False, uses environment variables or defaults (WARNING for root, INFO for bec).
    """
    if verbose:
        # Verbose mode: bec namespace at DEBUG, other loggers at INFO
        root_level = "INFO"
        app_level = "DEBUG"
    else:
        # Normal mode: use environment variables or defaults
        root_level = os.environ.get("BEC_ROOT_LOG_LEVEL", "WARNING").upper()
        app_level = os.environ.get("BEC_APP_LOG_LEVEL", "INFO").upper()

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())
    # Handler should not filter - let loggers control what gets through
    handler.setLevel(logging.DEBUG if verbose else logging.INFO)

    # Root logger: controls third-party libraries
    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(root_level)
    root.addHandler(handler)

    # Your application logger namespace
    app_logger = logging.getLogger("bec")
    app_logger.setLevel(app_level)
    app_logger.propagate = True  # still go to root handler
    
    # Timing/performance logger - ERROR by default (suppresses slow decode/wait warnings)
    # Can be set to WARNING/INFO via BEC_TIMINGS_LOG_LEVEL env var if needed
    timings_level = os.environ.get("BEC_TIMINGS_LOG_LEVEL", "ERROR").upper()
    timings_logger = logging.getLogger("bec_timings")
    timings_logger.setLevel(timings_level)
    timings_logger.propagate = True  # still go to root handler
    
    # Memory monitor logger - ERROR by default (suppresses periodic memory snapshots)
    # Can be set to INFO via BEC_MEMORY_LOG_LEVEL env var or --verbose flag for OOM debugging
    memory_level = os.environ.get("BEC_MEMORY_LOG_LEVEL", "ERROR").upper()
    if verbose:
        memory_level = "INFO"  # Enable memory logging in verbose mode
    memory_logger = logging.getLogger("bec_memory")
    memory_logger.setLevel(memory_level)
    memory_logger.propagate = True  # still go to root handler
    
    # Suppress third-party deprecation warnings we can't control
    # PyTorch's pynvml deprecation warning
    warnings.filterwarnings(
        "ignore",
        category=FutureWarning,
        message=".*pynvml package is deprecated.*",
    )