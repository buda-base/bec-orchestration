import json
import logging
import os
import sys
import time
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

def setup_logging() -> None:
    root_level = os.environ.get("BEC_ROOT_LOG_LEVEL", "WARNING").upper()
    app_level = os.environ.get("BEC_APP_LOG_LEVEL", "INFO").upper()

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())

    # Root logger: controls third-party libraries
    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(root_level)
    handler.setLevel(root_level)
    root.addHandler(handler)

    # Your application logger namespace
    app_logger = logging.getLogger("bec")
    app_logger.setLevel(app_level)
    app_logger.propagate = True  # still go to root handler
