"""Structured logging configuration for DeepResearch-Agent.

Provides:
- request_id propagation via contextvars
- JSON-formatted logs for production (parsable by log aggregators)
- Plain-text formatted logs for development (readable)
"""

import json
import logging
import sys
from contextvars import ContextVar
from datetime import datetime, timezone

# Context variable for request_id — flows through async tasks automatically
request_id_var: ContextVar[str] = ContextVar("request_id", default="")
task_id_var: ContextVar[str] = ContextVar("task_id", default="")


def get_request_id() -> str:
    """Get the current request_id from context."""
    return request_id_var.get()


def set_request_id(request_id: str) -> None:
    """Set request_id in the current context."""
    request_id_var.set(request_id)


def get_task_id() -> str:
    """Get the current task_id from context."""
    return task_id_var.get()


class StructuredFormatter(logging.Formatter):
    """Log formatter that produces JSON lines when LOG_FORMAT=json.

    In JSON mode each log line is a parsable JSON object with
    timestamp, level, logger, message, and context fields.
    Falls back to a readable text format for local development.
    """

    def __init__(self):
        super().__init__()
        self._json_mode = False

    def format(self, record: logging.LogRecord) -> str:
        rid = get_request_id()
        tid = get_task_id()
        ts = datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat()

        # JSON mode for production
        if self._json_mode:
            fields = {
                "timestamp": ts,
                "level": record.levelname,
                "logger": record.name,
                "message": record.getMessage(),
            }
            if rid:
                fields["request_id"] = rid
            if tid:
                fields["task_id"] = tid
            if record.exc_info and record.exc_info[0]:
                fields["exception"] = self.formatException(record.exc_info)
            return json.dumps(fields, ensure_ascii=False)

        # Text mode for development
        parts = [
            ts,
            f"{record.levelname:<8}",
            f"{record.name}:{record.lineno}",
        ]
        if rid:
            parts.append(f"[req={rid[:8]}]")
        parts.append(record.getMessage())
        return " | ".join(parts)


def setup_logging(level: str = "INFO") -> None:
    """Configure structured logging for the application.

    Sets up a consistent logging format with timestamps, log levels,
    module names, and request_id tracing. Reads LOG_FORMAT env var;
    set to "json" for structured JSON output.

    Args:
        level: Log level string (DEBUG, INFO, WARNING, ERROR, CRITICAL).
    """
    import os

    formatter = StructuredFormatter()
    formatter._json_mode = os.environ.get("LOG_FORMAT", "").lower() == "json"

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)

    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    # Remove default handlers and add ours
    for h in root_logger.handlers[:]:
        root_logger.removeHandler(h)
    root_logger.addHandler(handler)
