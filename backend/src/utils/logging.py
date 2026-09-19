"""Structured logging setup."""
import json
import logging
import sys
from typing import Any


class StructuredFormatter(logging.Formatter):
    """Format logs as JSON with request context."""

    def format(self, record: logging.LogRecord) -> str:
        """Format log record as structured JSON."""
        log_obj: dict[str, Any] = {
            "timestamp": self.formatTime(record),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        # Add extra fields from the record
        if hasattr(record, "request_id"):
            log_obj["request_id"] = record.request_id

        for key, value in record.__dict__.items():
            if key not in (
                "name",
                "msg",
                "args",
                "created",
                "filename",
                "funcName",
                "levelname",
                "levelno",
                "lineno",
                "module",
                "msecs",
                "message",
                "pathname",
                "process",
                "processName",
                "relativeCreated",
                "thread",
                "threadName",
                "exc_info",
                "exc_text",
                "stack_info",
            ):
                if not key.startswith("_"):
                    log_obj[key] = value

        return json.dumps(log_obj)


def get_logger(name: str) -> logging.Logger:
    """Get a logger with structured formatting."""
    logger = logging.getLogger(name)

    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        from src.config.settings import settings

        level = getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO)
        logger.setLevel(level)
        handler.setLevel(level)

        formatter = StructuredFormatter()
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger
