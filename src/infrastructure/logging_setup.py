"""Logging configuration.

Human-readable on the console, optionally line-delimited JSON for shipping
into a log aggregator, and always a rotating file inside the session
directory so a support engineer can reconstruct what happened.
"""

from __future__ import annotations

import json
import logging
import logging.handlers
import os
import sys
from pathlib import Path

__all__ = ["configure_logging", "JsonFormatter"]

_CONSOLE_FORMAT = "%(asctime)s %(levelname)-7s %(name)-22s %(message)s"
_DATE_FORMAT = "%H:%M:%S"

_RESERVED = frozenset(logging.LogRecord("", 0, "", 0, "", (), None).__dict__) | {
    "message", "asctime", "taskName",
}


class JsonFormatter(logging.Formatter):
    """Minimal structured formatter (no third-party dependency)."""

    def __init__(self, session_id: str | None = None) -> None:
        super().__init__()
        self.session_id = session_id

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, object] = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S%z"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if self.session_id:
            payload["session_id"] = self.session_id
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        for key, value in record.__dict__.items():
            if key not in _RESERVED and not key.startswith("_"):
                payload[key] = value
        return json.dumps(payload, default=str)


def configure_logging(
    level: str = "INFO",
    *,
    json_output: bool = False,
    log_file: Path | None = None,
    session_id: str | None = None,
) -> None:
    """Install handlers on the root logger (idempotent)."""
    root = logging.getLogger()
    for handler in list(root.handlers):
        root.removeHandler(handler)
        handler.close()
    root.setLevel(getattr(logging, level.upper(), logging.INFO))

    console = logging.StreamHandler(stream=sys.stderr)
    console.setFormatter(
        JsonFormatter(session_id)
        if json_output
        else logging.Formatter(_CONSOLE_FORMAT, datefmt=_DATE_FORMAT)
    )
    root.addHandler(console)

    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        rotating = logging.handlers.RotatingFileHandler(
            log_file, maxBytes=2_000_000, backupCount=3, encoding="utf-8"
        )
        rotating.setFormatter(JsonFormatter(session_id) if json_output else
                              logging.Formatter(_CONSOLE_FORMAT, datefmt=_DATE_FORMAT))
        root.addHandler(rotating)
        try:  # logs can contain paths and device details: keep them owner-only
            os.chmod(log_file, 0o600)
        except OSError:  # pragma: no cover - platform dependent
            pass

    # These libraries are chatty at INFO and drown out our own messages.
    logging.getLogger("mediapipe").setLevel(logging.WARNING)
    logging.getLogger("absl").setLevel(logging.ERROR)
    
    
    
    
    
    
        #     *** _ ***