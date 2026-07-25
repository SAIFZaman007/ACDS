"""Filesystem safety boundary.

Every byte this application writes goes through here. The rules are:

* **Sandboxing** – resolved paths must remain inside the configured output
  root. Traversal (``../``), absolute escapes and symlinked directories that
  point outside the root are rejected.
* **Sanitised names** – session/file names are reduced to a conservative
  ``[A-Za-z0-9._-]`` alphabet, so nothing that reaches the filesystem can be
  shell-, path- or unicode-confusable.
* **Atomic writes** – artefacts are written to a temporary file in the same
  directory and then ``os.replace``'d, so a crash or a full disk can never
  leave a half-written PNG behind.
* **Least privilege** – session directories are created ``0o700``.
"""

from __future__ import annotations

import os
import re
import tempfile
from pathlib import Path

__all__ = [
    "PathSecurityError",
    "sanitize_name",
    "ensure_directory",
    "safe_join",
    "atomic_write_bytes",
    "atomic_write_text",
]

_UNSAFE = re.compile(r"[^A-Za-z0-9._-]+")
_MULTI_DOT = re.compile(r"\.{2,}")
_MULTI_DASH = re.compile(r"-{2,}")
MAX_NAME_LENGTH = 96


class PathSecurityError(RuntimeError):
    """Raised when a requested path would escape the sandbox."""


def sanitize_name(name: str, *, fallback: str = "session") -> str:
    """Reduce an arbitrary string to a safe single path component."""
    candidate = _UNSAFE.sub("-", name.strip()).strip("-._")
    candidate = _MULTI_DOT.sub(".", candidate)
    candidate = _MULTI_DASH.sub("-", candidate).strip("-._")
    candidate = candidate[:MAX_NAME_LENGTH]
    if not candidate or candidate in {".", ".."}:
        return fallback
    return candidate


def ensure_directory(path: Path, *, mode: int = 0o700) -> Path:
    """Create ``path`` (and parents) with restrictive permissions."""
    path.mkdir(parents=True, exist_ok=True)
    try:
        os.chmod(path, mode)
    except (OSError, NotImplementedError):  # pragma: no cover - platform dependent
        pass
    return path


def safe_join(root: Path, *parts: str) -> Path:
    """Join ``parts`` under ``root``, guaranteeing the result stays inside it.

    Raises:
        PathSecurityError: if the resolved path escapes ``root``.
    """
    if not parts:
        raise ValueError("safe_join requires at least one path component")
    root_resolved = root.resolve()
    cleaned = [sanitize_name(part, fallback="file") for part in parts]
    candidate = root_resolved.joinpath(*cleaned)
    resolved = Path(os.path.realpath(candidate))
    real_root = Path(os.path.realpath(root_resolved))
    if resolved != real_root and real_root not in resolved.parents:
        raise PathSecurityError(f"refusing to write outside sandbox: {candidate}")
    return resolved


def atomic_write_bytes(path: Path, payload: bytes, *, mode: int = 0o600) -> Path:
    """Write ``payload`` to ``path`` atomically."""
    ensure_directory(path.parent)
    fd, tmp_name = tempfile.mkstemp(dir=str(path.parent), prefix=".tmp-", suffix=path.suffix)
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(tmp_path, mode)
        os.replace(tmp_path, path)
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise
    return path


def atomic_write_text(path: Path, text: str, *, encoding: str = "utf-8") -> Path:
    return atomic_write_bytes(path, text.encode(encoding))


    #     *** _ ***