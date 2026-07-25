"""
Session video recording.

Encoding happens on a **background thread** behind a bounded queue. If the
encoder momentarily falls behind, frames are dropped and counted rather than
stalling the interaction loop — a dropped frame is invisible, a stalled
cursor is not.
"""

from __future__ import annotations

import logging
import os
import queue
import threading
import time
from enum import Enum
from pathlib import Path

import cv2
import numpy as np

from src.core.canvas import canvas_on_background

log = logging.getLogger(__name__)

__all__ = ["Layout", "compose_output", "output_size", "VideoRecorder"]

_SENTINEL: object = object()

#: Ordered codec/extension candidates. The first that opens wins.
_CODECS: tuple[tuple[str, str], ...] = (
    ("mp4v", ".mp4"),
    ("avc1", ".mp4"),
    ("XVID", ".avi"),
    ("MJPG", ".avi"),
)


class Layout(str, Enum):
    OVERLAY = "overlay"
    SIDE_BY_SIDE = "side_by_side"
    PIP = "pip"
    CANVAS_ONLY = "canvas_only"


def output_size(layout: Layout | str, width: int, height: int) -> tuple[int, int]:
    """Frame size produced by :func:`compose_output` for a layout."""
    return (width * 2, height) if Layout(layout) is Layout.SIDE_BY_SIDE else (width, height)


def compose_output(
    view_bgr: np.ndarray,
    canvas_bgra: np.ndarray,
    layout: Layout | str = Layout.OVERLAY,
    *,
    paper: tuple[int, int, int] = (255, 255, 255),
    pip_scale: float = 0.26,
) -> np.ndarray:
    """Build the recorded frame for a given layout.

    ``view_bgr`` is the camera pane exactly as the user sees it (drawing
    already composited, HUD optionally included). Keeping this a pure
    function of its inputs makes every layout unit-testable.
    """
    layout = Layout(layout)
    height, width = view_bgr.shape[:2]
    if canvas_bgra.shape[:2] != (height, width):
        raise ValueError("canvas and frame dimensions must match")

    if layout is Layout.OVERLAY:
        return view_bgr
    if layout is Layout.CANVAS_ONLY:
        return canvas_on_background(canvas_bgra, paper)
    if layout is Layout.SIDE_BY_SIDE:
        return np.hstack([canvas_on_background(canvas_bgra, paper), view_bgr])

    # Picture-in-picture: artwork full-frame, camera inset bottom-right.
    board = canvas_on_background(canvas_bgra, paper)
    inset_w = max(96, int(width * pip_scale))
    inset_h = max(72, int(height * pip_scale))
    inset = cv2.resize(view_bgr, (inset_w, inset_h), interpolation=cv2.INTER_AREA)
    margin = 16
    y0, x0 = height - inset_h - margin, width - inset_w - margin
    board[y0 : y0 + inset_h, x0 : x0 + inset_w] = inset
    cv2.rectangle(board, (x0 - 2, y0 - 2), (x0 + inset_w + 1, y0 + inset_h + 1), (40, 40, 40), 2)
    return board


class VideoRecorder:
    """Asynchronous, crash-tolerant video writer."""

    def __init__(
        self,
        path: Path,
        *,
        fps: float = 30.0,
        frame_size: tuple[int, int],
        codec: str = "mp4v",
        queue_size: int = 64,
        max_seconds: float | None = 1800.0,
    ) -> None:
        self.requested_path = Path(path)
        self.fps = max(1.0, float(fps))
        self.frame_size = (int(frame_size[0]), int(frame_size[1]))
        self.max_seconds = max_seconds
        self.dropped = 0
        self.written = 0
        self.started_at = time.monotonic()

        self.path, self._writer = self._open_writer(codec)
        self._queue: queue.Queue = queue.Queue(maxsize=max(4, int(queue_size)))
        self._stopped = threading.Event()
        self._thread = threading.Thread(target=self._drain, name="video-writer", daemon=True)
        self._thread.start()
        log.info("Recording to %s (%dx%d @ %.1f fps)", self.path, *self.frame_size, self.fps)

    # ------------------------------------------------------------------ #
    def _open_writer(self, preferred: str) -> tuple[Path, cv2.VideoWriter]:
        candidates: list[tuple[str, str]] = []
        for fourcc, ext in _CODECS:
            if fourcc.lower() == preferred.lower():
                candidates.insert(0, (fourcc, ext))
            else:
                candidates.append((fourcc, ext))

        errors: list[str] = []
        for fourcc, ext in candidates:
            path = self.requested_path.with_suffix(ext)
            writer = cv2.VideoWriter(
                str(path), cv2.VideoWriter_fourcc(*fourcc), self.fps, self.frame_size
            )
            if writer.isOpened():
                if fourcc.lower() != preferred.lower():
                    log.warning("Codec %r unavailable; using %r instead", preferred, fourcc)
                return path, writer
            writer.release()
            errors.append(fourcc)
        raise RuntimeError(
            f"no usable video codec found (tried: {', '.join(errors)}). "
            "Install a full OpenCV/FFmpeg build or pass --no-record."
        )

    # ------------------------------------------------------------------ #
    @property
    def elapsed(self) -> float:
        return time.monotonic() - self.started_at

    @property
    def exhausted(self) -> bool:
        return self.max_seconds is not None and self.elapsed >= self.max_seconds

    def write(self, frame: np.ndarray) -> bool:
        """Queue a frame. Returns ``False`` if it was dropped."""
        if self._stopped.is_set():
            return False
        if frame.shape[1::-1] != self.frame_size:
            frame = cv2.resize(frame, self.frame_size, interpolation=cv2.INTER_AREA)
        try:
            self._queue.put_nowait(frame)
            return True
        except queue.Full:
            self.dropped += 1
            if self.dropped in (1, 10, 100) or self.dropped % 500 == 0:
                log.warning("Encoder is behind; dropped %d frame(s)", self.dropped)
            return False

    def _drain(self) -> None:
        while True:
            item = self._queue.get()
            if item is _SENTINEL:
                break
            try:
                self._writer.write(item)
                self.written += 1
            except Exception:  # pragma: no cover - defensive
                log.exception("Video writer failed; stopping recording")
                break
            finally:
                self._queue.task_done()

    def close(self) -> None:
        """Flush the queue and finalise the container."""
        if self._stopped.is_set():
            return
        self._stopped.set()
        self._queue.put(_SENTINEL)
        self._thread.join(timeout=10.0)
        self._writer.release()
        try:  # OpenCV creates the container with the process umask; tighten it.
            os.chmod(self.path, 0o600)
        except OSError:  # pragma: no cover - platform dependent
            log.debug("Could not tighten permissions on %s", self.path)
        log.info(
            "Recording finalised: %s (%d frames written, %d dropped)",
            self.path, self.written, self.dropped,
        )

    def __enter__(self) -> "VideoRecorder":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
        
        
        
    #     *** _ ***