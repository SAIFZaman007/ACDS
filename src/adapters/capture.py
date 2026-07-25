"""Frame acquisition.

Two sources implement the same :class:`FrameSource` protocol:

``ThreadedCameraSource``
    A live camera. A background thread continuously grabs frames and keeps
    only the newest one. Without this, OpenCV's internal buffer makes the
    cursor lag behind the hand by several frames whenever inference is
    slower than the sensor — the single biggest cause of "sluggish" feel in
    naive implementations.

``VideoFileSource``
    Deterministic playback for tests, demos and CI. Every frame is delivered
    in order; nothing is dropped.
"""

from __future__ import annotations

import logging
import platform
import threading
import time
from typing import Protocol, runtime_checkable

import cv2
import numpy as np

log = logging.getLogger(__name__)

__all__ = ["FrameSource", "ThreadedCameraSource", "VideoFileSource", "open_source", "probe_cameras"]


class CaptureError(RuntimeError):
    """Raised when a capture device or file cannot be opened."""


@runtime_checkable
class FrameSource(Protocol):
    """Minimal contract required by the application loop."""

    width: int
    height: int
    fps: float
    #: ``True`` for a live sensor (a momentary gap is normal and recoverable),
    #: ``False`` for a finite file (a gap means the stream really has ended).
    is_live: bool

    def read(self, timeout: float | None = None) -> np.ndarray | None:
        """Block until the next BGR frame is ready.

        Returns ``None`` **only** at genuine end-of-stream, never merely
        because a fresh frame has not arrived yet.
        """

    def release(self) -> None:
        """Release all underlying resources."""


def _backend_flag(name: str) -> int:
    """Map a backend name to an OpenCV capture API preference."""
    explicit = {
        "any": cv2.CAP_ANY,
        "v4l2": getattr(cv2, "CAP_V4L2", cv2.CAP_ANY),
        "dshow": getattr(cv2, "CAP_DSHOW", cv2.CAP_ANY),
        "msmf": getattr(cv2, "CAP_MSMF", cv2.CAP_ANY),
        "avfoundation": getattr(cv2, "CAP_AVFOUNDATION", cv2.CAP_ANY),
        "gstreamer": getattr(cv2, "CAP_GSTREAMER", cv2.CAP_ANY),
    }
    if name != "auto":
        if name not in explicit:
            raise ValueError(f"unknown capture backend: {name!r}")
        return explicit[name]
    system = platform.system()
    if system == "Windows":
        return explicit["dshow"]
    if system == "Darwin":
        return explicit["avfoundation"]
    if system == "Linux":
        return explicit["v4l2"]
    return cv2.CAP_ANY


class ThreadedCameraSource:
    """Low-latency live camera reader (always yields the freshest frame).

    The reader thread keeps only the newest frame; the consumer blocks on a
    condition variable until one is published. Crucially, an empty slot means
    "not yet", not "never again" — only a dead reader thread or a stall longer
    than ``read_timeout`` ends the stream.
    """

    is_live = True

    def __init__(
        self,
        index: int = 0,
        *,
        width: int = 1280,
        height: int = 720,
        fps: float = 30.0,
        backend: str = "auto",
        warmup_seconds: float = 5.0,
        read_timeout: float = 5.0,
    ) -> None:
        if index < 0:
            raise ValueError("camera index must be >= 0")
        self._capture = cv2.VideoCapture(index, _backend_flag(backend))
        if not self._capture.isOpened():
            raise CaptureError(
                f"could not open camera index {index}. "
                "Check that the device exists and no other application is using it."
            )
        self._capture.set(cv2.CAP_PROP_FRAME_WIDTH, float(width))
        self._capture.set(cv2.CAP_PROP_FRAME_HEIGHT, float(height))
        self._capture.set(cv2.CAP_PROP_FPS, float(fps))
        # A shallow driver buffer is what actually keeps latency low.
        self._capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        self.width = int(self._capture.get(cv2.CAP_PROP_FRAME_WIDTH)) or width
        self.height = int(self._capture.get(cv2.CAP_PROP_FRAME_HEIGHT)) or height
        self.fps = float(self._capture.get(cv2.CAP_PROP_FPS)) or fps
        self.read_timeout = float(read_timeout)

        self._frame: np.ndarray | None = None
        # One condition guards both the frame slot and the finished flag, so a
        # consumer can await "new frame OR stream ended" in a single wait().
        self._cond = threading.Condition()
        self._finished = False
        self._stopped = threading.Event()
        self._thread = threading.Thread(target=self._pump, name="camera-reader", daemon=True)
        self._thread.start()
        self._await_first_frame(warmup_seconds)
        log.info(
            "Camera %s opened at %dx%d @ %.1f fps", index, self.width, self.height, self.fps
        )

    def _await_first_frame(self, timeout: float) -> None:
        deadline = time.monotonic() + timeout
        with self._cond:
            while self._frame is None and not self._finished:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                self._cond.wait(remaining)
            arrived = self._frame is not None
        if arrived:
            return
        self.release()
        raise CaptureError("camera opened but delivered no frames (is it in use or blocked?)")

    def _pump(self) -> None:
        failures = 0
        try:
            while not self._stopped.is_set():
                ok, frame = self._capture.read()
                if not ok or frame is None:
                    failures += 1
                    if failures > 60:
                        log.error("Camera stopped delivering frames; shutting down reader")
                        break
                    time.sleep(0.01)
                    continue
                failures = 0
                with self._cond:
                    self._frame = frame
                    self._cond.notify_all()  # wake the consumer the instant it lands
        finally:
            # Never leave a consumer blocked forever, whatever went wrong.
            with self._cond:
                self._finished = True
                self._cond.notify_all()

    def read(self, timeout: float | None = None) -> np.ndarray | None:
        """Block until a *fresh* frame is available.

        Returns ``None`` only when the reader thread has stopped (genuine
        end-of-stream) or the sensor stalls past ``timeout``. An empty slot on
        a healthy camera simply means the consumer outran the sensor — the
        normal case on a fast machine — and is waited out, not reported as EOF.
        """
        limit = self.read_timeout if timeout is None else float(timeout)
        deadline = time.monotonic() + limit
        with self._cond:
            while self._frame is None:
                if self._finished:
                    return None  # reader thread gone: genuine end of stream
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    log.warning("Camera delivered no frame within %.1fs", limit)
                    return None  # genuine stall
                self._cond.wait(remaining)
            frame = self._frame
            self._frame = None  # never re-process an identical frame
            return frame

    def release(self) -> None:
        self._stopped.set()
        with self._cond:
            self._finished = True
            self._cond.notify_all()
        if self._thread.is_alive():
            self._thread.join(timeout=2.0)
        self._capture.release()


class VideoFileSource:
    """Deterministic frame-by-frame reader for a video file."""

    is_live = False

    def __init__(self, path: str, *, loop: bool = False, realtime: bool = False) -> None:
        self._capture = cv2.VideoCapture(path)
        if not self._capture.isOpened():
            raise CaptureError(f"could not open video file: {path}")
        self.path = path
        self.loop = loop
        self.realtime = realtime
        self.width = int(self._capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self._capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = float(self._capture.get(cv2.CAP_PROP_FPS)) or 30.0
        self._next_deadline = time.monotonic()

    def read(self, timeout: float | None = None) -> np.ndarray | None:
        del timeout  # a file is never "late"; it either has a frame or is done
        if self.realtime:
            delay = self._next_deadline - time.monotonic()
            if delay > 0:
                time.sleep(delay)
            self._next_deadline = time.monotonic() + 1.0 / max(self.fps, 1.0)
        ok, frame = self._capture.read()
        if not ok or frame is None:
            if self.loop:
                self._capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ok, frame = self._capture.read()
                if ok and frame is not None:
                    return frame
            return None
        return frame

    def release(self) -> None:
        self._capture.release()


def open_source(
    source: int | str,
    *,
    width: int = 1280,
    height: int = 720,
    fps: float = 30.0,
    backend: str = "auto",
    loop: bool = False,
    realtime: bool = False,
) -> FrameSource:
    """Factory returning the right :class:`FrameSource` for ``source``."""
    if isinstance(source, str) and source.isdigit():
        source = int(source)
    if isinstance(source, int):
        return ThreadedCameraSource(
            source, width=width, height=height, fps=fps, backend=backend
        )
    return VideoFileSource(str(source), loop=loop, realtime=realtime)


def probe_cameras(max_index: int = 6, backend: str = "auto") -> list[dict]:
    """Enumerate usable camera indices (best-effort; used by ``--list-cameras``)."""
    found: list[dict] = []
    flag = _backend_flag(backend)
    for index in range(max_index):
        capture = cv2.VideoCapture(index, flag)
        try:
            if not capture.isOpened():
                continue
            ok, frame = capture.read()
            if not ok or frame is None:
                continue
            found.append(
                {
                    "index": index,
                    "width": int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
                    "height": int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                    "fps": round(float(capture.get(cv2.CAP_PROP_FPS)), 1),
                }
            )
        finally:
            capture.release()
    return found


    #     *** _ ***