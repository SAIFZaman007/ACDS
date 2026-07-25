"""Regression tests for the frame-source liveness contract.

The original bug: ``ThreadedCameraSource.read()`` returned ``None`` whenever the
consumer outran the sensor, and ``AirCanvasApp`` treats ``None`` as
end-of-stream. On any machine where the pipeline is faster than the camera —
i.e. every healthy machine — the session died after one or two frames.

These tests pin the contract that fixes it: ``None`` means the stream is over,
and nothing else.
"""

from __future__ import annotations

import threading
import time

import numpy as np
import pytest

from src.adapters.capture import ThreadedCameraSource


class FakeCapture:
    """Stand-in for ``cv2.VideoCapture`` that emits frames at a fixed rate."""

    def __init__(self, period: float = 1 / 30.0, total: int | None = None) -> None:
        self.period = period
        self.total = total
        self.served = 0
        self.released = False
        self._props: dict[int, float] = {}

    def isOpened(self) -> bool:  # noqa: N802 - mirrors the OpenCV API
        return not self.released

    def set(self, prop: int, value: float) -> bool:  # noqa: A003
        self._props[prop] = value
        return True

    def get(self, prop: int) -> float:
        return self._props.get(prop, 0.0)

    def read(self):
        if self.total is not None and self.served >= self.total:
            return False, None
        time.sleep(self.period)
        self.served += 1
        return True, np.full((4, 4, 3), self.served % 256, dtype=np.uint8)

    def release(self) -> None:
        self.released = True


@pytest.fixture()
def source(monkeypatch):
    """A ThreadedCameraSource backed by FakeCapture, with cleanup."""
    made: list[ThreadedCameraSource] = []

    def build(period: float = 1 / 60.0, total: int | None = None) -> ThreadedCameraSource:
        monkeypatch.setattr(
            "src.adapters.capture.cv2.VideoCapture",
            lambda *a, **k: FakeCapture(period=period, total=total),
        )
        src = ThreadedCameraSource(0, width=4, height=4, fps=60.0, warmup_seconds=5.0)
        made.append(src)
        return src

    yield build
    for src in made:
        src.release()


def test_fast_consumer_does_not_end_the_stream(source):
    """The bug, stated as a test: a consumer faster than the camera must not EOF.

    Previously this returned None on the second read and the app logged
    "Source exhausted" after a single frame.
    """
    src = source(period=1 / 30.0)  # camera ~30 fps, consumer as fast as possible
    frames = [src.read(timeout=2.0) for _ in range(10)]
    assert all(f is not None for f in frames), "read() reported EOF on a healthy camera"
    assert len(frames) == 10


def test_read_blocks_rather_than_returning_none(source):
    """A read issued before the next frame exists must wait for it, not bail."""
    src = source(period=0.15)
    src.read(timeout=2.0)  # drain whatever warmup produced
    start = time.monotonic()
    frame = src.read(timeout=2.0)
    waited = time.monotonic() - start
    assert frame is not None
    assert waited >= 0.05, "read() returned instantly; it did not wait for a fresh frame"


def test_consecutive_reads_never_repeat_a_frame(source):
    """Freshness guarantee: the same frame is never handed out twice."""
    src = source(period=1 / 50.0)
    seen = [src.read(timeout=2.0)[0, 0, 0] for _ in range(6)]
    assert all(a != b for a, b in zip(seen, seen[1:])), f"duplicate frames served: {seen}"


def test_genuine_end_of_stream_still_returns_none(source):
    """The other half of the contract: a dead camera must terminate the loop."""
    src = source(period=1 / 60.0, total=3)
    drained = 0
    for _ in range(25):
        if src.read(timeout=1.5) is None:
            break
        drained += 1
    else:
        pytest.fail("read() never reported end-of-stream on a finished camera")
    assert drained >= 1


def test_release_unblocks_a_waiting_reader(source):
    """Shutdown must never deadlock a consumer parked inside read()."""
    src = source(period=1 / 60.0)
    src._capture.period = 5.0  # warmup succeeded; now the camera goes silent
    while src.read(timeout=0.3) is not None:
        pass  # drain anything already buffered
    result: list = []

    def consume() -> None:
        result.append(src.read(timeout=10.0))

    reader = threading.Thread(target=consume, daemon=True)
    reader.start()
    time.sleep(0.1)
    src.release()
    reader.join(timeout=3.0)
    assert not reader.is_alive(), "release() left a reader blocked"
    assert result == [None]


def test_is_live_flag_distinguishes_camera_from_file(source):
    """The app relies on this to tell a hiccup apart from a finished file."""
    from src.adapters.capture import VideoFileSource

    assert source().is_live is True
    assert VideoFileSource.is_live is False