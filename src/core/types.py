"""Core domain types shared across the application.

This module is intentionally dependency-light (numpy only) so that gesture,
canvas and UI logic can be unit-tested without a camera or MediaPipe.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Final, Iterable

import numpy as np

# --------------------------------------------------------------------------- #
# MediaPipe hand landmark indices (21-point topology).
# --------------------------------------------------------------------------- #
WRIST: Final = 0
THUMB_CMC: Final = 1
THUMB_MCP: Final = 2
THUMB_IP: Final = 3
THUMB_TIP: Final = 4
INDEX_MCP: Final = 5
INDEX_PIP: Final = 6
INDEX_DIP: Final = 7
INDEX_TIP: Final = 8
MIDDLE_MCP: Final = 9
MIDDLE_PIP: Final = 10
MIDDLE_DIP: Final = 11
MIDDLE_TIP: Final = 12
RING_MCP: Final = 13
RING_PIP: Final = 14
RING_DIP: Final = 15
RING_TIP: Final = 16
PINKY_MCP: Final = 17
PINKY_PIP: Final = 18
PINKY_DIP: Final = 19
PINKY_TIP: Final = 20

NUM_LANDMARKS: Final = 21

#: (tip, pip) index pairs for the four non-thumb fingers, in canonical order.
FINGER_JOINTS: Final = (
    (INDEX_TIP, INDEX_PIP),
    (MIDDLE_TIP, MIDDLE_PIP),
    (RING_TIP, RING_PIP),
    (PINKY_TIP, PINKY_PIP),
)

PALM_POINTS: Final = (WRIST, INDEX_MCP, MIDDLE_MCP, RING_MCP, PINKY_MCP)


class Gesture(str, Enum):
    """Recognised interaction gestures."""

    NONE = "none"
    DRAW = "draw"      # index finger only -> pen down
    SELECT = "select"  # index + middle -> hover / toolbar cursor
    ERASE = "erase"    # open palm -> eraser
    SAVE = "save"      # thumbs up (held) -> export artefacts

    @property
    def label(self) -> str:
        return self.value.upper()


class ToolKind(str, Enum):
    COLOR = "color"
    SIZE = "size"
    ACTION = "action"


@dataclass(frozen=True, slots=True)
class Point:
    """Integer pixel coordinate."""

    x: int
    y: int

    def as_tuple(self) -> tuple[int, int]:
        return (self.x, self.y)

    def distance_to(self, other: "Point") -> float:
        return float(np.hypot(self.x - other.x, self.y - other.y))


@dataclass(frozen=True, slots=True)
class FingerState:
    """Boolean extension state for each digit."""

    thumb: bool
    index: bool
    middle: bool
    ring: bool
    pinky: bool

    @property
    def fingers(self) -> tuple[bool, bool, bool, bool]:
        """The four non-thumb fingers, index -> pinky."""
        return (self.index, self.middle, self.ring, self.pinky)

    @property
    def extended_count(self) -> int:
        return sum((self.thumb, self.index, self.middle, self.ring, self.pinky))

    def as_tuple(self) -> tuple[bool, bool, bool, bool, bool]:
        return (self.thumb, self.index, self.middle, self.ring, self.pinky)


@dataclass(frozen=True, slots=True)
class HandObservation:
    """A single detected hand, expressed in **pixel** coordinates."""

    landmarks: np.ndarray  # shape (21, 3): x, y in px, z relative depth
    handedness: str = "Unknown"
    score: float = 0.0

    def __post_init__(self) -> None:
        if self.landmarks.shape != (NUM_LANDMARKS, 3):
            raise ValueError(
                f"landmarks must have shape ({NUM_LANDMARKS}, 3), "
                f"got {self.landmarks.shape}"
            )

    def point(self, index: int) -> Point:
        x, y = self.landmarks[index, 0], self.landmarks[index, 1]
        return Point(int(round(float(x))), int(round(float(y))))

    @property
    def index_tip(self) -> Point:
        return self.point(INDEX_TIP)

    @property
    def palm_center(self) -> Point:
        pts = self.landmarks[list(PALM_POINTS), :2].mean(axis=0)
        return Point(int(round(float(pts[0]))), int(round(float(pts[1]))))

    @property
    def palm_size(self) -> float:
        """Wrist -> middle-MCP distance; a scale-invariant reference length."""
        delta = self.landmarks[MIDDLE_MCP, :2] - self.landmarks[WRIST, :2]
        return float(np.linalg.norm(delta)) or 1.0


@dataclass(slots=True)
class Stroke:
    """A vector stroke; the canvas is rasterised from these."""

    color: tuple[int, int, int]  # BGR
    thickness: int
    erase: bool = False
    points: list[tuple[int, int]] = field(default_factory=list)

    def add(self, point: Point | tuple[int, int]) -> None:
        pt = point.as_tuple() if isinstance(point, Point) else tuple(point)
        if self.points and self.points[-1] == pt:
            return
        self.points.append((int(pt[0]), int(pt[1])))

    def to_dict(self) -> dict:
        return {
            "color": list(self.color),
            "thickness": self.thickness,
            "erase": self.erase,
            "points": self.points,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Stroke":
        return cls(
            color=tuple(int(c) for c in data["color"]),  # type: ignore[arg-type]
            thickness=int(data["thickness"]),
            erase=bool(data.get("erase", False)),
            points=[(int(x), int(y)) for x, y in data.get("points", [])],
        )


def landmarks_from_normalized(
    normalized: Iterable[tuple[float, float, float]],
    width: int,
    height: int,
) -> np.ndarray:
    """Convert normalised MediaPipe landmarks to a pixel-space ``(21, 3)`` array."""
    arr = np.asarray(list(normalized), dtype=np.float32)
    if arr.shape != (NUM_LANDMARKS, 3):
        raise ValueError(f"expected ({NUM_LANDMARKS}, 3) landmarks, got {arr.shape}")
    scaled = arr.copy()
    scaled[:, 0] *= width
    scaled[:, 1] *= height
    scaled[:, 2] *= width  # z uses the same scale as x by MediaPipe convention
    return scaled



    #     *** _ ***