"""Shared fixtures.

``make_hand`` synthesises anatomically plausible 21-point landmark sets so
that gesture logic can be tested exhaustively without a camera, a model or a
human hand. Folded fingers curl their tip back towards the wrist, which is
what the radial-distance heuristic keys on.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.core.types import HandObservation

# Lateral offsets of each finger's MCP joint, thumb side first.
_COLUMNS = {"thumb": -0.62, "index": -0.30, "middle": 0.0, "ring": 0.30, "pinky": 0.62}


def _finger_chain(column: float, extended: bool) -> list[tuple[float, float]]:
    """Return (mcp, pip, dip, tip) in wrist-relative units, y negative = up."""
    mcp = (column, -1.0)
    if extended:
        return [mcp, (column, -1.50), (column, -1.85), (column, -2.15)]
    # Folded: the tip curls back towards the palm, closer to the wrist than the PIP.
    return [mcp, (column, -1.42), (column, -1.20), (column, -0.92)]


def _thumb_chain(extended: bool, pointing_up: bool) -> list[tuple[float, float]]:
    """Return (cmc, mcp, ip, tip)."""
    if not extended:
        return [(-0.40, -0.45), (-0.62, -0.72), (-0.45, -0.86), (-0.02, -0.92)]
    if pointing_up:
        return [(-0.34, -0.40), (-0.22, -0.90), (-0.10, -1.45), (0.0, -2.05)]
    return [(-0.40, -0.45), (-0.78, -0.80), (-1.02, -1.02), (-1.30, -1.22)]


def make_hand(
    *,
    thumb: bool = False,
    index: bool = False,
    middle: bool = False,
    ring: bool = False,
    pinky: bool = False,
    thumb_up: bool = False,
    center: tuple[float, float] = (640.0, 620.0),
    scale: float = 120.0,
    rotation_degrees: float = 0.0,
    handedness: str = "Right",
    score: float = 0.95,
) -> HandObservation:
    """Build a :class:`HandObservation` with the requested digits extended."""
    points: list[tuple[float, float]] = [(0.0, 0.0)]  # wrist
    points += _thumb_chain(thumb, thumb_up)
    for name in ("index", "middle", "ring", "pinky"):
        extended = {"index": index, "middle": middle, "ring": ring, "pinky": pinky}[name]
        points += _finger_chain(_COLUMNS[name], extended)

    local = np.asarray(points, dtype=np.float32)
    theta = np.deg2rad(rotation_degrees)
    rotation = np.array(
        [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]], dtype=np.float32
    )
    pixels = local @ rotation.T * scale + np.asarray(center, dtype=np.float32)

    landmarks = np.zeros((21, 3), dtype=np.float32)
    landmarks[:, :2] = pixels
    return HandObservation(landmarks=landmarks, handedness=handedness, score=score)


@pytest.fixture
def hand_factory():
    return make_hand


@pytest.fixture
def blank_frame() -> np.ndarray:
    return np.full((360, 640, 3), 24, dtype=np.uint8)






    #     *** _ ***