"""Gesture recognition.

Design notes
------------
1. **Rotation invariance.** Naive implementations decide a finger is
   "extended" by comparing ``tip.y < pip.y``. That breaks the moment the user
   tilts their hand. Instead we compare *radial distance from the wrist*:
   an extended finger's tip is farther from the wrist than its PIP joint,
   whatever the hand's orientation.

2. **Scale invariance.** All thresholds are expressed as multiples of the
   palm size (wrist -> middle MCP), so the gestures work near and far from
   the camera.

3. **Temporal stability.** Per-frame classification flickers. A k-of-n
   majority vote plus a cooldown converts noisy per-frame labels into a
   stable interaction state, and destructive actions additionally require a
   deliberate *hold*.
"""

from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass
from typing import Deque

import numpy as np

from .types import (
    FINGER_JOINTS,
    INDEX_MCP,
    PINKY_MCP,
    THUMB_IP,
    THUMB_TIP,
    WRIST,
    FingerState,
    Gesture,
    HandObservation,
)

__all__ = [
    "finger_states",
    "classify",
    "GestureStabilizer",
    "HoldTimer",
    "GestureEngine",
]


def _dist(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a[:2] - b[:2]))


def finger_states(hand: HandObservation, margin: float = 1.05) -> FingerState:
    """Determine which digits are extended.

    Args:
        hand: Observation in pixel space.
        margin: Multiplicative hysteresis (``>1``) that suppresses borderline
            flapping between extended and folded.
    """
    lm = hand.landmarks
    wrist = lm[WRIST]

    fingers: list[bool] = []
    for tip_idx, pip_idx in FINGER_JOINTS:
        fingers.append(_dist(lm[tip_idx], wrist) > _dist(lm[pip_idx], wrist) * margin)

    # The thumb folds *across* the palm, so radial distance from the wrist is a
    # poor signal. Distance from the pinky MCP separates the two poses cleanly.
    pinky_mcp = lm[PINKY_MCP]
    thumb = _dist(lm[THUMB_TIP], pinky_mcp) > _dist(lm[THUMB_IP], pinky_mcp) * margin

    return FingerState(thumb, fingers[0], fingers[1], fingers[2], fingers[3])


def _is_thumbs_up(hand: HandObservation, min_rise: float = 0.35) -> bool:
    """True when the thumb points clearly upwards above the rest of the hand."""
    lm = hand.landmarks
    if int(np.argmin(lm[:, 1])) != THUMB_TIP:
        return False  # some other landmark is higher -> not a thumbs-up
    rise = float(lm[INDEX_MCP, 1] - lm[THUMB_TIP, 1])
    return rise > min_rise * hand.palm_size


def classify(
    hand: HandObservation,
    *,
    margin: float = 1.05,
    thumb_rise: float = 0.35,
) -> tuple[Gesture, FingerState]:
    """Map a single hand observation to a gesture.

    The four-finger poses deliberately ignore the thumb: users naturally leave
    it splayed while pointing, and requiring an exact five-digit pattern makes
    the system feel brittle.
    """
    state = finger_states(hand, margin=margin)
    index, middle, ring, pinky = state.fingers

    # Open palm -> eraser (all four fingers out).
    if index and middle and ring and pinky:
        return Gesture.ERASE, state

    # Thumbs up -> save (fist + raised thumb).
    if state.thumb and not any(state.fingers) and _is_thumbs_up(hand, thumb_rise):
        return Gesture.SAVE, state

    # Index only -> pen down.
    if index and not middle and not ring and not pinky:
        return Gesture.DRAW, state

    # Index + middle -> hover cursor (used for toolbar selection).
    if index and middle and not ring and not pinky:
        return Gesture.SELECT, state

    return Gesture.NONE, state


class GestureStabilizer:
    """k-of-n majority vote over recent frames.

    Emits a new gesture only when it has been observed ``min_votes`` times
    within the last ``window`` frames, which removes single-frame flicker
    without adding perceptible latency.
    """

    __slots__ = ("_window", "_min_votes", "_buffer", "_current")

    def __init__(self, window: int = 5, min_votes: int = 3) -> None:
        if window < 1:
            raise ValueError("window must be >= 1")
        if not 1 <= min_votes <= window:
            raise ValueError("min_votes must be within [1, window]")
        self._window = window
        self._min_votes = min_votes
        self._buffer: Deque[Gesture] = deque(maxlen=window)
        self._current = Gesture.NONE

    @property
    def current(self) -> Gesture:
        return self._current

    def reset(self) -> None:
        self._buffer.clear()
        self._current = Gesture.NONE

    def update(self, observed: Gesture) -> Gesture:
        self._buffer.append(observed)
        gesture, votes = Counter(self._buffer).most_common(1)[0]
        if votes >= self._min_votes:
            self._current = gesture
        return self._current


@dataclass(slots=True)
class HoldTimer:
    """Requires a gesture to persist before firing, then enforces a cooldown.

    Used for destructive/irreversible actions (save, clear) so that a
    transient misclassification can never trigger them.
    """

    hold_seconds: float = 0.9
    cooldown_seconds: float = 2.0
    _started_at: float | None = None
    _last_fired_at: float | None = None

    def progress(self, now: float) -> float:
        """Completion ratio in ``[0, 1]`` for UI feedback."""
        if self._started_at is None or self.hold_seconds <= 0:
            return 0.0
        return min(1.0, (now - self._started_at) / self.hold_seconds)

    def reset(self) -> None:
        self._started_at = None

    def update(self, active: bool, now: float) -> bool:
        """Advance the timer; returns ``True`` exactly once per completed hold."""
        if not active:
            self._started_at = None
            return False
        if self._last_fired_at is not None and now - self._last_fired_at < self.cooldown_seconds:
            self._started_at = None
            return False
        if self._started_at is None:
            self._started_at = now
            return False
        if now - self._started_at >= self.hold_seconds:
            self._started_at = None
            self._last_fired_at = now
            return True
        return False


class GestureEngine:
    """Convenience façade combining classification and stabilisation."""

    def __init__(
        self,
        *,
        window: int = 5,
        min_votes: int = 3,
        margin: float = 1.05,
        thumb_rise: float = 0.35,
    ) -> None:
        self._stabilizer = GestureStabilizer(window=window, min_votes=min_votes)
        self._margin = margin
        self._thumb_rise = thumb_rise
        self.last_state: FingerState | None = None
        self.last_raw: Gesture = Gesture.NONE

    def reset(self) -> None:
        self._stabilizer.reset()
        self.last_state = None
        self.last_raw = Gesture.NONE

    def update(self, hand: HandObservation | None) -> Gesture:
        if hand is None:
            self.last_state = None
            self.last_raw = Gesture.NONE
            return self._stabilizer.update(Gesture.NONE)
        raw, state = classify(hand, margin=self._margin, thumb_rise=self._thumb_rise)
        self.last_state = state
        self.last_raw = raw
        return self._stabilizer.update(raw)
    
    
    
    #     *** _ ***