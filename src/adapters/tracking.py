"""
Hand tracking.

MediaPipe is isolated behind the :class:`HandTracker` protocol and imported
lazily. Two consequences matter in practice:

* every other module (gestures, canvas, UI, recording) is unit-testable
  without the ~100 MB dependency or a camera, and
* swapping in a different backend later (MediaPipe Tasks API, an ONNX model,
  a remote inference service) is a single-file change with no call-site churn.
"""

from __future__ import annotations

import logging
from typing import Protocol, runtime_checkable

import numpy as np

from src.core.types import HandObservation, landmarks_from_normalized

log = logging.getLogger(__name__)

__all__ = ["HandTracker", "MediaPipeHandTracker", "NullHandTracker", "create_tracker"]


@runtime_checkable
class HandTracker(Protocol):
    """Contract for any hand-landmark backend."""

    def process(self, frame_bgr: np.ndarray) -> list[HandObservation]:
        """Return the hands detected in a BGR frame, best first."""

    def close(self) -> None:
        """Release backend resources."""


class NullHandTracker:
    """A tracker that never detects a hand.

    Used by CI and ``--tracker none`` to exercise the capture -> compositing
    -> recording pipeline deterministically, with no model and no camera.
    """

    def process(self, frame_bgr: np.ndarray) -> list[HandObservation]:  # noqa: ARG002
        return []

    def close(self) -> None:
        return None


class MediaPipeHandTracker:
    """Adapter over ``mediapipe.solutions.hands``."""

    def __init__(
        self,
        *,
        max_hands: int = 1,
        detection_confidence: float = 0.6,
        tracking_confidence: float = 0.6,
        model_complexity: int = 1,
    ) -> None:
        try:
            import mediapipe as mp  # noqa: PLC0415  (deliberately lazy)
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise RuntimeError(
                "MediaPipe is not installed. Install it with "
                "`pip install 'mediapipe>=0.10.9,<0.11'`, or run with "
                "`--tracker none` to test the pipeline without hand tracking."
            ) from exc

        self._mp = mp
        self._hands = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=max(1, int(max_hands)),
            model_complexity=int(model_complexity),
            min_detection_confidence=float(detection_confidence),
            min_tracking_confidence=float(tracking_confidence),
        )
        log.info(
            "MediaPipe hand tracker ready (max_hands=%d, complexity=%d)",
            max_hands,
            model_complexity,
        )

    def process(self, frame_bgr: np.ndarray) -> list[HandObservation]:
        import cv2  # noqa: PLC0415

        height, width = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False  # lets MediaPipe skip a copy
        result = self._hands.process(rgb)

        if not result.multi_hand_landmarks:
            return []

        observations: list[HandObservation] = []
        handedness_list = result.multi_handedness or []
        for i, hand_landmarks in enumerate(result.multi_hand_landmarks):
            label, score = "Unknown", 0.0
            if i < len(handedness_list) and handedness_list[i].classification:
                classification = handedness_list[i].classification[0]
                label, score = classification.label, float(classification.score)
            points = ((lm.x, lm.y, lm.z) for lm in hand_landmarks.landmark)
            observations.append(
                HandObservation(
                    landmarks=landmarks_from_normalized(points, width, height),
                    handedness=label,
                    score=score,
                )
            )
        observations.sort(key=lambda obs: obs.score, reverse=True)
        return observations

    def close(self) -> None:
        try:
            self._hands.close()
        except Exception:  # pragma: no cover - defensive
            log.debug("MediaPipe close() raised", exc_info=True)


def create_tracker(
    backend: str,
    *,
    max_hands: int = 1,
    detection_confidence: float = 0.6,
    tracking_confidence: float = 0.6,
    model_complexity: int = 1,
) -> HandTracker:
    """Instantiate a tracker by name (``mediapipe`` | ``none``)."""
    normalized = backend.strip().lower()
    if normalized in {"none", "null", "off"}:
        log.warning("Hand tracking disabled (backend=%s)", normalized)
        return NullHandTracker()
    if normalized == "mediapipe":
        return MediaPipeHandTracker(
            max_hands=max_hands,
            detection_confidence=detection_confidence,
            tracking_confidence=tracking_confidence,
            model_complexity=model_complexity,
        )
    raise ValueError(f"unknown tracking backend: {backend!r} (expected 'mediapipe' or 'none')")


def select_primary(
    hands: list[HandObservation], preferred: str = "any"
) -> HandObservation | None:
    """Pick the hand to drive the cursor.

    Note: frames are mirrored for natural interaction, so MediaPipe's
    ``Left``/``Right`` labels already match the user's own perspective.
    """
    if not hands:
        return None
    if preferred and preferred.lower() != "any":
        wanted = preferred.strip().lower()
        for hand in hands:
            if hand.handedness.lower() == wanted:
                return hand
    return hands[0]


    #     *** _ ***