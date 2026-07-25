"""Session artefact export.

A "save" produces a small bundle rather than a single image, because the
useful outputs differ by audience:

======================  =========================================================
``drawing-*.png``       Artwork only, real alpha channel — drop straight into a
                        design tool.
``composite-*.png``     Artwork over the last camera frame — the shareable one.
``strokes-*.json``      Vector history — reproducible, diffable, replayable.
``session.json``        Manifest: config echo, metrics, artefact inventory.
======================  =========================================================
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

from src.core.canvas import Canvas, canvas_on_background
from src.infrastructure.paths import atomic_write_bytes, atomic_write_text, ensure_directory, safe_join, sanitize_name

log = logging.getLogger(__name__)

__all__ = ["SessionStore", "SessionMetrics"]


@dataclass(slots=True)
class SessionMetrics:
    """Counters aggregated over a session and written to the manifest."""

    frames: int = 0
    frames_with_hand: int = 0
    dropped_recording_frames: int = 0
    saves: int = 0
    undos: int = 0
    clears: int = 0
    gesture_counts: dict[str, int] = field(default_factory=dict)
    started_at: float = field(default_factory=time.time)

    def note_gesture(self, gesture: str) -> None:
        self.gesture_counts[gesture] = self.gesture_counts.get(gesture, 0) + 1

    def as_dict(self, *, average_fps: float = 0.0) -> dict:
        duration = max(0.0, time.time() - self.started_at)
        return {
            "frames": self.frames,
            "frames_with_hand": self.frames_with_hand,
            "hand_detection_rate": round(
                self.frames_with_hand / self.frames, 4
            ) if self.frames else 0.0,
            "dropped_recording_frames": self.dropped_recording_frames,
            "saves": self.saves,
            "undos": self.undos,
            "clears": self.clears,
            "gesture_counts": dict(sorted(self.gesture_counts.items())),
            "duration_seconds": round(duration, 2),
            "average_fps": round(average_fps, 2),
        }


class SessionStore:
    """Owns the on-disk layout for one run, inside a sandboxed root."""

    def __init__(self, root: Path, session_name: str | None = None) -> None:
        self.root = Path(root).expanduser()
        ensure_directory(self.root)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        self.session_id = sanitize_name(session_name or f"session-{stamp}")
        self.directory = safe_join(self.root, self.session_id)
        ensure_directory(self.directory)
        self.artifacts: list[str] = []
        log.info("Session directory: %s", self.directory)

    # ------------------------------------------------------------------ #
    def path_for(self, filename: str) -> Path:
        return safe_join(self.directory, filename)

    def _register(self, path: Path) -> Path:
        self.artifacts.append(path.name)
        return path

    @staticmethod
    def _encode(image: np.ndarray, suffix: str = ".png") -> bytes:
        ok, buffer = cv2.imencode(suffix, image)
        if not ok:  # pragma: no cover - only on codec misconfiguration
            raise RuntimeError(f"failed to encode image as {suffix}")
        return buffer.tobytes()

    # ------------------------------------------------------------------ #
    def save_snapshot(
        self,
        canvas: Canvas,
        frame: np.ndarray | None = None,
        *,
        export_vectors: bool = True,
        background: tuple[int, int, int] = (255, 255, 255),
    ) -> list[Path]:
        """Write the full artefact bundle for the current canvas state."""
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S-%f")[:-3]
        written: list[Path] = []

        layer = canvas.copy()
        written.append(
            self._register(
                atomic_write_bytes(self.path_for(f"drawing-{stamp}.png"), self._encode(layer))
            )
        )

        composite = (
            canvas.composite(frame) if frame is not None else canvas_on_background(layer, background)
        )
        written.append(
            self._register(
                atomic_write_bytes(self.path_for(f"composite-{stamp}.png"), self._encode(composite))
            )
        )

        if export_vectors:
            written.append(
                self._register(
                    atomic_write_text(
                        self.path_for(f"strokes-{stamp}.json"), canvas.to_json(indent=2)
                    )
                )
            )

        log.info("Saved %d artefact(s) -> %s", len(written), self.directory)
        return written

    def write_manifest(self, payload: dict) -> Path:
        """Persist the session manifest (config echo + metrics + inventory)."""
        payload = {**payload, "artifacts": sorted(self.artifacts), "session_id": self.session_id}
        return atomic_write_text(
            self.path_for("session.json"), json.dumps(payload, indent=2, default=str)
        )
        
        
        
    #     *** _ ***