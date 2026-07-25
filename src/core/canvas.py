"""The drawing surface.

Architecture
------------
The canvas keeps **both** a rasterised BGRA layer and the vector stroke
history that produced it:

* Rasterising each new segment incrementally keeps the per-frame cost O(1),
  independent of how much has already been drawn.
* Retaining the vector history gives exact undo, a reproducible ``.json``
  export, and resolution-independent re-rendering.
* History is bounded: once ``max_history`` strokes accumulate, the oldest is
  permanently *baked* into an immutable base layer. Memory therefore stays
  flat during long sessions, which matters for unattended/kiosk deployments.

Transparency is first-class (BGRA), so the exported PNG contains the artwork
alone with a genuine alpha channel rather than artwork burned onto video.
"""

from __future__ import annotations

import json
from collections import deque
from typing import Deque, Iterable, Sequence

import cv2
import numpy as np

from .types import Point, Stroke

__all__ = ["Canvas", "canvas_on_background"]

_TRANSPARENT = (0, 0, 0, 0)


class Canvas:
    """A transparent BGRA drawing layer with bounded undo history."""

    def __init__(
        self,
        width: int,
        height: int,
        *,
        max_history: int = 128,
        antialias: bool = True,
    ) -> None:
        if width <= 0 or height <= 0:
            raise ValueError("canvas dimensions must be positive")
        if max_history < 1:
            raise ValueError("max_history must be >= 1")
        self.width = int(width)
        self.height = int(height)
        self.max_history = int(max_history)
        self._line_type = cv2.LINE_AA if antialias else cv2.LINE_8
        self._base = np.zeros((self.height, self.width, 4), dtype=np.uint8)
        self._layer = self._base.copy()
        self._strokes: Deque[Stroke] = deque()
        self._active: Stroke | None = None
        self._has_content = False

    # ------------------------------------------------------------------ #
    # Introspection
    # ------------------------------------------------------------------ #
    @property
    def image(self) -> np.ndarray:
        """The live BGRA layer (do not mutate; call :meth:`copy` instead)."""
        return self._layer

    @property
    def is_empty(self) -> bool:
        return not self._has_content

    @property
    def stroke_count(self) -> int:
        return len(self._strokes) + (1 if self._active else 0)

    @property
    def drawing(self) -> bool:
        return self._active is not None

    def copy(self) -> np.ndarray:
        return self._layer.copy()

    # ------------------------------------------------------------------ #
    # Stroke lifecycle
    # ------------------------------------------------------------------ #
    def begin_stroke(
        self,
        color: Sequence[int],
        thickness: int,
        *,
        erase: bool = False,
    ) -> None:
        """Start a new stroke, committing any stroke already in progress."""
        self.end_stroke()
        self._active = Stroke(
            color=(int(color[0]), int(color[1]), int(color[2])),
            thickness=max(1, int(thickness)),
            erase=erase,
        )

    def extend_stroke(self, point: Point | tuple[int, int]) -> None:
        """Append a point to the active stroke and rasterise the new segment."""
        if self._active is None:
            raise RuntimeError("extend_stroke() called with no active stroke")
        pt = point.as_tuple() if isinstance(point, Point) else (int(point[0]), int(point[1]))
        previous = self._active.points[-1] if self._active.points else None
        self._active.add(pt)
        if self._active.points[-1] != pt and previous == pt:
            return  # duplicate point, nothing to rasterise
        self._rasterise_segment(self._active, previous, pt, self._layer)
        if not self._active.erase:
            self._has_content = True

    def end_stroke(self) -> None:
        """Commit the active stroke to history (baking out overflow)."""
        if self._active is None:
            return
        if self._active.points:
            self._strokes.append(self._active)
            while len(self._strokes) > self.max_history:
                self._bake(self._strokes.popleft())
        self._active = None

    def stroke_at(
        self,
        point: Point | tuple[int, int],
        color: Sequence[int],
        thickness: int,
        *,
        erase: bool = False,
    ) -> None:
        """Continue drawing at ``point``, starting a stroke if needed."""
        if self._active is None or self._active.erase != erase:
            self.begin_stroke(color, thickness, erase=erase)
        elif not erase and (
            self._active.color != tuple(int(c) for c in color)
            or self._active.thickness != max(1, int(thickness))
        ):
            self.begin_stroke(color, thickness, erase=erase)
        elif erase and self._active.thickness != max(1, int(thickness)):
            self.begin_stroke(color, thickness, erase=erase)
        self.extend_stroke(point)

    # ------------------------------------------------------------------ #
    # History operations
    # ------------------------------------------------------------------ #
    def undo(self) -> bool:
        """Remove the most recent stroke. Returns ``True`` if anything changed."""
        if self._active is not None:
            self._active = None
            self._rerender()
            return True
        if not self._strokes:
            return False
        self._strokes.pop()
        self._rerender()
        return True

    def clear(self) -> None:
        """Erase everything, including baked history."""
        self._strokes.clear()
        self._active = None
        self._base[:] = 0
        self._layer[:] = 0
        self._has_content = False

    def strokes(self) -> list[Stroke]:
        out = list(self._strokes)
        if self._active is not None and self._active.points:
            out.append(self._active)
        return out

    def load_strokes(self, strokes: Iterable[Stroke]) -> None:
        """Replace the stroke history (used for replay / regression tests)."""
        self._strokes = deque(strokes)
        self._active = None
        self._base[:] = 0
        self._rerender()

    # ------------------------------------------------------------------ #
    # Serialisation
    # ------------------------------------------------------------------ #
    def to_json(self, *, indent: int | None = None) -> str:
        payload = {
            "version": 1,
            "width": self.width,
            "height": self.height,
            "strokes": [s.to_dict() for s in self.strokes()],
        }
        return json.dumps(payload, indent=indent)

    # ------------------------------------------------------------------ #
    # Compositing
    # ------------------------------------------------------------------ #
    def composite(self, frame: np.ndarray, opacity: float = 1.0) -> np.ndarray:
        """Alpha-blend the canvas over a BGR frame and return a new image."""
        if frame.shape[:2] != (self.height, self.width):
            raise ValueError(
                f"frame shape {frame.shape[:2]} does not match canvas "
                f"({self.height}, {self.width})"
            )
        if self.is_empty:
            return frame.copy()
        return alpha_over(frame, self._layer, opacity)

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #
    def _rasterise_segment(
        self,
        stroke: Stroke,
        previous: tuple[int, int] | None,
        current: tuple[int, int],
        target: np.ndarray,
    ) -> None:
        if stroke.erase:
            color = _TRANSPARENT
            line_type = cv2.LINE_8  # AA would leave partially-erased fringes
            radius = max(1, stroke.thickness // 2)
        else:
            color = (*stroke.color, 255)
            line_type = self._line_type
            radius = max(1, stroke.thickness // 2)

        if previous is None:
            cv2.circle(target, current, radius, color, -1, line_type)
        else:
            cv2.line(target, previous, current, color, stroke.thickness, line_type)
            cv2.circle(target, current, radius, color, -1, line_type)

    def _draw_stroke(self, stroke: Stroke, target: np.ndarray) -> None:
        previous: tuple[int, int] | None = None
        for pt in stroke.points:
            self._rasterise_segment(stroke, previous, pt, target)
            previous = pt

    def _bake(self, stroke: Stroke) -> None:
        self._draw_stroke(stroke, self._base)

    def _rerender(self) -> None:
        np.copyto(self._layer, self._base)
        for stroke in self._strokes:
            self._draw_stroke(stroke, self._layer)
        self._has_content = bool(np.any(self._layer[:, :, 3]))


def alpha_over(background_bgr: np.ndarray, overlay_bgra: np.ndarray, opacity: float = 1.0) -> np.ndarray:
    """Composite a BGRA overlay onto a BGR background using integer SIMD ops."""
    alpha = overlay_bgra[:, :, 3]
    if opacity < 1.0:
        alpha = cv2.multiply(alpha, np.full_like(alpha, 255), scale=float(opacity) / 255.0)
    mask3 = cv2.merge([alpha, alpha, alpha])
    inv3 = cv2.bitwise_not(mask3)
    foreground = cv2.multiply(overlay_bgra[:, :, :3], mask3, scale=1.0 / 255.0)
    background = cv2.multiply(background_bgr, inv3, scale=1.0 / 255.0)
    return cv2.add(foreground, background)


def canvas_on_background(
    overlay_bgra: np.ndarray,
    color: tuple[int, int, int] = (255, 255, 255),
) -> np.ndarray:
    """Flatten a BGRA canvas onto a solid BGR background (e.g. white paper)."""
    height, width = overlay_bgra.shape[:2]
    background = np.full((height, width, 3), color, dtype=np.uint8)
    return alpha_over(background, overlay_bgra)

#     *** _ ***