"""
Heads-up display and touchless toolbar.

Interaction model
-----------------
There is no click gesture, so selection uses **dwell**: hold the two-finger
SELECT cursor over a control and a progress bar fills; when it completes, the
control activates. Dwell is the standard solution for mid-air interfaces —
it is unambiguous, needs no second hand, and gives continuous feedback that
lets the user abort simply by moving away.

Drawing is suppressed inside the toolbar strip, so reaching for a colour can
never leave a stray mark on the canvas.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Sequence

import cv2
import numpy as np

from src.core.types import Point, ToolKind

__all__ = ["ToolbarItem", "Toolbar", "DwellSelector", "Hud", "Toast", "draw_landmarks"]

FONT = cv2.FONT_HERSHEY_SIMPLEX
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
PANEL = (32, 30, 28)
ACCENT = (255, 196, 84)
DANGER = (60, 60, 235)
MUTED = (168, 162, 158)

HAND_CONNECTIONS: tuple[tuple[int, int], ...] = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
)


# --------------------------------------------------------------------------- #
# Primitives
# --------------------------------------------------------------------------- #
def rounded_rect(
    image: np.ndarray,
    rect: tuple[int, int, int, int],
    color: Sequence[int],
    *,
    radius: int = 10,
    thickness: int = -1,
) -> None:
    """Draw a filled or outlined rounded rectangle."""
    x, y, w, h = (int(v) for v in rect)
    r = max(0, min(int(radius), w // 2, h // 2))
    color = tuple(int(c) for c in color)
    if thickness < 0:
        if w - 2 * r > 0:
            cv2.rectangle(image, (x + r, y), (x + w - r, y + h), color, -1, cv2.LINE_AA)
        if h - 2 * r > 0:
            cv2.rectangle(image, (x, y + r), (x + w, y + h - r), color, -1, cv2.LINE_AA)
        for cx, cy, start in (
            (x + r, y + r, 180), (x + w - r, y + r, 270),
            (x + w - r, y + h - r, 0), (x + r, y + h - r, 90),
        ):
            cv2.ellipse(image, (cx, cy), (r, r), start, 0, 90, color, -1, cv2.LINE_AA)
    else:
        cv2.line(image, (x + r, y), (x + w - r, y), color, thickness, cv2.LINE_AA)
        cv2.line(image, (x + r, y + h), (x + w - r, y + h), color, thickness, cv2.LINE_AA)
        cv2.line(image, (x, y + r), (x, y + h - r), color, thickness, cv2.LINE_AA)
        cv2.line(image, (x + w, y + r), (x + w, y + h - r), color, thickness, cv2.LINE_AA)
        for cx, cy, start in (
            (x + r, y + r, 180), (x + w - r, y + r, 270),
            (x + w - r, y + h - r, 0), (x + r, y + h - r, 90),
        ):
            cv2.ellipse(image, (cx, cy), (r, r), start, 0, 90, color, thickness, cv2.LINE_AA)


def translucent_panel(
    image: np.ndarray,
    rect: tuple[int, int, int, int],
    color: Sequence[int] = PANEL,
    alpha: float = 0.62,
    radius: int = 12,
) -> None:
    """Blend a rounded translucent panel into ``image`` in place."""
    x, y, w, h = (int(v) for v in rect)
    x, y = max(0, x), max(0, y)
    w = min(w, image.shape[1] - x)
    h = min(h, image.shape[0] - y)
    if w <= 0 or h <= 0:
        return
    region = image[y : y + h, x : x + w]
    layer = region.copy()
    rounded_rect(layer, (0, 0, w - 1, h - 1), color, radius=radius)
    cv2.addWeighted(layer, alpha, region, 1.0 - alpha, 0, dst=region)


def centered_text(
    image: np.ndarray,
    text: str,
    center: tuple[int, int],
    *,
    scale: float = 0.5,
    color: Sequence[int] = WHITE,
    thickness: int = 1,
) -> None:
    (tw, th), _ = cv2.getTextSize(text, FONT, scale, thickness)
    origin = (int(center[0] - tw / 2), int(center[1] + th / 2))
    cv2.putText(image, text, origin, FONT, scale, tuple(int(c) for c in color), thickness, cv2.LINE_AA)


def _readable_on(color: Sequence[int]) -> tuple[int, int, int]:
    """Pick black or white text for adequate contrast against ``color``."""
    b, g, r = (float(c) for c in color[:3])
    luminance = 0.114 * b + 0.587 * g + 0.299 * r
    return BLACK if luminance > 140 else WHITE


# --------------------------------------------------------------------------- #
# Toolbar
# --------------------------------------------------------------------------- #
@dataclass(frozen=True, slots=True)
class ToolbarItem:
    id: str
    kind: ToolKind
    rect: tuple[int, int, int, int]
    label: str = ""
    payload: Any = None

    def contains(self, point: Point) -> bool:
        x, y, w, h = self.rect
        return x <= point.x <= x + w and y <= point.y <= y + h

    @property
    def center(self) -> tuple[int, int]:
        x, y, w, h = self.rect
        return (x + w // 2, y + h // 2)


class Toolbar:
    """Adaptive top strip containing colours, brush sizes and actions."""

    ACTIONS: tuple[tuple[str, str], ...] = (
        ("eraser", "ERASE"),
        ("undo", "UNDO"),
        ("clear", "CLEAR"),
        ("save", "SAVE"),
        ("record", "REC"),
    )

    def __init__(
        self,
        width: int,
        *,
        height: int = 76,
        colors: Sequence[Sequence[int]] = (),
        sizes: Sequence[int] = (4, 10, 20),
    ) -> None:
        self.width = int(width)
        self.colors = [tuple(int(c) for c in color) for color in colors]
        self.sizes = [int(s) for s in sizes]
        self.items: list[ToolbarItem] = []
        self.height = self._layout(int(height))

    # ------------------------------------------------------------------ #
    def _layout(self, requested_height: int) -> int:
        pad, gap, group_gap = 10, 8, 20
        n_colors, n_sizes, n_actions = len(self.colors), len(self.sizes), len(self.ACTIONS)
        action_w = 62

        budget = self.width - 2 * pad - 2 * group_gap - n_actions * (action_w + gap)
        divisor = max(1, n_colors + n_sizes)
        cell = int(min(requested_height - 2 * pad, max(22, budget // divisor - gap)))
        height = cell + 2 * pad

        items: list[ToolbarItem] = []
        x = pad
        for i, color in enumerate(self.colors):
            items.append(
                ToolbarItem(f"color:{i}", ToolKind.COLOR, (x, pad, cell, cell), payload=color)
            )
            x += cell + gap

        x += group_gap - gap
        for i, size in enumerate(self.sizes):
            items.append(
                ToolbarItem(f"size:{i}", ToolKind.SIZE, (x, pad, cell, cell), payload=size)
            )
            x += cell + gap

        x = self.width - pad - n_actions * (action_w + gap) + gap
        for action_id, label in self.ACTIONS:
            items.append(
                ToolbarItem(
                    f"action:{action_id}", ToolKind.ACTION, (x, pad, action_w, cell), label=label,
                    payload=action_id,
                )
            )
            x += action_w + gap

        self.items = items
        return height

    # ------------------------------------------------------------------ #
    def contains(self, point: Point | None) -> bool:
        return point is not None and point.y <= self.height

    def hit(self, point: Point | None) -> ToolbarItem | None:
        if point is None:
            return None
        for item in self.items:
            if item.contains(point):
                return item
        return None

    def color_at(self, index: int) -> tuple[int, int, int]:
        return self.colors[index % len(self.colors)]

    def size_at(self, index: int) -> int:
        return self.sizes[index % len(self.sizes)]

    # ------------------------------------------------------------------ #
    def render(
        self,
        frame: np.ndarray,
        *,
        color_index: int,
        size_index: int,
        eraser: bool,
        recording: bool,
        hover: ToolbarItem | None = None,
        progress: float = 0.0,
    ) -> None:
        translucent_panel(frame, (0, 0, self.width, self.height), PANEL, 0.72, radius=0)

        for item in self.items:
            x, y, w, h = item.rect
            active = (
                (item.kind is ToolKind.COLOR and item.id == f"color:{color_index}")
                or (item.kind is ToolKind.SIZE and item.id == f"size:{size_index}")
                or (item.id == "action:eraser" and eraser)
                or (item.id == "action:record" and recording)
            )

            if item.kind is ToolKind.COLOR:
                rounded_rect(frame, item.rect, item.payload, radius=8)
            elif item.kind is ToolKind.SIZE:
                rounded_rect(frame, item.rect, (58, 55, 52), radius=8)
                radius = max(2, min(h // 2 - 4, int(item.payload) // 2 + 2))
                cv2.circle(frame, item.center, radius, WHITE, -1, cv2.LINE_AA)
            else:
                base = DANGER if (item.payload == "record" and recording) else (58, 55, 52)
                if item.payload == "clear":
                    base = (48, 48, 92)
                rounded_rect(frame, item.rect, base, radius=8)
                centered_text(frame, item.label, item.center, scale=0.42, color=WHITE)

            if active:
                rounded_rect(frame, (x - 3, y - 3, w + 6, h + 6), ACCENT, radius=10, thickness=2)

            if hover is not None and hover.id == item.id and progress > 0:
                bar_w = int(w * min(1.0, progress))
                cv2.rectangle(frame, (x, y + h - 4), (x + bar_w, y + h), ACCENT, -1, cv2.LINE_AA)
                rounded_rect(frame, (x - 2, y - 2, w + 4, h + 4), WHITE, radius=9, thickness=1)


class DwellSelector:
    """Fires an item id once the cursor has hovered it for ``dwell_seconds``."""

    __slots__ = ("dwell_seconds", "cooldown_seconds", "_item_id", "_since", "_last_fired")

    def __init__(self, dwell_seconds: float = 0.45, cooldown_seconds: float = 0.5) -> None:
        self.dwell_seconds = max(0.0, float(dwell_seconds))
        self.cooldown_seconds = max(0.0, float(cooldown_seconds))
        self._item_id: str | None = None
        self._since: float = 0.0
        self._last_fired: dict[str, float] = {}

    def reset(self) -> None:
        self._item_id = None
        self._since = 0.0

    def update(self, item_id: str | None, now: float) -> tuple[str | None, float]:
        """Returns ``(fired_id_or_None, progress_ratio)``."""
        if item_id is None:
            self.reset()
            return None, 0.0
        if item_id != self._item_id:
            self._item_id = item_id
            self._since = now
            return None, 0.0

        last = self._last_fired.get(item_id)
        if last is not None and now - last < self.cooldown_seconds:
            return None, 0.0

        elapsed = now - self._since
        if self.dwell_seconds <= 0 or elapsed >= self.dwell_seconds:
            self._last_fired[item_id] = now
            self._since = now
            return item_id, 1.0
        return None, elapsed / self.dwell_seconds


# --------------------------------------------------------------------------- #
# HUD
# --------------------------------------------------------------------------- #
@dataclass(slots=True)
class Toast:
    """Transient status message shown at the bottom of the frame."""

    message: str = ""
    expires_at: float = 0.0
    level: str = "info"

    def show(self, message: str, *, seconds: float = 2.0, level: str = "info") -> None:
        self.message = message
        self.expires_at = time.monotonic() + seconds
        self.level = level

    def active(self, now: float | None = None) -> bool:
        return bool(self.message) and (now or time.monotonic()) < self.expires_at


HELP_LINES: tuple[str, ...] = (
    "Index finger  -  draw",
    "Index + middle  -  move cursor / pick tool",
    "Open palm  -  erase",
    "Thumbs up (hold)  -  save",
    "Keys: [q]uit  [c]lear  [u]ndo  [s]ave  [r]ec  [h]elp  [1-9] colour",
)


class Hud:
    """Renders status text, cursor, progress rings and toasts."""

    def __init__(self, *, show_help: bool = True) -> None:
        self.show_help = show_help
        self.toast = Toast()

    def draw_cursor(
        self,
        frame: np.ndarray,
        point: Point,
        color: Sequence[int],
        thickness: int,
        *,
        mode: str = "draw",
    ) -> None:
        radius = max(6, int(thickness))
        if mode == "erase":
            cv2.circle(frame, point.as_tuple(), radius, WHITE, 2, cv2.LINE_AA)
            cv2.circle(frame, point.as_tuple(), radius + 2, (40, 40, 40), 1, cv2.LINE_AA)
        elif mode == "select":
            cv2.circle(frame, point.as_tuple(), 10, ACCENT, 2, cv2.LINE_AA)
            cv2.line(frame, (point.x - 16, point.y), (point.x - 6, point.y), ACCENT, 2, cv2.LINE_AA)
            cv2.line(frame, (point.x + 6, point.y), (point.x + 16, point.y), ACCENT, 2, cv2.LINE_AA)
        else:
            cv2.circle(frame, point.as_tuple(), radius, tuple(int(c) for c in color), -1, cv2.LINE_AA)
            cv2.circle(frame, point.as_tuple(), radius + 3, WHITE, 1, cv2.LINE_AA)

    def draw_progress_ring(
        self, frame: np.ndarray, center: tuple[int, int], progress: float, label: str = ""
    ) -> None:
        if progress <= 0:
            return
        radius = 46
        cv2.circle(frame, center, radius, (70, 70, 70), 5, cv2.LINE_AA)
        cv2.ellipse(
            frame, center, (radius, radius), -90, 0, int(360 * min(1.0, progress)),
            ACCENT, 5, cv2.LINE_AA,
        )
        if label:
            centered_text(frame, label, (center[0], center[1] + radius + 22), scale=0.6, color=WHITE)

    def draw_status(
        self,
        frame: np.ndarray,
        *,
        fps: float,
        gesture: str,
        color: Sequence[int],
        thickness: int,
        eraser: bool,
        recording: bool,
        elapsed: float,
        strokes: int,
    ) -> None:
        height, width = frame.shape[:2]
        panel_h = 78
        translucent_panel(frame, (12, height - panel_h - 12, 336, panel_h))
        top = height - panel_h + 8
        cv2.putText(
            frame, f"{fps:5.1f} FPS", (26, top), FONT, 0.6, WHITE, 1, cv2.LINE_AA
        )
        cv2.putText(
            frame, f"MODE {gesture.upper()}", (132, top), FONT, 0.6, ACCENT, 1, cv2.LINE_AA
        )
        cv2.putText(
            frame, f"strokes {strokes}", (26, top + 26), FONT, 0.48, MUTED, 1, cv2.LINE_AA
        )
        swatch = (250, height - 34)
        cv2.circle(frame, swatch, 11, tuple(int(c) for c in color), -1, cv2.LINE_AA)
        cv2.circle(frame, swatch, 13, WHITE if not eraser else DANGER, 2, cv2.LINE_AA)
        cv2.putText(
            frame, f"{thickness}px", (272, height - 28), FONT, 0.48, MUTED, 1, cv2.LINE_AA
        )

        if recording:
            badge = (width - 128, height - 34)
            cv2.circle(frame, badge, 7, DANGER, -1, cv2.LINE_AA)
            mins, secs = divmod(int(elapsed), 60)
            cv2.putText(
                frame, f"REC {mins:02d}:{secs:02d}", (badge[0] + 14, badge[1] + 5),
                FONT, 0.55, WHITE, 1, cv2.LINE_AA,
            )

        if self.show_help:
            self._draw_help(frame)

        now = time.monotonic()
        if self.toast.active(now):
            self._draw_toast(frame, now)

    def _draw_help(self, frame: np.ndarray) -> None:
        width = frame.shape[1]
        panel_w, line_h = 372, 22
        panel_h = line_h * len(HELP_LINES) + 18
        x = width - panel_w - 12
        translucent_panel(frame, (x, 92, panel_w, panel_h), alpha=0.55)
        for i, line in enumerate(HELP_LINES):
            cv2.putText(
                frame, line, (x + 16, 92 + 26 + i * line_h), FONT, 0.44, WHITE, 1, cv2.LINE_AA
            )

    def _draw_toast(self, frame: np.ndarray, now: float) -> None:
        height, width = frame.shape[:2]
        text = self.toast.message
        (tw, _), _ = cv2.getTextSize(text, FONT, 0.62, 2)
        box_w = tw + 44
        x = (width - box_w) // 2
        y = height - 132
        color = DANGER if self.toast.level == "error" else PANEL
        translucent_panel(frame, (x, y, box_w, 46), color, alpha=0.78)
        centered_text(frame, text, (width // 2, y + 24), scale=0.62, color=WHITE, thickness=2)


def draw_landmarks(
    frame: np.ndarray,
    landmarks: np.ndarray,
    *,
    color: Sequence[int] = (120, 220, 120),
    joint_color: Sequence[int] = (255, 255, 255),
) -> None:
    """Lightweight skeleton overlay (avoids depending on MediaPipe's drawing utils)."""
    points = [(int(x), int(y)) for x, y in landmarks[:, :2]]
    for a, b in HAND_CONNECTIONS:
        cv2.line(frame, points[a], points[b], tuple(int(c) for c in color), 2, cv2.LINE_AA)
    for pt in points:
        cv2.circle(frame, pt, 3, tuple(int(c) for c in joint_color), -1, cv2.LINE_AA)
        
        
    #     *** _ ***