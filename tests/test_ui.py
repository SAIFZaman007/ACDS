from __future__ import annotations

import numpy as np
import pytest

from src.core.types import Point, ToolKind
from src.adapters.ui import DwellSelector, Hud, Toolbar, draw_landmarks

from .conftest import make_hand

COLORS = [[0, 0, 255], [0, 255, 0], [255, 0, 0]]
SIZES = [4, 10, 20]


@pytest.fixture
def toolbar() -> Toolbar:
    return Toolbar(1280, height=76, colors=COLORS, sizes=SIZES)


class TestToolbarLayout:
    def test_item_inventory(self, toolbar: Toolbar) -> None:
        kinds = [item.kind for item in toolbar.items]
        assert kinds.count(ToolKind.COLOR) == len(COLORS)
        assert kinds.count(ToolKind.SIZE) == len(SIZES)
        assert kinds.count(ToolKind.ACTION) == len(Toolbar.ACTIONS)

    def test_items_stay_inside_the_frame(self, toolbar: Toolbar) -> None:
        for item in toolbar.items:
            x, y, w, h = item.rect
            assert x >= 0 and y >= 0
            assert x + w <= toolbar.width
            assert y + h <= toolbar.height

    def test_items_do_not_overlap(self, toolbar: Toolbar) -> None:
        spans = sorted((item.rect[0], item.rect[0] + item.rect[2]) for item in toolbar.items)
        for (_, end), (start, _) in zip(spans, spans[1:]):
            assert start >= end, "toolbar controls overlap"

    @pytest.mark.parametrize("width", [640, 800, 1024, 1280, 1920, 3840])
    def test_layout_adapts_to_any_width(self, width: int) -> None:
        bar = Toolbar(width, colors=COLORS, sizes=SIZES)
        for item in bar.items:
            assert item.rect[0] + item.rect[2] <= width
            assert item.rect[2] > 0 and item.rect[3] > 0

    def test_contains_only_the_top_strip(self, toolbar: Toolbar) -> None:
        assert toolbar.contains(Point(100, 5)) is True
        assert toolbar.contains(Point(100, toolbar.height + 1)) is False
        assert toolbar.contains(None) is False


class TestToolbarHitTesting:
    def test_hits_the_first_colour(self, toolbar: Toolbar) -> None:
        item = toolbar.items[0]
        assert toolbar.hit(Point(*item.center)) is item
        assert item.id == "color:0"

    def test_hits_every_item_at_its_centre(self, toolbar: Toolbar) -> None:
        for item in toolbar.items:
            assert toolbar.hit(Point(*item.center)) is item

    def test_gap_between_items_is_a_miss(self, toolbar: Toolbar) -> None:
        assert toolbar.hit(Point(toolbar.width // 2, toolbar.height - 1)) is None

    def test_none_cursor_is_a_miss(self, toolbar: Toolbar) -> None:
        assert toolbar.hit(None) is None


class TestDwellSelector:
    def test_requires_sustained_hover(self) -> None:
        dwell = DwellSelector(dwell_seconds=0.5, cooldown_seconds=1.0)
        assert dwell.update("color:0", 0.0) == (None, 0.0)
        fired, progress = dwell.update("color:0", 0.25)
        assert fired is None and progress == pytest.approx(0.5)
        assert dwell.update("color:0", 0.55)[0] == "color:0"

    def test_moving_away_cancels(self) -> None:
        dwell = DwellSelector(dwell_seconds=0.5)
        dwell.update("color:0", 0.0)
        dwell.update("color:1", 0.3)          # cursor moved to a different control
        assert dwell.update("color:0", 0.6)[0] is None

    def test_leaving_the_toolbar_resets(self) -> None:
        dwell = DwellSelector(dwell_seconds=0.5)
        dwell.update("color:0", 0.0)
        assert dwell.update(None, 0.3) == (None, 0.0)
        assert dwell.update("color:0", 0.6)[0] is None

    def test_cooldown_blocks_immediate_refire(self) -> None:
        dwell = DwellSelector(dwell_seconds=0.2, cooldown_seconds=1.0)
        dwell.update("action:undo", 0.0)
        assert dwell.update("action:undo", 0.25)[0] == "action:undo"
        assert dwell.update("action:undo", 0.5)[0] is None
        assert dwell.update("action:undo", 1.6)[0] == "action:undo"


class TestRendering:
    """Rendering must never raise or resize the frame — it runs 30x a second."""

    def test_toolbar_render(self, toolbar: Toolbar) -> None:
        frame = np.full((720, 1280, 3), 40, dtype=np.uint8)
        toolbar.render(
            frame, color_index=1, size_index=2, eraser=True, recording=True,
            hover=toolbar.items[0], progress=0.6,
        )
        assert frame.shape == (720, 1280, 3) and frame.dtype == np.uint8

    def test_hud_render(self) -> None:
        frame = np.full((720, 1280, 3), 40, dtype=np.uint8)
        hud = Hud(show_help=True)
        hud.toast.show("Saved 3 files")
        hud.draw_cursor(frame, Point(400, 300), (0, 0, 255), 12, mode="draw")
        hud.draw_cursor(frame, Point(400, 300), (0, 0, 255), 40, mode="erase")
        hud.draw_cursor(frame, Point(400, 300), (0, 0, 255), 12, mode="select")
        hud.draw_progress_ring(frame, (640, 360), 0.5, "SAVING")
        hud.draw_status(
            frame, fps=29.7, gesture="draw", color=(0, 0, 255), thickness=10,
            eraser=False, recording=True, elapsed=71.0, strokes=12,
        )
        assert frame.shape == (720, 1280, 3)

    def test_cursor_near_the_edge_does_not_crash(self) -> None:
        frame = np.zeros((240, 320, 3), dtype=np.uint8)
        Hud().draw_cursor(frame, Point(319, 239), (255, 255, 255), 30)

    def test_draw_landmarks(self) -> None:
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        draw_landmarks(frame, make_hand(index=True).landmarks)
        assert np.count_nonzero(frame) > 0
        
        
        
            #     *** _ ***