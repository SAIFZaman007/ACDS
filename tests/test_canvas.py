from __future__ import annotations

import json

import numpy as np
import pytest

from src.core.canvas import Canvas, canvas_on_background
from src.core.types import Point, Stroke

RED = (0, 0, 255)
BLUE = (255, 0, 0)


@pytest.fixture
def canvas() -> Canvas:
    return Canvas(200, 120, max_history=4, antialias=False)


def opaque_pixels(canvas: Canvas) -> int:
    return int(np.count_nonzero(canvas.image[:, :, 3]))


class TestDrawing:
    def test_starts_empty(self, canvas: Canvas) -> None:
        assert canvas.is_empty
        assert opaque_pixels(canvas) == 0
        assert canvas.stroke_count == 0

    def test_draw_marks_pixels(self, canvas: Canvas) -> None:
        canvas.begin_stroke(RED, 6)
        canvas.extend_stroke(Point(20, 20))
        canvas.extend_stroke(Point(80, 20))
        canvas.end_stroke()
        assert not canvas.is_empty
        assert opaque_pixels(canvas) > 0
        assert tuple(int(c) for c in canvas.image[20, 50]) == (*RED, 255)

    def test_stroke_at_autostarts(self, canvas: Canvas) -> None:
        canvas.stroke_at((10, 10), RED, 4)
        assert canvas.drawing
        assert canvas.stroke_count == 1

    def test_colour_change_splits_strokes(self, canvas: Canvas) -> None:
        canvas.stroke_at((10, 10), RED, 4)
        canvas.stroke_at((20, 20), BLUE, 4)
        canvas.end_stroke()
        strokes = canvas.strokes()
        assert len(strokes) == 2
        assert strokes[0].color == RED and strokes[1].color == BLUE

    def test_extend_without_begin_raises(self, canvas: Canvas) -> None:
        with pytest.raises(RuntimeError):
            canvas.extend_stroke(Point(1, 1))

    def test_rejects_invalid_dimensions(self) -> None:
        with pytest.raises(ValueError):
            Canvas(0, 10)


class TestErasing:
    def test_erase_removes_pixels(self, canvas: Canvas) -> None:
        canvas.begin_stroke(RED, 20)
        canvas.extend_stroke(Point(100, 60))
        canvas.end_stroke()
        before = opaque_pixels(canvas)

        canvas.begin_stroke((0, 0, 0), 40, erase=True)
        canvas.extend_stroke(Point(100, 60))
        canvas.end_stroke()
        assert opaque_pixels(canvas) < before
        assert canvas.image[60, 100, 3] == 0

    def test_erasing_alone_does_not_mark_content(self, canvas: Canvas) -> None:
        canvas.stroke_at((50, 50), (0, 0, 0), 30, erase=True)
        canvas.end_stroke()
        assert canvas.is_empty


class TestHistory:
    def test_undo_removes_last_stroke(self, canvas: Canvas) -> None:
        canvas.begin_stroke(RED, 6)
        canvas.extend_stroke(Point(10, 10))
        canvas.end_stroke()
        canvas.begin_stroke(BLUE, 6)
        canvas.extend_stroke(Point(150, 100))
        canvas.end_stroke()

        assert canvas.undo() is True
        assert canvas.stroke_count == 1
        assert canvas.image[100, 150, 3] == 0   # blue gone
        assert canvas.image[10, 10, 3] == 255   # red kept

    def test_undo_on_empty_canvas(self, canvas: Canvas) -> None:
        assert canvas.undo() is False

    def test_undo_discards_in_progress_stroke(self, canvas: Canvas) -> None:
        canvas.stroke_at((30, 30), RED, 8)
        assert canvas.undo() is True
        assert canvas.stroke_count == 0

    def test_history_is_bounded_but_pixels_survive(self, canvas: Canvas) -> None:
        """Old strokes bake into the base layer: memory stays flat, art stays put."""
        for i in range(10):
            canvas.begin_stroke(RED, 4)
            canvas.extend_stroke(Point(10 + i * 15, 60))
            canvas.end_stroke()
        assert canvas.stroke_count == canvas.max_history == 4
        assert canvas.image[60, 10, 3] == 255  # the very first dot is still visible

    def test_clear_removes_everything(self, canvas: Canvas) -> None:
        for i in range(8):
            canvas.stroke_at((10 + i * 10, 40), RED, 5)
            canvas.end_stroke()
        canvas.clear()
        assert canvas.is_empty
        assert canvas.stroke_count == 0
        assert opaque_pixels(canvas) == 0


class TestSerialisation:
    def test_round_trip(self, canvas: Canvas) -> None:
        canvas.begin_stroke(RED, 7)
        canvas.extend_stroke(Point(5, 5))
        canvas.extend_stroke(Point(60, 40))
        canvas.end_stroke()

        payload = json.loads(canvas.to_json())
        assert payload["width"] == 200 and payload["height"] == 120
        assert len(payload["strokes"]) == 1

        replay = Canvas(200, 120, antialias=False)
        replay.load_strokes([Stroke.from_dict(s) for s in payload["strokes"]])
        assert np.array_equal(replay.image, canvas.image)


class TestCompositing:
    def test_empty_canvas_leaves_frame_untouched(self, canvas: Canvas) -> None:
        frame = np.full((120, 200, 3), 30, dtype=np.uint8)
        assert np.array_equal(canvas.composite(frame), frame)

    def test_stroke_replaces_frame_pixels(self, canvas: Canvas) -> None:
        frame = np.full((120, 200, 3), 30, dtype=np.uint8)
        canvas.begin_stroke(RED, 12)
        canvas.extend_stroke(Point(100, 60))
        canvas.end_stroke()
        out = canvas.composite(frame)
        assert tuple(int(c) for c in out[60, 100]) == RED
        assert tuple(int(c) for c in out[5, 5]) == (30, 30, 30)
        assert frame[60, 100, 2] == 30  # input frame not mutated

    def test_dimension_mismatch_raises(self, canvas: Canvas) -> None:
        with pytest.raises(ValueError):
            canvas.composite(np.zeros((10, 10, 3), dtype=np.uint8))

    def test_canvas_on_background(self, canvas: Canvas) -> None:
        canvas.begin_stroke(RED, 10)
        canvas.extend_stroke(Point(20, 20))
        canvas.end_stroke()
        flat = canvas_on_background(canvas.image, (255, 255, 255))
        assert flat.shape == (120, 200, 3)
        assert tuple(int(c) for c in flat[119, 199]) == (255, 255, 255)
        assert tuple(int(c) for c in flat[20, 20]) == RED
        
        
            #     *** _ ***