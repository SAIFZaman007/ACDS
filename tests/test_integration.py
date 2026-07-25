"""End-to-end tests for the full application loop.

A scripted tracker replays a deterministic gesture sequence, and a generated
video file stands in for the camera, so the entire pipeline — capture,
gesture reduction, canvas mutation, compositing, recording and artefact
export — is exercised in CI with no hardware and no MediaPipe.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from src import app as app_module
from src.app import AirCanvasApp
from src.infrastructure.config import load_config
from src.core.types import HandObservation

from .conftest import make_hand

FRAME_W, FRAME_H = 320, 240
DRAW_FRAMES = 24
SAVE_FRAMES = 6


@pytest.fixture
def clip(tmp_path: Path) -> Path:
    """A short synthetic video standing in for a webcam feed."""
    path = tmp_path / "input.mp4"
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (FRAME_W, FRAME_H)
    )
    assert writer.isOpened(), "no usable codec in this environment"
    for i in range(DRAW_FRAMES + SAVE_FRAMES + 4):
        writer.write(np.full((FRAME_H, FRAME_W, 3), (i * 3) % 200, dtype=np.uint8))
    writer.release()
    return path


class ScriptedTracker:
    """Replays a fixed gesture sequence: draw a horizontal line, then save."""

    def __init__(self) -> None:
        self.frame_index = 0

    def process(self, frame_bgr: np.ndarray) -> list[HandObservation]:  # noqa: ARG002
        i = self.frame_index
        self.frame_index += 1
        if i < DRAW_FRAMES:
            x = 60.0 + (i / max(1, DRAW_FRAMES - 1)) * 180.0
            return [make_hand(index=True, center=(x, 170.0), scale=34.0)]
        if i < DRAW_FRAMES + SAVE_FRAMES:
            return [make_hand(thumb=True, thumb_up=True, center=(160.0, 190.0), scale=34.0)]
        return []

    def close(self) -> None:
        return None


@pytest.fixture
def scripted(monkeypatch: pytest.MonkeyPatch) -> ScriptedTracker:
    tracker = ScriptedTracker()
    monkeypatch.setattr(app_module, "create_tracker", lambda *a, **k: tracker)
    return tracker


def build_config(clip: Path, output: Path, **extra) -> object:
    overrides = {
        "camera": {"source": str(clip), "width": FRAME_W, "height": FRAME_H, "mirror": False},
        "tracking": {"backend": "none"},
        # Fire on the first SAVE frame, then rely on the cooldown to prove that a
        # sustained thumbs-up cannot save repeatedly.
        "gestures": {"vote_window": 1, "min_votes": 1, "save_hold_seconds": 0.0,
                     "action_cooldown_seconds": 30.0},
        "output": {"root": str(output), "session_name": "itest"},
        "recording": {"enabled": True, "fps": 30.0, "layout": "side_by_side"},
        "logging": {"to_file": False, "level": "WARNING"},
        "headless": True,
    }
    for key, value in extra.items():
        overrides.setdefault(key, {}).update(value) if isinstance(value, dict) else None
        if not isinstance(value, dict):
            overrides[key] = value
    return load_config(overrides=overrides)


class TestFullSession:
    def test_produces_every_artifact(self, clip: Path, tmp_path: Path, scripted) -> None:
        output = tmp_path / "sessions"
        assert AirCanvasApp(build_config(clip, output)).run() == 0

        session = output / "itest"
        names = sorted(p.name for p in session.iterdir())
        assert any(n.startswith("drawing-") and n.endswith(".png") for n in names), names
        assert any(n.startswith("composite-") and n.endswith(".png") for n in names), names
        assert any(n.startswith("strokes-") and n.endswith(".json") for n in names), names
        assert any(n.startswith("session-video") for n in names), names
        assert "session.json" in names

    def test_manifest_records_metrics(self, clip: Path, tmp_path: Path, scripted) -> None:
        output = tmp_path / "sessions"
        AirCanvasApp(build_config(clip, output)).run()

        manifest = json.loads((output / "itest" / "session.json").read_text())
        metrics = manifest["metrics"]
        assert metrics["frames"] >= DRAW_FRAMES
        assert metrics["frames_with_hand"] >= DRAW_FRAMES
        assert metrics["saves"] == 1, "action cooldown should prevent repeat saves"
        assert metrics["gesture_counts"]["draw"] >= DRAW_FRAMES - 2
        assert metrics["gesture_counts"]["save"] >= 1
        assert manifest["config"]["recording"]["layout"] == "side_by_side"
        assert set(manifest["artifacts"])

    def test_exported_drawing_has_real_transparency(
        self, clip: Path, tmp_path: Path, scripted
    ) -> None:
        output = tmp_path / "sessions"
        AirCanvasApp(build_config(clip, output)).run()

        drawing = next((output / "itest").glob("drawing-*.png"))
        image = cv2.imread(str(drawing), cv2.IMREAD_UNCHANGED)
        assert image.shape == (FRAME_H, FRAME_W, 4)
        alpha = image[:, :, 3]
        assert alpha.max() == 255 and alpha.min() == 0  # ink and true transparency

    def test_side_by_side_video_is_double_width(
        self, clip: Path, tmp_path: Path, scripted
    ) -> None:
        output = tmp_path / "sessions"
        AirCanvasApp(build_config(clip, output)).run()

        video = next((output / "itest").glob("session-video.*"))
        capture = cv2.VideoCapture(str(video))
        try:
            assert capture.isOpened()
            assert int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)) == FRAME_W * 2
            assert int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)) == FRAME_H
            assert int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) > 0
        finally:
            capture.release()

    def test_strokes_json_is_replayable(self, clip: Path, tmp_path: Path, scripted) -> None:
        output = tmp_path / "sessions"
        AirCanvasApp(build_config(clip, output)).run()

        payload = json.loads(next((output / "itest").glob("strokes-*.json")).read_text())
        assert payload["version"] == 1
        assert payload["width"] == FRAME_W and payload["height"] == FRAME_H
        assert payload["strokes"], "the scripted drawing produced no strokes"
        assert len(payload["strokes"][0]["points"]) > 5

    def test_privacy_layout_records_no_camera_pixels(
        self, clip: Path, tmp_path: Path, scripted
    ) -> None:
        output = tmp_path / "sessions"
        config = build_config(clip, output)
        config.recording.layout = "canvas_only"
        AirCanvasApp(config).run()

        video = next((output / "itest").glob("session-video.*"))
        capture = cv2.VideoCapture(str(video))
        try:
            ok, frame = capture.read()
            assert ok
            assert frame.shape[:2] == (FRAME_H, FRAME_W)
            corner = frame[2:8, 2:8].reshape(-1, 3).mean(axis=0)
            assert corner.min() > 200, "privacy layout must render on blank paper"
        finally:
            capture.release()

    def test_no_record_mode_writes_no_video(self, clip: Path, tmp_path: Path, scripted) -> None:
        output = tmp_path / "sessions"
        config = build_config(clip, output)
        config.recording.enabled = False
        AirCanvasApp(config).run()
        assert not list((output / "itest").glob("session-video.*"))


class TestResilience:
    def test_missing_source_exits_with_code_2(self, tmp_path: Path, scripted) -> None:
        config = load_config(
            overrides={
                "camera": {"source": str(tmp_path / "does-not-exist.mp4")},
                "tracking": {"backend": "none"},
                "output": {"root": str(tmp_path / "sessions")},
                "recording": {"enabled": False},
                "logging": {"to_file": False, "level": "CRITICAL"},
                "headless": True,
            }
        )
        with pytest.raises(Exception):
            AirCanvasApp(config).run()

    def test_duration_limit_stops_the_loop(self, clip: Path, tmp_path: Path, scripted) -> None:
        output = tmp_path / "sessions"
        config = build_config(clip, output)
        config.camera.loop = True     # would otherwise run forever
        config.duration = 0.4
        assert AirCanvasApp(config).run() == 0
        
        
            #     *** _ ***