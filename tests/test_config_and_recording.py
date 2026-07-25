from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.cli import build_parser, main
from src.infrastructure.config import AppConfig, ConfigError, load_config, merge, nested_override
from src.adapters.recording import Layout, compose_output, output_size


class TestConfigDefaults:
    def test_defaults_are_valid(self) -> None:
        config = AppConfig().validate()
        assert config.camera.width == 1280
        assert config.recording.layout == "overlay"
        assert config.tracking.backend == "mediapipe"

    def test_round_trips_through_dict(self) -> None:
        assert load_config().to_dict()["camera"]["fps"] == 30.0


class TestConfigLayering:
    def test_file_overrides_defaults(self, tmp_path: Path) -> None:
        path = tmp_path / "config.json"
        path.write_text(json.dumps({"camera": {"width": 640, "height": 480}}))
        config = load_config(path)
        assert (config.camera.width, config.camera.height) == (640, 480)
        assert config.camera.fps == 30.0  # untouched key keeps its default

    def test_yaml_is_supported(self, tmp_path: Path) -> None:
        path = tmp_path / "config.yaml"
        path.write_text("recording:\n  layout: pip\n  fps: 24\n")
        config = load_config(path)
        assert config.recording.layout == "pip" and config.recording.fps == 24

    def test_env_overrides_file(self, tmp_path: Path) -> None:
        path = tmp_path / "config.json"
        path.write_text(json.dumps({"camera": {"width": 640}}))
        config = load_config(path, environ={"AIRCANVAS_CAMERA_WIDTH": "800"})
        assert config.camera.width == 800

    def test_cli_overrides_env(self, tmp_path: Path) -> None:
        config = load_config(
            None,
            overrides={"camera": {"width": 1024}},
            environ={"AIRCANVAS_CAMERA_WIDTH": "800"},
        )
        assert config.camera.width == 1024

    def test_booleans_from_env(self) -> None:
        config = load_config(environ={"AIRCANVAS_RECORDING_ENABLED": "false"})
        assert config.recording.enabled is False

    def test_missing_file_is_an_error(self, tmp_path: Path) -> None:
        with pytest.raises(ConfigError, match="not found"):
            load_config(tmp_path / "nope.yaml")

    def test_unknown_key_is_rejected(self, tmp_path: Path) -> None:
        path = tmp_path / "config.json"
        path.write_text(json.dumps({"camera": {"nonsense": 1}}))
        with pytest.raises(ConfigError, match="unknown configuration key"):
            load_config(path)


class TestConfigValidation:
    @pytest.mark.parametrize(
        "overrides,message",
        [
            ({"camera": {"width": 10}}, "camera.width"),
            ({"camera": {"backend": "magic"}}, "backend"),
            ({"tracking": {"detection_confidence": 1.5}}, "detection_confidence"),
            ({"gestures": {"vote_window": 3, "min_votes": 9}}, "min_votes"),
            ({"drawing": {"colors": []}}, "colors"),
            ({"drawing": {"colors": [[0, 0, 300]]}}, "colors"),
            ({"drawing": {"default_color_index": 99}}, "default_color_index"),
            ({"recording": {"layout": "hologram"}}, "layout"),
            ({"logging": {"level": "LOUD"}}, "logging.level"),
        ],
    )
    def test_invalid_values_fail_fast(self, overrides: dict, message: str) -> None:
        with pytest.raises(ConfigError, match=message):
            load_config(overrides=overrides)


class TestOverrideHelpers:
    def test_nested_override(self) -> None:
        assert nested_override("a.b.c", 1) == {"a": {"b": {"c": 1}}}

    def test_merge_is_recursive(self) -> None:
        base = {"a": {"b": 1, "c": 2}}
        assert merge(base, {"a": {"c": 9, "d": 3}}) == {"a": {"b": 1, "c": 9, "d": 3}}


class TestLayouts:
    @pytest.fixture
    def view(self) -> np.ndarray:
        return np.full((120, 200, 3), 60, dtype=np.uint8)

    @pytest.fixture
    def canvas(self) -> np.ndarray:
        layer = np.zeros((120, 200, 4), dtype=np.uint8)
        layer[40:60, 40:60] = (0, 0, 255, 255)
        return layer

    @pytest.mark.parametrize("layout", list(Layout))
    def test_size_matches_declaration(self, view, canvas, layout: Layout) -> None:
        out = compose_output(view, canvas, layout)
        assert out.shape[1::-1] == output_size(layout, 200, 120)
        assert out.dtype == np.uint8

    def test_overlay_passthrough(self, view, canvas) -> None:
        assert np.array_equal(compose_output(view, canvas, Layout.OVERLAY), view)

    def test_canvas_only_hides_the_camera(self, view, canvas) -> None:
        """Privacy mode must contain no camera pixels at all."""
        out = compose_output(view, canvas, Layout.CANVAS_ONLY, paper=(255, 255, 255))
        assert tuple(int(c) for c in out[5, 5]) == (255, 255, 255)
        assert tuple(int(c) for c in out[50, 50]) == (0, 0, 255)
        assert not np.any(np.all(out == 60, axis=-1))

    def test_side_by_side_places_artwork_left(self, view, canvas) -> None:
        out = compose_output(view, canvas, Layout.SIDE_BY_SIDE)
        assert tuple(int(c) for c in out[50, 50]) == (0, 0, 255)      # artwork pane
        assert tuple(int(c) for c in out[5, 205]) == (60, 60, 60)     # camera pane

    def test_pip_inset_contains_the_camera(self, view, canvas) -> None:
        out = compose_output(view, canvas, Layout.PIP)
        assert tuple(int(c) for c in out[70, 130]) == (60, 60, 60)   # inside the inset
        assert tuple(int(c) for c in out[5, 5]) == (255, 255, 255)   # paper elsewhere

    def test_dimension_mismatch_raises(self, view) -> None:
        with pytest.raises(ValueError):
            compose_output(view, np.zeros((10, 10, 4), dtype=np.uint8), Layout.OVERLAY)


class TestCli:
    def test_parser_defaults_to_none(self) -> None:
        args = build_parser().parse_args([])
        assert args.source is None and args.layout is None

    def test_flags_map_to_config(self) -> None:
        from src.cli import _overrides

        args = build_parser().parse_args(
            ["--source", "2", "--layout", "pip", "--no-record", "--headless"]
        )
        config = load_config(overrides=_overrides(args))
        assert config.camera.source == 2
        assert config.recording.layout == "pip"
        assert config.recording.enabled is False
        assert config.headless is True

    def test_video_file_source_is_kept_as_a_string(self) -> None:
        assert build_parser().parse_args(["-s", "clip.mp4"]).source == "clip.mp4"

    def test_print_config_exits_cleanly(self, capsys) -> None:
        assert main(["--print-config", "--width", "800"]) == 0
        assert json.loads(capsys.readouterr().out)["camera"]["width"] == 800

    def test_invalid_config_returns_exit_code_2(self, capsys) -> None:
        assert main(["--width", "1"]) == 2
        assert "configuration error" in capsys.readouterr().err
        
        
        
            #     *** _ ***