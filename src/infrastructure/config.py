"""Layered configuration.

Precedence (lowest to highest)::

    dataclass defaults  <  YAML/JSON file  <  AIRCANVAS_* env vars  <  CLI flags

Everything is a typed dataclass validated at construction, so a bad value
fails fast at start-up with a precise message instead of surfacing as a
mystery exception 40 frames into the render loop. No third-party settings
library is required.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

__all__ = ["AppConfig", "ConfigError", "load_config"]

ENV_PREFIX = "AIRCANVAS_"


class ConfigError(ValueError):
    """Raised when configuration is structurally or semantically invalid."""


def _check(condition: bool, message: str) -> None:
    if not condition:
        raise ConfigError(message)


def _in_range(name: str, value: float, low: float, high: float) -> None:
    _check(low <= value <= high, f"{name} must be within [{low}, {high}], got {value}")


# --------------------------------------------------------------------------- #
# Sections
# --------------------------------------------------------------------------- #
@dataclass(slots=True)
class CameraConfig:
    source: int | str = 0
    width: int = 1280
    height: int = 720
    fps: float = 30.0
    backend: str = "auto"
    mirror: bool = True
    loop: bool = False
    realtime: bool = False

    def validate(self) -> None:
        _in_range("camera.width", self.width, 160, 7680)
        _in_range("camera.height", self.height, 120, 4320)
        _in_range("camera.fps", self.fps, 1, 240)
        _check(
            self.backend in {"auto", "any", "v4l2", "dshow", "msmf", "avfoundation", "gstreamer"},
            f"camera.backend {self.backend!r} is not a known OpenCV backend",
        )
        if isinstance(self.source, int):
            _check(self.source >= 0, "camera.source index must be >= 0")


@dataclass(slots=True)
class TrackingConfig:
    backend: str = "mediapipe"
    max_hands: int = 1
    detection_confidence: float = 0.6
    tracking_confidence: float = 0.6
    model_complexity: int = 1
    preferred_hand: str = "any"

    def validate(self) -> None:
        _check(
            self.backend in {"mediapipe", "none"},
            "tracking.backend must be 'mediapipe' or 'none'",
        )
        _in_range("tracking.max_hands", self.max_hands, 1, 4)
        _in_range("tracking.detection_confidence", self.detection_confidence, 0.0, 1.0)
        _in_range("tracking.tracking_confidence", self.tracking_confidence, 0.0, 1.0)
        _check(self.model_complexity in (0, 1), "tracking.model_complexity must be 0 or 1")
        _check(
            self.preferred_hand.lower() in {"any", "left", "right"},
            "tracking.preferred_hand must be any|left|right",
        )


@dataclass(slots=True)
class GestureConfig:
    vote_window: int = 5
    min_votes: int = 3
    extension_margin: float = 1.05
    thumb_rise: float = 0.35
    save_hold_seconds: float = 0.9
    clear_hold_seconds: float = 1.6
    action_cooldown_seconds: float = 2.0
    dwell_seconds: float = 0.45

    def validate(self) -> None:
        _in_range("gestures.vote_window", self.vote_window, 1, 30)
        _check(
            1 <= self.min_votes <= self.vote_window,
            "gestures.min_votes must be within [1, vote_window]",
        )
        _in_range("gestures.extension_margin", self.extension_margin, 1.0, 2.0)
        _in_range("gestures.thumb_rise", self.thumb_rise, 0.0, 3.0)
        _in_range("gestures.save_hold_seconds", self.save_hold_seconds, 0.0, 10.0)
        _in_range("gestures.dwell_seconds", self.dwell_seconds, 0.0, 5.0)


@dataclass(slots=True)
class SmoothingConfig:
    enabled: bool = True
    min_cutoff: float = 1.2
    beta: float = 0.02
    d_cutoff: float = 1.0

    def validate(self) -> None:
        _in_range("smoothing.min_cutoff", self.min_cutoff, 0.001, 100.0)
        _in_range("smoothing.beta", self.beta, 0.0, 10.0)
        _in_range("smoothing.d_cutoff", self.d_cutoff, 0.001, 100.0)


DEFAULT_COLORS: list[list[int]] = [
    [80, 80, 245],    # red
    [80, 200, 255],   # amber
    [90, 220, 120],   # green
    [235, 180, 70],   # blue
    [220, 120, 220],  # violet
    [250, 250, 250],  # white
]


@dataclass(slots=True)
class DrawingConfig:
    colors: list[list[int]] = field(default_factory=lambda: [c[:] for c in DEFAULT_COLORS])
    thickness_steps: list[int] = field(default_factory=lambda: [4, 10, 20])
    default_color_index: int = 0
    default_size_index: int = 1
    eraser_radius: int = 48
    max_history: int = 128
    antialias: bool = True

    def validate(self) -> None:
        _check(bool(self.colors), "drawing.colors must not be empty")
        for i, color in enumerate(self.colors):
            _check(
                len(color) == 3 and all(0 <= int(c) <= 255 for c in color),
                f"drawing.colors[{i}] must be a BGR triple within 0-255",
            )
        _check(bool(self.thickness_steps), "drawing.thickness_steps must not be empty")
        for i, size in enumerate(self.thickness_steps):
            _in_range(f"drawing.thickness_steps[{i}]", size, 1, 128)
        _check(
            0 <= self.default_color_index < len(self.colors),
            "drawing.default_color_index out of range",
        )
        _check(
            0 <= self.default_size_index < len(self.thickness_steps),
            "drawing.default_size_index out of range",
        )
        _in_range("drawing.eraser_radius", self.eraser_radius, 4, 400)
        _in_range("drawing.max_history", self.max_history, 1, 5000)


@dataclass(slots=True)
class UIConfig:
    toolbar_height: int = 76
    show_landmarks: bool = True
    show_help: bool = True
    window_title: str = "AirCanvas"

    def validate(self) -> None:
        _in_range("ui.toolbar_height", self.toolbar_height, 40, 240)


@dataclass(slots=True)
class RecordingConfig:
    enabled: bool = True
    layout: str = "overlay"
    fps: float = 30.0
    codec: str = "mp4v"
    queue_size: int = 64
    max_seconds: float = 1800.0
    filename: str = "session-video"

    def validate(self) -> None:
        _check(
            self.layout in {"overlay", "side_by_side", "pip", "canvas_only"},
            "recording.layout must be overlay|side_by_side|pip|canvas_only",
        )
        _in_range("recording.fps", self.fps, 1, 120)
        _in_range("recording.queue_size", self.queue_size, 4, 4096)
        _in_range("recording.max_seconds", self.max_seconds, 1, 86400)


@dataclass(slots=True)
class OutputConfig:
    root: str = "./sessions"
    session_name: str | None = None
    export_vectors: bool = True
    paper_color: list[int] = field(default_factory=lambda: [255, 255, 255])

    def validate(self) -> None:
        _check(bool(self.root), "output.root must not be empty")
        _check(
            len(self.paper_color) == 3 and all(0 <= int(c) <= 255 for c in self.paper_color),
            "output.paper_color must be a BGR triple within 0-255",
        )


@dataclass(slots=True)
class LoggingConfig:
    level: str = "INFO"
    json: bool = False
    to_file: bool = True

    def validate(self) -> None:
        _check(
            self.level.upper() in {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"},
            "logging.level must be a standard logging level name",
        )


@dataclass(slots=True)
class AppConfig:
    camera: CameraConfig = field(default_factory=CameraConfig)
    tracking: TrackingConfig = field(default_factory=TrackingConfig)
    gestures: GestureConfig = field(default_factory=GestureConfig)
    smoothing: SmoothingConfig = field(default_factory=SmoothingConfig)
    drawing: DrawingConfig = field(default_factory=DrawingConfig)
    ui: UIConfig = field(default_factory=UIConfig)
    recording: RecordingConfig = field(default_factory=RecordingConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    headless: bool = False
    duration: float | None = None

    def validate(self) -> "AppConfig":
        for f in fields(self):
            value = getattr(self, f.name)
            if is_dataclass(value) and hasattr(value, "validate"):
                value.validate()
        if self.duration is not None:
            _in_range("duration", self.duration, 0.1, 86400)
        return self

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# --------------------------------------------------------------------------- #
# Loading & merging
# --------------------------------------------------------------------------- #
def _coerce(target_type: Any, value: Any, path: str) -> Any:
    """Best-effort coercion of a raw config value to the declared type."""
    if value is None:
        return None
    # `from __future__ import annotations` means dataclass field types arrive as
    # strings ("int", "float | None", ...). Only simple scalars are coerced;
    # unions and containers are passed through untouched.
    type_name = getattr(target_type, "__name__", str(target_type))
    try:
        if type_name == "bool":
            if isinstance(value, str):
                return value.strip().lower() in {"1", "true", "yes", "on"}
            return bool(value)
        if type_name == "int":
            return int(value)
        if type_name == "float":
            return float(value)
        if type_name == "str":
            return str(value)
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"{path}: cannot interpret {value!r} as {type_name}") from exc
    return value


def _apply_mapping(instance: Any, data: Mapping[str, Any], prefix: str = "") -> None:
    known = {f.name: f for f in fields(instance)}
    for key, value in data.items():
        path = f"{prefix}{key}"
        if key not in known:
            raise ConfigError(f"unknown configuration key: {path}")
        current = getattr(instance, key)
        if is_dataclass(current) and isinstance(value, Mapping):
            _apply_mapping(current, value, prefix=f"{path}.")
        elif isinstance(current, list) or isinstance(value, (list, tuple)):
            setattr(instance, key, list(value))
        else:
            setattr(instance, key, _coerce(known[key].type, value, path))


def _read_file(path: Path) -> Mapping[str, Any]:
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() in {".yaml", ".yml"}:
        try:
            import yaml  # noqa: PLC0415
        except ImportError as exc:  # pragma: no cover
            raise ConfigError(
                "PyYAML is required to read YAML config files (`pip install pyyaml`), "
                "or use a .json config instead."
            ) from exc
        # safe_load never constructs arbitrary Python objects.
        data = yaml.safe_load(text) or {}
    elif path.suffix.lower() == ".json":
        data = json.loads(text or "{}")
    else:
        raise ConfigError(f"unsupported config format: {path.suffix!r} (use .yaml or .json)")
    if not isinstance(data, Mapping):
        raise ConfigError(f"{path}: top level must be a mapping")
    return data


#: Environment overrides, deliberately narrow: deployment knobs only.
ENV_MAP: dict[str, tuple[str, ...]] = {
    "CAMERA_SOURCE": ("camera", "source"),
    "CAMERA_WIDTH": ("camera", "width"),
    "CAMERA_HEIGHT": ("camera", "height"),
    "TRACKING_BACKEND": ("tracking", "backend"),
    "RECORDING_ENABLED": ("recording", "enabled"),
    "RECORDING_LAYOUT": ("recording", "layout"),
    "OUTPUT_ROOT": ("output", "root"),
    "LOG_LEVEL": ("logging", "level"),
    "LOG_JSON": ("logging", "json"),
    "HEADLESS": ("headless",),
}


def _apply_env(config: AppConfig, environ: Mapping[str, str] | None = None) -> None:
    env = os.environ if environ is None else environ
    for suffix, path in ENV_MAP.items():
        raw = env.get(ENV_PREFIX + suffix)
        if raw is None:
            continue
        target: Any = config
        for part in path[:-1]:
            target = getattr(target, part)
        leaf = path[-1]
        declared = {f.name: f.type for f in fields(target)}[leaf]
        setattr(target, leaf, _coerce(declared, raw, ENV_PREFIX + suffix))


def load_config(
    config_path: str | Path | None = None,
    *,
    overrides: Mapping[str, Any] | None = None,
    environ: Mapping[str, str] | None = None,
) -> AppConfig:
    """Build a validated :class:`AppConfig` from all configuration layers."""
    config = AppConfig()

    if config_path:
        path = Path(config_path).expanduser()
        if not path.is_file():
            raise ConfigError(f"config file not found: {path}")
        _apply_mapping(config, _read_file(path))

    _apply_env(config, environ)

    if overrides:
        cleaned = {k: v for k, v in overrides.items() if v is not None}
        _apply_mapping(config, cleaned)

    return config.validate()


def nested_override(dotted: str, value: Any) -> dict[str, Any]:
    """Turn ``"camera.width", 640`` into ``{"camera": {"width": 640}}``."""
    parts: Sequence[str] = dotted.split(".")
    result: dict[str, Any] = {}
    cursor = result
    for part in parts[:-1]:
        cursor = cursor.setdefault(part, {})
    cursor[parts[-1]] = value
    return result


def merge(base: dict[str, Any], extra: Mapping[str, Any]) -> dict[str, Any]:
    """Recursively merge ``extra`` into ``base`` (returns ``base``)."""
    for key, value in extra.items():
        if isinstance(value, Mapping) and isinstance(base.get(key), dict):
            merge(base[key], value)
        else:
            base[key] = value
    return base


    #     *** _ ***