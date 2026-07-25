"""Command-line interface."""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any, Sequence

from . import __version__
from .infrastructure.config import ConfigError, load_config, merge, nested_override
from .infrastructure.logging_setup import configure_logging

__all__ = ["build_parser", "main"]

EPILOG = """\
gestures:
  index finger            draw
  index + middle          move cursor / dwell on a toolbar control to pick it
  open palm               erase
  thumbs up (hold ~1s)    save the artwork

keys:
  q / Esc quit    c clear    u undo    s save    e eraser    r pause recording
  h help          l landmarks           1-9 colour            [ ] brush size

examples:
  aircanvas                                   # default camera, overlay recording
  aircanvas --layout side_by_side --fps 24    # artwork next to the live video
  aircanvas --source clip.mp4 --headless      # batch/CI processing, no window
  aircanvas --layout canvas_only              # privacy: record artwork only
"""


def _parse_source(value: str) -> int | str:
    return int(value) if value.isdigit() else value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="aircanvas",
        description="Draw in the air with your hand — no touch, no stylus.",
        epilog=EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--version", action="version", version=f"aircanvas {__version__}")
    parser.add_argument("-c", "--config", metavar="PATH", help="YAML or JSON config file")

    source = parser.add_argument_group("input")
    source.add_argument("-s", "--source", type=_parse_source,
                        help="camera index (e.g. 0) or path to a video file")
    source.add_argument("--width", type=int, help="requested capture width")
    source.add_argument("--height", type=int, help="requested capture height")
    source.add_argument("--fps", type=float, help="requested capture frame rate")
    source.add_argument("--no-mirror", action="store_true",
                        help="disable the mirrored (selfie) view")
    source.add_argument("--loop", action="store_true", help="loop video-file input")

    tracking = parser.add_argument_group("tracking")
    tracking.add_argument("--tracker", choices=("mediapipe", "none"),
                          help="hand-tracking backend ('none' disables detection)")
    tracking.add_argument("--max-hands", type=int, help="maximum hands to detect")
    tracking.add_argument("--hand", choices=("any", "left", "right"),
                          help="which hand drives the cursor")
    tracking.add_argument("--model-complexity", type=int, choices=(0, 1),
                          help="0 = fastest, 1 = most accurate")

    output = parser.add_argument_group("output")
    output.add_argument("-o", "--output", metavar="DIR", help="session root directory")
    output.add_argument("--session", metavar="NAME", help="session directory name")
    output.add_argument("--layout", choices=("overlay", "side_by_side", "pip", "canvas_only"),
                        help="recorded video layout")
    output.add_argument("--record-fps", type=float, help="frame rate written to the video file")
    output.add_argument("--codec", help="preferred FourCC (mp4v, avc1, XVID, MJPG)")
    output.add_argument("--no-record", action="store_true", help="disable video recording")
    output.add_argument("--no-vectors", action="store_true", help="skip stroke JSON export")

    runtime = parser.add_argument_group("runtime")
    runtime.add_argument("--headless", action="store_true", help="run without a preview window")
    runtime.add_argument("--duration", type=float, metavar="SEC",
                         help="stop automatically after SEC seconds")
    runtime.add_argument("--no-landmarks", action="store_true", help="hide the hand skeleton")
    runtime.add_argument("--no-help", action="store_true", help="hide the on-screen help panel")
    runtime.add_argument("--log-level", choices=("DEBUG", "INFO", "WARNING", "ERROR"),
                         help="console log verbosity")
    runtime.add_argument("--json-logs", action="store_true", help="emit structured JSON logs")

    utility = parser.add_argument_group("utilities")
    utility.add_argument("--list-cameras", action="store_true",
                         help="probe camera indices and exit")
    utility.add_argument("--print-config", action="store_true",
                         help="print the effective configuration and exit")
    return parser


def _overrides(args: argparse.Namespace) -> dict[str, Any]:
    """Translate CLI flags into a nested override mapping."""
    direct: list[tuple[str, Any]] = [
        ("camera.source", args.source),
        ("camera.width", args.width),
        ("camera.height", args.height),
        ("camera.fps", args.fps),
        ("tracking.backend", args.tracker),
        ("tracking.max_hands", args.max_hands),
        ("tracking.preferred_hand", args.hand),
        ("tracking.model_complexity", args.model_complexity),
        ("output.root", args.output),
        ("output.session_name", args.session),
        ("recording.layout", args.layout),
        ("recording.fps", args.record_fps),
        ("recording.codec", args.codec),
        ("duration", args.duration),
        ("logging.level", args.log_level),
    ]
    flags: list[tuple[str, Any]] = [
        ("camera.mirror", False if args.no_mirror else None),
        ("camera.loop", True if args.loop else None),
        ("recording.enabled", False if args.no_record else None),
        ("output.export_vectors", False if args.no_vectors else None),
        ("ui.show_landmarks", False if args.no_landmarks else None),
        ("ui.show_help", False if args.no_help else None),
        ("logging.json", True if args.json_logs else None),
        ("headless", True if args.headless else None),
    ]

    result: dict[str, Any] = {}
    for dotted, value in [*direct, *flags]:
        if value is not None:
            merge(result, nested_override(dotted, value))
    return result


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    configure_logging(args.log_level or "INFO", json_output=bool(args.json_logs))

    if args.list_cameras:
        from .adapters.capture import probe_cameras  # noqa: PLC0415  (import cost only when used)

        cameras = probe_cameras()
        if not cameras:
            print("No cameras detected.", file=sys.stderr)
            return 1
        for cam in cameras:
            print(f"  index {cam['index']}: {cam['width']}x{cam['height']} @ {cam['fps']} fps")
        return 0

    try:
        config = load_config(args.config, overrides=_overrides(args))
    except ConfigError as exc:
        print(f"configuration error: {exc}", file=sys.stderr)
        return 2

    if args.print_config:
        print(json.dumps(config.to_dict(), indent=2, default=str))
        return 0

    from .app import AirCanvasApp  # noqa: PLC0415  (defer heavy imports)

    try:
        return AirCanvasApp(config).run()
    except KeyboardInterrupt:
        return 130
    except Exception as exc:  # pragma: no cover - top-level guard
        print(f"fatal: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


    #     *** _ ***