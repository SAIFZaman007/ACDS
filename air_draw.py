#!/usr/bin/env python3
"""
Air Drawing System
==================
Draw in mid-air with your index finger, erase with an open palm, pick
colors by pointing at the on-screen toolbar, and save your artwork with
a thumbs-up. Real-time hand tracking is powered by MediaPipe; compositing
and I/O by OpenCV.

The whole session — your drawing blended live over your own video feed —
is recorded to a video file automatically. Flashing a thumbs-up also saves
a clean PNG of just the artwork.

Usage:
    uv run air_draw.py
    uv run air_draw.py --camera 1 --width 1920 --height 1080
    uv run air_draw.py --no-record

Controls (gestures do the real work; these are just backups):
    q / ESC   quit
    c         clear canvas
    s         save drawing
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime
from enum import Enum, auto
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("air_draw")

# Pinned to a specific model version (not "latest") so behavior is
# reproducible across runs and machines.
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)

# The 21-point hand skeleton's bone connectivity, used only to draw the
# on-screen hand overlay (MediaPipe no longer ships a drawing helper for
# the modern Tasks API, so this is a small, self-contained replacement).
HAND_CONNECTIONS: tuple[tuple[int, int], ...] = (
    (0, 1), (1, 2), (2, 3), (3, 4),            # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),            # index
    (5, 9), (9, 10), (10, 11), (11, 12),       # middle
    (9, 13), (13, 14), (14, 15), (15, 16),     # ring
    (13, 17), (17, 18), (18, 19), (19, 20),    # pinky
    (0, 17),                                    # palm base
)


# ============================================================================
# Configuration
# ============================================================================

@dataclass(frozen=True)
class Config:
    """All the knobs in one place. Defaults are tuned for a laptop webcam
    at arm's length; override any of them from the CLI (see parse_args)."""

    camera_index: int = 0
    frame_width: int = 1280
    frame_height: int = 720
    header_height: int = 90

    brush_thickness: int = 8
    eraser_radius: int = 45
    smoothing: float = 0.5          # 0 = raw/jittery, closer to 1 = smoother/laggier

    min_hand_detection_confidence: float = 0.6
    min_hand_presence_confidence: float = 0.5
    min_tracking_confidence: float = 0.5

    save_cooldown_s: float = 2.0    # min seconds between thumbs-up saves
    gesture_hold_frames: int = 8    # consecutive frames thumbs-up must hold to fire
    miss_tolerance_frames: int = 4  # brief tracking dropouts mid-stroke are absorbed, not treated as "stop drawing"

    record: bool = True
    model_path: Path = Path(__file__).resolve().parent / "models" / "hand_landmarker.task"
    output_root: Path = Path(__file__).resolve().parent / "outputs"

    palette: tuple[tuple[str, tuple[int, int, int]], ...] = (
        ("Red", (0, 0, 230)),
        ("Orange", (0, 140, 255)),
        ("Yellow", (0, 220, 220)),
        ("Green", (0, 180, 0)),
        ("Blue", (255, 120, 0)),
        ("Purple", (200, 0, 160)),
        ("Eraser", (30, 30, 30)),
    )

    @property
    def drawings_dir(self) -> Path:
        return self.output_root / "drawings"

    @property
    def recordings_dir(self) -> Path:
        return self.output_root / "recordings"


class Gesture(Enum):
    IDLE = auto()
    DRAW = auto()
    ERASE = auto()
    SAVE = auto()


# ============================================================================
# Gesture recognition — pure functions, no OpenCV/MediaPipe objects touched.
# Each takes the 21-item landmark list MediaPipe returns (each landmark has
# normalized .x/.y in [0, 1]) so this logic is trivially unit-testable.
# ============================================================================

def _dist(a, b) -> float:
    return ((a.x - b.x) ** 2 + (a.y - b.y) ** 2) ** 0.5


def fingers_extended(lm: list) -> list[bool]:
    """Returns [thumb, index, middle, ring, pinky] as extended/curled."""
    # "Palm length" (wrist -> middle knuckle) as a scale reference, so the
    # thumb threshold works regardless of hand size or camera distance.
    scale = _dist(lm[0], lm[9]) or 1e-6
    # Thumb: distance from tip to the index knuckle, relative to palm
    # length. Tucked into a fist -> small. Splayed out / pointing up ->
    # large. This avoids needing left/right handedness at all.
    thumb = _dist(lm[4], lm[5]) > scale * 0.5
    index = lm[8].y < lm[6].y
    middle = lm[12].y < lm[10].y
    ring = lm[16].y < lm[14].y
    pinky = lm[20].y < lm[18].y
    return [thumb, index, middle, ring, pinky]


def classify_gesture(lm: list, fingers: list[bool]) -> Gesture:
    thumb, index, middle, ring, pinky = fingers

    if thumb and not (index or middle or ring or pinky):
        # Require the thumb to also point upward (tip above knuckle above
        # wrist), so a loose fist with the thumb resting off to the side
        # doesn't false-trigger a save.
        if lm[4].y < lm[2].y < lm[0].y:
            return Gesture.SAVE

    if index and middle and ring and pinky:
        return Gesture.ERASE

    if index and not (middle or ring or pinky):
        return Gesture.DRAW  # DRAW vs. toolbar SELECT is decided by y-position

    return Gesture.IDLE


# ============================================================================
# Model download
# ============================================================================

def ensure_model(path: Path) -> Path:
    """Downloads the hand-landmark model once and caches it locally."""
    if path.exists() and path.stat().st_size > 0:
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    log.info("Hand-tracking model not found locally — downloading it once...")
    try:
        urllib.request.urlretrieve(MODEL_URL, path)
    except (urllib.error.URLError, OSError) as exc:
        raise RuntimeError(
            f"Could not download the hand-tracking model automatically ({exc}).\n"
            f"  Download it manually and save it to: {path}\n"
            f"  curl -L -o \"{path}\" {MODEL_URL}"
        ) from exc
    log.info("Model ready at %s", path)
    return path


# ============================================================================
# Hand tracking
# ============================================================================

class HandTracker:
    """Wraps MediaPipe's HandLandmarker (Tasks API) and turns raw landmarks
    into the handful of primitives the app cares about."""

    def __init__(self, cfg: Config):
        model_path = ensure_model(cfg.model_path)

        base_options = mp.tasks.BaseOptions(model_asset_path=str(model_path))
        options = mp.tasks.vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=mp.tasks.vision.RunningMode.VIDEO,
            num_hands=1,
            min_hand_detection_confidence=cfg.min_hand_detection_confidence,
            min_hand_presence_confidence=cfg.min_hand_presence_confidence,
            min_tracking_confidence=cfg.min_tracking_confidence,
        )
        self._landmarker = mp.tasks.vision.HandLandmarker.create_from_options(options)
        self._last_ts_ms = -1

    def _next_timestamp_ms(self) -> int:
        # VIDEO mode requires strictly increasing timestamps; guard against
        # clock-resolution ties rather than trusting wall-clock precision.
        ts = int(time.time() * 1000)
        if ts <= self._last_ts_ms:
            ts = self._last_ts_ms + 1
        self._last_ts_ms = ts
        return ts

    def find_hand(self, frame_bgr: np.ndarray) -> list | None:
        """Returns the 21 landmarks of the first detected hand, or None."""
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
        result = self._landmarker.detect_for_video(mp_image, self._next_timestamp_ms())
        if not result.hand_landmarks:
            return None
        return result.hand_landmarks[0]

    @staticmethod
    def draw_skeleton(frame: np.ndarray, landmarks: list, width: int, height: int) -> None:
        pts = [(int(p.x * width), int(p.y * height)) for p in landmarks]
        for a, b in HAND_CONNECTIONS:
            cv2.line(frame, pts[a], pts[b], (255, 255, 255), 2, cv2.LINE_AA)
        for x, y in pts:
            cv2.circle(frame, (x, y), 4, (0, 255, 255), -1, cv2.LINE_AA)

    def close(self) -> None:
        self._landmarker.close()


# ============================================================================
# Drawing canvas
# ============================================================================

class Canvas:
    """Owns the persistent drawing surface and knows how to blend it onto
    a live camera frame (or a plain white background for saved artwork)."""

    def __init__(self, width: int, height: int):
        self.surface = np.zeros((height, width, 3), dtype=np.uint8)

    def draw_line(self, pt1: tuple[int, int], pt2: tuple[int, int],
                  color: tuple[int, int, int], thickness: int) -> None:
        cv2.line(self.surface, pt1, pt2, color, thickness, lineType=cv2.LINE_AA)

    def erase(self, center: tuple[int, int], radius: int) -> None:
        cv2.circle(self.surface, center, radius, (0, 0, 0), thickness=-1)

    def clear(self) -> None:
        self.surface[:] = 0

    def composite_onto(self, frame: np.ndarray) -> np.ndarray:
        """Overlay strokes onto `frame` — painted pixels replace the video,
        everything else shows the live feed untouched."""
        gray = cv2.cvtColor(self.surface, cv2.COLOR_BGR2GRAY)
        _, mask_inv = cv2.threshold(gray, 15, 255, cv2.THRESH_BINARY_INV)
        mask_inv_3ch = cv2.cvtColor(mask_inv, cv2.COLOR_GRAY2BGR)
        background = cv2.bitwise_and(frame, mask_inv_3ch)
        return cv2.bitwise_or(background, self.surface)

    def on_white(self) -> np.ndarray:
        """Render just the drawing on a clean white background — used when
        saving 'the work' itself rather than a video frame with it on."""
        white = np.full_like(self.surface, 255)
        gray = cv2.cvtColor(self.surface, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, 15, 255, cv2.THRESH_BINARY)
        mask_3ch = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        bg = cv2.bitwise_and(white, cv2.bitwise_not(mask_3ch))
        fg = cv2.bitwise_and(self.surface, mask_3ch)
        return cv2.bitwise_or(bg, fg)


# ============================================================================
# Color toolbar
# ============================================================================

class Toolbar:
    """Renders the color-swatch header and hit-tests a fingertip against it."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.height = cfg.header_height
        self._n = len(cfg.palette)
        self._swatch_width = cfg.frame_width // self._n
        self.active_index = 0

    @property
    def active_color(self) -> tuple[int, int, int]:
        return self.cfg.palette[self.active_index][1]

    @property
    def active_name(self) -> str:
        return self.cfg.palette[self.active_index][0]

    def _bounds(self, i: int) -> tuple[int, int]:
        x1 = i * self._swatch_width
        x2 = self.cfg.frame_width if i == self._n - 1 else x1 + self._swatch_width
        return x1, x2

    def hit_test(self, x: int, y: int) -> int | None:
        if y > self.height or not (0 <= x < self.cfg.frame_width):
            return None
        return min(x // self._swatch_width, self._n - 1)

    def select(self, x: int, y: int) -> bool:
        idx = self.hit_test(x, y)
        if idx is None:
            return False
        self.active_index = idx
        return True

    def draw(self, frame: np.ndarray) -> None:
        for i, (_, color) in enumerate(self.cfg.palette):
            x1, x2 = self._bounds(i)
            cv2.rectangle(frame, (x1, 0), (x2, self.height), color, thickness=-1)
            is_active = i == self.active_index
            border_color = (255, 255, 255) if is_active else (70, 70, 70)
            cv2.rectangle(frame, (x1, 0), (x2, self.height), border_color, 4 if is_active else 1)


# ============================================================================
# Session recorder
# ============================================================================

class Recorder:
    """Wraps cv2.VideoWriter with a codec fallback so a missing codec
    disables recording gracefully instead of crashing the session."""

    _CODECS = (("mp4v", ".mp4"), ("XVID", ".avi"))

    def __init__(self, cfg: Config, fps: float = 20.0):
        self.cfg = cfg
        self._fps = fps
        self._writer: cv2.VideoWriter | None = None
        self.path: Path | None = None

    @property
    def is_active(self) -> bool:
        return self._writer is not None

    def start(self) -> Path | None:
        self.cfg.recordings_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        for codec, ext in self._CODECS:
            path = self.cfg.recordings_dir / f"session_{stamp}{ext}"
            fourcc = cv2.VideoWriter_fourcc(*codec)
            writer = cv2.VideoWriter(str(path), fourcc, self._fps,
                                      (self.cfg.frame_width, self.cfg.frame_height))
            if writer.isOpened():
                self._writer, self.path = writer, path
                log.info("Recording session to %s", path)
                return path
            writer.release()

        log.warning("No usable video codec found — recording disabled for this session.")
        return None

    def write(self, frame: np.ndarray) -> None:
        if self._writer is not None:
            self._writer.write(frame)

    def stop(self) -> None:
        if self._writer is not None:
            self._writer.release()
            log.info("Recording saved: %s", self.path)
            self._writer = None


# ============================================================================
# Application
# ============================================================================

class AirDrawApp:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.tracker = HandTracker(cfg)
        self.canvas = Canvas(cfg.frame_width, cfg.frame_height)
        self.toolbar = Toolbar(cfg)
        self.recorder = Recorder(cfg) if cfg.record else None

        self._prev_point: tuple[int, int] | None = None
        self._smoothed_point: tuple[int, int] | None = None
        self._miss_count = 0
        self._save_hold = 0
        self._last_save_time = 0.0
        self._flash_message = ""
        self._flash_until = 0.0

        self._cap = self._open_camera()

    def _open_camera(self) -> cv2.VideoCapture:
        cap = cv2.VideoCapture(self.cfg.camera_index)
        if not cap.isOpened():
            raise RuntimeError(
                f"Could not open camera index {self.cfg.camera_index}. "
                "Check it's connected, not in use by another app, and that "
                "this app has camera permission."
            )
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.cfg.frame_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.cfg.frame_height)
        return cap

    # -- main loop -----------------------------------------------------

    def run(self) -> None:
        if self.recorder:
            self.recorder.start()
        log.info("Running. Index finger draws, open palm erases, thumbs-up saves. "
                  "Press 'q' to quit.")
        try:
            self._loop()
        except KeyboardInterrupt:
            log.info("Interrupted by user.")
        finally:
            self._shutdown()

    def _loop(self) -> None:
        prev_time = time.time()
        while True:
            ok, frame = self._cap.read()
            if not ok:
                log.warning("Camera frame grab failed; skipping frame.")
                continue

            frame = cv2.flip(frame, 1)  # mirror, so it feels natural to use
            frame = cv2.resize(frame, (self.cfg.frame_width, self.cfg.frame_height))

            landmarks = self.tracker.find_hand(frame)
            gesture = Gesture.IDLE
            point = None

            if landmarks is not None:
                fingers = fingers_extended(landmarks)
                gesture = classify_gesture(landmarks, fingers)
                tip = landmarks[8]
                point = (int(tip.x * self.cfg.frame_width), int(tip.y * self.cfg.frame_height))
                self.tracker.draw_skeleton(frame, landmarks, self.cfg.frame_width, self.cfg.frame_height)

            self._handle_gesture(gesture, landmarks, point)

            frame = self.canvas.composite_onto(frame)
            self.toolbar.draw(frame)
            fps = self._draw_hud(frame, gesture, prev_time)
            prev_time = time.time()

            if self.recorder:
                self.recorder.write(frame)

            cv2.imshow("Air Drawing System", frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            elif key == ord("c"):
                self.canvas.clear()
                log.info("Canvas cleared.")
            elif key == ord("s"):
                self._save_drawing()

    # -- gesture handling ------------------------------------------------

    def _end_stroke_now(self) -> None:
        """Immediately ends the current stroke — used for *deliberate* mode
        changes (switching to erase/save/toolbar), where starting fresh is
        correct, not a bug."""
        self._prev_point = None
        self._smoothed_point = None
        self._miss_count = 0

    def _handle_gesture(self, gesture: Gesture, landmarks, point: tuple[int, int] | None) -> None:
        if gesture == Gesture.SAVE:
            self._save_hold += 1
            self._end_stroke_now()
            if (self._save_hold >= self.cfg.gesture_hold_frames
                    and time.time() - self._last_save_time > self.cfg.save_cooldown_s):
                self._save_drawing()
                self._last_save_time = time.time()
                self._save_hold = 0
            return
        self._save_hold = 0

        if gesture == Gesture.ERASE and landmarks is not None:
            palm = landmarks[9]  # middle-finger MCP ~= palm center
            center = (int(palm.x * self.cfg.frame_width), int(palm.y * self.cfg.frame_height))
            self.canvas.erase(center, self.cfg.eraser_radius)
            self._end_stroke_now()
            return

        if gesture == Gesture.DRAW and point is not None:
            x, y = point
            if y <= self.cfg.header_height:
                self.toolbar.select(x, y)
                self._end_stroke_now()
                return
            self._miss_count = 0
            smoothed = self._smooth(point)
            if self._prev_point is not None:
                self.canvas.draw_line(self._prev_point, smoothed,
                                       self.toolbar.active_color, self.cfg.brush_thickness)
            self._prev_point = smoothed
            return

        # No hand this frame, or an ambiguous/transitional hand pose. This
        # is often just a flaky frame in the middle of a stroke (motion
        # blur, a momentary tracking miss) rather than the user actually
        # stopping, so coast for a few frames before ending the stroke —
        # otherwise a single dropped frame splits one line into a dot and
        # a gap.
        if self._prev_point is not None:
            self._miss_count += 1
            if self._miss_count > self.cfg.miss_tolerance_frames:
                self._end_stroke_now()

    def _smooth(self, point: tuple[int, int]) -> tuple[int, int]:
        if self._smoothed_point is None:
            self._smoothed_point = point
        else:
            a = self.cfg.smoothing
            px, py = self._smoothed_point
            x, y = point
            self._smoothed_point = (int(px * a + x * (1 - a)), int(py * a + y * (1 - a)))
        return self._smoothed_point

    def _save_drawing(self) -> None:
        self.cfg.drawings_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = self.cfg.drawings_dir / f"drawing_{stamp}.png"
        cv2.imwrite(str(path), self.canvas.on_white())
        log.info("Drawing saved: %s", path)
        self._flash_message = f"Saved {path.name}"
        self._flash_until = time.time() + 1.5

    # -- HUD ---------------------------------------------------------------

    def _draw_hud(self, frame: np.ndarray, gesture: Gesture, prev_time: float) -> float:
        fps = 1.0 / max(time.time() - prev_time, 1e-6)

        label = f"Mode: {gesture.name}   Color: {self.toolbar.active_name}"
        cv2.putText(frame, label, (10, self.cfg.header_height + 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

        cv2.rectangle(frame, (0, self.cfg.frame_height - 30),
                      (self.cfg.frame_width, self.cfg.frame_height), (0, 0, 0), -1)
        legend = "Index: Draw | Palm: Erase | Thumbs Up: Save  |  q: Quit  c: Clear  s: Save"
        cv2.putText(frame, legend, (10, self.cfg.frame_height - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1, cv2.LINE_AA)

        cv2.putText(frame, f"{fps:4.1f} FPS", (self.cfg.frame_width - 110, self.cfg.frame_height - 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)

        if self.recorder and self.recorder.is_active:
            cv2.circle(frame, (self.cfg.frame_width - 20, 20), 7, (0, 0, 255), -1, cv2.LINE_AA)
            cv2.putText(frame, "REC", (self.cfg.frame_width - 60, 26),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)

        if time.time() < self._flash_until:
            cv2.putText(frame, self._flash_message, (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)

        return fps

    # -- cleanup -------------------------------------------------------

    def _shutdown(self) -> None:
        self._cap.release()
        self.tracker.close()
        if self.recorder:
            self.recorder.stop()
        cv2.destroyAllWindows()
        log.info("Shut down cleanly.")


# ============================================================================
# CLI
# ============================================================================

def parse_args(argv: list[str] | None = None) -> Config:
    defaults = Config()
    p = argparse.ArgumentParser(description="Air Drawing System — draw in the air with hand gestures.")
    p.add_argument("--camera", type=int, default=defaults.camera_index, help="Camera index (default: 0)")
    p.add_argument("--width", type=int, default=defaults.frame_width, help="Capture width")
    p.add_argument("--height", type=int, default=defaults.frame_height, help="Capture height")
    p.add_argument("--no-record", action="store_true", help="Disable session video recording")
    p.add_argument("--output-dir", type=str, default=None, help="Base folder for saved drawings/recordings")
    p.add_argument("--model", type=str, default=None, help="Path to a local hand_landmarker.task file")
    args = p.parse_args(argv)

    return Config(
        camera_index=args.camera,
        frame_width=args.width,
        frame_height=args.height,
        record=not args.no_record,
        output_root=Path(args.output_dir) if args.output_dir else defaults.output_root,
        model_path=Path(args.model) if args.model else defaults.model_path,
    )


def main() -> None:
    cfg = parse_args()
    try:
        app = AirDrawApp(cfg)
    except RuntimeError as exc:
        log.error(str(exc))
        sys.exit(1)
    app.run()


if __name__ == "__main__":
    main()