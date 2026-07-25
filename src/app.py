"""Application orchestration.

The render loop is deliberately linear and side-effect-light: every frame it
(1) acquires, (2) infers, (3) reduces to a stable gesture, (4) mutates canvas
state, (5) renders, (6) records. All the hard parts live in the modules it
composes, so this file stays readable.
"""

from __future__ import annotations

import logging
import time
from contextlib import ExitStack
from pathlib import Path

import cv2
import numpy as np

from . import __version__
from .adapters.artifacts import SessionMetrics, SessionStore
from .core.canvas import Canvas
from .adapters.capture import open_source
from .infrastructure.config import AppConfig
from .core.filters import FpsMeter, OneEuroFilter
from .core.gestures import GestureEngine, HoldTimer
from .infrastructure.logging_setup import configure_logging
from .adapters.recording import Layout, VideoRecorder, compose_output, output_size
from .adapters.tracking import create_tracker, select_primary
from .core.types import Gesture, Point
from .adapters.ui import DwellSelector, Hud, Toolbar, draw_landmarks

log = logging.getLogger(__name__)

__all__ = ["AirCanvasApp"]

#: Break the stroke if the cursor teleports further than this fraction of the
#: frame diagonal in one frame — it is a tracking glitch, not a hand movement.
MAX_JUMP_RATIO = 0.22

#: Consecutive empty reads tolerated from a *live* camera before giving up.
#: Each read already blocks for its own timeout, so this is a backstop against
#: a genuinely wedged device, not a busy-wait.
MAX_CONSECUTIVE_STALLS = 3


class AirCanvasApp:
    """Owns the lifetime of one drawing session."""

    def __init__(self, config: AppConfig) -> None:
        self.config = config.validate()
        self.metrics = SessionMetrics()

        self.color_index = config.drawing.default_color_index
        self.size_index = config.drawing.default_size_index
        self.eraser_mode = False
        self.recording_paused = False
        self._should_quit = False
        self._last_cursor: Point | None = None

        self.store: SessionStore | None = None
        self.canvas: Canvas | None = None
        self.toolbar: Toolbar | None = None
        self.recorder: VideoRecorder | None = None
        self.hud = Hud(show_help=config.ui.show_help)

    # ------------------------------------------------------------------ 
    # Derived state
    # ------------------------------------------------------------------ 
    @property
    def color(self) -> tuple[int, int, int]:
        colors = self.config.drawing.colors
        return tuple(int(c) for c in colors[self.color_index % len(colors)])  # type: ignore[return-value]

    @property
    def thickness(self) -> int:
        steps = self.config.drawing.thickness_steps
        return int(steps[self.size_index % len(steps)])

    # ------------------------------------------------------------------ 
    # Lifecycle
    # ------------------------------------------------------------------ 
    def run(self) -> int:
        cfg = self.config
        self.store = SessionStore(Path(cfg.output.root), cfg.output.session_name)
        configure_logging(
            cfg.logging.level,
            json_output=cfg.logging.json,
            log_file=self.store.path_for("session.log") if cfg.logging.to_file else None,
            session_id=self.store.session_id,
        )
        log.info("AirCanvas %s starting (session %s)", __version__, self.store.session_id)

        exit_code = 0
        with ExitStack() as stack:
            source = open_source(
                cfg.camera.source,
                width=cfg.camera.width,
                height=cfg.camera.height,
                fps=cfg.camera.fps,
                backend=cfg.camera.backend,
                loop=cfg.camera.loop,
                realtime=cfg.camera.realtime,
            )
            stack.callback(source.release)

            first = source.read()
            if first is None:
                log.error("No frames available from source %r", cfg.camera.source)
                return 2
            height, width = first.shape[:2]
            log.info("Stream resolution: %dx%d", width, height)
            # Model load below takes seconds; this probe frame would be stale
            # by the time the loop starts, so drop it and pull a fresh one.
            del first

            tracker = create_tracker(
                cfg.tracking.backend,
                max_hands=cfg.tracking.max_hands,
                detection_confidence=cfg.tracking.detection_confidence,
                tracking_confidence=cfg.tracking.tracking_confidence,
                model_complexity=cfg.tracking.model_complexity,
            )
            stack.callback(tracker.close)

            self.canvas = Canvas(
                width, height,
                max_history=cfg.drawing.max_history,
                antialias=cfg.drawing.antialias,
            )
            self.toolbar = Toolbar(
                width,
                height=cfg.ui.toolbar_height,
                colors=cfg.drawing.colors,
                sizes=cfg.drawing.thickness_steps,
            )

            if cfg.recording.enabled:
                self.recorder = VideoRecorder(
                    self.store.path_for(cfg.recording.filename),
                    fps=cfg.recording.fps,
                    frame_size=output_size(cfg.recording.layout, width, height),
                    codec=cfg.recording.codec,
                    queue_size=cfg.recording.queue_size,
                    max_seconds=cfg.recording.max_seconds,
                )
                stack.callback(self._finalise_recording)

            engine = GestureEngine(
                window=cfg.gestures.vote_window,
                min_votes=cfg.gestures.min_votes,
                margin=cfg.gestures.extension_margin,
                thumb_rise=cfg.gestures.thumb_rise,
            )
            dwell = DwellSelector(
                dwell_seconds=cfg.gestures.dwell_seconds,
                cooldown_seconds=max(0.35, cfg.gestures.dwell_seconds),
            )
            save_timer = HoldTimer(
                hold_seconds=cfg.gestures.save_hold_seconds,
                cooldown_seconds=cfg.gestures.action_cooldown_seconds,
            )
            smoother = OneEuroFilter(
                min_cutoff=cfg.smoothing.min_cutoff,
                beta=cfg.smoothing.beta,
                d_cutoff=cfg.smoothing.d_cutoff,
                freq=max(1.0, cfg.camera.fps),
            )
            fps_meter = FpsMeter()
            max_jump = MAX_JUMP_RATIO * float(np.hypot(width, height))

            if not cfg.headless:
                cv2.namedWindow(cfg.ui.window_title, cv2.WINDOW_NORMAL)
                cv2.resizeWindow(cfg.ui.window_title, width, height)
                stack.callback(cv2.destroyAllWindows)

            started = time.monotonic()
            frame: np.ndarray | None = None
            stalls = 0
            try:
                while not self._should_quit:
                    if cfg.duration is not None and time.monotonic() - started >= cfg.duration:
                        log.info("Reached --duration limit of %.1fs", cfg.duration)
                        break

                    if frame is None:
                        frame = source.read()
                    if frame is None:
                        # A live camera that is merely late is not a finished
                        # stream. Only a finite source, or a camera that stays
                        # silent across several blocking reads, ends the session.
                        stalls += 1
                        if getattr(source, "is_live", False) and stalls < MAX_CONSECUTIVE_STALLS:
                            log.warning(
                                "Camera stalled (%d/%d); retrying",
                                stalls, MAX_CONSECUTIVE_STALLS,
                            )
                            continue
                        log.info("Source exhausted after %d frames", self.metrics.frames)
                        break
                    stalls = 0

                    display = self._process(
                        frame, tracker, engine, dwell, save_timer, smoother, fps_meter, max_jump
                    )

                    if self.recorder is not None and not self.recording_paused:
                        if self.recorder.exhausted:
                            log.warning("Recording length cap reached; stopping recorder")
                            self._finalise_recording()
                        else:
                            written = self.recorder.write(
                                compose_output(
                                    display,
                                    self.canvas.image,
                                    cfg.recording.layout,
                                    paper=tuple(cfg.output.paper_color),  # type: ignore[arg-type]
                                )
                            )
                            if not written:
                                self.metrics.dropped_recording_frames += 1

                    if not cfg.headless:
                        cv2.imshow(cfg.ui.window_title, display)
                        if self._handle_key(cv2.waitKey(1) & 0xFF):
                            break
                        if cv2.getWindowProperty(cfg.ui.window_title, cv2.WND_PROP_VISIBLE) < 1:
                            log.info("Window closed by user")
                            break

                    frame = None
            except KeyboardInterrupt:
                log.info("Interrupted by user")
            except Exception:
                log.exception("Fatal error in render loop")
                exit_code = 1
            finally:
                self._shutdown(fps_meter)

        return exit_code

    # ------------------------------------------------------------------ 
    # Per-frame pipeline
    # ------------------------------------------------------------------ 
    def _process(
        self,
        frame: np.ndarray,
        tracker,
        engine: GestureEngine,
        dwell: DwellSelector,
        save_timer: HoldTimer,
        smoother: OneEuroFilter,
        fps_meter: FpsMeter,
        max_jump: float,
    ) -> np.ndarray:
        assert self.canvas is not None and self.toolbar is not None
        cfg = self.config
        now = time.monotonic()

        if cfg.camera.mirror:
            frame = cv2.flip(frame, 1)
        if frame.shape[:2] != (self.canvas.height, self.canvas.width):
            frame = cv2.resize(frame, (self.canvas.width, self.canvas.height))

        hands = tracker.process(frame)
        hand = select_primary(hands, cfg.tracking.preferred_hand)
        gesture = engine.update(hand)

        self.metrics.frames += 1
        self.metrics.note_gesture(gesture.value)
        if hand is not None:
            self.metrics.frames_with_hand += 1

        cursor: Point | None = None
        if hand is not None:
            tip = hand.index_tip
            if cfg.smoothing.enabled:
                cursor = Point(*smoother.filter_point(float(tip.x), float(tip.y), now))
            else:
                cursor = tip
            if self._last_cursor is not None and cursor.distance_to(self._last_cursor) > max_jump:
                self.canvas.end_stroke()  # tracking jump: never bridge it with a line
            self._last_cursor = cursor
        else:
            smoother.reset()
            self._last_cursor = None
            self.canvas.end_stroke()

        hover = None
        progress = 0.0

        if gesture is Gesture.DRAW and cursor is not None:
            if self.toolbar.contains(cursor):
                self.canvas.end_stroke()
            else:
                self.canvas.stroke_at(
                    cursor, self.color, self.thickness, erase=self.eraser_mode
                )
        elif gesture is Gesture.SELECT and cursor is not None:
            self.canvas.end_stroke()
            hover = self.toolbar.hit(cursor)
            fired, progress = dwell.update(hover.id if hover else None, now)
            if fired:
                self._activate(fired)
        elif gesture is Gesture.ERASE and hand is not None:
            self.canvas.end_stroke()
            self.canvas.stroke_at(
                hand.palm_center, (0, 0, 0), cfg.drawing.eraser_radius, erase=True
            )
        else:
            self.canvas.end_stroke()

        if save_timer.update(gesture is Gesture.SAVE, now):
            self._save(frame)

        return self._render(
            frame, hand, gesture, cursor, hover, progress,
            save_timer.progress(now), fps_meter.tick(now),
        )

    # ------------------------------------------------------------------ 
    # Rendering
    # ------------------------------------------------------------------ 
    def _render(
        self,
        frame: np.ndarray,
        hand,
        gesture: Gesture,
        cursor: Point | None,
        hover,
        dwell_progress: float,
        save_progress: float,
        fps: float,
    ) -> np.ndarray:
        assert self.canvas is not None and self.toolbar is not None
        display = self.canvas.composite(frame)

        if self.config.ui.show_landmarks and hand is not None:
            draw_landmarks(display, hand.landmarks)

        if cursor is not None:
            mode = {
                Gesture.ERASE: "erase",
                Gesture.SELECT: "select",
                Gesture.DRAW: "erase" if self.eraser_mode else "draw",
            }.get(gesture, "select")
            point = hand.palm_center if (gesture is Gesture.ERASE and hand) else cursor
            radius = (
                self.config.drawing.eraser_radius
                if mode == "erase"
                else self.thickness
            )
            self.hud.draw_cursor(display, point, self.color, radius, mode=mode)

        self.toolbar.render(
            display,
            color_index=self.color_index,
            size_index=self.size_index,
            eraser=self.eraser_mode,
            recording=self.recorder is not None and not self.recording_paused,
            hover=hover,
            progress=dwell_progress,
        )

        if save_progress > 0:
            centre = (display.shape[1] // 2, display.shape[0] // 2)
            self.hud.draw_progress_ring(display, centre, save_progress, "SAVING")

        self.hud.draw_status(
            display,
            fps=fps,
            gesture=gesture.value,
            color=self.color,
            thickness=self.thickness,
            eraser=self.eraser_mode,
            recording=self.recorder is not None and not self.recording_paused,
            elapsed=self.recorder.elapsed if self.recorder else 0.0,
            strokes=self.canvas.stroke_count,
        )
        return display

    # ------------------------------------------------------------------ 
    # Commands
    # ------------------------------------------------------------------ 
    def _activate(self, item_id: str) -> None:
        """Apply a toolbar selection (from dwell or keyboard)."""
        assert self.toolbar is not None
        kind, _, value = item_id.partition(":")
        if kind == "color":
            self.color_index = int(value)
            self.eraser_mode = False
            self.hud.toast.show("Colour changed")
        elif kind == "size":
            self.size_index = int(value)
            self.hud.toast.show(f"Brush {self.thickness}px")
        elif kind == "action":
            self._run_action(value)

    def _run_action(self, action: str) -> None:
        assert self.canvas is not None
        if action == "eraser":
            self.eraser_mode = not self.eraser_mode
            self.hud.toast.show("Eraser on" if self.eraser_mode else "Eraser off")
        elif action == "undo":
            changed = self.canvas.undo()
            self.metrics.undos += int(changed)
            self.hud.toast.show("Undo" if changed else "Nothing to undo")
        elif action == "clear":
            self.canvas.clear()
            self.metrics.clears += 1
            self.hud.toast.show("Canvas cleared")
        elif action == "save":
            self._save(None)
        elif action == "record":
            if self.recorder is None:
                self.hud.toast.show("Recording disabled", level="error")
            else:
                self.recording_paused = not self.recording_paused
                self.hud.toast.show("Recording paused" if self.recording_paused else "Recording")

    def _save(self, frame: np.ndarray | None) -> None:
        assert self.canvas is not None and self.store is not None
        if self.canvas.is_empty:
            self.hud.toast.show("Nothing to save yet", level="error")
            return
        self.canvas.end_stroke()
        try:
            written = self.store.save_snapshot(
                self.canvas,
                frame,
                export_vectors=self.config.output.export_vectors,
                background=tuple(self.config.output.paper_color),  # type: ignore[arg-type]
            )
        except Exception:
            log.exception("Failed to save artefacts")
            self.hud.toast.show("Save failed - see log", level="error")
            return
        self.metrics.saves += 1
        self.hud.toast.show(f"Saved {len(written)} file(s)", seconds=2.5)

    def _handle_key(self, key: int) -> bool:
        """Handle a keypress. Returns ``True`` to exit the loop."""
        if key in (255, -1):
            return False
        char = chr(key) if 32 <= key < 127 else ""
        if key == 27 or char == "q":
            return True
        if char == "c":
            self._run_action("clear")
        elif char == "u":
            self._run_action("undo")
        elif char == "s":
            self._run_action("save")
        elif char == "e":
            self._run_action("eraser")
        elif char == "r":
            self._run_action("record")
        elif char == "h":
            self.hud.show_help = not self.hud.show_help
        elif char == "l":
            self.config.ui.show_landmarks = not self.config.ui.show_landmarks
        elif char in "123456789":
            index = int(char) - 1
            if index < len(self.config.drawing.colors):
                self._activate(f"color:{index}")
        elif char in "[]":
            steps = len(self.config.drawing.thickness_steps)
            delta = -1 if char == "[" else 1
            self._activate(f"size:{(self.size_index + delta) % steps}")
        return False

    # ------------------------------------------------------------------ 
    # Shutdown
    # ------------------------------------------------------------------ 
    def _finalise_recording(self) -> None:
        if self.recorder is not None:
            self.metrics.dropped_recording_frames = max(
                self.metrics.dropped_recording_frames, self.recorder.dropped
            )
            self.recorder.close()
            self.recorder = None

    def _shutdown(self, fps_meter: FpsMeter) -> None:
        if self.canvas is not None:
            self.canvas.end_stroke()
            if not self.canvas.is_empty and self.metrics.saves == 0:
                log.info("Auto-saving artwork on exit")
                self._save(None)

        self._finalise_recording()

        if self.store is not None:
            payload = {
                "version": __version__,
                "config": self.config.to_dict(),
                "metrics": self.metrics.as_dict(average_fps=fps_meter.fps),
            }
            try:
                self.store.write_manifest(payload)
            except Exception:  # pragma: no cover - defensive
                log.exception("Could not write session manifest")
            log.info(
                "Session complete: %d frames, %.1f fps avg, artefacts in %s",
                self.metrics.frames, fps_meter.fps, self.store.directory,
            )
            
            
            
            
            #   *** _ ***