# AirCanvas

**Draw in the air with your hand. No touch, no stylus, no mouse.**

A real-time computer-vision drawing studio. Your index finger is the pen, your
open palm is the eraser, a two-finger cursor picks colours off an on-screen
toolbar, and a thumbs-up saves your work. The whole session is recorded, so the
output contains both the artwork and the live video.

Built with Python, OpenCV, MediaPipe and NumPy.

---

## Contents

- [Quickstart](#quickstart)
- [Installation](#installation)
- [How to use it](#how-to-use-it)
- [Command line](#command-line)
- [Configuration](#configuration)
- [What a session produces](#what-a-session-produces)
- [Architecture](#architecture)
- [Design decisions](#design-decisions-and-why)
- [Development](#development)
- [Troubleshooting](#troubleshooting)
- [Privacy and security](#privacy-and-security)
- [Extending it](#extending-it)

---

## Quickstart

```bash
git clone <your-repo-url> aircanvas && cd aircanvas
make setup            # virtualenv + editable install + dev tooling
make cameras          # confirm which index your webcam actually is
make run              # go
```

Press `q` to quit. Your artwork and the session video land in `sessions/`.

No `make`? The three equivalent commands:

```bash
python3 -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
aircanvas
```

---

## Installation

**Requirements:** Python 3.10–3.12, a webcam, and roughly 500 MB of disk for
dependencies (MediaPipe ships its own models).

> Python 3.13 is not supported yet — MediaPipe does not publish reliable wheels
> for it. The version bound in `pyproject.toml` is deliberate, not laziness.

### Platform notes

| Platform | Notes |
|---|---|
| **Linux** | Works out of the box. If `cv2` fails to import, install `libgl1` and `libglib2.0-0`. Camera permission comes from group `video`. |
| **macOS** | The first run triggers a camera-permission prompt. If you launch from a terminal, grant camera access to *the terminal app*, not to Python — System Settings → Privacy & Security → Camera. Apple Silicon works natively. |
| **Windows** | Works out of the box. If the camera takes several seconds to open, force the DirectShow backend: `--config` with `camera.backend: dshow`. |

### Servers, containers and CI

Install the headless OpenCV build instead — no GUI libraries required:

```bash
make setup-headless
aircanvas --source clip.mp4 --headless --layout side_by_side
```

Or use the container (see the usage notes at the bottom of the `Dockerfile`):

```bash
docker build -t aircanvas:1.0.0 .
docker run --rm -v "$PWD/clips:/clips:ro" -v "$PWD/sessions:/sessions" \
  aircanvas:1.0.0 --source /clips/demo.mp4 --layout side_by_side
```

### Verify without a camera

The pipeline runs end to end with no hardware and no hand-tracking model:

```bash
aircanvas --source any-video.mp4 --tracker none --headless --duration 5
```

---

## How to use it

Sit about an arm's length from the camera, with your hand well lit and fully in
frame. The view is mirrored, so moving right moves the cursor right.

### Gestures

| Gesture | What it does | Notes |
|---|---|---|
| ☝️ **Index finger only** | **Draw** | Pen down. The thumb may stick out; it is ignored. |
| ✌️ **Index + middle** | **Cursor** | Pen up. Hover a toolbar control for ~0.45 s to activate it. |
| ✋ **Open palm** | **Erase** | Erases a disc at your palm centre. |
| 👍 **Thumbs-up, held ~1 s** | **Save** | A progress ring fills, then the artwork is exported. |
| ✊ **Fist / anything else** | **Idle** | Pen up. Safe resting position. |

Drawing is suppressed inside the toolbar strip, so reaching for a colour can
never leave a stray mark across your canvas.

### Keyboard

| Key | Action | Key | Action |
|---|---|---|---|
| `q` / `Esc` | Quit | `1`–`9` | Pick colour |
| `c` | Clear canvas | `[` `]` | Brush size down / up |
| `u` | Undo last stroke | `h` | Toggle help panel |
| `s` | Save now | `l` | Toggle hand skeleton |
| `e` | Toggle eraser | `r` | Pause / resume recording |

### Getting good results

- **Light your hand, not the wall behind it.** Backlighting is the single most
  common cause of dropouts.
- **Keep your whole hand in frame.** MediaPipe needs the wrist to anchor the
  skeleton; a cropped palm makes gestures unreliable.
- **Move deliberately when switching gestures.** Recognition needs 3 agreeing
  frames out of 5 — roughly a tenth of a second — before it commits.
- **If lines look shaky**, lower `smoothing.min_cutoff`. **If they feel laggy**,
  raise `smoothing.beta`.

---

## Command line

```
aircanvas [-h] [--version] [-c PATH]
          [-s SOURCE] [--width W] [--height H] [--fps F] [--no-mirror] [--loop]
          [--tracker {mediapipe,none}] [--max-hands N] [--hand {any,left,right}]
          [--model-complexity {0,1}]
          [-o DIR] [--session NAME] [--layout LAYOUT] [--record-fps F]
          [--codec FOURCC] [--no-record] [--no-vectors]
          [--headless] [--duration SEC] [--no-landmarks] [--no-help]
          [--log-level LEVEL] [--json-logs]
          [--list-cameras] [--print-config]
```

Recipes:

```bash
aircanvas                                    # default camera, overlay recording
aircanvas --layout side_by_side              # artwork beside the live video
aircanvas --layout canvas_only               # privacy: record artwork, no camera
aircanvas --source 1 --width 1920 --height 1080
aircanvas --source clip.mp4 --headless       # batch processing, no window
aircanvas --model-complexity 0               # ~2x faster on a weak CPU
aircanvas --print-config                     # dump the effective settings
aircanvas --list-cameras                     # which indices actually open
aircanvas --json-logs --log-level DEBUG      # structured logs for aggregation
```

---

## Configuration

Four layers, each overriding the one before:

```
dataclass defaults  <  YAML/JSON file  <  AIRCANVAS_* env vars  <  CLI flags
```

`configs/default.yaml` documents every tunable. Copy it and edit:

```bash
cp configs/default.yaml configs/local.yaml
aircanvas --config configs/local.yaml
```

Everything is a typed dataclass validated at start-up, so a bad value fails
immediately with a precise message instead of surfacing forty frames into the
render loop:

```
$ aircanvas --width 1
configuration error: camera.width must be within [160, 7680], got 1
```

Environment overrides cover the deployment knobs:

| Variable | Maps to |
|---|---|
| `AIRCANVAS_CAMERA_SOURCE` | `camera.source` |
| `AIRCANVAS_CAMERA_WIDTH` / `_HEIGHT` | `camera.width` / `.height` |
| `AIRCANVAS_TRACKING_BACKEND` | `tracking.backend` |
| `AIRCANVAS_RECORDING_ENABLED` / `_LAYOUT` | `recording.enabled` / `.layout` |
| `AIRCANVAS_OUTPUT_ROOT` | `output.root` |
| `AIRCANVAS_LOG_LEVEL` / `AIRCANVAS_LOG_JSON` | `logging.level` / `.json` |
| `AIRCANVAS_HEADLESS` | `headless` |

### Tuning cheat-sheet

| Symptom | Change |
|---|---|
| Lines are jittery | `smoothing.min_cutoff` ↓ (try `0.6`) |
| Cursor lags behind the hand | `smoothing.beta` ↑ (try `0.05`) |
| Gestures switch too eagerly | `gestures.min_votes` ↑ |
| Gestures feel sluggish to switch | `gestures.vote_window` ↓ |
| Fingers flicker extended/folded | `gestures.extension_margin` ↑ (try `1.12`) |
| Toolbar activates by accident | `gestures.dwell_seconds` ↑ |
| Saves fire accidentally | `gestures.save_hold_seconds` ↑ |
| Low frame rate | `tracking.model_complexity: 0`, or drop capture resolution |

---

## What a session produces

```
sessions/session-20260722-035443/
├── drawing-20260722-035501-882.png    # artwork only, true alpha channel
├── composite-20260722-035501-882.png  # artwork over the live frame — shareable
├── strokes-20260722-035501-882.json   # vector history: replayable, diffable
├── session-video.mp4                  # the full recorded session
├── session.json                       # manifest: config echo + metrics
└── session.log                        # rotating log for this session
```

Recording layouts (`--layout`):

| Layout | Output |
|---|---|
| `overlay` *(default)* | Camera with the artwork composited on top |
| `side_by_side` | Artwork on paper beside the live overlay (double width) |
| `pip` | Artwork full-frame, camera as a picture-in-picture inset |
| `canvas_only` | Artwork only — **records no video of the user** |

A save writes a bundle rather than one image because the useful output differs
by audience: the transparent PNG drops straight into a design tool, the
composite is what you post, and the JSON makes the session reproducible.

`session.json` is the operational record:

```json
{
  "version": "1.0.0",
  "config": { "...": "full effective configuration" },
  "metrics": {
    "frames": 1842, "frames_with_hand": 1665, "hand_detection_rate": 0.9039,
    "dropped_recording_frames": 0, "saves": 2, "undos": 3, "clears": 1,
    "gesture_counts": {"draw": 921, "erase": 88, "none": 640, "save": 12, "select": 181},
    "duration_seconds": 61.4, "average_fps": 29.8
  },
  "artifacts": ["composite-....png", "drawing-....png", "strokes-....json"]
}
```

---

## Architecture

```
                 ┌──────────────┐
   webcam ──────►│   capture    │  threaded reader, always the freshest frame
   or file       └──────┬───────┘
                        ▼
                 ┌──────────────┐
                 │   tracking   │  HandTracker protocol → MediaPipe adapter
                 └──────┬───────┘
                        ▼  HandObservation (21 landmarks, pixel space)
                 ┌──────────────┐
                 │   gestures   │  finger geometry → classify → k-of-n vote
                 └──────┬───────┘
                        ▼  stable Gesture
   ┌────────────────────┴────────────────────┐
   ▼                    ▼                    ▼
┌────────┐        ┌──────────┐        ┌─────────────┐
│ canvas │        │    ui    │        │  artifacts  │
│ BGRA + │        │ toolbar, │        │ PNG, JSON,  │
│ vector │        │ dwell,   │        │ manifest    │
│ history│        │ HUD      │        │ (sandboxed) │
└───┬────┘        └────┬─────┘        └─────────────┘
    └──────────┬───────┘
               ▼
        ┌─────────────┐
        │  recording  │  threaded encoder, 4 layouts, codec fallback
        └─────────────┘
```

```
src/aircanvas/
├── types.py          Landmark constants, Gesture enum, Point, Stroke, HandObservation
├── filters.py        One Euro filter (adaptive smoothing) + FPS meter
├── gestures.py       Finger geometry → classifier → stabiliser → hold timer
├── canvas.py         BGRA layer: O(1) incremental raster + vector history + undo
├── ui.py             Adaptive toolbar, dwell selection, HUD, landmark overlay
├── capture.py        Threaded camera / deterministic file reader
├── tracking.py       HandTracker protocol; MediaPipe adapter + null tracker
├── recording.py      Threaded encoder, layout composition, codec fallback
├── paths.py          Sandbox, sanitisation, atomic writes  ← security boundary
├── artifacts.py      Artefact bundle + session manifest
├── config.py         Typed dataclasses; defaults < file < env < CLI
├── logging_setup.py  Console + rotating file, optional JSON
├── app.py            The loop that composes all of the above
└── cli.py            Argument parsing and layered overrides
```

Only `tracking.py` imports MediaPipe, and it does so lazily. Everything else is
testable without the model, without a camera, and without a display.

---

## Design decisions (and why)

**Rotation-invariant finger detection.** The common approach tests
`tip.y < pip.y`, which breaks the moment you tilt your hand. AirCanvas compares
*radial distance from the wrist* instead — an extended finger's tip is farther
from the wrist than its PIP joint at any orientation — and scales every
threshold by palm size so it works near and far from the camera. Covered by
tests at seven rotations and three distances.

**One Euro filter rather than an exponential moving average.** An EMA forces a
choice between jitter and lag: smooth enough to kill tremor means visibly
trailing the hand. The One Euro filter adapts its cutoff frequency to observed
speed — heavy smoothing when the hand is slow, almost none when it is fast.
That is precisely the behaviour a drawing cursor needs.

**Threaded capture that discards stale frames.** OpenCV buffers frames inside
the driver. When inference is slower than the sensor, `read()` hands you an old
frame and the cursor trails several frames behind your hand. A background
thread that keeps only the newest frame removes the single biggest cause of
"sluggish" feel in naive implementations.

**Vector history plus a baked base layer.** The canvas keeps both a rasterised
BGRA image and the strokes that produced it. Rasterising incrementally keeps
per-frame cost O(1) no matter how much is already drawn; the vector history
gives exact undo and a replayable export. When history exceeds `max_history`,
the oldest stroke bakes permanently into an immutable base layer, so memory
stays flat during long unattended sessions.

**Dwell selection instead of a click gesture.** There is no click in mid-air.
Hovering a control with a visible progress bar is unambiguous, needs only one
hand, and lets the user abort by simply moving away.

**Holds and cooldowns on destructive actions.** Save requires a sustained
thumbs-up and then enters a cooldown. A single misclassified frame can never
trigger something irreversible.

**Teleport guard.** If the cursor jumps more than 22% of the frame diagonal in
one frame, that is a tracking glitch, not a hand movement — the stroke breaks
rather than drawing a long straight line across your artwork.

---

## Development

```bash
make setup       # venv + editable install + dev tooling
make test        # pytest
make cov         # coverage report → htmlcov/index.html
make lint        # ruff, including bandit security rules
make typecheck   # mypy, strict
make audit       # pip-audit against known CVEs
make check       # everything CI runs
make format      # auto-fix and normalise
```

Install the git hooks once: `pre-commit install`.

### Test strategy

**147 tests, 82% coverage**, and they run in about two seconds with no camera
and no MediaPipe. The uncovered remainder is exactly the hardware-bound code —
real camera enumeration and the MediaPipe adapter.

| Suite | Covers |
|---|---|
| `test_gestures.py` | Finger geometry, rotation/scale invariance, classifier, stabiliser, hold timer |
| `test_filters.py` | One Euro jitter suppression and lag behaviour, FPS meter |
| `test_canvas.py` | Drawing, erasing, undo, history baking, compositing, JSON round-trip |
| `test_ui.py` | Toolbar layout at six resolutions, hit testing, dwell, render safety |
| `test_paths.py` | Traversal, absolute-path and symlink escapes; atomic writes; permissions |
| `test_config_and_recording.py` | Config layering and validation, all four layouts, CLI mapping |
| `test_integration.py` | Full app loop end to end via a scripted tracker |

Two techniques make this possible:

- **`tests/conftest.py` synthesises anatomically plausible 21-point landmark
  sets.** Folded fingers curl their tip back toward the wrist, which is exactly
  what the radial heuristic keys on — so gesture logic can be tested
  exhaustively without a camera or a human hand.
- **`ScriptedTracker` replays a deterministic gesture sequence** against a
  generated video file, exercising capture → gesture reduction → canvas
  mutation → compositing → recording → export in CI with no hardware.

---

## Troubleshooting

| Symptom | Cause and fix |
|---|---|
| `could not open camera index 0` | Another app holds the camera, or the index is wrong. Run `aircanvas --list-cameras`. |
| `camera opened but delivered no frames` | Usually an OS permission prompt that was dismissed. macOS: grant camera access to your **terminal app**. |
| Black window, no error | Some Windows drivers need DirectShow. Set `camera.backend: dshow`. |
| `ImportError: libGL.so.1` | Headless Linux: `apt-get install libgl1 libglib2.0-0`, or `make setup-headless`. |
| MediaPipe fails to install | Check your Python version — it must be 3.10–3.12. |
| Hand not detected | Improve lighting on the hand itself; keep the whole hand including the wrist in frame; lower `tracking.detection_confidence` to `0.5`. |
| Very low FPS | `--model-complexity 0`, reduce capture resolution, or set `drawing.antialias: false`. |
| Video plays too fast or slow | `recording.fps` must match your real throughput. Check `average_fps` in `session.json` and set `--record-fps` to match. |
| `no usable video codec found` | Your OpenCV build lacks FFmpeg. Try `--codec MJPG`, or run with `--no-record`. |
| Encoder drops frames | Raise `recording.queue_size`, or use a lighter layout than `side_by_side`. |

---

## Privacy and security

This application points a camera at a person, so those concerns are structural
rather than an afterthought.

**Nothing leaves your machine.** There are no network calls at runtime. All
inference is local; MediaPipe ships its models with the package.

**`canvas_only` layout is a genuine privacy mode.** It records the artwork on
blank paper with zero camera pixels — verified by a test that asserts no camera
pixel value survives into the output.

**Every write goes through one audited boundary** (`paths.py`):

- **Sandboxed** — resolved paths must stay inside the configured output root.
  Traversal (`../`), absolute escapes and symlinked directories pointing
  outside the root are all rejected, with tests for each.
- **Sanitised names** — session and file names are reduced to
  `[A-Za-z0-9._-]`, so nothing shell-, path- or unicode-confusable reaches the
  filesystem.
- **Atomic** — artefacts are written to a temporary file and then `os.replace`d,
  so a crash or a full disk never leaves a half-written PNG.
- **Least privilege** — session directories are `0700`, files `0600`. That
  includes the video and log files, which OpenCV and the logging handler would
  otherwise create with the process umask.

**Configuration is inert.** YAML is parsed with `safe_load`, which cannot
construct arbitrary Python objects. Unknown keys are rejected rather than
silently ignored.

**Resource limits.** `recording.max_seconds` caps runaway disk usage;
`drawing.max_history` bounds memory; the encoder queue is bounded and drops
frames rather than growing without limit.

**Supply chain.** Dependencies carry deliberate upper bounds, `pip-audit` runs
as a CI gate, `ruff` includes bandit security rules, and the container runs as
a non-root user with a single writable volume.

**In `.gitignore` by default:** `sessions/` and all video extensions. A
pre-commit hook additionally blocks them, because recorded sessions contain
video of a person and must never reach a repository.

---

## Extending it

The seams are already in place.

**A different tracking backend.** Implement the `HandTracker` protocol —
`process(frame_bgr) -> list[HandObservation]` and `close()` — and register it in
`create_tracker()`. Nothing else changes. That is the migration path to
MediaPipe's newer Tasks API, an ONNX model, or a remote inference service.

**A new gesture.** Add a member to the `Gesture` enum, a branch in
`classify()`, and a case in `AirCanvasApp._process`. The synthetic
`make_hand()` factory makes it straightforward to test first.

**A new toolbar control.** Append to `Toolbar.ACTIONS` and handle the id in
`_run_action()`. Layout, hit testing and dwell come for free.

**A new recording layout.** Add a `Layout` member and a branch in
`compose_output()` — it is a pure function, so the test is three lines.

**Multi-user or networked use.** `FrameSource` already abstracts the input, so
an RTSP or shared-memory source slots in; the session store is per-instance and
sandboxed, so several instances can write to one root safely.

Natural next steps: shape recognition (snap a rough circle to a clean one),
two-handed gestures such as pinch-to-zoom, an undo *redo* stack, and pressure
emulation via hand-to-camera distance (the `z` landmark component is already
carried through `HandObservation`).

---

## Licence

MIT — see [LICENSE](LICENSE).