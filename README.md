# Air Drawing System

Draw on screen without touching anything. Point your index finger to draw,
open your palm to erase, tap the on-screen toolbar to change colors, and
flash a thumbs-up to save your work — all tracked in real time from your
webcam.

| Gesture | Action |
|---|---|
| ☝️ Index finger only | Draw |
| ✋ Open palm (4 fingers) | Erase |
| 👆 Index finger over the top toolbar | Pick a color |
| 👍 Thumbs up (held briefly) | Save the drawing as a PNG |

The whole session — your drawing blended live over your own webcam feed —
is also recorded to a video file automatically.

## Tech stack

Python · OpenCV · MediaPipe (Hand Landmarker) · NumPy · [uv](https://docs.astral.sh/uv/)

## How it's built

Everything lives in **one file**, `air_draw.py`, organized into small,
single-purpose classes so it stays easy to read end to end:

```
Config          all tunable settings in one place
HandTracker     wraps MediaPipe's HandLandmarker, exposes 21 hand landmarks
fingers_extended() / classify_gesture()   pure functions: landmarks -> gesture
Canvas          the persistent drawing surface + video/white-background blending
Toolbar         the color header + fingertip hit-testing
Recorder        session video writer, with a codec fallback
AirDrawApp      wires it all together: camera -> gesture -> canvas -> screen
```

Gesture recognition is deliberately simple heuristics (finger-curl angles
and relative distances) rather than a trained gesture-classifier model —
that's what keeps this fast, dependency-light, and easy to tune. The
gesture functions take plain landmark lists and don't touch OpenCV or the
camera at all, so they're trivial to unit test in isolation.

One deliberate detail: a stroke doesn't end the instant hand tracking has
one flaky frame (motion blur, a brief occlusion). `Config.miss_tolerance_frames`
lets it coast for a few frames before actually breaking the line — otherwise
a single dropped frame mid-stroke splits what should be one line into a dot
and a gap. A *deliberate* mode change (switching to erase, save, or the
toolbar) still ends the stroke immediately, since that's intentional, not
a tracking hiccup.

## Requirements

- Python 3.10–3.13
- A webcam
- macOS, Windows, or Linux with a display (this is a desktop app — it opens
  a live window, so it won't run headless or over SSH without a display)

## Setup (uv — recommended)

[Install uv](https://docs.astral.sh/uv/getting-started/installation/) if you
don't have it, then:

```bash
git clone <your-repo-url> air-draw   # or just download the 4 project files
cd air-draw
uv sync                              # creates .venv, installs pinned deps
uv run air_draw.py                   # first run auto-downloads the hand model (~10 MB, one time)
```

That's it — `uv sync` reads `pyproject.toml` / `uv.lock` and builds an
isolated environment; `uv run` executes inside it without you needing to
activate anything manually.

### Setup (Poetry — alternative)

```bash
poetry init --no-interaction
poetry add opencv-python mediapipe numpy
poetry run python air_draw.py
```

## Usage

```bash
uv run air_draw.py                          # defaults: camera 0, 1280x720, recording on
uv run air_draw.py --camera 1                # use a different webcam
uv run air_draw.py --width 1920 --height 1080
uv run air_draw.py --no-record               # skip session recording
uv run air_draw.py --output-dir ~/Desktop/art
```

| Flag | Default | Description |
|---|---|---|
| `--camera` | `0` | Webcam index |
| `--width`, `--height` | `1280`, `720` | Capture resolution |
| `--no-record` | off | Disable session video recording |
| `--output-dir` | `./outputs` | Where drawings/ and recordings/ get saved |
| `--model` | auto-downloaded | Path to a local `hand_landmarker.task`, if you already have one |

**Keyboard backups** (gestures do the real work, these are just convenient
while testing): `q` / `Esc` quit · `c` clear canvas · `s` save drawing.

## Output

```
outputs/
├── drawings/     drawing_20260728_143205.png   <- one per thumbs-up (or 's')
└── recordings/   session_20260728_142950.mp4   <- one per run, drawing + video combined
```

## Troubleshooting

- **"Could not open camera index 0"** — another app may be using the webcam,
  or you need to grant camera permission (macOS: System Settings → Privacy
  & Security → Camera. Windows: Settings → Privacy & security → Camera →
  make sure **"Let desktop apps access your camera"** is on, not just the
  Microsoft Store toggle above it). Try `--camera 1` if you have more than
  one device.
- **Model download fails** — you're likely offline or behind a firewall.
  The error message prints an exact `curl` command to fetch the model
  manually and drop it in `models/hand_landmarker.task`.
- **Low FPS / laggy drawing** — lower the resolution (`--width 960 --height 540`)
  or close other apps using the GPU/CPU. The FPS counter in the bottom-right
  of the window tells you where you stand.
- **Gestures feel unreliable** — keep your hand roughly upright and fully in
  frame; the recognizer uses simple, fast heuristics rather than a trained
  classifier, so it favors clear, deliberate gestures over subtlety.
- **No window appears** — this is a desktop GUI app (`cv2.imshow`); it needs
  a real display and won't work over a headless SSH session.

## A note on "deployment"

This app talks directly to your webcam and pops up a live OpenCV window, so
"deploying" it doesn't mean shipping it to a cloud server the way you would
a web app — there's no browser or server in the loop. In practice,
"deployment" here means **distribution**:

- **Share the code** — push this repo to GitHub; anyone with `uv` can be
  running it in two commands (`uv sync && uv run air_draw.py`).
- **Ship a standalone executable** (no Python required on the other end) —
  package it with [PyInstaller](https://pyinstaller.org/):
  ```bash
  uv run pyinstaller --onefile --name air-draw air_draw.py
  ```
  Bundle the `models/hand_landmarker.task` file alongside the executable
  (or let it auto-download on first run, as it does today).

If you ever did want a browser-based, many-users version, that's a genuinely
different architecture — MediaPipe also ships a JS/WASM build that runs
hand tracking client-side in the browser, which is the natural path for a
web product. That's a bigger rebuild, not a tweak to this script, so it's
out of scope here on purpose.

## Possible extensions

Left out to keep this simple and reviewable as a single file — but the
class boundaries above make each of these a self-contained addition:

- Two-hand support (e.g., left hand picks colors, right hand draws)
- Brush-size control via pinch distance (thumb–index gap)
- Undo/redo (keep a small stack of canvas snapshots)
- A `tests/` folder with `pytest` around `fingers_extended` / `classify_gesture`
  (they're pure functions — no camera needed to test them)

## License

MIT — do whatever you'd like with this.