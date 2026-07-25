# AirCanvas — headless container image.

# --------------------------------------------------------------------------- #
# Stage 1: build a wheel
# --------------------------------------------------------------------------- #
FROM python:3.12-slim-bookworm AS builder

WORKDIR /build
RUN pip install --no-cache-dir build==1.2.2

COPY pyproject.toml README.md ./
COPY src ./src
RUN python -m build --wheel --outdir /dist

# --------------------------------------------------------------------------- #
# Stage 2: runtime
# --------------------------------------------------------------------------- #
FROM python:3.12-slim-bookworm AS runtime

# MediaPipe links against these; opencv-python-headless does not need libGL, but mediapipe's own OpenCV dependency does.
RUN apt-get update \
    && apt-get install --no-install-recommends -y \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Run unprivileged. The session directory is the only writable location.
RUN useradd --create-home --uid 10001 aircanvas
WORKDIR /app

COPY --from=builder /dist/*.whl /tmp/
RUN pip install --no-cache-dir /tmp/*.whl \
    && pip install --no-cache-dir "opencv-python-headless>=4.8,<5" \
    && rm -f /tmp/*.whl

COPY --chown=aircanvas:aircanvas configs ./configs

USER aircanvas
ENV AIRCANVAS_OUTPUT_ROOT=/sessions \
    AIRCANVAS_HEADLESS=true \
    AIRCANVAS_LOG_JSON=true \
    PYTHONUNBUFFERED=1
VOLUME ["/sessions"]

HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD ["python", "-c", "import aircanvas, sys; sys.exit(0)"]

ENTRYPOINT ["python", "-m", "aircanvas"]
CMD ["--help"]

# --------------------------------------------------------------------------- #
# Usage
#
#   docker build -t aircanvas:1.0.0 .
#
#   # Process a recorded clip (no camera, no display needed):
#   docker run --rm \
#     -v "$PWD/clips:/clips:ro" -v "$PWD/sessions:/sessions" \
#     aircanvas:1.0.0 --source /clips/demo.mp4 --layout side_by_side
#
#   # Live webcam on Linux (add --device; add the X11 mounts for a preview):
#   docker run --rm --device /dev/video0:/dev/video0 \
#     -e AIRCANVAS_HEADLESS=false -e DISPLAY="$DISPLAY" \
#     -v /tmp/.X11-unix:/tmp/.X11-unix -v "$PWD/sessions:/sessions" \
#     aircanvas:1.0.0
#
# macOS and Windows cannot pass a USB camera into a Linux container; run the
# app natively there instead.
# --------------------------------------------------------------------------- 