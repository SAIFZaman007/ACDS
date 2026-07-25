"""Signal-smoothing utilities.

The fingertip signal returned by any hand tracker is noisy. A naive
exponential moving average forces a trade-off: heavy smoothing removes
jitter but adds visible lag when the hand moves quickly.

The *One Euro* filter (Casiez et al., CHI 2012) solves this by adapting its
cutoff frequency to the observed speed: slow movements are smoothed hard,
fast movements are barely touched. That is exactly the behaviour required
for a drawing cursor.
"""

from __future__ import annotations

import math
from collections import deque
from typing import Deque

import numpy as np

__all__ = ["OneEuroFilter", "FpsMeter", "ema"]


def ema(previous: float, current: float, alpha: float) -> float:
    """Standard exponential moving average."""
    return alpha * current + (1.0 - alpha) * previous


class _LowPass:
    """Scalar/vector low-pass filter with an explicit alpha."""

    __slots__ = ("_value", "_initialised")

    def __init__(self) -> None:
        self._value: np.ndarray | None = None
        self._initialised = False

    @property
    def value(self) -> np.ndarray | None:
        return self._value

    @property
    def initialised(self) -> bool:
        return self._initialised

    def __call__(self, x: np.ndarray, alpha: float) -> np.ndarray:
        if not self._initialised:
            self._value = x
            self._initialised = True
        else:
            assert self._value is not None
            self._value = alpha * x + (1.0 - alpha) * self._value
        return self._value

    def reset(self) -> None:
        self._value = None
        self._initialised = False


class OneEuroFilter:
    """Adaptive low-pass filter for 1-D or N-D points.

    Args:
        min_cutoff: Baseline cutoff (Hz). Lower -> smoother when still.
        beta: Speed coefficient. Higher -> less lag when moving fast.
        d_cutoff: Cutoff for the derivative estimate (Hz).
        freq: Fallback sampling frequency used before two samples exist.
    """

    __slots__ = ("min_cutoff", "beta", "d_cutoff", "freq", "_x", "_dx", "_last_t")

    def __init__(
        self,
        min_cutoff: float = 1.0,
        beta: float = 0.0,
        d_cutoff: float = 1.0,
        freq: float = 30.0,
    ) -> None:
        if min_cutoff <= 0 or d_cutoff <= 0 or freq <= 0:
            raise ValueError("min_cutoff, d_cutoff and freq must be positive")
        if beta < 0:
            raise ValueError("beta must be non-negative")
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)
        self.freq = float(freq)
        self._x = _LowPass()
        self._dx = _LowPass()
        self._last_t: float | None = None

    @staticmethod
    def _alpha(cutoff: float, dt: float) -> float:
        tau = 1.0 / (2.0 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def reset(self) -> None:
        self._x.reset()
        self._dx.reset()
        self._last_t = None

    def __call__(self, x: np.ndarray | tuple[float, ...] | float, t: float) -> np.ndarray:
        """Filter sample ``x`` observed at timestamp ``t`` (seconds)."""
        value = np.atleast_1d(np.asarray(x, dtype=np.float64))

        if self._last_t is None or t <= self._last_t:
            dt = 1.0 / self.freq
        else:
            dt = t - self._last_t
        self._last_t = t

        prev = self._x.value
        raw_derivative = (
            np.zeros_like(value) if prev is None else (value - prev) / dt
        )
        derivative = self._dx(raw_derivative, self._alpha(self.d_cutoff, dt))

        speed = float(np.linalg.norm(derivative))
        cutoff = self.min_cutoff + self.beta * speed
        return self._x(value, self._alpha(cutoff, dt))

    def filter_point(self, x: float, y: float, t: float) -> tuple[int, int]:
        """Convenience wrapper returning rounded integer pixel coordinates."""
        out = self(np.array([x, y], dtype=np.float64), t)
        return int(round(float(out[0]))), int(round(float(out[1])))


class FpsMeter:
    """Rolling frames-per-second estimator."""

    __slots__ = ("_timestamps", "_count")

    def __init__(self, window: int = 60) -> None:
        if window < 2:
            raise ValueError("window must be >= 2")
        self._timestamps: Deque[float] = deque(maxlen=window)
        self._count = 0

    def tick(self, now: float) -> float:
        self._timestamps.append(now)
        self._count += 1
        return self.fps

    @property
    def frames(self) -> int:
        return self._count

    @property
    def fps(self) -> float:
        if len(self._timestamps) < 2:
            return 0.0
        span = self._timestamps[-1] - self._timestamps[0]
        if span <= 0:
            return 0.0
        return (len(self._timestamps) - 1) / span 

#     *** _ ***