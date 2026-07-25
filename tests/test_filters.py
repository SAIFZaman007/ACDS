from __future__ import annotations

import numpy as np
import pytest

from src.core.filters import FpsMeter, OneEuroFilter, ema


def test_ema_interpolates() -> None:
    assert ema(0.0, 10.0, 0.5) == pytest.approx(5.0)
    assert ema(0.0, 10.0, 1.0) == pytest.approx(10.0)


class TestOneEuroFilter:
    def test_first_sample_passes_through(self) -> None:
        f = OneEuroFilter()
        assert f(np.array([100.0, 50.0]), 0.0) == pytest.approx([100.0, 50.0])

    def test_suppresses_jitter_around_a_stationary_point(self) -> None:
        rng = np.random.default_rng(1234)
        f = OneEuroFilter(min_cutoff=0.6, beta=0.0)
        truth = np.array([300.0, 200.0])
        raw, filtered = [], []
        for i in range(120):
            noisy = truth + rng.normal(0.0, 4.0, size=2)
            raw.append(noisy)
            filtered.append(f(noisy, i / 30.0))
        warm = slice(30, None)
        raw_error = np.linalg.norm(np.array(raw)[warm] - truth, axis=1).mean()
        filtered_error = np.linalg.norm(np.array(filtered)[warm] - truth, axis=1).mean()
        assert filtered_error < raw_error * 0.5

    def test_beta_reduces_lag_on_fast_motion(self) -> None:
        """Higher beta must track a ramp more closely — the whole point of One Euro."""
        def endpoint_error(beta: float) -> float:
            f = OneEuroFilter(min_cutoff=0.6, beta=beta)
            out = np.zeros(2)
            for i in range(60):
                target = np.array([10.0 * i, 0.0])
                out = f(target, i / 30.0)
            return abs(float(out[0]) - 10.0 * 59)

        assert endpoint_error(0.05) < endpoint_error(0.0)

    def test_reset_clears_state(self) -> None:
        f = OneEuroFilter()
        f(np.array([10.0, 10.0]), 0.0)
        f.reset()
        assert f(np.array([900.0, 5.0]), 1.0) == pytest.approx([900.0, 5.0])

    def test_filter_point_returns_ints(self) -> None:
        f = OneEuroFilter()
        x, y = f.filter_point(12.4, 88.6, 0.0)
        assert (x, y) == (12, 89)
        assert isinstance(x, int) and isinstance(y, int)

    def test_non_monotonic_timestamps_do_not_explode(self) -> None:
        f = OneEuroFilter()
        f(np.array([0.0, 0.0]), 5.0)
        out = f(np.array([10.0, 10.0]), 4.0)  # clock went backwards
        assert np.all(np.isfinite(out))

    @pytest.mark.parametrize("kwargs", [{"min_cutoff": 0}, {"d_cutoff": -1}, {"beta": -0.1}])
    def test_rejects_invalid_parameters(self, kwargs: dict) -> None:
        with pytest.raises(ValueError):
            OneEuroFilter(**kwargs)


class TestFpsMeter:
    def test_reports_zero_before_two_samples(self) -> None:
        meter = FpsMeter()
        assert meter.tick(0.0) == 0.0

    def test_measures_steady_rate(self) -> None:
        meter = FpsMeter(window=10)
        fps = 0.0
        for i in range(10):
            fps = meter.tick(i / 25.0)
        assert fps == pytest.approx(25.0, rel=1e-6)
        assert meter.frames == 10

    def test_rejects_tiny_window(self) -> None:
        with pytest.raises(ValueError):
            FpsMeter(window=1)
            
            
            
            
                #     *** _ ***