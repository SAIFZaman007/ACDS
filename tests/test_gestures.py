from __future__ import annotations

import pytest

from src.core.gestures import (
    GestureEngine,
    GestureStabilizer,
    HoldTimer,
    classify,
    finger_states,
)
from src.core.types import Gesture

from .conftest import make_hand


class TestFingerStates:
    def test_all_folded(self) -> None:
        state = finger_states(make_hand())
        assert state.as_tuple() == (False, False, False, False, False)
        assert state.extended_count == 0

    def test_all_extended(self) -> None:
        state = finger_states(
            make_hand(thumb=True, index=True, middle=True, ring=True, pinky=True)
        )
        assert state.fingers == (True, True, True, True)
        assert state.thumb is True

    def test_index_only(self) -> None:
        state = finger_states(make_hand(index=True))
        assert state.fingers == (True, False, False, False)

    @pytest.mark.parametrize("rotation", [-90, -45, 0, 45, 90, 135, 180])
    def test_extension_is_rotation_invariant(self, rotation: float) -> None:
        """The whole point of the radial heuristic: tilt must not change the result."""
        hand = make_hand(index=True, middle=True, rotation_degrees=rotation)
        assert finger_states(hand).fingers == (True, True, False, False)

    @pytest.mark.parametrize("scale", [40.0, 120.0, 300.0])
    def test_extension_is_scale_invariant(self, scale: float) -> None:
        hand = make_hand(index=True, scale=scale)
        assert finger_states(hand).fingers == (True, False, False, False)


class TestClassify:
    def test_draw(self) -> None:
        gesture, _ = classify(make_hand(index=True))
        assert gesture is Gesture.DRAW

    def test_draw_ignores_splayed_thumb(self) -> None:
        """Users naturally leave the thumb out while pointing."""
        gesture, _ = classify(make_hand(index=True, thumb=True))
        assert gesture is Gesture.DRAW

    def test_select(self) -> None:
        gesture, _ = classify(make_hand(index=True, middle=True))
        assert gesture is Gesture.SELECT

    def test_erase_open_palm(self) -> None:
        gesture, _ = classify(
            make_hand(thumb=True, index=True, middle=True, ring=True, pinky=True)
        )
        assert gesture is Gesture.ERASE

    def test_erase_without_thumb(self) -> None:
        gesture, _ = classify(make_hand(index=True, middle=True, ring=True, pinky=True))
        assert gesture is Gesture.ERASE

    def test_save_thumbs_up(self) -> None:
        gesture, _ = classify(make_hand(thumb=True, thumb_up=True))
        assert gesture is Gesture.SAVE

    def test_sideways_thumb_is_not_save(self) -> None:
        """A thumb held horizontally must not trigger an irreversible action."""
        gesture, _ = classify(make_hand(thumb=True, thumb_up=False))
        assert gesture is not Gesture.SAVE

    def test_fist_is_none(self) -> None:
        gesture, _ = classify(make_hand())
        assert gesture is Gesture.NONE

    def test_ring_only_is_none(self) -> None:
        gesture, _ = classify(make_hand(ring=True))
        assert gesture is Gesture.NONE


class TestStabilizer:
    def test_requires_minimum_votes(self) -> None:
        stabilizer = GestureStabilizer(window=5, min_votes=3)
        assert stabilizer.update(Gesture.DRAW) is Gesture.NONE
        assert stabilizer.update(Gesture.DRAW) is Gesture.NONE
        assert stabilizer.update(Gesture.DRAW) is Gesture.DRAW

    def test_single_frame_glitch_is_ignored(self) -> None:
        stabilizer = GestureStabilizer(window=5, min_votes=3)
        for _ in range(5):
            stabilizer.update(Gesture.DRAW)
        assert stabilizer.update(Gesture.ERASE) is Gesture.DRAW

    def test_sustained_change_wins(self) -> None:
        stabilizer = GestureStabilizer(window=5, min_votes=3)
        for _ in range(5):
            stabilizer.update(Gesture.DRAW)
        for _ in range(3):
            result = stabilizer.update(Gesture.ERASE)
        assert result is Gesture.ERASE

    def test_rejects_bad_configuration(self) -> None:
        with pytest.raises(ValueError):
            GestureStabilizer(window=3, min_votes=4)


class TestHoldTimer:
    def test_fires_only_after_hold(self) -> None:
        timer = HoldTimer(hold_seconds=1.0, cooldown_seconds=2.0)
        assert timer.update(True, 0.0) is False
        assert timer.update(True, 0.5) is False
        assert timer.progress(0.5) == pytest.approx(0.5)
        assert timer.update(True, 1.01) is True

    def test_release_resets_progress(self) -> None:
        timer = HoldTimer(hold_seconds=1.0)
        timer.update(True, 0.0)
        timer.update(False, 0.5)
        assert timer.progress(0.5) == 0.0
        assert timer.update(True, 1.2) is False

    def test_cooldown_prevents_double_fire(self) -> None:
        timer = HoldTimer(hold_seconds=0.5, cooldown_seconds=3.0)
        timer.update(True, 0.0)
        assert timer.update(True, 0.6) is True
        timer.update(True, 0.7)
        assert timer.update(True, 1.5) is False  # still inside the cooldown
        timer.update(True, 4.0)
        assert timer.update(True, 4.6) is True


class TestGestureEngine:
    def test_missing_hand_decays_to_none(self) -> None:
        engine = GestureEngine(window=3, min_votes=2)
        for _ in range(3):
            engine.update(make_hand(index=True))
        assert engine.update(None) is Gesture.DRAW  # one missed frame is tolerated
        for _ in range(3):
            result = engine.update(None)
        assert result is Gesture.NONE
        assert engine.last_state is None
        
        
            #     *** _ ***