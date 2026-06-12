"""
Unit tests for tracker.py — GojoHandTracker and HandLandmark.

These tests do NOT require a real webcam or a GPU.  MediaPipe inference is
bypassed by patching the Hands.process() call so the suite runs in any CI
environment.
"""

import types
import unittest
from unittest.mock import MagicMock, patch

import numpy as np

from tracker import GojoHandTracker, HandLandmark, THUMB_TIP, INDEX_TIP


def _make_landmark(x=0.5, y=0.5, z=0.0):
    """Return a minimal object that looks like a MediaPipe NormalizedLandmark."""
    lm = types.SimpleNamespace(x=x, y=y, z=z)
    return lm


def _make_mediapipe_result(hand_data):
    """
    Build a fake mediapipe result object.

    hand_data: list of lists of (x, y, z) tuples, one list per detected hand.
    """
    result = types.SimpleNamespace(multi_hand_landmarks=None)
    if hand_data:
        hands = []
        for hand in hand_data:
            mp_hand = types.SimpleNamespace(
                landmark=[_make_landmark(*lm) for lm in hand]
            )
            hands.append(mp_hand)
        result.multi_hand_landmarks = hands
    return result


# ---------------------------------------------------------------------------
# HandLandmark dataclass
# ---------------------------------------------------------------------------

class TestHandLandmarkDataclass(unittest.TestCase):
    def test_fields_accessible(self):
        lm = HandLandmark(id=8, x=0.3, y=0.7, z=-0.02, visibility=0.9)
        self.assertEqual(lm.id, 8)
        self.assertAlmostEqual(lm.x, 0.3)
        self.assertAlmostEqual(lm.y, 0.7)
        self.assertAlmostEqual(lm.z, -0.02)
        self.assertAlmostEqual(lm.visibility, 0.9)

    def test_default_construction(self):
        lm = HandLandmark(id=0, x=0.0, y=0.0, z=0.0, visibility=1.0)
        self.assertIsInstance(lm, HandLandmark)


# ---------------------------------------------------------------------------
# GojoHandTracker
# ---------------------------------------------------------------------------

class TestGojoHandTrackerInit(unittest.TestCase):
    def test_hand_connections_not_none(self):
        """HAND_CONNECTIONS must be a non-empty set of pairs."""
        tracker = GojoHandTracker()
        self.assertIsNotNone(tracker.hand_connections)
        self.assertGreater(len(tracker.hand_connections), 0)

    def test_tracker_creates_hands_object(self):
        tracker = GojoHandTracker()
        self.assertIsNotNone(tracker.hands)


class TestFindHandsNoDetection(unittest.TestCase):
    """find_hands returns empty list when no hands are visible."""

    def setUp(self):
        self.tracker = GojoHandTracker()
        # A small black image
        self.frame = np.zeros((240, 320, 3), dtype=np.uint8)

    def test_no_hands_returns_empty_list(self):
        empty_result = _make_mediapipe_result([])
        with patch.object(self.tracker.hands, "process", return_value=empty_result):
            _, hands = self.tracker.find_hands(self.frame, flip=False)
        self.assertEqual(hands, [])

    def test_returns_image(self):
        empty_result = _make_mediapipe_result([])
        with patch.object(self.tracker.hands, "process", return_value=empty_result):
            img, _ = self.tracker.find_hands(self.frame, flip=False)
        self.assertIsInstance(img, np.ndarray)


class TestFindHandsWithDetection(unittest.TestCase):
    """find_hands correctly converts MediaPipe results to HandLandmark objects."""

    def setUp(self):
        self.tracker = GojoHandTracker()
        self.frame = np.zeros((240, 320, 3), dtype=np.uint8)
        # 21 landmarks per hand — use distinct x values so we can verify them
        self.fake_hand = [(i / 21.0, 0.5, 0.0) for i in range(21)]

    def test_one_hand_detected(self):
        result = _make_mediapipe_result([self.fake_hand])
        with patch.object(self.tracker.hands, "process", return_value=result):
            _, hands = self.tracker.find_hands(self.frame, flip=False)
        self.assertEqual(len(hands), 1)

    def test_landmark_count_per_hand(self):
        result = _make_mediapipe_result([self.fake_hand])
        with patch.object(self.tracker.hands, "process", return_value=result):
            _, hands = self.tracker.find_hands(self.frame, flip=False)
        self.assertEqual(len(hands[0]), 21)

    def test_landmark_ids_sequential(self):
        result = _make_mediapipe_result([self.fake_hand])
        with patch.object(self.tracker.hands, "process", return_value=result):
            _, hands = self.tracker.find_hands(self.frame, flip=False)
        ids = [lm.id for lm in hands[0]]
        self.assertEqual(ids, list(range(21)))

    def test_landmark_x_values_preserved(self):
        result = _make_mediapipe_result([self.fake_hand])
        with patch.object(self.tracker.hands, "process", return_value=result):
            _, hands = self.tracker.find_hands(self.frame, flip=False)
        for i, lm in enumerate(hands[0]):
            self.assertAlmostEqual(lm.x, i / 21.0, places=5)

    def test_two_hands_detected(self):
        result = _make_mediapipe_result([self.fake_hand, self.fake_hand])
        with patch.object(self.tracker.hands, "process", return_value=result):
            _, hands = self.tracker.find_hands(self.frame, flip=False)
        self.assertEqual(len(hands), 2)

    def test_flip_does_not_crash(self):
        """Flipping should work without error on a valid colour frame."""
        result = _make_mediapipe_result([])
        with patch.object(self.tracker.hands, "process", return_value=result):
            img, _ = self.tracker.find_hands(self.frame, flip=True)
        self.assertIsInstance(img, np.ndarray)


# ---------------------------------------------------------------------------
# draw_landmarks
# ---------------------------------------------------------------------------

class TestDrawLandmarks(unittest.TestCase):
    def setUp(self):
        self.tracker = GojoHandTracker()
        self.frame = np.zeros((480, 640, 3), dtype=np.uint8)
        # Build 21 landmarks placed in a grid so none are out of bounds
        self.hand = [
            HandLandmark(id=i, x=(i % 7) / 7.0, y=(i // 7) / 3.0, z=0.0, visibility=1.0)
            for i in range(21)
        ]

    def test_returns_ndarray(self):
        out = self.tracker.draw_landmarks(self.frame, [self.hand])
        self.assertIsInstance(out, np.ndarray)

    def test_returns_same_object(self):
        """draw_landmarks should mutate and return the same frame object."""
        out = self.tracker.draw_landmarks(self.frame, [self.hand])
        self.assertIs(out, self.frame)

    def test_empty_hand_list_no_error(self):
        """Passing an empty list of hands must not raise."""
        out = self.tracker.draw_landmarks(self.frame, [])
        self.assertIsNotNone(out)

    def test_pixels_changed(self):
        """After drawing, at least some pixels should differ from the black background."""
        frame_copy = self.frame.copy()
        self.tracker.draw_landmarks(frame_copy, [self.hand])
        self.assertFalse(np.all(frame_copy == 0), "Expected some pixels to be drawn")


# ---------------------------------------------------------------------------
# Landmark index constants
# ---------------------------------------------------------------------------

class TestLandmarkConstants(unittest.TestCase):
    def test_thumb_tip_index(self):
        self.assertEqual(THUMB_TIP, 4)

    def test_index_tip_index(self):
        self.assertEqual(INDEX_TIP, 8)


if __name__ == "__main__":
    unittest.main()
