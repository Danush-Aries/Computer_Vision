"""
Unit tests for mapper.py — SpatialMapper and ControlSignal.
"""

import unittest

from tracker import HandLandmark
from mapper import SpatialMapper, ControlSignal


def _make_hand(index_x=0.5, index_y=0.5, index_z=0.0,
               thumb_x=0.6, thumb_y=0.5, thumb_z=0.0):
    """
    Build a minimal 21-landmark hand with meaningful positions only for
    INDEX_TIP (id 8) and THUMB_TIP (id 4); the rest are at the origin.
    """
    hand = [HandLandmark(id=i, x=0.0, y=0.0, z=0.0, visibility=1.0) for i in range(21)]
    hand[8] = HandLandmark(id=8, x=index_x, y=index_y, z=index_z, visibility=1.0)
    hand[4] = HandLandmark(id=4, x=thumb_x, y=thumb_y, z=thumb_z, visibility=1.0)
    return hand


# ---------------------------------------------------------------------------
# ControlSignal dataclass
# ---------------------------------------------------------------------------

class TestControlSignalDataclass(unittest.TestCase):
    def test_fields(self):
        sig = ControlSignal(cursor_pos=(0.3, 0.7), pinch_active=True, gesture_id=1, confidence=0.9)
        self.assertEqual(sig.cursor_pos, (0.3, 0.7))
        self.assertTrue(sig.pinch_active)
        self.assertEqual(sig.gesture_id, 1)
        self.assertAlmostEqual(sig.confidence, 0.9)


# ---------------------------------------------------------------------------
# SpatialMapper._calculate_distance
# ---------------------------------------------------------------------------

class TestCalculateDistance(unittest.TestCase):
    def setUp(self):
        self.mapper = SpatialMapper()

    def test_zero_distance(self):
        d = self.mapper._calculate_distance((0, 0, 0), (0, 0, 0))
        self.assertAlmostEqual(d, 0.0)

    def test_unit_distance_x(self):
        d = self.mapper._calculate_distance((0, 0, 0), (1, 0, 0))
        self.assertAlmostEqual(d, 1.0)

    def test_3d_distance(self):
        d = self.mapper._calculate_distance((0, 0, 0), (1, 1, 1))
        import math
        self.assertAlmostEqual(d, math.sqrt(3), places=5)


# ---------------------------------------------------------------------------
# SpatialMapper.normalize_coordinates
# ---------------------------------------------------------------------------

class TestNormalizeCoordinates(unittest.TestCase):
    def setUp(self):
        self.mapper = SpatialMapper()

    def test_clamp_above_one(self):
        nx, ny = self.mapper.normalize_coordinates(1.5, 2.0)
        self.assertEqual(nx, 1.0)
        self.assertEqual(ny, 1.0)

    def test_clamp_below_zero(self):
        nx, ny = self.mapper.normalize_coordinates(-0.5, -1.0)
        self.assertEqual(nx, 0.0)
        self.assertEqual(ny, 0.0)

    def test_passthrough_in_range(self):
        nx, ny = self.mapper.normalize_coordinates(0.4, 0.6)
        self.assertAlmostEqual(nx, 0.4)
        self.assertAlmostEqual(ny, 0.6)


# ---------------------------------------------------------------------------
# SpatialMapper.apply_smoothing
# ---------------------------------------------------------------------------

class TestApplySmoothing(unittest.TestCase):
    def setUp(self):
        self.mapper = SpatialMapper(smoothing_factor=0.5)

    def test_first_call_returns_half_of_input(self):
        # prev = (0, 0), alpha = 0.5 → result = 0.5 * input + 0.5 * 0 = 0.5 * input
        sx, sy = self.mapper.apply_smoothing((1.0, 1.0))
        self.assertAlmostEqual(sx, 0.5)
        self.assertAlmostEqual(sy, 0.5)

    def test_converges_toward_target(self):
        mapper = SpatialMapper(smoothing_factor=0.5)
        # Run 20 iterations pushing toward (1, 1)
        for _ in range(20):
            sx, sy = mapper.apply_smoothing((1.0, 1.0))
        self.assertGreater(sx, 0.99)
        self.assertGreater(sy, 0.99)

    def test_prev_cursor_updated(self):
        self.mapper.apply_smoothing((0.8, 0.6))
        # After one call prev_cursor_pos should no longer be (0, 0)
        self.assertNotEqual(self.mapper.prev_cursor_pos, (0.0, 0.0))


# ---------------------------------------------------------------------------
# SpatialMapper.map_to_signal — cursor position
# ---------------------------------------------------------------------------

class TestMapToSignalCursorPosition(unittest.TestCase):
    def setUp(self):
        self.mapper = SpatialMapper(smoothing_factor=1.0)  # alpha=1 → no lag

    def test_cursor_within_unit_square(self):
        hand = _make_hand(index_x=0.3, index_y=0.7)
        sig = self.mapper.map_to_signal(hand)
        self.assertGreaterEqual(sig.cursor_pos[0], 0.0)
        self.assertLessEqual(sig.cursor_pos[0], 1.0)
        self.assertGreaterEqual(sig.cursor_pos[1], 0.0)
        self.assertLessEqual(sig.cursor_pos[1], 1.0)

    def test_cursor_tracks_index_tip(self):
        hand = _make_hand(index_x=0.2, index_y=0.8)
        sig = self.mapper.map_to_signal(hand)
        self.assertAlmostEqual(sig.cursor_pos[0], 0.2, places=4)
        self.assertAlmostEqual(sig.cursor_pos[1], 0.8, places=4)

    def test_empty_landmarks_returns_default(self):
        sig = self.mapper.map_to_signal([])
        self.assertEqual(sig.cursor_pos, (0.0, 0.0))
        self.assertFalse(sig.pinch_active)
        self.assertEqual(sig.gesture_id, -1)


# ---------------------------------------------------------------------------
# SpatialMapper.map_to_signal — pinch detection
# ---------------------------------------------------------------------------

class TestMapToSignalPinch(unittest.TestCase):
    def setUp(self):
        self.mapper = SpatialMapper(smoothing_factor=1.0)

    def test_pinch_active_when_tips_close(self):
        # Index and thumb at almost the same position → pinch
        hand = _make_hand(index_x=0.5, index_y=0.5, index_z=0.0,
                          thumb_x=0.5, thumb_y=0.5, thumb_z=0.0)
        sig = self.mapper.map_to_signal(hand)
        self.assertTrue(sig.pinch_active)

    def test_pinch_inactive_when_tips_far(self):
        # Index and thumb far apart → no pinch
        hand = _make_hand(index_x=0.1, index_y=0.5, index_z=0.0,
                          thumb_x=0.9, thumb_y=0.5, thumb_z=0.0)
        sig = self.mapper.map_to_signal(hand)
        self.assertFalse(sig.pinch_active)

    def test_gesture_id_pinch(self):
        hand = _make_hand(index_x=0.5, index_y=0.5, index_z=0.0,
                          thumb_x=0.5, thumb_y=0.5, thumb_z=0.0)
        sig = self.mapper.map_to_signal(hand)
        self.assertEqual(sig.gesture_id, 1)

    def test_gesture_id_open_palm(self):
        # Very large separation → open palm
        hand = _make_hand(index_x=0.0, index_y=0.0, index_z=0.0,
                          thumb_x=1.0, thumb_y=0.0, thumb_z=0.0)
        sig = self.mapper.map_to_signal(hand)
        self.assertEqual(sig.gesture_id, 2)

    def test_gesture_id_neutral(self):
        # Moderate separation → neutral
        hand = _make_hand(index_x=0.5, index_y=0.5, index_z=0.0,
                          thumb_x=0.6, thumb_y=0.5, thumb_z=0.0)
        sig = self.mapper.map_to_signal(hand)
        self.assertEqual(sig.gesture_id, 0)


# ---------------------------------------------------------------------------
# SpatialMapper.get_screen_coordinates
# ---------------------------------------------------------------------------

class TestGetScreenCoordinates(unittest.TestCase):
    def test_full_screen(self):
        mapper = SpatialMapper(screen_width=1920, screen_height=1080)
        sx, sy = mapper.get_screen_coordinates((1.0, 1.0))
        self.assertEqual(sx, 1920)
        self.assertEqual(sy, 1080)

    def test_origin(self):
        mapper = SpatialMapper(screen_width=1920, screen_height=1080)
        sx, sy = mapper.get_screen_coordinates((0.0, 0.0))
        self.assertEqual(sx, 0)
        self.assertEqual(sy, 0)

    def test_midpoint(self):
        mapper = SpatialMapper(screen_width=1920, screen_height=1080)
        sx, sy = mapper.get_screen_coordinates((0.5, 0.5))
        self.assertEqual(sx, 960)
        self.assertEqual(sy, 540)


if __name__ == "__main__":
    unittest.main()
