import numpy as np
from typing import List, Tuple, Dict
from dataclasses import dataclass

@dataclass
class ControlSignal:
    cursor_pos: Tuple[float, float]  # Normalized (0,0) to (1,1)
    pinch_active: bool
    gesture_id: int
    confidence: float

class SpatialMapper:
    """
    Translates spatial hand landmarks into normalized control signals.
    Implements coordinate normalization and gesture recognition logic.
    """
    def __init__(self,
                 screen_width: int = 1920,
                 screen_height: int = 1080,
                 smoothing_factor: float = 0.5):
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.smoothing_factor = smoothing_factor
        self.prev_cursor_pos = (0.0, 0.0)

    def _calculate_distance(self, p1: Tuple[float, float, float], p2: Tuple[float, float, float]) -> float:
        return np.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2 + (p1[2]-p2[2])**2)

    def normalize_coordinates(self, x: float, y: float) -> Tuple[float, float]:
        """
        Normalizes raw coordinates to screen space with boundaries.
        """
        # Clamp to [0, 1]
        nx = max(0.0, min(1.0, x))
        ny = max(0.0, min(1.0, y))
        return (nx, ny)

    def apply_smoothing(self, current_pos: Tuple[float, float]) -> Tuple[float, float]:
        """
        Exponential Moving Average (EMA) for cursor stabilization.
        """
        smoothed_x = (self.smoothing_factor * current_pos[0]) + ((1 - self.smoothing_factor) * self.prev_cursor_pos[0])
        smoothed_y = (self.smoothing_factor * current_pos[1]) + ((1 - self.smoothing_factor) * self.prev_cursor_pos[1])
        self.prev_cursor_pos = (smoothed_x, smoothed_y)
        return (smoothed_x, smoothed_y)

    def map_to_signal(self, landmarks: List) -> ControlSignal:
        """
        Maps landmarks to a control signal.
        Focuses on the Index Finger Tip (ID 8) for cursor and
        distance between Index and Thumb (ID 4) for pinch.
        """
        if not landmarks:
            return ControlSignal((0.0, 0.0), False, -1, 0.0)

        # Index Finger Tip (ID 8)
        index_tip = landmarks[8]
        thumb_tip = landmarks[4]

        # Normalize and smooth cursor position
        raw_pos = self.normalize_coordinates(index_tip.x, index_tip.y)
        smoothed_pos = self.apply_smoothing(raw_pos)

        # Pinch Detection: Distance between index and thumb
        # Using 3D distance for better precision
        dist = self._calculate_distance(
            (index_tip.x, index_tip.y, index_tip.z),
            (thumb_tip.x, thumb_tip.y, thumb_tip.z)
        )

        # Threshold for pinch (calibrated based on normalized units)
        pinch_active = dist < 0.05

        # Basic Gesture ID mapping (example)
        # 0: Neutral, 1: Pinch, 2: Open Palm
        gesture_id = 1 if pinch_active else 0
        if dist > 0.2:
            gesture_id = 2

        return ControlSignal(
            cursor_pos=smoothed_pos,
            pinch_active=pinch_active,
            gesture_id=gesture_id,
            confidence=index_tip.visibility if hasattr(index_tip, 'visibility') else 1.0
        )

    def get_screen_coordinates(self, normalized_pos: Tuple[float, float]) -> Tuple[int, int]:
        """
        Translates normalized (0,1) coordinates to absolute screen pixels.
        """
        sx = int(normalized_pos[0] * self.screen_width)
        sy = int(normalized_pos[1] * self.screen_height)
        return (sx, sy)
