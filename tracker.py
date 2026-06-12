import cv2
import mediapipe as mp
import numpy as np
import time
from dataclasses import dataclass
from typing import List, Tuple, Optional

@dataclass
class HandLandmark:
    id: int
    x: float
    y: float
    z: float
    visibility: float


# Landmark index constants (MediaPipe 21-point model)
WRIST = 0
THUMB_TIP = 4
INDEX_TIP = 8
MIDDLE_TIP = 12
RING_TIP = 16
PINKY_TIP = 20


class GojoHandTracker:
    """
    High-precision hand tracking system using MediaPipe and OpenCV.
    Designed for real-time Human-Computer Interaction (HCI).

    Detects up to `max_num_hands` hands per frame and returns a list of
    21-point landmark sets in normalized (x, y, z) coordinates.
    """

    def __init__(
        self,
        static_image_mode: bool = False,
        max_num_hands: int = 2,
        min_detection_confidence: float = 0.7,
        min_tracking_confidence: float = 0.5,
    ):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=static_image_mode,
            max_num_hands=max_num_hands,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self.mp_draw = mp.solutions.drawing_utils
        # Correct connection set used for skeleton drawing
        self.hand_connections = self.mp_hands.HAND_CONNECTIONS

    def find_hands(
        self, image: np.ndarray, flip: bool = True
    ) -> Tuple[np.ndarray, List[List[HandLandmark]]]:
        """
        Process a BGR frame, detect hands, and return landmarks.

        Args:
            image: BGR frame from OpenCV.
            flip:  Mirror the frame horizontally so it acts like a mirror
                   (natural for self-facing webcam use).

        Returns:
            Tuple of (annotated image, list-of-hands), where each hand is a
            list of 21 HandLandmark objects in MediaPipe landmark order.
        """
        if flip:
            image = cv2.flip(image, 1)

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # Make the array non-writeable to pass by reference and avoid a copy
        image_rgb.flags.writeable = False
        results = self.hands.process(image_rgb)
        image_rgb.flags.writeable = True

        all_hand_landmarks: List[List[HandLandmark]] = []

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                landmarks = [
                    HandLandmark(
                        id=idx,
                        x=lm.x,
                        y=lm.y,
                        z=lm.z,
                        visibility=getattr(lm, "visibility", 1.0),
                    )
                    for idx, lm in enumerate(hand_landmarks.landmark)
                ]
                all_hand_landmarks.append(landmarks)

        return image, all_hand_landmarks

    def draw_landmarks(
        self,
        image: np.ndarray,
        hand_landmarks_list: List[List[HandLandmark]],
        dot_color: Tuple[int, int, int] = (0, 255, 0),
        line_color: Tuple[int, int, int] = (255, 255, 255),
        dot_radius: int = 4,
        line_thickness: int = 1,
    ) -> np.ndarray:
        """
        Draw all 21 landmarks and the hand skeleton on *image*.

        Connections are drawn using the canonical MediaPipe HAND_CONNECTIONS
        set so the skeleton matches the model topology exactly.

        Args:
            image:               BGR frame to draw on (modified in place).
            hand_landmarks_list: Output from :meth:`find_hands`.
            dot_color:           BGR colour for landmark dots.
            line_color:          BGR colour for connection lines.
            dot_radius:          Radius of each landmark dot in pixels.
            line_thickness:      Thickness of skeleton lines in pixels.

        Returns:
            The annotated image (same object as *image*).
        """
        h, w = image.shape[:2]

        for landmarks in hand_landmarks_list:
            # Build a quick lookup: landmark id -> pixel position
            pts: dict = {}
            for lm in landmarks:
                cx, cy = int(lm.x * w), int(lm.y * h)
                pts[lm.id] = (cx, cy)
                cv2.circle(image, (cx, cy), dot_radius, dot_color, cv2.FILLED)

            # Draw skeleton lines along every canonical connection
            for start_id, end_id in self.hand_connections:
                if start_id in pts and end_id in pts:
                    cv2.line(image, pts[start_id], pts[end_id], line_color, line_thickness)

        return image


def main():
    """Standalone demo: opens the default webcam and runs the tracker."""
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError(
            "Could not open webcam (source 0). "
            "Ensure a camera is connected and not in use by another application."
        )

    tracker = GojoHandTracker()
    prev_time = 0.0

    print("Gojo Hand Tracker running — press 'q' to quit.")

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Warning: failed to read frame, retrying…")
            continue

        image, hands_data = tracker.find_hands(image)
        image = tracker.draw_landmarks(image, hands_data)

        curr_time = time.time()
        elapsed = curr_time - prev_time
        fps = 1.0 / elapsed if elapsed > 0 else 0.0
        prev_time = curr_time

        cv2.putText(
            image, f"FPS: {int(fps)}", (10, 70),
            cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3,
        )
        cv2.imshow("Gojo Hand Tracking", image)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
