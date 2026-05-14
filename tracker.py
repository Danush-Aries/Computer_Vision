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

class GojoHandTracker:
    """
    High-precision hand tracking system using MediaPipe and OpenCV.
    Designed for real-time Human-Computer Interaction (HCI).
    """
    def __init__(self,
                 static_image_mode=False,
                 max_num_hands=2,
                 min_detection_confidence=0.7,
                 min_tracking_confidence=0.5):

        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=static_image_mode,
            max_num_hands=max_num_hands,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
        self.mp_draw = mp.solutions.drawing_utils
        self.mp_hands_landmark = self.mp_hands.HAND_LANDMARKS

    def find_hands(self, image: np.ndarray, flip: bool = True) -> Tuple[np.ndarray, List[List[HandLandmark]]]:
        """
        Processes a frame to detect hands and extract landmarks.
        """
        if flip:
            image = cv2.flip(image, 1)

        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.hands.process(image_rgb)

        all_hand_landmarks = []

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                landmarks = []
                for id, lm in enumerate(hand_landmarks.landmark):
                    landmarks.append(HandLandmark(
                        id=id,
                        x=lm.x,
                        y=lm.y,
                        z=lm.z,
                        visibility=lm.visibility if hasattr(lm, 'visibility') else 1.0
                    ))
                all_hand_landmarks.append(landmarks)

        return image, all_hand_landmarks

    def draw_landmarks(self, image: np.ndarray, hand_landmarks: List[List[HandLandmark]]):
        """
        Visualizes the detected hand landmarks on the image.
        """
        # We need the original mediapipe results for drawing,
        # but we can recreate the landmark objects or use the results object directly.
        # For simplicity in this class, we'll let the main loop handle the results object.
        pass

def main():
    # Initialize camera
    cap = cv2.VideoCapture(0)
    tracker = GojoHandTracker()

    prev_time = 0

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            break

        image, hands_data = tracker.find_hands(image)

        # Drawing (Using MediaPipe results directly for drawing is more efficient)
        # Here we just visualize that something is happening
        for hand in hands_data:
            for lm in hand:
                h, w, c = image.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                cv2.circle(image, (cx, cy), 5, (0, 255, 0), cv2.FILLED)

        # Calculate FPS
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time)
        prev_time = curr_time

        cv2.putText(image, f'FPS: {int(fps)}', (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)
        cv2.imshow("Gojo Hand Tracking", image)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
