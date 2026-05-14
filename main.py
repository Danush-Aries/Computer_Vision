import cv2
import mediapipe as mp
import numpy as np
import time
import threading
from queue import Queue
from typing import Optional

from tracker import GojoHandTracker
from mapper import SpatialMapper, ControlSignal

class OptimizedGojoSystem:
    """
    Production-ready optimized pipeline for Gojo Hand Tracking.
    Implements a Producer-Consumer pattern to decouple capture from processing.
    """
    def __init__(self,
                 video_source=0,
                 width=640,
                 height=480,
                 processing_res=(320, 240)):
        self.cap = cv2.VideoCapture(video_source)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

        self.processing_res = processing_res
        self.frame_queue = Queue(maxsize=2) # Minimize lag by keeping queue small
        self.result_queue = Queue(maxsize=2)

        self.tracker = GojoHandTracker()
        self.mapper = SpatialMapper()

        self.stopped = False
        self.fps = 0
        self.prev_time = time.time()

    def start(self):
        # Start capture thread
        self.capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.capture_thread.start()

        # Start processing thread
        self.processing_thread = threading.Thread(target=self._processing_loop, daemon=True)
        self.processing_thread.start()

    def stop(self):
        self.stopped = True
        self.cap.release()

    def _capture_loop(self):
        while not self.stopped:
            success, frame = self.cap.read()
            if not success:
                continue

            # Use non-blocking put to discard frames if processing is slow
            if not self.frame_queue.full():
                self.frame_queue.put(frame)

    def _processing_loop(self):
        while not self.stopped:
            if self.frame_queue.empty():
                continue

            frame = self.frame_queue.get()

            # Performance Optimization 1: Downscale frame for MediaPipe processing
            # This significantly boosts FPS with minimal precision loss for hand tracking
            proc_frame = cv2.resize(frame, self.processing_res)

            # Pipeline
            processed_image, hands_data = self.tracker.find_hands(proc_frame)

            # Map spatial data to control signals
            signals = []
            for hand in hands_data:
                signals.append(self.mapper.map_to_signal(hand))

            # Store result (frame and associated signals)
            if not self.result_queue.full():
                self.result_queue.put((frame, hands_data, signals))

    def get_latest_result(self):
        if not self.result_queue.empty():
            return self.result_queue.get()
        return None

    def run_interactive(self):
        """
        Main loop for visualization and interaction.
        """
        self.start()

        while True:
            result = self.get_latest_result()
            if result is None:
                continue

            frame, hands_data, signals = result

            # Visualization (draw on original high-res frame)
            # We need to scale the landmarks back to original size
            h, w, _ = frame.shape
            pw, ph = self.processing_res
            scale_x, scale_y = w/pw, h/ph

            for hand in hands_data:
                for lm in hand:
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    cv2.circle(frame, (cx, cy), 4, (0, 255, 0), cv2.FILLED)

            # Display Control Signals
            if signals:
                sig = signals[0]
                status = "PINCH" if sig.pinch_active else "NEUTRAL"
                cv2.putText(frame, f"State: {status}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                # Draw cursor simulation
                cursor_x = int(sig.cursor_pos[0] * w)
                cursor_y = int(sig.cursor_pos[1] * h)
                cv2.circle(frame, (cursor_x, cursor_y), 10, (255, 0, 0), 2)

            # Calculate FPS
            curr_time = time.time()
            self.fps = 1 / (curr_time - self.prev_time)
            self.prev_time = curr_time
            cv2.putText(frame, f"FPS: {int(self.fps)}", (10, 60),
                      cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

            cv2.imshow("Gojo Hand Tracking Optimized", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    system = OptimizedGojoSystem()
    system.run_interactive()
