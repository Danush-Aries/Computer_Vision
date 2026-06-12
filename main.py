import cv2
import time
import threading
from queue import Queue, Empty

from tracker import GojoHandTracker
from mapper import SpatialMapper, ControlSignal

# Sentinel value used to signal worker threads to stop
_STOP = object()


class OptimizedGojoSystem:
    """
    Production-ready optimized pipeline for Gojo Hand Tracking.

    Implements a Producer-Consumer pattern to decouple frame capture from
    ML inference.  Two daemon threads run concurrently:

    * **Capture thread** – reads raw frames from the webcam as fast as
      possible and drops frames when the processing thread falls behind.
    * **Processing thread** – pulls frames, runs MediaPipe inference, maps
      landmarks to control signals, and publishes results.

    The main thread only handles visualization and user input, keeping the
    display loop responsive at all times.
    """

    def __init__(
        self,
        video_source: int = 0,
        width: int = 640,
        height: int = 480,
        processing_res: tuple = (320, 240),
    ):
        self.cap = cv2.VideoCapture(video_source)
        if not self.cap.isOpened():
            raise RuntimeError(
                f"Could not open video source {video_source!r}. "
                "Ensure a camera is connected and not in use by another application."
            )
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

        self.width = width
        self.height = height
        self.processing_res = processing_res  # (pw, ph) used for ML inference

        # Small queues deliberately drop stale frames to minimise latency
        self.frame_queue: Queue = Queue(maxsize=2)
        self.result_queue: Queue = Queue(maxsize=2)

        self.tracker = GojoHandTracker()
        self.mapper = SpatialMapper()

        self._stopped = False
        self.fps = 0.0
        self._prev_time = time.time()

    # ------------------------------------------------------------------
    # Thread management
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start the capture and processing background threads."""
        self._capture_thread = threading.Thread(
            target=self._capture_loop, daemon=True, name="capture"
        )
        self._processing_thread = threading.Thread(
            target=self._processing_loop, daemon=True, name="processing"
        )
        self._capture_thread.start()
        self._processing_thread.start()

    def stop(self) -> None:
        """Signal threads to stop and release resources."""
        self._stopped = True
        # Unblock any waiting get() calls with a sentinel
        self.frame_queue.put_nowait(_STOP)
        self.cap.release()

    # ------------------------------------------------------------------
    # Background threads
    # ------------------------------------------------------------------

    def _capture_loop(self) -> None:
        """Continuously read frames from the webcam."""
        while not self._stopped:
            success, frame = self.cap.read()
            if not success:
                # Camera may have been momentarily unavailable; keep trying
                time.sleep(0.005)
                continue

            # Drop oldest frame if the queue is full so we never block here
            if self.frame_queue.full():
                try:
                    self.frame_queue.get_nowait()
                except Empty:
                    pass
            self.frame_queue.put(frame)

    def _processing_loop(self) -> None:
        """Pull frames, run inference, and publish (frame, landmarks, signals)."""
        while not self._stopped:
            try:
                frame = self.frame_queue.get(timeout=0.1)
            except Empty:
                continue

            # Sentinel check
            if frame is _STOP:
                break

            # Downscale for faster ML inference
            pw, ph = self.processing_res
            proc_frame = cv2.resize(frame, (pw, ph))

            processed_image, hands_data = self.tracker.find_hands(proc_frame)

            # Scale landmark coordinates back to the full-resolution frame
            h, w = frame.shape[:2]
            scale_x, scale_y = w / pw, h / ph
            scaled_hands = []
            for hand in hands_data:
                scaled_hand = []
                for lm in hand:
                    from tracker import HandLandmark
                    scaled_hand.append(
                        HandLandmark(
                            id=lm.id,
                            x=lm.x,   # already normalised 0-1, no scaling needed
                            y=lm.y,
                            z=lm.z,
                            visibility=lm.visibility,
                        )
                    )
                scaled_hands.append(scaled_hand)

            signals = [self.mapper.map_to_signal(hand) for hand in scaled_hands]

            # Flip the original frame to match the mirrored inference frame
            display_frame = cv2.flip(frame, 1)

            if self.result_queue.full():
                try:
                    self.result_queue.get_nowait()
                except Empty:
                    pass
            self.result_queue.put((display_frame, scaled_hands, signals))

    # ------------------------------------------------------------------
    # Results access
    # ------------------------------------------------------------------

    def get_latest_result(self):
        """Return the most recent (frame, hands_data, signals) tuple or None."""
        try:
            return self.result_queue.get_nowait()
        except Empty:
            return None

    # ------------------------------------------------------------------
    # Main interactive loop
    # ------------------------------------------------------------------

    def run_interactive(self) -> None:
        """
        Start the pipeline and open the OpenCV display window.

        Controls
        --------
        q  – quit
        """
        self.start()
        print("Gojo Hand Tracking System running — press 'q' to quit.")

        while True:
            result = self.get_latest_result()
            if result is None:
                # Sleep briefly to avoid busy-waiting and burning CPU
                time.sleep(0.005)
                continue

            frame, hands_data, signals = result

            # Draw skeleton landmarks on the full-resolution display frame
            frame = self.tracker.draw_landmarks(frame, hands_data)

            # Overlay control signal information
            if signals:
                sig: ControlSignal = signals[0]
                status = "PINCH" if sig.pinch_active else "NEUTRAL"
                gesture_labels = {0: "Neutral", 1: "Pinch", 2: "Open Palm"}
                gesture_label = gesture_labels.get(sig.gesture_id, "Unknown")

                cv2.putText(
                    frame, f"State: {status}  [{gesture_label}]",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2,
                )

                # Draw cursor indicator at the index-finger tip position
                h, w = frame.shape[:2]
                cursor_x = int(sig.cursor_pos[0] * w)
                cursor_y = int(sig.cursor_pos[1] * h)
                color = (0, 0, 255) if sig.pinch_active else (255, 0, 0)
                cv2.circle(frame, (cursor_x, cursor_y), 12, color, 2)
                cv2.circle(frame, (cursor_x, cursor_y), 3, color, cv2.FILLED)

            # FPS overlay
            curr_time = time.time()
            elapsed = curr_time - self._prev_time
            self.fps = 1.0 / elapsed if elapsed > 0 else self.fps
            self._prev_time = curr_time
            cv2.putText(
                frame, f"FPS: {int(self.fps)}", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2,
            )

            cv2.imshow("Gojo Hand Tracking", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        self.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    system = OptimizedGojoSystem()
    system.run_interactive()
