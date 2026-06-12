# Gojo Hand Tracking System

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python)
![OpenCV](https://img.shields.io/badge/OpenCV-4.8%2B-green?logo=opencv)
![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10%2B-orange)
![License](https://img.shields.io/badge/License-MIT-yellow)

A real-time hand tracking and Human-Computer Interaction (HCI) pipeline that translates physical hand gestures into digital control signals using MediaPipe and OpenCV.

---

## What It Does

The system captures a live webcam feed, detects up to two hands per frame, and extracts 21 3D landmark points per hand using MediaPipe's ML model. Those landmarks are then mapped to normalized control signals — cursor position and click/pinch events — which can drive mouse emulation, presentation control, or any other gesture-based interface.

A producer-consumer threading architecture keeps the display loop running smoothly even when ML inference momentarily slows down.

---

## Features

- **21-point hand skeleton** — Full landmark extraction (wrist, knuckles, fingertips) in normalised (x, y, z) coordinates.
- **Pinch-to-click detection** — 3D Euclidean distance between index-finger tip (landmark 8) and thumb tip (landmark 4) triggers a click event below a tunable threshold.
- **EMA cursor smoothing** — Exponential Moving Average filter eliminates high-frequency jitter from the raw webcam signal.
- **Producer-Consumer pipeline** — Frame capture and ML inference run on separate daemon threads; the display thread never blocks on camera I/O or model inference.
- **Resolution downscaling for ML** — Frames are downscaled to 320x240 for inference and the results are projected back onto the full-resolution display frame, giving a large FPS boost with minimal precision loss.
- **Skeleton visualisation** — All 21 landmarks and the canonical hand-connection skeleton are drawn on the live feed.
- **Gesture classification** — Three gesture IDs out of the box: Neutral (0), Pinch (1), Open Palm (2).
- **Graceful camera error handling** — Raises a clear `RuntimeError` when the camera cannot be opened instead of crashing silently.

---

## Tech Stack

| Component | Library | Version |
|-----------|---------|---------|
| Hand landmark detection | [MediaPipe Hands](https://developers.google.com/mediapipe/solutions/vision/hand_landmarker) | >= 0.10 |
| Image capture & display | [OpenCV](https://opencv.org/) | >= 4.8 |
| Numerical operations | [NumPy](https://numpy.org/) | >= 1.24 |
| Concurrency | Python `threading` + `queue` | stdlib |

---

## Installation

### Prerequisites

- Python 3.8 or later
- A working webcam

### Install Dependencies

```bash
pip install -r requirements.txt
```

Or manually:

```bash
pip install opencv-python>=4.8.0 mediapipe>=0.10.0 numpy>=1.24.0
```

---

## Usage

### Run the Full Optimized Pipeline

```bash
python main.py
```

Press `q` to quit.

### Run the Standalone Tracker Demo

```bash
python tracker.py
```

### Use Modules Individually

```python
from tracker import GojoHandTracker
from mapper import SpatialMapper
import cv2

tracker = GojoHandTracker(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5,
)
mapper = SpatialMapper(smoothing_factor=0.5)

cap = cv2.VideoCapture(0)
while cap.isOpened():
    ok, frame = cap.read()
    if not ok:
        break

    frame, hands = tracker.find_hands(frame)
    frame = tracker.draw_landmarks(frame, hands)

    for hand in hands:
        signal = mapper.map_to_signal(hand)
        print(f"Cursor: {signal.cursor_pos}  Pinch: {signal.pinch_active}")

    cv2.imshow("Tracking", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
```

---

## Architecture

```
Webcam
  |
  v
[Capture Thread]  ──── frame_queue (maxsize=2) ────> [Processing Thread]
                                                            |
                                              MediaPipe Hands (320x240)
                                                            |
                                              SpatialMapper (EMA smooth)
                                                            |
                                         result_queue (maxsize=2)
                                                            |
                                                            v
                                                   [Display Thread]
                                              draw_landmarks + overlay
                                                     cv2.imshow
```

### Module Overview

| File | Responsibility |
|------|---------------|
| `tracker.py` | `GojoHandTracker` — wraps MediaPipe Hands, exposes `find_hands()` and `draw_landmarks()` |
| `mapper.py` | `SpatialMapper` — coordinate normalisation, EMA smoothing, pinch detection, gesture classification |
| `main.py` | `OptimizedGojoSystem` — producer-consumer pipeline, display loop, FPS counter |

---

## Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `max_num_hands` | `2` | Maximum simultaneous hands tracked |
| `min_detection_confidence` | `0.7` | Minimum score to accept a new detection |
| `min_tracking_confidence` | `0.5` | Minimum score to keep tracking an existing hand |
| `processing_res` | `(320, 240)` | Resolution used for ML inference |
| `smoothing_factor` | `0.5` | EMA alpha: 0 = no movement, 1 = no smoothing |
| `pinch_threshold` | `0.05` | 3D normalised distance below which a pinch is triggered |

---

## Running Tests

```bash
python -m pytest tests/ -v
```

No camera or GPU required — all tests run with a synthetic black frame and mocked MediaPipe results.

---

## Performance Notes

- Typical latency: < 30 ms on a standard laptop CPU.
- Typical FPS: 30-60+ depending on hardware (downscaling to 320x240 for inference is the main lever).
- The queue `maxsize=2` deliberately drops stale frames so the system always processes the most recent data rather than building up backlog.

---

## HCI Applications

- Virtual mouse and touchless cursor control
- Gesture-based presentation remote
- Sterile-environment interfaces (medical, lab, food-safe)
- Accessibility input devices
- AR/VR hand interaction prototypes

---

## License

This project is licensed under the [MIT License](https://opensource.org/licenses/MIT).
