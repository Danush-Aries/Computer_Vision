# Gojo Hand Tracking System

A production-ready, high-precision hand tracking and Human-Computer Interaction (HCI) pipeline developed for real-time control signals using Computer Vision.

## 🚀 Overview

The Gojo Hand Tracking system implements a low-latency pipeline that translates physical hand gestures into digital control signals. By leveraging MediaPipe's ML-based landmark detection and a custom spatial mapping layer, it achieves smooth, stabilized cursor control and gesture-based interaction.

## 🛠️ Architecture

### 1. Real-time Pipeline (`tracker.py`)
- **ML Backend**: Utilizes MediaPipe Hands for 21-point 3D landmark extraction.
- **Confidence Thresholding**: Implements strict detection (0.7) and tracking (0.5) confidence to minimize jitter and false positives.
- **Coordinate Extraction**: Captures normalized $(x, y, z)$ coordinates for every landmark in real-time.

### 2. Spatial Mapping Layer (`mapper.py`)
- **Coordinate Normalization**: Maps raw sensor data to a normalized $[0, 1]$ screen space.
- **EMA Stabilization**: Uses an Exponential Moving Average (EMA) filter:
  $S_t = \alpha \cdot X_t + (1 - \alpha) \cdot S_{t-1}$
  This removes high-frequency jitter common in webcam-based tracking.
- **Pinch Detection**: Computes the Euclidean distance between the index finger tip (ID 8) and thumb tip (ID 4) in 3D space to trigger "Click/Pinch" events.

### 3. Performance Optimization (`main.py`)
- **Producer-Consumer Pattern**: Decouples frame acquisition from processing using Python `threading` and `Queue`. This prevents the GUI from freezing and ensures the latest frame is always processed.
- **Resolution Scaling**: Downsamples frames to $320 \times 240$ for the ML inference stage, significantly increasing FPS while maintaining spatial accuracy.
- **Non-blocking I/O**: Implements frame dropping to prioritize low latency over frame-perfect processing.

## 💻 Installation & Usage

### Prerequisites
- Python 3.8+
- OpenCV
- MediaPipe
- NumPy

```bash
pip install opencv-python mediapipe numpy
```

### Running the System
```bash
python main.py
```

## 📈 Performance Metrics
- **Latency**: < 30ms processing delay.
- **Stability**: High cursor stability via EMA smoothing.
- **FPS**: Optimized for 30+ FPS on standard laptop hardware.

## 🎯 HCI Applications
- Virtual Mouse/Keyboard control.
- Touchless interface for sterile environments (e.g., medical software).
- Gesture-based presentation control.
