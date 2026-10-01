# Gojo Hand Tracking

**Point your hand. Click with a pinch. No mouse.**

![Python](https://img.shields.io/badge/Python-3.8%2B-3776AB?logo=python&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-4.8%2B-5C3EE8?logo=opencv&logoColor=white)
![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10%2B-orange)
![License](https://img.shields.io/badge/License-MIT-yellow)

A real-time hand-tracking pipeline that turns a webcam feed into a gesture-driven HCI controller. MediaPipe extracts 21 3D landmarks per hand; a smoothing filter converts them into cursor coordinates and pinch-to-click events you can wire into any application.

---

## Why this exists

Every hand-tracking demo online is a toy — grab code from a tutorial, run it, watch it jitter and drop frames. Gojo is the version that survives real use: producer/consumer threading so ML inference never blocks the display loop, EMA smoothing so the cursor stops shaking, downscaled inference for a real FPS boost, and clean error handling when the camera isn't available. Built as a foundation for gesture-driven interfaces, not a screenshot for LinkedIn.

---

## Try it in 60 seconds

```bash
git clone https://github.com/Danush-Aries/Computer_Vision
cd Computer_Vision
pip install -r requirements.txt
python main.py
```

Point your hand at the camera. Pinch index + thumb to click. Press `q` to quit.

---

## How it works

```
webcam
   |
   v
+-- producer thread ------+
|  cv2.VideoCapture       |
|  frames -> queue        |
+-----------|-------------+
            v
+-- ML thread ------------+
|  downscale 320x240      |
|  MediaPipe Hands (21x3) |
|  gesture classifier     |
+-----------|-------------+
            v
+-- display thread -------+
|  EMA cursor smoothing   |
|  skeleton overlay       |
|  pinch-to-click event   |
+-------------------------+
```

Gestures out of the box: Neutral, Pinch (click), Open Palm.

---

## Stack

| Layer | Tech |
|---|---|
| Landmark detection | MediaPipe Hands |
| Video I/O | OpenCV |
| Math | NumPy |
| Concurrency | stdlib `threading` + `queue` |
| Tests | pytest |

---

## More from Danush

Part of a broader stack of AI + security tooling:

- [jarvis](https://github.com/Danush-Aries/jarvis) — portable multi-provider AI assistant (voice/web/CLI)
- [breachintel](https://github.com/Danush-Aries/breachintel) — OSINT breach intelligence aggregator
- [cve-advisor](https://github.com/Danush-Aries/cve-advisor) — AI-powered CVE triage and patch recommendation
- [llm-fragility-lab](https://github.com/Danush-Aries/llm-fragility-lab) — adversarial testing lab for LLM robustness
- [network-intrusion-analyzer](https://github.com/Danush-Aries/network-intrusion-analyzer) — Suricata + Claude AI intrusion triage
- [autonomous-coding-agent](https://github.com/Danush-Aries/autonomous-coding-agent) — two-agent autonomous coding system

Built by [Dhanush](https://github.com/Danush-Aries) — AI engineering + cybersecurity.

## License

MIT.
