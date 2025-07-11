# 🛡️ Face Anti-Spoofing and Identity Matching System

This project implements a real-time face **anti-spoofing** and **identity verification** system with:

- ✅ Blink-based liveness detection
- ✅ Real-time face capture via webcam
- ✅ Matching captured faces against known identities in `face_db/`
- ✅ A modern web interface for monitoring and control

> Ideal for use cases such as exam proctoring, biometric access control, and identity validation in secure zones.

---

## 🚀 Features

- **Liveness Detection** – Uses blink detection and a trained anti-spoofing model to ensure the face is real
- **Auto-Capture** – Captures images only after confirming real, live faces
- **Capture Gallery** – View captured images directly from a web UI
- **Face Matching** – Compares each capture with known identities in `face_db/` using DeepFace + ArcFace
- **Web Control** – Start and stop detection from a browser

---

## 🧠 Tech Stack

- Python 3.8+
- Flask (Web UI backend)
- OpenCV (Video capture & image preprocessing)
- MediaPipe (Face & pose landmark detection)
- TensorFlow (Anti-spoofing model)
- DeepFace (Face matching using ArcFace)
- HTML/CSS/JS (Frontend interface)

---



