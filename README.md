<<<<<<< HEAD
# Real‑time Attention Detector (Python 3.13/3.14)

A pure‑Python, MediaPipe‑based solution that:

* reads the built‑in webcam,
* extracts face landmarks,
* estimates head pose (yaw/pitch),
* computes Eye‑Aspect‑Ratio for both eyes,
* returns a **0 – 1** attention score in real time,
* shows the video with live scores.

## 1️⃣  Install

```bash
python -m venv .venv          # optional – creates a clean environment
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
=======
# 🚀 Real-Time AI Attention Tracking System for Online Classrooms

## 🎯 The Problem
In online classes, instructors lack objective visibility into student engagement. Manual observation is unreliable, non-scalable, and impractical at scale.

## 💡 The Solution
A lightweight, privacy-first AI system that runs entirely inside the student’s browser. No video is transmitted. No biometric data is stored. Only a real-time attention score is sent to the server.

## 🧠 Technical Architecture
### AI Layer (Client-Side)
- MediaPipe Face Mesh running 100% in-browser via WebAssembly.
- No GPU servers required.
- Zero video streaming to backend.

### 📐 Attention Algorithm
The system combines multiple signals into a normalized score (0.0 – 1.0):
- Head Pose Estimation (Yaw & Pitch via 6-point facial landmark ratio) → detects looking away.
- Eye Aspect Ratio (EAR) → detects drowsiness / closed eyes.
- Weighted fusion → real-time attention score.

### ⚡ Backend Infrastructure
- FastAPI (Python 3.13).
- WebSocket-based real-time ingestion.
- Per-student rolling score history.
- Designed for high concurrency.

## 🔄 End-to-End Data Flow
Webcam → MediaPipe → Attention Score → WebSocket → FastAPI → Instructor Dashboard.

## 🏗 Key Design Decisions
- ✅ Client-side computation (privacy-first architecture).
- ✅ Only lightweight numeric scores transmitted.
- ✅ Automatic camera detection (external vs built-in with manual override).
- ✅ Real-time color-coded alerts (Green → Yellow → Red).
- ✅ Agora SDK-ready architecture for live classroom integration
>>>>>>> efa7c792e54212e63a7313f4f20a737e35ded138
