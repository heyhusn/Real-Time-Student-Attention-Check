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
