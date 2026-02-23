"""
main.py
-------
FastAPI application.

Endpoints
---------
GET  /                        → serves frontend/index.html
GET  /frontend/{path}         → serves static frontend files
WS   /ws/{student_id}         → receives attention scores from browser
GET  /scores/{student_id}     → last N scores for a student (JSON)
GET  /scores                  → latest score per every connected student
"""

import json
import logging
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles

from backend.attention_store import store

# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
log = logging.getLogger("attention")

app = FastAPI(title="Attention Checking System", version="1.0.0")

# ---------------------------------------------------------------------------
# Static files — serve the entire frontend/ folder
# ---------------------------------------------------------------------------
FRONTEND_DIR = Path(__file__).parent.parent / "frontend"

app.mount("/frontend", StaticFiles(directory=str(FRONTEND_DIR)), name="frontend")


@app.get("/", response_class=HTMLResponse)
async def index():
    """Serve the main test page."""
    html_path = FRONTEND_DIR / "index.html"
    return HTMLResponse(content=html_path.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# WebSocket endpoint — one connection per student
# ---------------------------------------------------------------------------
@app.websocket("/ws/{student_id}")
async def websocket_attention(websocket: WebSocket, student_id: str):
    await websocket.accept()
    log.info(f"[{student_id}] WebSocket connected")

    try:
        while True:
            raw = await websocket.receive_text()
            try:
                data = json.loads(raw)
                score = float(data.get("score", 0.0))
                timestamp = str(data.get("timestamp", ""))
                store.record(student_id, score, timestamp)
                log.info(f"[{student_id}] score={score:.2f}  ts={timestamp}")

                # Echo back a simple ACK so the browser knows the server is alive
                await websocket.send_text(json.dumps({"ack": True, "score": score}))

            except (ValueError, KeyError, json.JSONDecodeError) as exc:
                log.warning(f"[{student_id}] Bad payload: {exc} — raw={raw!r}")

    except WebSocketDisconnect:
        log.info(f"[{student_id}] WebSocket disconnected")


# ---------------------------------------------------------------------------
# REST: read scores back (useful for dashboard / testing)
# ---------------------------------------------------------------------------
@app.get("/scores/{student_id}")
async def get_scores(student_id: str, n: int = 20):
    """Return the last *n* attention scores for a student."""
    return {
        "student_id": student_id,
        "scores": store.get_latest(student_id, n),
    }


@app.get("/scores")
async def get_all_scores():
    """Return the most-recent score for every student currently tracked."""
    return store.get_all_latest()


@app.get("/students")
async def get_students():
    """Return a list of all student IDs currently tracked."""
    return {"students": store.get_student_ids()}
