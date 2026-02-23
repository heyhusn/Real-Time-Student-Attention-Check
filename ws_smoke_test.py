"""
ws_smoke_test.py
Connects to the running FastAPI WebSocket, sends 4 scores, reads ACKs,
then GETs /scores/student_test to confirm persistence.
"""
import asyncio
import json
import urllib.request

import websockets


async def test_websocket():
    url = "ws://localhost:8000/ws/student_test"
    print(f"Connecting to {url} ...")
    async with websockets.connect(url) as ws:
        test_scores = [0.9, 0.5, 0.1, 0.0]
        for score in test_scores:
            payload = json.dumps({"score": score, "timestamp": "2026-02-23T20:25:00Z"})
            await ws.send(payload)
            raw_ack = await asyncio.wait_for(ws.recv(), timeout=5)
            ack = json.loads(raw_ack)
            status = "✅" if ack.get("ack") else "❌"
            print(f"  Sent score={score:.2f}  |  ACK={ack}  {status}")
    print("\nWebSocket test DONE.")


def test_rest():
    print("\nChecking REST endpoint /scores/student_test ...")
    with urllib.request.urlopen("http://localhost:8000/scores/student_test?n=10") as r:
        data = json.loads(r.read())
    print(f"  student_id : {data['student_id']}")
    print(f"  scores     : {data['scores']}")
    assert len(data["scores"]) == 4, f"Expected 4 scores, got {len(data['scores'])}"
    print("  REST test  : ✅ PASSED\n")


if __name__ == "__main__":
    asyncio.run(test_websocket())
    test_rest()
    print("=" * 48)
    print("ALL SMOKE TESTS PASSED ✅")
    print("=" * 48)
