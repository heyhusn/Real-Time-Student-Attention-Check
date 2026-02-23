"""
run_attention.py
----------------

A tiny demo that opens the webcam, runs AttentionDetector on each
frame and displays a live window with the attention score.

Run:
    python run_attention.py
"""

import cv2
from attention_detector import AttentionDetector


def main():
    # ------------------- camera init -------------------------------
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # CAP_DSHOW fixes a Windows bug
    if not cap.isOpened():
        print("❌ Could not open webcam – check device permissions.")
        return

    # force a known resolution (helps MediaPipe stay fast)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # ------------------- detector -------------------------------
    detector = AttentionDetector(img_width=640, img_height=480)

    print("👀  Starting – press 'q' to quit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("⚠️  Failed to grab a frame – exiting.")
            break

        annotated, score = detector.process(frame)

        cv2.imshow("Attention Detector", annotated)

        # quit on 'q'
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # ------------------- cleanup -------------------------------
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
