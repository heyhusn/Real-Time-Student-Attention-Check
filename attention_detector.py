"""
attention_detector.py
---------------------

A lightweight class that:
* receives a BGR OpenCV frame,
* runs MediaPipe Face‑Mesh,
* extracts 6 landmarks for head‑pose (solvePnP),
* computes Eye‑Aspect‑Ratio (EAR) for both eyes,
* builds a simple 0‑1 attention score,
* draws debug overlays on the image.

No TensorFlow, only MediaPipe 0.10+.
"""

from typing import Tuple, List
import numpy as np
import cv2
import mediapipe as mp


class AttentionDetector:
    """
    Core object – call ``process(frame)`` for each video frame.
    """

    # -----------------------------------------------------------------
    # Landmarks used for head‑pose – indices from MediaPipe 0.10+ face mesh.
    # -----------------------------------------------------------------
    _POSE_LANDMARKS = {
        "nose_tip": 1,
        "chin": 152,
        "left_eye_left_corner": 33,
        "right_eye_right_corner": 263,
        "left_mouth_corner": 61,
        "right_mouth_corner": 291,
    }

    # -----------------------------------------------------------------
    # Six points per eye – ordering matches the classic EAR formula.
    # -----------------------------------------------------------------
    _LEFT_EYE_IDX = [33, 160, 158, 133, 153, 144]
    _RIGHT_EYE_IDX = [362, 385, 387, 263, 373, 380]

    # -----------------------------------------------------------------
    def __init__(
        self,
        img_width: int = 640,
        img_height: int = 480,
        head_pose_thresholds: Tuple[float, float] = (15.0, 15.0),
        ear_threshold: float = 0.22,
    ):
        """
        Parameters
        ----------
        img_width, img_height : expected frame size (pixels).
        head_pose_thresholds : (yaw_thr, pitch_thr) in degrees.
        ear_threshold : EAR below this → eyes considered closed.
        """
        self.img_width = img_width
        self.img_height = img_height
        self.head_yaw_thr, self.head_pitch_thr = head_pose_thresholds
        self.ear_thr = ear_threshold

        # -----------------------------------------------------------------
        # MediaPipe Face Mesh (no TensorFlow dependency)
        # -----------------------------------------------------------------
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

        # -----------------------------------------------------------------
        # 3‑D model points of a generic face (mm). Order matches _POSE_LANDMARKS.
        # -----------------------------------------------------------------
        self._model_points_3d = np.array(
            [
                (0.0, 0.0, 0.0),          # nose tip
                (0.0, -63.6, -12.5),      # chin
                (-43.3, 32.7, -26.0),    # left eye corner
                (43.3, 32.7, -26.0),     # right eye corner
                (-28.9, -28.9, -24.1),   # left mouth
                (28.9, -28.9, -24.1),    # right mouth
            ],
            dtype=np.float32,
        )

        # -----------------------------------------------------------------
        # Simple pinhole camera matrix for solvePnP.
        # -----------------------------------------------------------------
        focal = self.img_width
        self.camera_matrix = np.array(
            [
                [focal, 0, self.img_width / 2],
                [0, focal, self.img_height / 2],
                [0, 0, 1],
            ],
            dtype=np.float64,
        )
        self.dist_coeffs = np.zeros((4, 1))

    # -----------------------------------------------------------------
    def _landmarks_to_np(self, landmarks, img_shape) -> np.ndarray:
        """Convert normalized MediaPipe landmarks to pixel (int) coordinates."""
        h, w = img_shape[:2]
        return np.array(
            [(int(lm.x * w), int(lm.y * h)) for lm in landmarks],
            dtype=np.int32,
        )

    # -----------------------------------------------------------------
    def _get_pose(self, img_pts: np.ndarray) -> Tuple[float, float, float]:
        """solvePnP → yaw, pitch, roll (degrees)."""
        success, rvec, tvec = cv2.solvePnP(
            self._model_points_3d,
            img_pts.astype(np.float64),
            self.camera_matrix,
            self.dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        if not success:
            return 0.0, 0.0, 0.0

        rot_mat, _ = cv2.Rodrigues(rvec)

        sy = np.sqrt(rot_mat[0, 0] ** 2 + rot_mat[1, 0] ** 2)
        singular = sy < 1e-6

        if not singular:
            x = np.arctan2(rot_mat[2, 1], rot_mat[2, 2])
            y = np.arctan2(-rot_mat[2, 0], sy)
            z = np.arctan2(rot_mat[1, 0], rot_mat[0, 0])
        else:
            x = np.arctan2(-rot_mat[1, 2], rot_mat[1, 1])
            y = np.arctan2(-rot_mat[2, 0], sy)
            z = 0

        pitch = np.degrees(x)
        yaw = np.degrees(y)
        roll = np.degrees(z)
        return yaw, pitch, roll

    # -----------------------------------------------------------------
    @staticmethod
    def _eye_aspect_ratio(eye_pts: List[Tuple[int, int]]) -> float:
        """EAR = (|p2‑p6| + |p3‑p5|) / (2·|p1‑p4|)"""
        eye = np.array(eye_pts, dtype=np.float32)
        horiz = np.linalg.norm(eye[0] - eye[3])
        vert1 = np.linalg.norm(eye[1] - eye[5])
        vert2 = np.linalg.norm(eye[2] - eye[4])
        ear = (vert1 + vert2) / (2.0 * horiz)
        return ear

    # -----------------------------------------------------------------
    def process(self, frame: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Return (annotated_frame, attention_score∈[0,1]).
        """
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb)

        if not results.multi_face_landmarks:
            cv2.putText(frame, "No face", (30, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                        (0, 0, 255), 2)
            return frame, 0.0

        landmarks = results.multi_face_landmarks[0].landmark
        img_pts = self._landmarks_to_np(landmarks, frame.shape)

        # ----- head pose -------------------------------------------------
        pose_img_pts = np.array(
            [
                img_pts[self._POSE_LANDMARKS["nose_tip"]],
                img_pts[self._POSE_LANDMARKS["chin"]],
                img_pts[self._POSE_LANDMARKS["left_eye_left_corner"]],
                img_pts[self._POSE_LANDMARKS["right_eye_right_corner"]],
                img_pts[self._POSE_LANDMARKS["left_mouth_corner"]],
                img_pts[self._POSE_LANDMARKS["right_mouth_corner"]],
            ],
            dtype=np.float32,
        )
        yaw, pitch, roll = self._get_pose(pose_img_pts)
        cv2.putText(frame,
                    f"Yaw:{yaw:5.1f} Pitch:{pitch:5.1f}",
                    (30, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 0), 2)

        # ----- eyes ------------------------------------------------------
        left_eye = [tuple(img_pts[i]) for i in self._LEFT_EYE_IDX]
        right_eye = [tuple(img_pts[i]) for i in self._RIGHT_EYE_IDX]

        left_ear = self._eye_aspect_ratio(left_eye)
        right_ear = self._eye_aspect_ratio(right_eye)
        ear = (left_ear + right_ear) / 2.0
        cv2.putText(frame,
                    f"EAR:{ear:0.2f}",
                    (30, 60), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (255, 255, 0), 2)

        # ----- attention score -------------------------------------------
        score = 1.0
        if abs(yaw) > self.head_yaw_thr or abs(pitch) > self.head_pitch_thr:
            score *= 0.5
        if ear < self.ear_thr:
            score *= 0.2
        score = max(0.0, min(1.0, score))

        cv2.putText(frame,
                    f"Attention:{score:.2f}",
                    (30, 90),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (0, 255, 0), 2)

        # ----- optional visualisation ------------------------------------
        for idx in self._POSE_LANDMARKS.values():
            cv2.circle(frame, tuple(img_pts[idx]), 2, (0, 0, 255), -1)

        for pt in left_eye + right_eye:
            cv2.circle(frame, pt, 1, (255, 0, 255), -1)

        return frame, score

    def __del__(self):
        self.face_mesh.close()
