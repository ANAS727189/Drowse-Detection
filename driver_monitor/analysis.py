from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import cv2
import mediapipe as mp
import numpy as np


@dataclass
class AnalysisResult:
    ear: float
    mouth_open: float
    pitch: float
    yaw: float
    nose_point: tuple[int, int]
    left_eye_box: Optional[tuple[int, int, int, int]]
    right_eye_box: Optional[tuple[int, int, int, int]]


class FaceAnalyzer:
    LEFT_EYE = (33, 160, 158, 133, 153, 144)
    RIGHT_EYE = (362, 385, 387, 263, 373, 380)
    HEAD_POSE_IDS = (1, 199, 33, 263, 61, 291)

    def __init__(self) -> None:
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.face_3d = np.array(
            [
                (0.0, 0.0, 0.0),
                (0.0, -330.0, -65.0),
                (-225.0, 170.0, -135.0),
                (225.0, 170.0, -135.0),
                (-150.0, -150.0, -125.0),
                (150.0, -150.0, -125.0),
            ],
            dtype=np.float64,
        )

    def close(self) -> None:
        self.face_mesh.close()

    @staticmethod
    def _distance(p1: np.ndarray, p2: np.ndarray) -> float:
        return float(np.linalg.norm(p1 - p2))

    def _ear(self, landmarks, eye_indices: tuple[int, ...], width: int, height: int) -> float:
        coords = np.array(
            [(landmarks[i].x * width, landmarks[i].y * height) for i in eye_indices],
            dtype=np.float64,
        )
        vertical_1 = self._distance(coords[1], coords[5])
        vertical_2 = self._distance(coords[2], coords[4])
        horizontal = self._distance(coords[0], coords[3])
        if horizontal == 0:
            return 0.0
        return (vertical_1 + vertical_2) / (2.0 * horizontal)

    def _head_pose(self, landmarks, width: int, height: int) -> tuple[float, float, tuple[int, int]]:
        face_2d = np.array(
            [[landmarks[idx].x * width, landmarks[idx].y * height] for idx in self.HEAD_POSE_IDS],
            dtype=np.float64,
        )
        nose_point = (int(face_2d[0][0]), int(face_2d[0][1]))

        cam_matrix = np.array(
            [[float(width), 0, width / 2], [0, float(width), height / 2], [0, 0, 1]],
            dtype=np.float64,
        )
        dist_matrix = np.zeros((4, 1), dtype=np.float64)
        success, rotation_vector, _ = cv2.solvePnP(
            self.face_3d, face_2d, cam_matrix, dist_matrix
        )
        if not success:
            return 0.0, 0.0, nose_point

        rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
        rq = cv2.RQDecomp3x3(rotation_matrix)
        angles = rq[0]

        pitch = float(angles[0]) * 360.0
        yaw = float(angles[1]) * 360.0
        return pitch, yaw, nose_point

    def _eye_box(
        self,
        landmarks,
        eye_indices: tuple[int, ...],
        width: int,
        height: int,
        padding_ratio: float = 0.45,
    ) -> Optional[tuple[int, int, int, int]]:
        coords = np.array(
            [(landmarks[i].x * width, landmarks[i].y * height) for i in eye_indices],
            dtype=np.float64,
        )
        min_xy = coords.min(axis=0)
        max_xy = coords.max(axis=0)

        eye_w = max(float(max_xy[0] - min_xy[0]), 1.0)
        eye_h = max(float(max_xy[1] - min_xy[1]), 1.0)
        pad_x = int(eye_w * padding_ratio)
        pad_y = int(eye_h * padding_ratio) + 2

        x1 = max(int(min_xy[0]) - pad_x, 0)
        y1 = max(int(min_xy[1]) - pad_y, 0)
        x2 = min(int(max_xy[0]) + pad_x, width)
        y2 = min(int(max_xy[1]) + pad_y, height)
        if x2 - x1 < 2 or y2 - y1 < 2:
            return None
        return (x1, y1, x2, y2)

    def analyze(self, frame: np.ndarray) -> Optional[AnalysisResult]:
        height, width = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb)
        if not results.multi_face_landmarks:
            return None

        landmarks = results.multi_face_landmarks[0].landmark
        left_ear = self._ear(landmarks, self.LEFT_EYE, width, height)
        right_ear = self._ear(landmarks, self.RIGHT_EYE, width, height)
        ear = (left_ear + right_ear) / 2.0

        top_lip = np.array([landmarks[13].x * width, landmarks[13].y * height], dtype=np.float64)
        bottom_lip = np.array([landmarks[14].x * width, landmarks[14].y * height], dtype=np.float64)
        mouth_open = self._distance(top_lip, bottom_lip)

        pitch, yaw, nose_point = self._head_pose(landmarks, width, height)
        left_eye_box = self._eye_box(landmarks, self.LEFT_EYE, width, height)
        right_eye_box = self._eye_box(landmarks, self.RIGHT_EYE, width, height)
        return AnalysisResult(
            ear=ear,
            mouth_open=mouth_open,
            pitch=pitch,
            yaw=yaw,
            nose_point=nose_point,
            left_eye_box=left_eye_box,
            right_eye_box=right_eye_box,
        )
