import math
import threading
import time
from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np


CAMERA_WIDTH = 1920
CAMERA_HEIGHT = 1080
CAMERA_FPS = 60
PROCESS_WIDTH = 960
PROCESS_HEIGHT = 540

LEFT_IRIS = [468, 469, 470, 471, 472]
RIGHT_IRIS = [473, 474, 475, 476, 477]
LEFT_EYE = {"outer": 33, "inner": 133, "top": 159, "bottom": 145}
RIGHT_EYE = {"outer": 362, "inner": 263, "top": 386, "bottom": 374}
LEFT_EYE_CONTOUR = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_CONTOUR = [362, 385, 387, 263, 373, 380]
NOSE_TIP = 1
CHIN = 152
LEFT_EYE_OUTER = 33
RIGHT_EYE_OUTER = 263
LEFT_MOUTH = 61
RIGHT_MOUTH = 291
FACE_3D_MODEL = np.array(
    [
        (0.0, 0.0, 0.0),
        (0.0, -330.0, -65.0),
        (-225.0, 170.0, -135.0),
        (225.0, 170.0, -135.0),
        (-150.0, -150.0, -125.0),
        (150.0, -150.0, -125.0),
    ],
    dtype=np.float32,
)


@dataclass
class FaceObservation:
    timestamp: float
    frame_size: Tuple[int, int]
    face_detected: bool
    gaze_vector: Optional[Tuple[float, float]]
    left_pupil: Optional[Tuple[float, float]]
    right_pupil: Optional[Tuple[float, float]]
    yaw: float
    pitch: float
    roll: float
    ear: float
    blink: bool
    confidence: float


class FaceTracker:
    def __init__(self, camera_index: int = 0) -> None:
        self.camera_index = camera_index
        self.capture: Optional[cv2.VideoCapture] = None
        self.running = False
        self.thread: Optional[threading.Thread] = None
        self.latest_observation: Optional[FaceObservation] = None
        self.lock = threading.Lock()
        self._face_mesh = None
        self._gaze_vector_history: list[np.ndarray] = []
        self._history_limit = 4
        self._neutral_pitch = 0.0
        self._neutral_yaw = 0.0
        self._neutral_head_offset = np.zeros(2, dtype=np.float32)
        self._neutral_ready = False

    def start(self) -> None:
        if self.running:
            return
        self._gaze_vector_history.clear()
        self._neutral_pitch = 0.0
        self._neutral_yaw = 0.0
        self._neutral_head_offset = np.zeros(2, dtype=np.float32)
        self._neutral_ready = False
        self.running = True
        self.thread = threading.Thread(target=self._worker, daemon=True)
        self.thread.start()

    def stop(self) -> None:
        self.running = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=1.5)
        self.thread = None
        if self._face_mesh is not None:
            self._face_mesh.close()
            self._face_mesh = None
        if self.capture is not None:
            self.capture.release()
            self.capture = None

    def get_latest_observation(self) -> Optional[FaceObservation]:
        with self.lock:
            return self.latest_observation

    def collect_gaze_vector_sample(
        self,
        duration_seconds: float = 0.30,
    ) -> Optional[Tuple[float, float]]:
        deadline = time.time() + duration_seconds
        samples = []
        while time.time() < deadline:
            observation = self.get_latest_observation()
            if observation and observation.gaze_vector and not observation.blink:
                samples.append(observation.gaze_vector)
            time.sleep(0.004)

        if len(samples) < 6:
            return None

        sample_array = np.array(samples, dtype=np.float32)
        median_vector = np.median(sample_array, axis=0)
        return float(median_vector[0]), float(median_vector[1])

    def _open_camera(self) -> Optional[cv2.VideoCapture]:
        capture = cv2.VideoCapture(self.camera_index, cv2.CAP_DSHOW)
        if not capture.isOpened():
            capture = cv2.VideoCapture(self.camera_index)
        if not capture.isOpened():
            return None

        capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
        capture.set(cv2.CAP_PROP_FPS, CAMERA_FPS)
        capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        return capture

    def _worker(self) -> None:
        self.capture = self._open_camera()
        if self.capture is None:
            self.running = False
            return

        self._face_mesh = mp.solutions.face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.45,
            min_tracking_confidence=0.6,
        )

        while self.running:
            ok, frame = self.capture.read()
            if not ok:
                time.sleep(0.004)
                continue

            frame = cv2.flip(frame, 1)
            observation = self._process_frame(frame)
            with self.lock:
                self.latest_observation = observation

    def _process_frame(self, frame: np.ndarray) -> FaceObservation:
        timestamp = time.time()
        frame_height, frame_width = frame.shape[:2]

        process_frame = cv2.resize(
            frame,
            (PROCESS_WIDTH, PROCESS_HEIGHT),
            interpolation=cv2.INTER_AREA,
        )
        rgb_frame = cv2.cvtColor(process_frame, cv2.COLOR_BGR2RGB)
        results = self._face_mesh.process(rgb_frame)

        if not results.multi_face_landmarks:
            return self._empty_observation(timestamp, frame_width, frame_height)

        scale_x = frame_width / PROCESS_WIDTH
        scale_y = frame_height / PROCESS_HEIGHT
        points = np.array(
            [
                (landmark.x * PROCESS_WIDTH * scale_x, landmark.y * PROCESS_HEIGHT * scale_y)
                for landmark in results.multi_face_landmarks[0].landmark
            ],
            dtype=np.float32,
        )

        yaw, pitch, roll = self._estimate_head_pose(points, frame_width, frame_height)
        left_pupil = tuple(points[LEFT_IRIS].mean(axis=0))
        right_pupil = tuple(points[RIGHT_IRIS].mean(axis=0))
        left_eye_center = self._eye_center(points, LEFT_EYE)
        right_eye_center = self._eye_center(points, RIGHT_EYE)
        left_eye_width = np.linalg.norm(points[LEFT_EYE["outer"]] - points[LEFT_EYE["inner"]])
        right_eye_width = np.linalg.norm(points[RIGHT_EYE["outer"]] - points[RIGHT_EYE["inner"]])
        left_eye_height = np.linalg.norm(points[LEFT_EYE["top"]] - points[LEFT_EYE["bottom"]])
        right_eye_height = np.linalg.norm(points[RIGHT_EYE["top"]] - points[RIGHT_EYE["bottom"]])

        eye_width = max((left_eye_width + right_eye_width) / 2.0, 1.0)
        eye_height = max((left_eye_height + right_eye_height) / 2.0, 1.0)
        iris_center = (np.array(left_pupil, dtype=np.float32) + np.array(right_pupil, dtype=np.float32)) / 2.0
        eye_center = (left_eye_center + right_eye_center) / 2.0
        eye_offset = (iris_center - eye_center) / np.array([eye_width, eye_height], dtype=np.float32)

        face_center = points[NOSE_TIP]
        frame_center = np.array([frame_width / 2.0, frame_height / 2.0], dtype=np.float32)
        head_offset = (face_center - frame_center) / np.array([frame_width, frame_height], dtype=np.float32)
        self._update_neutral_pose(yaw, pitch, head_offset)

        adjusted_yaw = yaw - self._neutral_yaw
        adjusted_pitch = pitch - self._neutral_pitch
        adjusted_head_offset = head_offset - self._neutral_head_offset

        raw_x = 0.5 + (eye_offset[0] * 2.18) + (adjusted_head_offset[0] * 1.20) + (adjusted_yaw * 0.08)
        raw_y = 0.5 + (eye_offset[1] * 1.92) + (adjusted_head_offset[1] * 0.85) - (adjusted_pitch * 0.04)
        stabilized_vector = self._stabilize_gaze_vector(raw_x, raw_y)

        ear = (
            self._eye_aspect_ratio(points, LEFT_EYE_CONTOUR)
            + self._eye_aspect_ratio(points, RIGHT_EYE_CONTOUR)
        ) / 2.0
        blink = ear < 0.19
        confidence = float(
            np.clip(
                1.05
                - (abs(adjusted_yaw) + abs(adjusted_pitch)) * 0.10
                - np.linalg.norm(eye_offset) * 0.05,
                0.45,
                1.0,
            )
        )

        return FaceObservation(
            timestamp=timestamp,
            frame_size=(frame_width, frame_height),
            face_detected=True,
            gaze_vector=stabilized_vector,
            left_pupil=(float(left_pupil[0]), float(left_pupil[1])),
            right_pupil=(float(right_pupil[0]), float(right_pupil[1])),
            yaw=yaw,
            pitch=pitch,
            roll=roll,
            ear=float(ear),
            blink=blink,
            confidence=confidence,
        )

    def _stabilize_gaze_vector(self, raw_x: float, raw_y: float) -> Tuple[float, float]:
        vector = np.array([raw_x, raw_y], dtype=np.float32)
        self._gaze_vector_history.append(vector)
        if len(self._gaze_vector_history) > self._history_limit:
            self._gaze_vector_history.pop(0)

        history = np.stack(self._gaze_vector_history, axis=0)
        median = np.median(history, axis=0)
        if len(self._gaze_vector_history) == 1:
            stabilized = vector
        else:
            stabilized = (0.68 * vector) + (0.32 * median)
        return float(stabilized[0]), float(stabilized[1])

    def _update_neutral_pose(
        self,
        yaw: float,
        pitch: float,
        head_offset: np.ndarray,
    ) -> None:
        alpha = 0.015 if self._neutral_ready else 0.08
        self._neutral_yaw = ((1.0 - alpha) * self._neutral_yaw) + (alpha * yaw)
        self._neutral_pitch = ((1.0 - alpha) * self._neutral_pitch) + (alpha * pitch)
        self._neutral_head_offset = ((1.0 - alpha) * self._neutral_head_offset) + (alpha * head_offset)
        self._neutral_ready = True

    def _estimate_head_pose(
        self,
        points: np.ndarray,
        frame_width: int,
        frame_height: int,
    ) -> Tuple[float, float, float]:
        image_points = np.array(
            [
                points[NOSE_TIP],
                points[CHIN],
                points[LEFT_EYE_OUTER],
                points[RIGHT_EYE_OUTER],
                points[LEFT_MOUTH],
                points[RIGHT_MOUTH],
            ],
            dtype=np.float32,
        )

        focal_length = frame_width
        camera_matrix = np.array(
            [
                [focal_length, 0, frame_width / 2.0],
                [0, focal_length, frame_height / 2.0],
                [0, 0, 1],
            ],
            dtype=np.float32,
        )
        dist_coeffs = np.zeros((4, 1), dtype=np.float32)

        ok, rotation_vector, _ = cv2.solvePnP(
            FACE_3D_MODEL,
            image_points,
            camera_matrix,
            dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        if not ok:
            return 0.0, 0.0, 0.0

        rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
        sy = math.sqrt(rotation_matrix[0, 0] ** 2 + rotation_matrix[1, 0] ** 2)
        singular = sy < 1e-6

        if not singular:
            pitch = math.atan2(rotation_matrix[2, 1], rotation_matrix[2, 2])
            yaw = math.atan2(-rotation_matrix[2, 0], sy)
            roll = math.atan2(rotation_matrix[1, 0], rotation_matrix[0, 0])
        else:
            pitch = math.atan2(-rotation_matrix[1, 2], rotation_matrix[1, 1])
            yaw = math.atan2(-rotation_matrix[2, 0], sy)
            roll = 0.0

        return float(yaw), float(pitch), float(roll)

    @staticmethod
    def _empty_observation(
        timestamp: float,
        frame_width: int,
        frame_height: int,
    ) -> FaceObservation:
        return FaceObservation(
            timestamp=timestamp,
            frame_size=(frame_width, frame_height),
            face_detected=False,
            gaze_vector=None,
            left_pupil=None,
            right_pupil=None,
            yaw=0.0,
            pitch=0.0,
            roll=0.0,
            ear=0.0,
            blink=False,
            confidence=0.0,
        )

    @staticmethod
    def _eye_center(points: np.ndarray, eye_indices: dict) -> np.ndarray:
        return (
            points[eye_indices["outer"]]
            + points[eye_indices["inner"]]
            + points[eye_indices["top"]]
            + points[eye_indices["bottom"]]
        ) / 4.0

    @staticmethod
    def _eye_aspect_ratio(points: np.ndarray, contour: list[int]) -> float:
        p1, p2, p3, p4, p5, p6 = [points[index] for index in contour]
        vertical = np.linalg.norm(p2 - p6) + np.linalg.norm(p3 - p5)
        horizontal = 2.0 * np.linalg.norm(p1 - p4)
        return float(vertical / max(horizontal, 1e-6))
