import threading
import time
from dataclasses import dataclass
from statistics import median
from typing import Dict, Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np


LEFT_IRIS = [468, 469, 470, 471, 472]
RIGHT_IRIS = [473, 474, 475, 476, 477]
LEFT_EYE = {"outer": 33, "inner": 133, "top": 159, "bottom": 145}
RIGHT_EYE = {"outer": 362, "inner": 263, "top": 386, "bottom": 374}
NOSE_TIP = 1
CALIBRATION_KEYS = {
    "top_left",
    "top_center",
    "top_right",
    "middle_left",
    "center",
    "middle_right",
    "bottom_left",
    "bottom_center",
    "bottom_right",
}


@dataclass
class GazeSample:
    screen_point: Tuple[int, int]
    confidence: float
    timestamp: float
    normalized_point: Tuple[float, float]


class GazeTracker:
    def __init__(self, screen_size: Tuple[int, int], camera_index: int = 0) -> None:
        self.screen_width, self.screen_height = screen_size
        self.camera_index = camera_index

        self.capture: Optional[cv2.VideoCapture] = None
        self.running = False
        self.thread: Optional[threading.Thread] = None
        self.latest_sample: Optional[GazeSample] = None
        self.lock = threading.Lock()

        center = np.array(
            [self.screen_width / 2.0, self.screen_height / 2.0], dtype=np.float32
        )
        self.smoothed_point = center.copy()
        self.previous_raw_point: Optional[np.ndarray] = None
        self.velocity = np.zeros(2, dtype=np.float32)
        self.previous_timestamp: Optional[float] = None
        self.calibration_samples: Dict[str, Tuple[float, float]] = {}

    def start(self) -> None:
        if self.running:
            return

        self.running = True
        self.thread = threading.Thread(target=self._worker, daemon=True)
        self.thread.start()

    def stop(self) -> None:
        self.running = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=1.5)
        self.thread = None

        if self.capture is not None:
            self.capture.release()
            self.capture = None

    def get_latest_gaze_point(self) -> Optional[Tuple[int, int]]:
        with self.lock:
            if self.latest_sample is None:
                return None
            return self.latest_sample.screen_point

    def get_latest_normalized_point(self) -> Optional[Tuple[float, float]]:
        with self.lock:
            if self.latest_sample is None:
                return None
            return self.latest_sample.normalized_point

    def collect_calibration_sample(
        self, label: str, duration_seconds: float = 0.75
    ) -> Optional[Tuple[float, float]]:
        deadline = time.time() + duration_seconds
        xs = []
        ys = []

        while time.time() < deadline:
            point = self.get_latest_normalized_point()
            if point is not None:
                xs.append(point[0])
                ys.append(point[1])
            time.sleep(0.01)

        if len(xs) < 10:
            return None

        sample = (float(median(xs)), float(median(ys)))
        self.calibration_samples[label] = sample
        return sample

    def has_calibration(self) -> bool:
        return CALIBRATION_KEYS.issubset(self.calibration_samples)

    def clear_calibration(self) -> None:
        self.calibration_samples.clear()

    def _open_camera(self) -> Optional[cv2.VideoCapture]:
        capture = cv2.VideoCapture(self.camera_index, cv2.CAP_DSHOW)
        if not capture.isOpened():
            capture = cv2.VideoCapture(self.camera_index)
        if not capture.isOpened():
            return None

        capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        capture.set(cv2.CAP_PROP_FPS, 60)
        capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        return capture

    def _worker(self) -> None:
        self.capture = self._open_camera()
        if self.capture is None:
            self.running = False
            return

        face_mesh = mp.solutions.face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

        try:
            while self.running:
                ok, frame = self.capture.read()
                if not ok:
                    time.sleep(0.01)
                    continue

                frame = cv2.flip(frame, 1)
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = face_mesh.process(rgb_frame)

                if not results.multi_face_landmarks:
                    time.sleep(0.003)
                    continue

                sample = self._estimate_gaze(results.multi_face_landmarks[0], frame.shape)
                if sample is not None:
                    with self.lock:
                        self.latest_sample = sample
        finally:
            face_mesh.close()
            if self.capture is not None:
                self.capture.release()
                self.capture = None

    def _estimate_gaze(self, face_landmarks, frame_shape: Tuple[int, int, int]) -> Optional[GazeSample]:
        frame_height, frame_width = frame_shape[:2]
        pts = np.array(
            [(lm.x * frame_width, lm.y * frame_height) for lm in face_landmarks.landmark],
            dtype=np.float32,
        )

        left_iris_center = pts[LEFT_IRIS].mean(axis=0)
        right_iris_center = pts[RIGHT_IRIS].mean(axis=0)
        iris_center = (left_iris_center + right_iris_center) / 2.0

        left_eye_center = self._eye_center(pts, LEFT_EYE)
        right_eye_center = self._eye_center(pts, RIGHT_EYE)
        eye_center = (left_eye_center + right_eye_center) / 2.0

        face_center = pts[NOSE_TIP]
        frame_center = np.array([frame_width / 2.0, frame_height / 2.0], dtype=np.float32)

        left_eye_width = np.linalg.norm(pts[LEFT_EYE["outer"]] - pts[LEFT_EYE["inner"]])
        right_eye_width = np.linalg.norm(pts[RIGHT_EYE["outer"]] - pts[RIGHT_EYE["inner"]])
        eye_width = max((left_eye_width + right_eye_width) / 2.0, 1.0)

        left_eye_height = np.linalg.norm(pts[LEFT_EYE["top"]] - pts[LEFT_EYE["bottom"]])
        right_eye_height = np.linalg.norm(pts[RIGHT_EYE["top"]] - pts[RIGHT_EYE["bottom"]])
        eye_height = max((left_eye_height + right_eye_height) / 2.0, 1.0)

        head_offset = (face_center - frame_center) / np.array([frame_width, frame_height], dtype=np.float32)
        eye_offset = (iris_center - eye_center) / np.array([eye_width, eye_height], dtype=np.float32)

        normalized_x = 0.5 + (head_offset[0] * 1.35) + (eye_offset[0] * 1.6)
        normalized_y = 0.5 + (head_offset[1] * 1.15) + (eye_offset[1] * 1.4)
        normalized_point = self._apply_calibration(
            float(np.clip(normalized_x, 0.0, 1.0)),
            float(np.clip(normalized_y, 0.0, 1.0)),
        )

        raw_point = np.array(
            [
                normalized_point[0] * self.screen_width,
                normalized_point[1] * self.screen_height,
            ],
            dtype=np.float32,
        )

        now = time.time()
        if self.previous_timestamp is None or self.previous_raw_point is None:
            predicted_point = raw_point
            self.velocity[:] = 0.0
        else:
            dt = max(now - self.previous_timestamp, 1e-3)
            instantaneous_velocity = (raw_point - self.previous_raw_point) / dt
            self.velocity = (0.78 * self.velocity) + (0.22 * instantaneous_velocity)
            prediction_horizon = min(0.014, dt)
            predicted_point = raw_point + (self.velocity * prediction_horizon)

        delta = predicted_point - self.smoothed_point
        movement = float(np.linalg.norm(delta))
        if movement < 5.0:
            predicted_point = self.smoothed_point

        smoothing_alpha = 0.30 if movement < 35.0 else 0.52
        self.smoothed_point = (1.0 - smoothing_alpha) * self.smoothed_point + smoothing_alpha * predicted_point
        self.smoothed_point = np.clip(
            self.smoothed_point,
            [0.0, 0.0],
            [self.screen_width - 1.0, self.screen_height - 1.0],
        )

        self.previous_raw_point = raw_point
        self.previous_timestamp = now
        confidence = float(np.clip(1.0 - np.linalg.norm(eye_offset) * 0.15, 0.4, 1.0))

        return GazeSample(
            screen_point=(int(self.smoothed_point[0]), int(self.smoothed_point[1])),
            confidence=confidence,
            timestamp=now,
            normalized_point=normalized_point,
        )

    def _apply_calibration(self, normalized_x: float, normalized_y: float) -> Tuple[float, float]:
        if not self.has_calibration():
            return normalized_x, normalized_y

        left = self._average_points("top_left", "middle_left", "bottom_left", axis=0)
        center_x = self._average_points("top_center", "center", "bottom_center", axis=0)
        right = self._average_points("top_right", "middle_right", "bottom_right", axis=0)
        top = self._average_points("top_left", "top_center", "top_right", axis=1)
        center_y = self._average_points("middle_left", "center", "middle_right", axis=1)
        bottom = self._average_points("bottom_left", "bottom_center", "bottom_right", axis=1)

        calibrated_x = self._piecewise_map(normalized_x, left, center_x, right)
        calibrated_y = self._piecewise_map(normalized_y, top, center_y, bottom)
        return calibrated_x, calibrated_y

    def _average_points(self, *keys: str, axis: int) -> float:
        values = [self.calibration_samples[key][axis] for key in keys]
        return float(sum(values) / len(values))

    @staticmethod
    def _piecewise_map(value: float, low: float, mid: float, high: float) -> float:
        low = min(low, mid - 1e-3)
        high = max(high, mid + 1e-3)

        if value <= mid:
            span = max(mid - low, 1e-3)
            normalized = 0.5 * ((value - low) / span)
        else:
            span = max(high - mid, 1e-3)
            normalized = 0.5 + 0.5 * ((value - mid) / span)

        return float(np.clip(normalized, 0.0, 1.0))

    @staticmethod
    def _eye_center(points: np.ndarray, eye_indices: dict) -> np.ndarray:
        return (
            points[eye_indices["outer"]]
            + points[eye_indices["inner"]]
            + points[eye_indices["top"]]
            + points[eye_indices["bottom"]]
        ) / 4.0
