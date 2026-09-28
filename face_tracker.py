import math
import threading
import time
from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np

CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720
CAMERA_FPS = 30
PROCESS_WIDTH = 640
PROCESS_HEIGHT = 360

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


@dataclass(frozen=True)
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
    gaze_features: Optional[Tuple[float, ...]]


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
        self._history_limit = 3
        self._neutral_pitch = 0.0
        self._neutral_yaw = 0.0
        self._neutral_head_offset = np.zeros(2, dtype=np.float32)
        self._neutral_ready = False
        self._neutral_samples = 0
        self._neutral_warmup_frames = 30
        self._feature_history: list[np.ndarray] = []
        self._frame_count = 0
        self._fps_started = time.perf_counter()
        self.processing_fps = 0.0
        self._gaze_motion_history: list[np.ndarray] = []

    def start(self) -> None:
        if self.running:
            return
        self._gaze_vector_history.clear()
        self._neutral_pitch = 0.0
        self._neutral_yaw = 0.0
        self._neutral_head_offset = np.zeros(2, dtype=np.float32)
        self._neutral_ready = False
        self.latest_observation = None
        self._feature_history.clear()
        self._frame_count = 0
        self._fps_started = time.perf_counter()
        self.processing_fps = 0.0
        self._gaze_motion_history.clear()
        self.running = True
        self.thread = threading.Thread(target=self._worker, daemon=True)
        self.thread.start()

    def stop(self) -> None:
        self.running = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=1.0)
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

    def collect_gaze_sample(
        self,
        duration_seconds: float = 0.45,
    ) -> Optional[tuple[Tuple[float, float], Tuple[float, ...]]]:
        deadline = time.perf_counter() + duration_seconds
        samples: list[tuple[Tuple[float, float], Tuple[float, ...]]] = []
        last_timestamp = -1.0

        while time.perf_counter() < deadline:
            observation = self.get_latest_observation()
            if (
                observation
                and observation.gaze_vector
                and not observation.blink
                and observation.confidence >= 0.62
                and observation.timestamp != last_timestamp
            ):
                if observation.gaze_features is not None:
                    samples.append((observation.gaze_vector, observation.gaze_features))
                last_timestamp = observation.timestamp
            time.sleep(0.006)

        if len(samples) < 6:
            return None

        vectors = np.asarray([sample[0] for sample in samples], dtype=np.float32)
        features = np.asarray([sample[1] for sample in samples], dtype=np.float32)
        center = np.median(vectors, axis=0)
        distances = np.linalg.norm(vectors - center, axis=1)
        cutoff = np.percentile(distances, 80)
        keep = distances <= cutoff
        if int(keep.sum()) < 4:
            keep = np.ones(len(samples), dtype=bool)
        median_vector = np.median(vectors[keep], axis=0)
        median_features = np.median(features[keep], axis=0)
        return (
            (float(median_vector[0]), float(median_vector[1])),
            tuple(float(value) for value in median_features),
        )

    def collect_gaze_vector_sample(self, duration_seconds: float = 0.45) -> Optional[Tuple[float, float]]:
        sample = self.collect_gaze_sample(duration_seconds)
        return sample[0] if sample else None

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
            min_detection_confidence=0.50,
            min_tracking_confidence=0.65,
        )

        while self.running:
            ok, frame = self.capture.read()
            if not ok:
                time.sleep(0.003)
                continue

            frame = cv2.flip(frame, 1)
            observation = self._process_frame(frame)
            with self.lock:
                self.latest_observation = observation

            self._frame_count += 1
            elapsed = time.perf_counter() - self._fps_started
            if elapsed >= 1.0:
                self.processing_fps = self._frame_count / elapsed
                self._frame_count = 0
                self._fps_started = time.perf_counter()

    def _process_frame(self, frame: np.ndarray) -> FaceObservation:
        timestamp = time.perf_counter()
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
        left_eye_offset = (np.asarray(left_pupil, dtype=np.float32) - left_eye_center) / np.array(
            [max(left_eye_width, 1.0), max(left_eye_height, 1.0)], dtype=np.float32
        )
        right_eye_offset = (np.asarray(right_pupil, dtype=np.float32) - right_eye_center) / np.array(
            [max(right_eye_width, 1.0), max(right_eye_height, 1.0)], dtype=np.float32
        )
        eye_offset = (left_eye_offset + right_eye_offset) / 2.0

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
        self._gaze_motion_history.append(np.asarray(stabilized_vector, dtype=np.float32))
        if len(self._gaze_motion_history) > 30:
            self._gaze_motion_history.pop(0)

        # Keep both eyes and head-pose signals for calibration. The learned mapper
        # can compensate for per-eye asymmetry instead of relying only on raw_x/raw_y.
        features = np.array(
            [
                left_eye_offset[0], left_eye_offset[1],
                right_eye_offset[0], right_eye_offset[1],
                adjusted_head_offset[0], adjusted_head_offset[1],
                adjusted_yaw, adjusted_pitch, adjusted_yaw * adjusted_head_offset[0],
                adjusted_pitch * adjusted_head_offset[1],
            ],
            dtype=np.float32,
        )
        self._feature_history.append(features)
        if len(self._feature_history) > 3:
            self._feature_history.pop(0)
        stable_features = np.median(np.stack(self._feature_history), axis=0)

        ear = (
            self._eye_aspect_ratio(points, LEFT_EYE_CONTOUR)
            + self._eye_aspect_ratio(points, RIGHT_EYE_CONTOUR)
        ) / 2.0
        blink = ear < 0.185

        yaw_error = abs(adjusted_yaw)
        pitch_error = abs(adjusted_pitch)
        eye_motion = float(np.linalg.norm(eye_offset))
        confidence = float(
            np.clip(
                1.0
                - (yaw_error * 0.16)
                - (pitch_error * 0.16)
                - (eye_motion * 0.08),
                0.0,
                1.0,
            )
        )

        return FaceObservation(
            timestamp=timestamp,
            frame_size=(frame_width, frame_height),
            face_detected=True,
            gaze_vector=stabilized_vector,
            gaze_features=tuple(float(value) for value in stable_features),
            left_pupil=(float(left_pupil[0]), float(left_pupil[1])),
            right_pupil=(float(right_pupil[0]), float(right_pupil[1])),
            yaw=yaw,
            pitch=pitch,
            roll=roll,
            ear=float(ear),
            blink=blink,
            confidence=confidence,
        )

    def get_diagnostics(self) -> dict:
        observation = self.get_latest_observation()
        camera_open = bool(self.capture is not None and self.capture.isOpened())
        if observation is None:
            return {
                "camera_open": camera_open,
                "face_detected": False,
                "iris_detected": False,
                "left_pupil": None,
                "right_pupil": None,
                "gaze_vector": None,
                "yaw": 0.0,
                "pitch": 0.0,
                "roll": 0.0,
                "ear": 0.0,
                "blink": False,
                "confidence": 0.0,
                "processing_fps": self.processing_fps,
                "gaze_motion": 0.0,
            }

        motion = 0.0
        if len(self._gaze_motion_history) >= 2:
            history = np.stack(self._gaze_motion_history, axis=0)
            motion = float(np.mean(np.linalg.norm(np.diff(history, axis=0), axis=1)))

        return {
            "camera_open": camera_open,
            "face_detected": observation.face_detected,
            "iris_detected": observation.left_pupil is not None and observation.right_pupil is not None,
            "left_pupil": observation.left_pupil,
            "right_pupil": observation.right_pupil,
            "gaze_vector": observation.gaze_vector,
            "yaw": observation.yaw,
            "pitch": observation.pitch,
            "roll": observation.roll,
            "ear": observation.ear,
            "blink": observation.blink,
            "confidence": observation.confidence,
            "processing_fps": self.processing_fps,
            "gaze_motion": motion,
            "neutral_ready": self._neutral_ready,
        }

    def _stabilize_gaze_vector(self, raw_x: float, raw_y: float) -> Tuple[float, float]:
        vector = np.asarray([raw_x, raw_y], dtype=np.float32)
        self._gaze_vector_history.append(vector)
        if len(self._gaze_vector_history) > self._history_limit:
            self._gaze_vector_history.pop(0)

        if len(self._gaze_vector_history) == 1:
            stabilized = vector
        else:
            history = np.stack(self._gaze_vector_history, axis=0)
            median = np.median(history, axis=0)
            stabilized = (0.78 * vector) + (0.22 * median)

        return float(stabilized[0]), float(stabilized[1])

    def _update_neutral_pose(self, yaw: float, pitch: float, head_offset: np.ndarray) -> None:
        # Establish the neutral face/head baseline only during the initial warm-up.
        # Continuously adapting it would absorb intentional gaze/head movement and
        # make the gaze signal appear almost static.
        if self._neutral_ready:
            return

        self._neutral_samples += 1
        alpha = 1.0 / float(self._neutral_samples)
        self._neutral_yaw = ((1.0 - alpha) * self._neutral_yaw) + (alpha * yaw)
        self._neutral_pitch = ((1.0 - alpha) * self._neutral_pitch) + (alpha * pitch)
        self._neutral_head_offset = ((1.0 - alpha) * self._neutral_head_offset) + (alpha * head_offset)

        if self._neutral_samples >= self._neutral_warmup_frames:
            self._neutral_ready = True

    @staticmethod
    def _estimate_head_pose(points: np.ndarray, frame_width: int, frame_height: int) -> Tuple[float, float, float]:
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
    def _empty_observation(timestamp: float, frame_width: int, frame_height: int) -> FaceObservation:
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
            gaze_features=None,
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
