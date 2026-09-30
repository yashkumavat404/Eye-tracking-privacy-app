import math
import threading
import time
from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np

from pupil_tracker import PupilDetector

CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720
CAMERA_FPS = 30
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


@dataclass(frozen=True)
class FaceObservation:
    timestamp: float
    frame_size: Tuple[int, int]
    face_detected: bool
    gaze_vector: Optional[Tuple[float, float]]
    left_pupil: Optional[Tuple[float, float]]
    right_pupil: Optional[Tuple[float, float]]
    left_pupil_confidence: float
    right_pupil_confidence: float
    pupil_confidence: float
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
        self._pupil_detector = PupilDetector()
        self._history_limit = 2
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

        read_failures = 0
        while self.running:
            if self.capture is None or not self.capture.isOpened():
                self.capture = self._open_camera()
                if self.capture is None:
                    with self.lock:
                        self.latest_observation = self._empty_observation(
                            time.perf_counter(),
                            CAMERA_WIDTH,
                            CAMERA_HEIGHT,
                        )
                    time.sleep(0.25)
                    continue

            ok, frame = self.capture.read()
            if not ok or frame is None or frame.size == 0:
                read_failures += 1
                if read_failures >= 15:
                    self.capture.release()
                    self.capture = None
                    read_failures = 0
                    time.sleep(0.15)
                else:
                    time.sleep(0.003)
                continue

            read_failures = 0
            frame = cv2.flip(frame, 1)
            try:
                observation = self._process_frame(frame)
            except (cv2.error, RuntimeError, ValueError, TypeError):
                observation = self._empty_observation(
                    time.perf_counter(),
                    frame.shape[1],
                    frame.shape[0],
                )

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

        scale = min(
            1.0,
            PROCESS_WIDTH / max(float(frame_width), 1.0),
            PROCESS_HEIGHT / max(float(frame_height), 1.0),
        )
        process_width = max(1, int(round(frame_width * scale)))
        process_height = max(1, int(round(frame_height * scale)))
        if process_width == frame_width and process_height == frame_height:
            process_frame = frame
        else:
            process_frame = cv2.resize(
                frame,
                (process_width, process_height),
                interpolation=cv2.INTER_AREA,
            )

        rgb_frame = cv2.cvtColor(process_frame, cv2.COLOR_BGR2RGB)
        results = self._face_mesh.process(rgb_frame)

        if not results.multi_face_landmarks:
            return self._empty_observation(timestamp, frame_width, frame_height)

        scale_x = frame_width / float(process_width)
        scale_y = frame_height / float(process_height)
        points = np.array(
            [
                (landmark.x * PROCESS_WIDTH * scale_x, landmark.y * PROCESS_HEIGHT * scale_y)
                for landmark in results.multi_face_landmarks[0].landmark
            ],
            dtype=np.float32,
        )

        process_points = points / np.array([scale_x, scale_y], dtype=np.float32)
        left_detection, right_detection = self._pupil_detector.detect_both(
            process_frame,
            process_points,
            LEFT_EYE,
            RIGHT_EYE,
            LEFT_IRIS,
            RIGHT_IRIS,
        )

        left_pupil = None if left_detection is None else (
            float(left_detection.center[0] * scale_x),
            float(left_detection.center[1] * scale_y),
        )
        right_pupil = None if right_detection is None else (
            float(right_detection.center[0] * scale_x),
            float(right_detection.center[1] * scale_y),
        )
        left_eye_offset = None if left_detection is None else np.asarray(
            [left_detection.normalized[0] - 0.5, left_detection.normalized[1] - 0.5],
            dtype=np.float32,
        )
        right_eye_offset = None if right_detection is None else np.asarray(
            [right_detection.normalized[0] - 0.5, right_detection.normalized[1] - 0.5],
            dtype=np.float32,
        )

        pupil_confidences = [
            detection.confidence
            for detection in (left_detection, right_detection)
            if detection is not None
        ]
        pupil_confidence = float(max(pupil_confidences)) if pupil_confidences else 0.0

        valid_offsets = []
        valid_weights = []
        if left_eye_offset is not None and left_detection is not None:
            valid_offsets.append(left_eye_offset)
            valid_weights.append(left_detection.confidence)
        if right_eye_offset is not None and right_detection is not None:
            valid_offsets.append(right_eye_offset)
            valid_weights.append(right_detection.confidence)

        if not valid_offsets:
            return self._empty_observation(timestamp, frame_width, frame_height)

        if len(valid_offsets) == 1:
            eye_offset = valid_offsets[0]
        else:
            weights = np.asarray(valid_weights, dtype=np.float32)
            weights /= max(float(weights.sum()), 1e-6)
            eye_offset = (
                valid_offsets[0] * weights[0]
                + valid_offsets[1] * weights[1]
            )

        # Head pose is diagnostic/auxiliary only. It is deliberately excluded
        # from the gaze vector and calibration features.
        yaw, pitch, roll = self._estimate_head_pose(points, frame_width, frame_height)
        face_center = points[NOSE_TIP]
        frame_center = np.array([frame_width / 2.0, frame_height / 2.0], dtype=np.float32)
        head_offset = (face_center - frame_center) / np.array([frame_width, frame_height], dtype=np.float32)
        self._update_neutral_pose(yaw, pitch, head_offset)

        # The screen-coordinate input is derived from normalized pupil position
        # inside the eye, never from face/head position or raw camera pixels.
        raw_x = 0.5 + (eye_offset[0] * 2.25)
        raw_y = 0.5 + (eye_offset[1] * 2.10)
        stabilized_vector = self._stabilize_gaze_vector(raw_x, raw_y)
        self._gaze_motion_history.append(np.asarray(stabilized_vector, dtype=np.float32))
        if len(self._gaze_motion_history) > 30:
            self._gaze_motion_history.pop(0)

        # Calibration features are pupil-only. Per-eye normalized pupil
        # coordinates remain available so calibration can learn binocular
        # asymmetry without ever learning head position as gaze.
        features = np.array(
            [
                left_detection.normalized[0] if left_detection else 0.5,
                left_detection.normalized[1] if left_detection else 0.5,
                right_detection.normalized[0] if right_detection else 0.5,
                right_detection.normalized[1] if right_detection else 0.5,
            ],
            dtype=np.float32,
        )
        stable_features = features

        ear = (
            self._eye_aspect_ratio(points, LEFT_EYE_CONTOUR)
            + self._eye_aspect_ratio(points, RIGHT_EYE_CONTOUR)
        ) / 2.0
        blink = ear < 0.185

        confidence = float(np.clip(
            pupil_confidence * (0.85 if blink else 1.0),
            0.0,
            1.0,
        ))

        return FaceObservation(
            timestamp=timestamp,
            frame_size=(frame_width, frame_height),
            face_detected=True,
            gaze_vector=stabilized_vector,
            gaze_features=tuple(float(value) for value in stable_features),
            left_pupil=left_pupil,
            right_pupil=right_pupil,
            left_pupil_confidence=0.0 if left_detection is None else float(left_detection.confidence),
            right_pupil_confidence=0.0 if right_detection is None else float(right_detection.confidence),
            pupil_confidence=pupil_confidence,
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
                "left_pupil_confidence": 0.0,
                "right_pupil_confidence": 0.0,
                "pupil_confidence": 0.0,
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
            "left_pupil_confidence": observation.left_pupil_confidence,
            "right_pupil_confidence": observation.right_pupil_confidence,
            "pupil_confidence": observation.pupil_confidence,
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
            stabilized = (0.90 * vector) + (0.10 * median)

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
    def _normalized_iris_offset(
        pupil: np.ndarray,
        points: np.ndarray,
        eye: dict[str, int],
    ) -> np.ndarray:
        """Return a consistent left-to-right, top-to-bottom iris offset."""
        first = points[eye["outer"]]
        second = points[eye["inner"]]
        left_corner, right_corner = (
            (first, second) if first[0] <= second[0] else (second, first)
        )
        top_point, bottom_point = (
            (points[eye["top"]], points[eye["bottom"]])
            if points[eye["top"]][1] <= points[eye["bottom"]][1]
            else (points[eye["bottom"]], points[eye["top"]])
        )

        horizontal = right_corner - left_corner
        vertical = bottom_point - top_point
        horizontal_norm = max(float(np.dot(horizontal, horizontal)), 1.0)
        vertical_norm = max(float(np.dot(vertical, vertical)), 1.0)

        h = float(np.dot(pupil - left_corner, horizontal) / horizontal_norm)
        v = float(np.dot(pupil - top_point, vertical) / vertical_norm)

        return np.asarray([h - 0.5, v - 0.5], dtype=np.float32)

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
            left_pupil_confidence=0.0,
            right_pupil_confidence=0.0,
            pupil_confidence=0.0,
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
