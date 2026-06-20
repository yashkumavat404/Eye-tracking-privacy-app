from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import gradio as gr
import mediapipe as mp
import numpy as np

from calibration import CalibrationMapper
from smoothing_filter import ExponentialSmoothingFilter


PROCESS_WIDTH = 640
PROCESS_HEIGHT = 360
OUTPUT_SIZE = (960, 540)

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
class PrivacyResult:
    frame: np.ndarray
    status_text: str
    looking: bool
    confidence: float


class WebPrivacyProcessor:
    def __init__(self) -> None:
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.45,
            min_tracking_confidence=0.5,
        )
        self.mapper = CalibrationMapper(OUTPUT_SIZE)
        self.filter = ExponentialSmoothingFilter()
        self.filter.reset((OUTPUT_SIZE[0] // 2, OUTPUT_SIZE[1] // 2))
        self.gaze_history: list[np.ndarray] = []
        self.history_limit = 4
        self.neutral_yaw = 0.0
        self.neutral_pitch = 0.0
        self.neutral_head_offset = np.zeros(2, dtype=np.float32)
        self.neutral_ready = False
        self.last_focus_point = (OUTPUT_SIZE[0] // 2, OUTPUT_SIZE[1] // 2)

    def process(self, frame: np.ndarray, privacy_strength: float, focus_radius: int) -> PrivacyResult:
        resized = self._prepare_frame(frame)
        observation = self._analyze_frame(resized)
        result = self._render_privacy_view(resized, observation, privacy_strength, focus_radius)
        return result

    def _prepare_frame(self, frame: np.ndarray) -> np.ndarray:
        if frame is None:
            raise gr.Error("No webcam frame received. Allow webcam access and try again.")
        if frame.ndim != 3:
            raise gr.Error("Unexpected webcam frame format.")
        bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        return cv2.resize(bgr, OUTPUT_SIZE, interpolation=cv2.INTER_AREA)

    def _analyze_frame(self, frame: np.ndarray) -> dict:
        timestamp = time.time()
        frame_height, frame_width = frame.shape[:2]

        process_frame = cv2.resize(frame, (PROCESS_WIDTH, PROCESS_HEIGHT), interpolation=cv2.INTER_AREA)
        rgb_frame = cv2.cvtColor(process_frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)

        if not results.multi_face_landmarks:
            return {
                "timestamp": timestamp,
                "face_detected": False,
                "blink": False,
                "confidence": 0.0,
                "gaze_vector": None,
                "focus_point": self.last_focus_point,
                "yaw": 0.0,
                "pitch": 0.0,
            }

        scale_x = frame_width / PROCESS_WIDTH
        scale_y = frame_height / PROCESS_HEIGHT
        points = np.array(
            [
                (landmark.x * PROCESS_WIDTH * scale_x, landmark.y * PROCESS_HEIGHT * scale_y)
                for landmark in results.multi_face_landmarks[0].landmark
            ],
            dtype=np.float32,
        )

        yaw, pitch, _ = self._estimate_head_pose(points, frame_width, frame_height)
        left_pupil = points[LEFT_IRIS].mean(axis=0)
        right_pupil = points[RIGHT_IRIS].mean(axis=0)
        left_eye_center = self._eye_center(points, LEFT_EYE)
        right_eye_center = self._eye_center(points, RIGHT_EYE)
        left_eye_width = np.linalg.norm(points[LEFT_EYE["outer"]] - points[LEFT_EYE["inner"]])
        right_eye_width = np.linalg.norm(points[RIGHT_EYE["outer"]] - points[RIGHT_EYE["inner"]])
        left_eye_height = np.linalg.norm(points[LEFT_EYE["top"]] - points[LEFT_EYE["bottom"]])
        right_eye_height = np.linalg.norm(points[RIGHT_EYE["top"]] - points[RIGHT_EYE["bottom"]])

        eye_width = max((left_eye_width + right_eye_width) / 2.0, 1.0)
        eye_height = max((left_eye_height + right_eye_height) / 2.0, 1.0)
        iris_center = (left_pupil + right_pupil) / 2.0
        eye_center = (left_eye_center + right_eye_center) / 2.0
        eye_offset = (iris_center - eye_center) / np.array([eye_width, eye_height], dtype=np.float32)

        face_center = points[NOSE_TIP]
        frame_center = np.array([frame_width / 2.0, frame_height / 2.0], dtype=np.float32)
        head_offset = (face_center - frame_center) / np.array([frame_width, frame_height], dtype=np.float32)
        self._update_neutral_pose(yaw, pitch, head_offset)

        adjusted_yaw = yaw - self.neutral_yaw
        adjusted_pitch = pitch - self.neutral_pitch
        adjusted_head_offset = head_offset - self.neutral_head_offset

        raw_x = 0.5 + (eye_offset[0] * 2.18) + (adjusted_head_offset[0] * 1.20) + (adjusted_yaw * 0.08)
        raw_y = 0.5 + (eye_offset[1] * 1.92) + (adjusted_head_offset[1] * 0.85) - (adjusted_pitch * 0.04)
        gaze_vector = self._stabilize_gaze_vector(raw_x, raw_y)

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

        raw_point = self.mapper.map_vector_to_screen(gaze_vector)
        if blink:
            self.filter.freeze(timestamp)
            raw_point = self.last_focus_point

        filtered = self.filter.update(raw_point, confidence, timestamp)
        self.last_focus_point = filtered.point

        return {
            "timestamp": timestamp,
            "face_detected": True,
            "blink": blink,
            "confidence": confidence,
            "gaze_vector": gaze_vector,
            "focus_point": filtered.point,
            "yaw": yaw,
            "pitch": pitch,
        }

    def _render_privacy_view(
        self,
        frame: np.ndarray,
        observation: dict,
        privacy_strength: float,
        focus_radius: int,
    ) -> PrivacyResult:
        display = frame.copy()
        dim_alpha = float(np.clip(privacy_strength, 0.2, 1.0))

        if not observation["face_detected"]:
            protected = self._full_privacy(display, dim_alpha)
            protected = self._draw_status_chip(protected, "Face not detected", (48, 48, 230))
            status_text = "Face not detected. Center your face in the webcam."
            return PrivacyResult(
                frame=cv2.cvtColor(protected, cv2.COLOR_BGR2RGB),
                status_text=status_text,
                looking=False,
                confidence=0.0,
            )

        focus_point = observation["focus_point"]
        gaze_vector = observation["gaze_vector"]
        center_distance = float(np.linalg.norm(np.array(gaze_vector, dtype=np.float32) - np.array([0.5, 0.5], dtype=np.float32)))
        looking = (not observation["blink"]) and center_distance < 0.24 and observation["confidence"] >= 0.55

        if looking:
            protected = self._spotlight_privacy(display, focus_point, focus_radius, dim_alpha)
            protected = self._draw_focus_marker(protected, focus_point, focus_radius)
            protected = self._draw_status_chip(protected, "Focused on screen", (46, 170, 90))
            status_text = (
                f"Focused | confidence {observation['confidence']:.2f} | "
                f"yaw {observation['yaw']:.2f} | pitch {observation['pitch']:.2f}"
            )
        else:
            protected = self._full_privacy(display, dim_alpha)
            protected = self._draw_status_chip(protected, "Privacy shield active", (48, 48, 230))
            status_text = (
                f"Looking away or blinking | confidence {observation['confidence']:.2f} | "
                f"yaw {observation['yaw']:.2f} | pitch {observation['pitch']:.2f}"
            )

        return PrivacyResult(
            frame=cv2.cvtColor(protected, cv2.COLOR_BGR2RGB),
            status_text=status_text,
            looking=looking,
            confidence=observation["confidence"],
        )

    def _spotlight_privacy(
        self,
        frame: np.ndarray,
        center: Tuple[int, int],
        radius: int,
        dim_alpha: float,
    ) -> np.ndarray:
        blurred = cv2.GaussianBlur(frame, (0, 0), sigmaX=21, sigmaY=21)
        darkened = cv2.addWeighted(blurred, 1.0 - (0.45 * dim_alpha), np.zeros_like(frame), 0.45 * dim_alpha, 0)
        mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        cv2.circle(mask, center, radius, 255, -1, lineType=cv2.LINE_AA)
        feathered = cv2.GaussianBlur(mask, (0, 0), sigmaX=18, sigmaY=18)
        alpha = feathered.astype(np.float32) / 255.0
        alpha = alpha[:, :, None]
        composed = (frame.astype(np.float32) * alpha) + (darkened.astype(np.float32) * (1.0 - alpha))
        return composed.astype(np.uint8)

    def _full_privacy(self, frame: np.ndarray, dim_alpha: float) -> np.ndarray:
        blurred = cv2.GaussianBlur(frame, (0, 0), sigmaX=27, sigmaY=27)
        return cv2.addWeighted(blurred, 1.0 - (0.60 * dim_alpha), np.zeros_like(frame), 0.60 * dim_alpha, 0)

    @staticmethod
    def _draw_focus_marker(frame: np.ndarray, center: Tuple[int, int], radius: int) -> np.ndarray:
        cv2.circle(frame, center, radius, (255, 255, 255), 2, lineType=cv2.LINE_AA)
        cv2.circle(frame, center, 5, (255, 255, 255), -1, lineType=cv2.LINE_AA)
        return frame

    @staticmethod
    def _draw_status_chip(frame: np.ndarray, text: str, color: Tuple[int, int, int]) -> np.ndarray:
        x1, y1, x2, y2 = 20, 18, 350, 68
        overlay = frame.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1, lineType=cv2.LINE_AA)
        frame = cv2.addWeighted(overlay, 0.24, frame, 0.76, 0)
        cv2.putText(frame, text, (34, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (255, 255, 255), 2, cv2.LINE_AA)
        return frame

    def _stabilize_gaze_vector(self, raw_x: float, raw_y: float) -> Tuple[float, float]:
        vector = np.array([raw_x, raw_y], dtype=np.float32)
        self.gaze_history.append(vector)
        if len(self.gaze_history) > self.history_limit:
            self.gaze_history.pop(0)

        history = np.stack(self.gaze_history, axis=0)
        median = np.median(history, axis=0)
        stabilized = vector if len(self.gaze_history) == 1 else (0.68 * vector) + (0.32 * median)
        return float(stabilized[0]), float(stabilized[1])

    def _update_neutral_pose(self, yaw: float, pitch: float, head_offset: np.ndarray) -> None:
        alpha = 0.015 if self.neutral_ready else 0.08
        self.neutral_yaw = ((1.0 - alpha) * self.neutral_yaw) + (alpha * yaw)
        self.neutral_pitch = ((1.0 - alpha) * self.neutral_pitch) + (alpha * pitch)
        self.neutral_head_offset = ((1.0 - alpha) * self.neutral_head_offset) + (alpha * head_offset)
        self.neutral_ready = True

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
        sy = float(np.sqrt(rotation_matrix[0, 0] ** 2 + rotation_matrix[1, 0] ** 2))
        singular = sy < 1e-6

        if not singular:
            pitch = np.arctan2(rotation_matrix[2, 1], rotation_matrix[2, 2])
            yaw = np.arctan2(-rotation_matrix[2, 0], sy)
            roll = np.arctan2(rotation_matrix[1, 0], rotation_matrix[0, 0])
        else:
            pitch = np.arctan2(-rotation_matrix[1, 2], rotation_matrix[1, 1])
            yaw = np.arctan2(-rotation_matrix[2, 0], sy)
            roll = 0.0

        return float(yaw), float(pitch), float(roll)


PROCESSOR = WebPrivacyProcessor()


def process_webcam_frame(frame: np.ndarray, privacy_strength: float, focus_radius: int):
    result = PROCESSOR.process(frame, privacy_strength, focus_radius)
    mode = "Focused" if result.looking else "Protected"
    score = f"{result.confidence:.2f}"
    return result.frame, result.status_text, mode, score


with gr.Blocks(theme=gr.themes.Soft(), title="Eye Tracking Privacy Screen") as demo:
    gr.Markdown(
        """
        # Eye Tracking Privacy Screen
        Browser-based eye-tracking privacy demo for Hugging Face Spaces.
        """
    )

    with gr.Row():
        webcam = gr.Image(
            sources=["webcam"],
            type="numpy",
            streaming=True,
            label="Webcam Input",
        )
        output = gr.Image(
            type="numpy",
            label="Privacy View",
            streaming=True,
        )

    with gr.Row():
        privacy_strength = gr.Slider(0.2, 1.0, value=0.75, step=0.05, label="Privacy Strength")
        focus_radius = gr.Slider(80, 240, value=150, step=10, label="Clear Focus Radius")

    with gr.Row():
        status = gr.Textbox(label="Status", value="Allow webcam access to start.", interactive=False)
        mode = gr.Textbox(label="Mode", value="Waiting", interactive=False)
        confidence = gr.Textbox(label="Confidence", value="0.00", interactive=False)

    webcam.stream(
        fn=process_webcam_frame,
        inputs=[webcam, privacy_strength, focus_radius],
        outputs=[output, status, mode, confidence],
        queue=False,
        time_limit=60,
        stream_every=0.1,
    )


if __name__ == "__main__":
    demo.launch()
