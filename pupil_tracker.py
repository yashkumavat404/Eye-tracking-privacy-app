from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np


@dataclass(frozen=True)
class PupilDetection:
    center: tuple[float, float]
    normalized: tuple[float, float]
    confidence: float
    radius: float


class PupilDetector:
    """Fast pupil localization inside MediaPipe eye/iris regions.

    MediaPipe landmarks are used only to define the eye/iris ROI. The gaze
    signal itself comes from the darkest pupil pixels inside that ROI.
    """

    MIN_IRIS_RADIUS = 2.5
    MAX_CENTER_DISTANCE = 0.92

    def __init__(self) -> None:
        self._clahe = cv2.createCLAHE(clipLimit=1.6, tileGridSize=(4, 4))

    def detect_both(
        self,
        frame: np.ndarray,
        points: np.ndarray,
        left_eye: dict[str, int],
        right_eye: dict[str, int],
        left_iris: list[int],
        right_iris: list[int],
    ) -> tuple[Optional[PupilDetection], Optional[PupilDetection]]:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return (
            self._detect_one(gray, points, left_eye, left_iris),
            self._detect_one(gray, points, right_eye, right_iris),
        )

    def _detect_one(
        self,
        gray: np.ndarray,
        points: np.ndarray,
        eye: dict[str, int],
        iris_indices: list[int],
    ) -> Optional[PupilDetection]:
        iris = points[iris_indices].astype(np.float32)
        iris_center = iris.mean(axis=0)
        iris_radius = float(np.mean(np.linalg.norm(iris - iris_center, axis=1)))
        iris_radius = max(iris_radius, 0.0)
        if iris_radius < self.MIN_IRIS_RADIUS:
            return None

        eye_points = np.asarray(
            [
                points[eye["outer"]],
                points[eye["inner"]],
                points[eye["top"]],
                points[eye["bottom"]],
            ],
            dtype=np.float32,
        )
        eye_width = float(np.linalg.norm(eye_points[0] - eye_points[1]))
        eye_height = float(np.linalg.norm(eye_points[2] - eye_points[3]))
        if eye_width < 4.0 or eye_height < 2.5:
            return None

        # Work only on a small iris-centered ROI. This avoids treating the
        # distorted full camera image or face position as a gaze coordinate.
        roi_radius = max(4.0, iris_radius * 1.65)
        x0 = max(0, int(np.floor(iris_center[0] - roi_radius)))
        y0 = max(0, int(np.floor(iris_center[1] - roi_radius)))
        x1 = min(gray.shape[1], int(np.ceil(iris_center[0] + roi_radius + 1)))
        y1 = min(gray.shape[0], int(np.ceil(iris_center[1] + roi_radius + 1)))
        if x1 - x0 < 5 or y1 - y0 < 5:
            return None

        crop = gray[y0:y1, x0:x1]
        enhanced = self._clahe.apply(crop)
        blurred = cv2.GaussianBlur(enhanced, (3, 3), 0)

        local_center = iris_center - np.array([x0, y0], dtype=np.float32)
        yy, xx = np.ogrid[:crop.shape[0], :crop.shape[1]]
        dx = (xx - local_center[0]) / max(roi_radius * 0.96, 1.0)
        dy = (yy - local_center[1]) / max(roi_radius * 0.82, 1.0)
        roi_mask = (dx * dx + dy * dy) <= 1.0

        values = blurred[roi_mask]
        if values.size < 12:
            return None

        q20 = float(np.percentile(values, 20))
        mean = float(np.mean(values))
        std = float(np.std(values))
        threshold = float(np.clip(min(q20 + 4.0, mean - 0.35 * std), 18.0, 125.0))

        dark = np.zeros_like(blurred, dtype=np.uint8)
        dark[(blurred <= threshold) & roi_mask] = 255
        dark = cv2.morphologyEx(dark, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
        dark = cv2.morphologyEx(dark, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))

        count, labels, stats, centroids = cv2.connectedComponentsWithStats(dark, 8)
        iris_area = np.pi * max(iris_radius, 1.0) ** 2
        best_score = -1.0
        best_center: Optional[np.ndarray] = None
        best_area = 0.0
        best_darkness = 0.0

        for label in range(1, count):
            area = float(stats[label, cv2.CC_STAT_AREA])
            if area < 2.0 or area > iris_area * 0.60:
                continue

            center = np.asarray(centroids[label], dtype=np.float32)
            distance = float(np.linalg.norm(center - local_center) / max(iris_radius, 1.0))
            if distance > self.MAX_CENTER_DISTANCE:
                continue

            width = float(stats[label, cv2.CC_STAT_WIDTH])
            height = float(stats[label, cv2.CC_STAT_HEIGHT])
            aspect = min(width, height) / max(width, height, 1.0)
            darkness = 1.0 - float(np.mean(blurred[labels == label])) / 255.0
            center_score = max(0.0, 1.0 - distance / self.MAX_CENTER_DISTANCE)
            area_score = min(1.0, area / max(iris_area * 0.12, 1.0))
            score = (
                0.16 * center_score
                + 0.46 * darkness
                + 0.22 * aspect
                + 0.16 * area_score
            )

            if score > best_score:
                best_score = score
                best_center = center
                best_area = area
                best_darkness = darkness

        # Low-quality cameras can blur the pupil into one dark blob. When no
        # clean component survives, use a darkness-weighted centroid, but only
        # when the dark pixels have enough contrast to be a plausible pupil.
        if best_center is None:
            inner = roi_mask & (
                ((xx - local_center[0]) ** 2 + (yy - local_center[1]) ** 2)
                <= (roi_radius * 0.88) ** 2
            )
            inner_values = blurred[inner]
            if inner_values.size:
                cutoff = float(np.percentile(inner_values, 12))
                weights = np.clip(cutoff + 6.0 - blurred.astype(np.float32), 0.0, None)
                weights *= inner.astype(np.float32)
                total = float(weights.sum())
                if total > 1.0:
                    ys, xs = np.indices(blurred.shape, dtype=np.float32)
                    weighted_center = np.array(
                        [(xs * weights).sum() / total, (ys * weights).sum() / total],
                        dtype=np.float32,
                    )
                    distance = float(
                        np.linalg.norm(weighted_center - local_center)
                        / max(iris_radius, 1.0)
                    )
                    if distance <= self.MAX_CENTER_DISTANCE:
                        best_center = weighted_center
                        best_area = float(np.count_nonzero(weights))
                        best_darkness = max(
                            0.0,
                            1.0 - float(np.mean(inner_values)) / 255.0,
                        )
                        best_score = 0.48 * (1.0 - distance / self.MAX_CENTER_DISTANCE) + 0.52 * best_darkness

        if best_center is None:
            return None

        pupil_center = best_center + np.array([x0, y0], dtype=np.float32)
        normalized = self._normalize_to_eye(pupil_center, points, eye)

        # Reject detections that land outside the visible eye or are too weak.
        if not (-0.18 <= normalized[0] <= 1.18 and -0.22 <= normalized[1] <= 1.22):
            return None

        confidence = float(
            np.clip(
                0.55 * max(0.0, min(1.0, best_score))
                + 0.30 * max(0.0, min(1.0, best_darkness))
                + 0.15 * min(1.0, best_area / max(iris_area * 0.08, 1.0)),
                0.0,
                1.0,
            )
        )
        if confidence < 0.42:
            return None

        return PupilDetection(
            center=(float(pupil_center[0]), float(pupil_center[1])),
            normalized=(float(normalized[0]), float(normalized[1])),
            confidence=confidence,
            radius=iris_radius,
        )

    @staticmethod
    def _normalize_to_eye(
        pupil: np.ndarray,
        points: np.ndarray,
        eye: dict[str, int],
    ) -> np.ndarray:
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
        h_den = max(float(np.dot(horizontal, horizontal)), 1.0)
        v_den = max(float(np.dot(vertical, vertical)), 1.0)
        x = float(np.dot(pupil - left_corner, horizontal) / h_den)
        y = float(np.dot(pupil - top_point, vertical) / v_den)
        return np.asarray([x, y], dtype=np.float32)
