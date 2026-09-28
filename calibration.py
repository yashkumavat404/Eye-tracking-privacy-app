import json
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np


CalibrationPoint = Tuple[str, Tuple[int, int], str]
CALIBRATION_FILE = Path(__file__).with_name("calibration_data.json")

ROW_NAMES = ["top", "upper", "middle", "lower", "bottom"]
COL_NAMES = ["left", "mid_left", "center", "mid_right", "right"]


def _grid_label(row: int, col: int) -> str:
    return f"{ROW_NAMES[row]}_{COL_NAMES[col]}"


@dataclass
class CalibrationSample:
    label: str
    screen_point: Tuple[int, int]
    gaze_vector: Tuple[float, float]
    gaze_features: Optional[Tuple[float, ...]] = None


class CalibrationMapper:
    GRID_SIZE = 5

    def __init__(self, screen_size: Tuple[int, int]) -> None:
        self.screen_width, self.screen_height = screen_size
        self.samples: Dict[str, CalibrationSample] = {}
        self._coefficients: Optional[np.ndarray] = None
        self._rmse: Optional[float] = None
        self._feature_coefficients: Optional[np.ndarray] = None

    @classmethod
    def build_grid(cls, screen_size: Tuple[int, int]) -> List[CalibrationPoint]:
        width, height = screen_size
        margin_x = max(90, width // 14)
        margin_y = max(90, height // 12)
        xs = np.linspace(margin_x, width - margin_x, cls.GRID_SIZE).astype(int)
        ys = np.linspace(margin_y, height - margin_y, cls.GRID_SIZE).astype(int)

        targets: List[CalibrationPoint] = []
        for row, y in enumerate(ys):
            for col, x in enumerate(xs):
                label = _grid_label(row, col)
                message = f"Look at the dot at row {row + 1}, column {col + 1}, then click."
                targets.append((label, (int(x), int(y)), message))
        return targets

    @classmethod
    def required_labels(cls) -> set[str]:
        return {_grid_label(row, col) for row in range(cls.GRID_SIZE) for col in range(cls.GRID_SIZE)}

    def clear(self) -> None:
        self.samples.clear()
        self._coefficients = None
        self._rmse = None
        self._feature_coefficients = None

    def load(self, path: Path = CALIBRATION_FILE) -> bool:
        if not path.exists():
            return False

        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return False

        saved_width = payload.get("screen_width")
        saved_height = payload.get("screen_height")
        sample_rows = payload.get("samples", [])
        if saved_width != self.screen_width or saved_height != self.screen_height:
            return False

        loaded_samples: Dict[str, CalibrationSample] = {}
        try:
            for row in sample_rows:
                label = row["label"]
                screen_point = (int(row["screen_point"][0]), int(row["screen_point"][1]))
                gaze_vector = (float(row["gaze_vector"][0]), float(row["gaze_vector"][1]))
                features = row.get("gaze_features")
                gaze_features = tuple(float(v) for v in features) if features else None
                loaded_samples[label] = CalibrationSample(label, screen_point, gaze_vector, gaze_features)
        except (KeyError, TypeError, ValueError, IndexError):
            return False

        self.samples = loaded_samples
        self._fit_regression()
        return self.is_complete()

    def save(self, path: Path = CALIBRATION_FILE) -> None:
        payload = {
            "screen_width": self.screen_width,
            "screen_height": self.screen_height,
            "samples": [
                {
                    "label": sample.label,
                    "screen_point": [sample.screen_point[0], sample.screen_point[1]],
                    "gaze_vector": [sample.gaze_vector[0], sample.gaze_vector[1]],
                    "gaze_features": list(sample.gaze_features) if sample.gaze_features else None,
                }
                for sample in self.samples.values()
            ],
        }
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def add_sample(
        self,
        label: str,
        screen_point: Tuple[int, int],
        gaze_vector: Tuple[float, float],
        gaze_features: Optional[Tuple[float, ...]] = None,
    ) -> None:
        self.samples[label] = CalibrationSample(label, screen_point, gaze_vector, gaze_features)
        self._fit_regression()

    def _fit_regression(self) -> None:
        """Fit a quadratic gaze-to-screen model from the calibration samples."""
        if len(self.samples) < 6:
            self._coefficients = None
            self._rmse = None
            self._feature_coefficients = None
            return

        vectors = np.asarray(
            [sample.gaze_vector for sample in self.samples.values()],
            dtype=np.float64,
        )
        points = np.asarray(
            [sample.screen_point for sample in self.samples.values()],
            dtype=np.float64,
        )
        x = vectors[:, 0]
        y = vectors[:, 1]
        design = np.column_stack(
            [np.ones(len(vectors)), x, y, x * y, x * x, y * y]
        )

        try:
            coefficients, _, _, _ = np.linalg.lstsq(
                design, points, rcond=None
            )
        except np.linalg.LinAlgError:
            self._coefficients = None
            self._rmse = None
            return

        predicted = design @ coefficients
        errors = np.linalg.norm(predicted - points, axis=1)
        self._coefficients = coefficients
        self._rmse = float(np.sqrt(np.mean(errors ** 2)))

        feature_rows = [sample.gaze_features for sample in self.samples.values()]
        if feature_rows and all(row is not None for row in feature_rows):
            feature_matrix = np.asarray(feature_rows, dtype=np.float64)
            design_features = np.column_stack([np.ones(len(feature_matrix)), feature_matrix])
            try:
                feature_coefficients, _, _, _ = np.linalg.lstsq(
                    design_features, points, rcond=None
                )
                self._feature_coefficients = feature_coefficients
            except np.linalg.LinAlgError:
                self._feature_coefficients = None
        else:
            self._feature_coefficients = None

    def is_complete(self) -> bool:
        return self.required_labels().issubset(self.samples)

    def map_vector_to_screen(self, gaze_vector: Tuple[float, float]) -> Tuple[int, int]:
        if not self.samples:
            x = int(np.clip(gaze_vector[0], 0.0, 1.0) * self.screen_width)
            y = int(np.clip(gaze_vector[1], 0.0, 1.0) * self.screen_height)
            return x, y

        target = np.asarray(gaze_vector, dtype=np.float64)

        if self._coefficients is not None and self.is_complete():
            mapped = self._map_quadratic(target)
        elif self._coefficients is not None and self.is_complete():
            mapped = self._map_quadratic(target)
        elif self.is_complete():
            mapped = self._grid_map(gaze_vector)
        else:
            mapped = self._weighted_map(gaze_vector)

        mapped[0] = np.clip(mapped[0], 0, self.screen_width - 1)
        mapped[1] = np.clip(mapped[1], 0, self.screen_height - 1)
        return int(mapped[0]), int(mapped[1])

    def map_observation(self, gaze_vector: Tuple[float, float], gaze_features: Optional[Tuple[float, ...]]) -> Tuple[int, int]:
        if self._feature_coefficients is not None and self.is_complete() and gaze_features is not None:
            feature_array = np.asarray(gaze_features, dtype=np.float64)
            if feature_array.size == self._feature_coefficients.shape[0] - 1:
                design = np.concatenate(([1.0], feature_array))
                mapped = design @ self._feature_coefficients
                mapped[0] = np.clip(mapped[0], 0, self.screen_width - 1)
                mapped[1] = np.clip(mapped[1], 0, self.screen_height - 1)
                return int(mapped[0]), int(mapped[1])
        return self.map_vector_to_screen(gaze_vector)

    def _map_quadratic(self, target: np.ndarray) -> np.ndarray:
        x, y = target
        features = np.array([1.0, x, y, x * y, x * x, y * y], dtype=np.float64)
        return features @ self._coefficients

    def calibration_rmse(self) -> Optional[float]:        return self._rmse

    def _weighted_map(self, gaze_vector: Tuple[float, float]) -> np.ndarray:
        vectors = np.array([sample.gaze_vector for sample in self.samples.values()], dtype=np.float32)
        points = np.array([sample.screen_point for sample in self.samples.values()], dtype=np.float32)
        target = np.array(gaze_vector, dtype=np.float32)
        distances = np.linalg.norm(vectors - target, axis=1)
        if float(distances.min()) < 1e-6:
            return points[int(distances.argmin())]
        weights = 1.0 / np.maximum(distances, 1e-3)
        weights /= weights.sum()
        return (points * weights[:, None]).sum(axis=0)

    def _grid_map(self, gaze_vector: Tuple[float, float]) -> np.ndarray:
        raw_x, raw_y = gaze_vector

        col_values = []
        for col in range(self.GRID_SIZE):
            labels = [_grid_label(row, col) for row in range(self.GRID_SIZE)]
            col_values.append(self._mean_axis(labels, axis=0))

        row_values = []
        for row in range(self.GRID_SIZE):
            labels = [_grid_label(row, col) for col in range(self.GRID_SIZE)]
            row_values.append(self._mean_axis(labels, axis=1))

        normalized_x = self._multi_segment_normalize(raw_x, col_values)
        normalized_y = self._multi_segment_normalize(raw_y, row_values)

        return np.array(
            [normalized_x * self.screen_width, normalized_y * self.screen_height],
            dtype=np.float32,
        )

    def _mean_axis(self, labels: List[str], axis: int) -> float:
        return float(sum(self.samples[label].gaze_vector[axis] for label in labels) / len(labels))

    @staticmethod
    def _multi_segment_normalize(value: float, anchors: List[float]) -> float:
        count = len(anchors)
        sorted_anchors = [anchors[0]]
        for anchor in anchors[1:]:
            sorted_anchors.append(max(anchor, sorted_anchors[-1] + 1e-4))

        if value <= sorted_anchors[0]:
            return 0.0
        if value >= sorted_anchors[-1]:
            return 1.0

        for idx in range(count - 1):
            left = sorted_anchors[idx]
            right = sorted_anchors[idx + 1]
            if left <= value <= right:
                span = max(right - left, 1e-4)
                local = (value - left) / span
                return float((idx + local) / (count - 1))

        return 1.0


class CalibrationSession:
    def __init__(self, screen_size: Tuple[int, int]) -> None:
        self.targets = CalibrationMapper.build_grid(screen_size)
        self.index = 0
        self.active = False

    def start(self) -> None:
        self.index = 0
        self.active = True

    def stop(self) -> None:
        self.active = False

    def current_target(self) -> Optional[CalibrationPoint]:
        if not self.active or self.index >= len(self.targets):
            return None
        return self.targets[self.index]

    def advance(self) -> bool:
        self.index += 1
        if self.index >= len(self.targets):
            self.active = False
            return False
        return True

    def progress_text(self) -> str:
        return f"Calibration {self.index + 1}/{len(self.targets)}"
