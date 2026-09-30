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
    MODEL_VERSION = 2
    RIDGE_LAMBDA = 0.08

    def __init__(self, screen_size: Tuple[int, int]) -> None:
        self.screen_width, self.screen_height = screen_size
        self.samples: Dict[str, CalibrationSample] = {}
        self._coefficients: Optional[np.ndarray] = None
        self._rmse: Optional[float] = None
        self._feature_mean: Optional[np.ndarray] = None
        self._feature_scale: Optional[np.ndarray] = None

    @classmethod
    def build_grid(cls, screen_size: Tuple[int, int]) -> List[CalibrationPoint]:
        width, height = screen_size
        # Keep all 25 targets comfortably inside the drawable display area.
        # Explicitly generate five rows and five columns: 5 x 5 = 25.
        margin_x = max(80, int(width * 0.07))
        margin_y = max(80, int(height * 0.07))
        xs = np.linspace(margin_x, width - margin_x, 5, dtype=np.int32)
        ys = np.linspace(margin_y, height - margin_y, 5, dtype=np.int32)

        targets: List[CalibrationPoint] = []
        for row in range(5):
            for col in range(5):
                x = int(xs[col])
                y = int(ys[row])
                label = _grid_label(row, col)
                message = f"Look at the dot at row {row + 1} of 5, column {col + 1} of 5, then click."
                targets.append((label, (x, y), message))

        assert len(targets) == 25
        assert len({point[0] for point in targets}) == 25
        return targets

    @classmethod
    def required_labels(cls) -> set[str]:
        return {_grid_label(row, col) for row in range(cls.GRID_SIZE) for col in range(cls.GRID_SIZE)}

    def clear(self) -> None:
        self.samples.clear()
        self._coefficients = None
        self._rmse = None
        self._feature_mean = None
        self._feature_scale = None

    def load(self, path: Path = CALIBRATION_FILE) -> bool:
        if not path.exists():
            return False
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return False

        if payload.get("screen_width") != self.screen_width or payload.get("screen_height") != self.screen_height:
            return False

        loaded: Dict[str, CalibrationSample] = {}
        try:
            for row in payload.get("samples", []):
                label = row["label"]
                point = (int(row["screen_point"][0]), int(row["screen_point"][1]))
                vector = (float(row["gaze_vector"][0]), float(row["gaze_vector"][1]))
                features = row.get("gaze_features")
                loaded[label] = CalibrationSample(
                    label,
                    point,
                    vector,
                    tuple(float(v) for v in features) if features else None,
                )
        except (KeyError, TypeError, ValueError, IndexError):
            return False

        self.samples = loaded
        self._fit_regression()
        return self.is_complete()

    def save(self, path: Path = CALIBRATION_FILE) -> None:
        payload = {
            "model_version": self.MODEL_VERSION,
            "screen_width": self.screen_width,
            "screen_height": self.screen_height,
            "samples": [
                {
                    "label": sample.label,
                    "screen_point": list(sample.screen_point),
                    "gaze_vector": list(sample.gaze_vector),
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

    def _design_matrix(
        self,
        vectors: np.ndarray,
        features: Optional[np.ndarray],
        fit_scaler: bool = False,
    ) -> np.ndarray:
        x = vectors[:, 0]
        y = vectors[:, 1]
        parts = [
            np.ones(len(vectors), dtype=np.float64),
            x,
            y,
            x * x,
            x * y,
            y * y,
        ]

        if features is not None and features.size:
            if fit_scaler or self._feature_mean is None or self._feature_scale is None:
                self._feature_mean = np.mean(features, axis=0)
                scale = np.std(features, axis=0)
                self._feature_scale = np.where(scale < 1e-5, 1.0, scale)
            normalized = (features - self._feature_mean) / self._feature_scale
            parts.extend([normalized[:, index] for index in range(normalized.shape[1])])

        return np.column_stack(parts)

    def _fit_regression(self) -> None:
        if len(self.samples) < 6:
            self._coefficients = None
            self._rmse = None
            return

        ordered = list(self.samples.values())
        vectors = np.asarray([sample.gaze_vector for sample in ordered], dtype=np.float64)
        points = np.asarray([sample.screen_point for sample in ordered], dtype=np.float64)

        feature_rows = [sample.gaze_features for sample in ordered]
        features: Optional[np.ndarray]
        if feature_rows and all(row is not None for row in feature_rows):
            features = np.asarray(feature_rows, dtype=np.float64)
        else:
            features = None

        self._feature_mean = None
        self._feature_scale = None
        design = self._design_matrix(vectors, features, fit_scaler=True)

        try:
            regularization = self.RIDGE_LAMBDA * np.eye(design.shape[1], dtype=np.float64)
            regularization[0, 0] = 0.0
            gram = design.T @ design + regularization
            rhs = design.T @ points
            self._coefficients = np.linalg.solve(gram, rhs)
        except np.linalg.LinAlgError:
            self._coefficients = None
            self._rmse = None
            return

        predicted = design @ self._coefficients
        errors = np.linalg.norm(predicted - points, axis=1)
        self._rmse = float(np.sqrt(np.mean(errors ** 2)))

    def is_complete(self) -> bool:
        return self.required_labels().issubset(self.samples)

    def sample_count(self) -> int:
        return len(self.samples)

    def calibration_rmse(self) -> Optional[float]:
        return self._rmse

    def _predict(self, gaze_vector: Tuple[float, float], gaze_features: Optional[Tuple[float, ...]]) -> np.ndarray:
        vector = np.asarray([gaze_vector], dtype=np.float64)
        features = None
        if self._feature_mean is not None and self._feature_scale is not None:
            if gaze_features is None:
                # Mean feature vector is the neutral/average eye state.
                features = self._feature_mean.reshape(1, -1)
            else:
                raw = np.asarray([gaze_features], dtype=np.float64)
                if raw.shape[1] == self._feature_mean.shape[0]:
                    features = raw
        design = self._design_matrix(vector, features, fit_scaler=False)
        if self._coefficients is None:
            return np.asarray([gaze_vector[0] * self.screen_width, gaze_vector[1] * self.screen_height], dtype=np.float64)
        return design[0] @ self._coefficients

    def map_vector_to_screen(self, gaze_vector: Tuple[float, float]) -> Tuple[int, int]:
        if not self.samples:
            mapped = np.asarray(
                [gaze_vector[0] * self.screen_width, gaze_vector[1] * self.screen_height],
                dtype=np.float64,
            )
        elif self._coefficients is not None and self.is_complete():
            mapped = self._predict(gaze_vector, None)
        else:
            mapped = self._weighted_map(gaze_vector)

        mapped[0] = np.clip(mapped[0], 0, self.screen_width - 1)
        mapped[1] = np.clip(mapped[1], 0, self.screen_height - 1)
        return int(mapped[0]), int(mapped[1])

    def map_observation(
        self,
        gaze_vector: Tuple[float, float],
        gaze_features: Optional[Tuple[float, ...]],
    ) -> Tuple[int, int]:
        if self._coefficients is not None and self.is_complete():
            mapped = self._predict(gaze_vector, gaze_features)
        else:
            mapped = np.asarray(self.map_vector_to_screen(gaze_vector), dtype=np.float64)

        mapped[0] = np.clip(mapped[0], 0, self.screen_width - 1)
        mapped[1] = np.clip(mapped[1], 0, self.screen_height - 1)
        return int(mapped[0]), int(mapped[1])

    def _weighted_map(self, gaze_vector: Tuple[float, float]) -> np.ndarray:
        vectors = np.asarray([sample.gaze_vector for sample in self.samples.values()], dtype=np.float64)
        points = np.asarray([sample.screen_point for sample in self.samples.values()], dtype=np.float64)
        target = np.asarray(gaze_vector, dtype=np.float64)
        distances = np.linalg.norm(vectors - target, axis=1)
        if float(distances.min()) < 1e-6:
            return points[int(distances.argmin())]
        weights = 1.0 / np.maximum(distances, 1e-3)
        weights /= weights.sum()
        return (points * weights[:, None]).sum(axis=0)


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

    def progress_percent(self) -> int:
        return int(round((min(self.index, len(self.targets)) / len(self.targets)) * 100.0))

    def progress_text(self) -> str:
        return f"Calibration {min(self.index + 1, len(self.targets))}/{len(self.targets)}"
