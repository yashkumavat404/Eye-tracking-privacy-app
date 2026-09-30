import json
import os
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
    MODEL_VERSION = 5
    # The pupil-only model has six terms and 25 calibration samples. A small
    # ridge term keeps the fit stable without pulling legitimate edge/center
    # gaze positions toward one side of the display.
    RIDGE_LAMBDA = 0.003
    GAZE_FEATURE_COUNT = 4

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
        # Place calibration targets close to the real display boundaries so the
        # model learns edge/corner pupil behavior instead of only the center.
        # Keep a small safety margin so the target remains fully clickable.
        margin_x = max(32, int(width * 0.025))
        margin_y = max(32, int(height * 0.025))
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

        if payload.get("model_version") != self.MODEL_VERSION:
            return False
        if payload.get("screen_width") != self.screen_width or payload.get("screen_height") != self.screen_height:
            return False

        loaded: Dict[str, CalibrationSample] = {}
        try:
            rows = payload.get("samples", [])
            if not isinstance(rows, list) or len(rows) != self.GRID_SIZE ** 2:
                return False
            for row in rows:
                label = row["label"]
                point = (int(row["screen_point"][0]), int(row["screen_point"][1]))
                vector = (float(row["gaze_vector"][0]), float(row["gaze_vector"][1]))
                features = row.get("gaze_features")
                if features is not None and len(features) != self.GAZE_FEATURE_COUNT:
                    return False
                loaded[label] = CalibrationSample(
                    label,
                    point,
                    vector,
                    tuple(float(v) for v in features) if features else None,
                )
        except (KeyError, TypeError, ValueError, IndexError):
            return False

        if set(loaded) != self.required_labels() or len(loaded) != self.GRID_SIZE ** 2:
            return False

        self.samples = loaded
        self._fit_regression()
        return self.is_complete()

    def save(self, path: Path = CALIBRATION_FILE) -> None:
        if not self.is_complete():
            raise ValueError("Refusing to save an incomplete calibration")
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
        temporary_path = path.with_suffix(path.suffix + ".tmp")
        temporary_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        os.replace(temporary_path, path)

    def add_sample(
        self,
        label: str,
        screen_point: Tuple[int, int],
        gaze_vector: Tuple[float, float],
        gaze_features: Optional[Tuple[float, ...]] = None,
    ) -> None:
        if label not in self.required_labels():
            raise ValueError(f"Unknown calibration target: {label}")
        if not all(np.isfinite(value) for value in gaze_vector):
            raise ValueError("Calibration gaze vector must be finite")
        if gaze_features is not None and not all(np.isfinite(value) for value in gaze_features):
            raise ValueError("Calibration gaze features must be finite")
        if gaze_features is not None and len(gaze_features) != self.GAZE_FEATURE_COUNT:
            raise ValueError(f"Calibration requires {self.GAZE_FEATURE_COUNT} gaze features")
        self.samples[label] = CalibrationSample(label, screen_point, gaze_vector, gaze_features)
        self._fit_regression()

    def _design_matrix(
        self,
        vectors: np.ndarray,
        features: Optional[np.ndarray],
        fit_scaler: bool = False,
    ) -> np.ndarray:
        if features is None:
            features = np.column_stack(
                [
                    0.5 + vectors[:, 0],
                    0.5 + vectors[:, 1],
                    0.5 + vectors[:, 0],
                    0.5 + vectors[:, 1],
                ]
            )

        values = np.asarray(features, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != self.GAZE_FEATURE_COUNT:
            raise ValueError(f"Calibration requires {self.GAZE_FEATURE_COUNT} pupil features")

        # Use binocular pupil position plus a small inter-eye asymmetry term.
        # This is intentionally low-order: 25 calibration points should define
        # the screen geometry without allowing webcam noise to create wild
        # high-order oscillations between neighboring targets.
        left_x, left_y, right_x, right_y = [values[:, index] for index in range(4)]
        avg_x = (left_x + right_x) * 0.5
        avg_y = (left_y + right_y) * 0.5
        diff_x = left_x - right_x
        diff_y = left_y - right_y
        parts = [
            np.ones(len(values), dtype=np.float64),
            avg_x,
            avg_y,
            avg_x * avg_x,
            avg_y * avg_y,
            diff_x,
            diff_y,
        ]

        del fit_scaler
        return np.column_stack(parts)

    def _fit_regression(self) -> None:
        if len(self.samples) < 6:
            self._coefficients = None
            self._rmse = None
            return

        ordered = list(self.samples.values())
        vectors = np.asarray([sample.gaze_vector for sample in ordered], dtype=np.float64)
        points = np.asarray([sample.screen_point for sample in ordered], dtype=np.float64)
        if any(sample.gaze_features is None for sample in ordered):
            self._coefficients = None
            self._rmse = None
            return
        features = np.asarray([sample.gaze_features for sample in ordered], dtype=np.float64)

        self._feature_mean = None
        self._feature_scale = None
        design = self._design_matrix(vectors, features)

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
        if gaze_features is None:
            return np.asarray(
                [gaze_vector[0] * self.screen_width, gaze_vector[1] * self.screen_height],
                dtype=np.float64,
            )
        features = np.asarray([gaze_features], dtype=np.float64)
        design = self._design_matrix(vector, features)
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
        if self._coefficients is not None and self.is_complete() and gaze_features is not None:
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
