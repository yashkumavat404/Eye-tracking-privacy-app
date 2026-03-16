from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


@dataclass
class FilteredPoint:
    point: Tuple[int, int]
    confidence: float
    timestamp: float


class ExponentialSmoothingFilter:
    def __init__(
        self,
        alpha_slow: float = 0.58,
        alpha_fast: float = 0.94,
        fast_threshold: float = 14.0,
        freeze_duration: float = 0.035,
        deadzone: float = 1.0,
        prediction_gain: float = 0.30,
    ) -> None:
        self.alpha_slow = alpha_slow
        self.alpha_fast = alpha_fast
        self.fast_threshold = fast_threshold
        self.freeze_duration = freeze_duration
        self.deadzone = deadzone
        self.prediction_gain = prediction_gain

        self._point: Optional[np.ndarray] = None
        self._previous_input: Optional[np.ndarray] = None
        self._freeze_until = 0.0
        self._previous_timestamp: Optional[float] = None

    def reset(self, point: Optional[Tuple[int, int]] = None) -> None:
        self._freeze_until = 0.0
        self._previous_timestamp = None
        self._previous_input = None
        if point is None:
            self._point = None
        else:
            point_array = np.array(point, dtype=np.float32)
            self._point = point_array
            self._previous_input = point_array.copy()

    def freeze(self, timestamp: float) -> None:
        self._freeze_until = max(self._freeze_until, timestamp + self.freeze_duration)

    def update(
        self,
        point: Tuple[int, int],
        confidence: float,
        timestamp: float,
    ) -> FilteredPoint:
        incoming = np.array(point, dtype=np.float32)
        if self._point is None:
            self._point = incoming
            self._previous_input = incoming
            self._previous_timestamp = timestamp
        elif timestamp < self._freeze_until:
            incoming = self._point.copy()
        else:
            dt = max(timestamp - (self._previous_timestamp or timestamp), 1e-3)
            velocity = np.zeros(2, dtype=np.float32)
            if self._previous_input is not None:
                velocity = (incoming - self._previous_input) / dt
            predicted = incoming + (velocity * min(self.prediction_gain * dt, 0.020))
            distance = float(np.linalg.norm(predicted - self._point))
            if distance < self.deadzone:
                predicted = self._point.copy()
                distance = 0.0

            alpha = self.alpha_fast if distance >= self.fast_threshold else self.alpha_slow
            alpha *= max(0.60, min(1.0, confidence))
            self._point = ((1.0 - alpha) * self._point) + (alpha * predicted)
            self._previous_input = incoming
            self._previous_timestamp = timestamp

        output = (int(self._point[0]), int(self._point[1]))
        return FilteredPoint(point=output, confidence=confidence, timestamp=timestamp)
