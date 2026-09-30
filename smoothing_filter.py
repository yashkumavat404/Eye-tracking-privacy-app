from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


@dataclass(frozen=True)
class FilteredPoint:
    point: Tuple[int, int]
    confidence: float
    timestamp: float


class ExponentialSmoothingFilter:
    """Adaptive low-latency gaze filter."""

    def __init__(
        self,
        alpha_slow: float = 0.90,
        alpha_fast: float = 0.985,
        fast_threshold: float = 18.0,
        freeze_duration: float = 0.030,
        deadzone: float = 0.45,
        prediction_gain: float = 0.18,
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
        self._point = None if point is None else np.asarray(point, dtype=np.float32)
        if self._point is not None:
            self._previous_input = self._point.copy()

    def freeze(self, timestamp: float) -> None:
        self._freeze_until = max(self._freeze_until, timestamp + self.freeze_duration)

    def update(self, point: Tuple[int, int], confidence: float, timestamp: float) -> FilteredPoint:
        incoming = np.asarray(point, dtype=np.float32)

        if self._point is None:
            self._point = incoming
            self._previous_input = incoming.copy()
            self._previous_timestamp = timestamp
        elif timestamp < self._freeze_until:
            incoming = self._point.copy()
        else:
            dt = max(timestamp - (self._previous_timestamp or timestamp), 1e-3)

            velocity = np.zeros(2, dtype=np.float32)
            if self._previous_input is not None:
                velocity = (incoming - self._previous_input) / dt

            prediction_horizon = min(self.prediction_gain * dt, 0.020)
            predicted = incoming + (velocity * prediction_horizon)
            distance = float(np.linalg.norm(predicted - self._point))

            if distance < self.deadzone:
                predicted = self._point.copy()
                distance = 0.0

            alpha = self.alpha_fast if distance >= self.fast_threshold else self.alpha_slow
            quality = float(np.clip(confidence, 0.0, 1.0))
            alpha *= 0.90 + (0.10 * quality)

            self._point = ((1.0 - alpha) * self._point) + (alpha * predicted)
            self._previous_input = incoming
            self._previous_timestamp = timestamp

        output = (int(self._point[0]), int(self._point[1]))
        return FilteredPoint(point=output, confidence=confidence, timestamp=timestamp)
