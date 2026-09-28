from dataclasses import dataclass
from typing import Optional, Tuple

from calibration import CALIBRATION_FILE, CalibrationMapper
from face_tracker import FaceObservation
from smoothing_filter import ExponentialSmoothingFilter


@dataclass
class GazeEstimate:
    screen_point: Tuple[int, int]
    raw_point: Tuple[int, int]
    gaze_vector: Optional[Tuple[float, float]]
    blink: bool
    confidence: float


class GazeEstimator:
    def __init__(self, screen_size: Tuple[int, int]) -> None:
        self.screen_width, self.screen_height = screen_size
        self.mapper = CalibrationMapper(screen_size)
        self.filter = ExponentialSmoothingFilter()
        self.last_point = (self.screen_width // 2, self.screen_height // 2)

    def reset(self) -> None:
        self.last_point = (self.screen_width // 2, self.screen_height // 2)
        self.filter.reset(self.last_point)

    def clear_calibration(self) -> None:
        self.mapper.clear()
        if CALIBRATION_FILE.exists():
            CALIBRATION_FILE.unlink()

    def has_calibration(self) -> bool:
        return self.mapper.is_complete()

    def load_calibration(self) -> bool:
        return self.mapper.load()

    def add_calibration_sample(
        self,
        label: str,
        screen_point: Tuple[int, int],
        gaze_vector: Tuple[float, float],
        gaze_features: Optional[Tuple[float, ...]] = None,
    ) -> None:
        self.mapper.add_sample(label, screen_point, gaze_vector, gaze_features)
        if self.mapper.is_complete():
            self.mapper.save()

    def estimate(self, observation: Optional[FaceObservation]) -> Optional[GazeEstimate]:
        if observation is None or not observation.face_detected or observation.gaze_vector is None:
            return None

        raw_point = self.mapper.map_observation(observation.gaze_vector, observation.gaze_features)
        if observation.blink:
            self.filter.freeze(observation.timestamp)
            raw_point = self.last_point

        filtered = self.filter.update(raw_point, observation.confidence, observation.timestamp)
        self.last_point = filtered.point
        return GazeEstimate(
            screen_point=filtered.point,
            raw_point=raw_point,
            gaze_vector=observation.gaze_vector,
            blink=observation.blink,
            confidence=observation.confidence,
        )
