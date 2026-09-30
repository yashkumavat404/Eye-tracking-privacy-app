import cv2
import numpy as np

from pupil_tracker import PupilDetector


LEFT_EYE = {"outer": 33, "inner": 133, "top": 159, "bottom": 145}
RIGHT_EYE = {"outer": 362, "inner": 263, "top": 386, "bottom": 374}
LEFT_IRIS = [468, 469, 470, 471, 472]
RIGHT_IRIS = [473, 474, 475, 476, 477]


def _points() -> np.ndarray:
    points = np.zeros((478, 2), dtype=np.float32)
    points[33] = [15, 40]
    points[133] = [65, 40]
    points[159] = [40, 30]
    points[145] = [40, 50]
    points[362] = [95, 40]
    points[263] = [135, 40]
    points[386] = [115, 30]
    points[374] = [115, 50]
    for indices, cx in ((LEFT_IRIS, 40), (RIGHT_IRIS, 115)):
        points[indices[0]] = [cx, 40]
        points[indices[1]] = [cx + 10, 40]
        points[indices[2]] = [cx, 50]
        points[indices[3]] = [cx - 10, 40]
        points[indices[4]] = [cx, 30]
    return points


def test_pupil_detector_tracks_black_pupil_not_iris_center():
    frame = np.full((80, 150, 3), 160, dtype=np.uint8)
    cv2.circle(frame, (40, 40), 10, (90, 90, 90), -1)
    cv2.circle(frame, (46, 40), 4, (12, 12, 12), -1)
    cv2.circle(frame, (115, 40), 10, (90, 90, 90), -1)
    cv2.circle(frame, (109, 40), 4, (12, 12, 12), -1)

    left, right = PupilDetector().detect_both(
        frame, _points(), LEFT_EYE, RIGHT_EYE, LEFT_IRIS, RIGHT_IRIS
    )

    assert left is not None
    assert right is not None
    assert left.normalized[0] > 0.52
    assert right.normalized[0] < 0.48


def test_pupil_detector_can_reject_flat_eye_region():
    frame = np.full((80, 150, 3), 150, dtype=np.uint8)
    left, right = PupilDetector().detect_both(
        frame, _points(), LEFT_EYE, RIGHT_EYE, LEFT_IRIS, RIGHT_IRIS
    )
    assert left is None or left.confidence < 0.55
    assert right is None or right.confidence < 0.55
