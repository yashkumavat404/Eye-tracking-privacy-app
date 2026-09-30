import cv2
import numpy as np

from pupil_tracker import PupilDetector


LEFT_EYE = {"outer": 33, "inner": 133, "top": 159, "bottom": 145}
RIGHT_EYE = {"outer": 362, "inner": 263, "top": 386, "bottom": 374}
LEFT_IRIS = [468, 469, 470, 471, 472]
RIGHT_IRIS = [473, 474, 475, 476, 477]


def _points(center_x: float, center_y: float, radius: float = 10.0) -> np.ndarray:
    points = np.zeros((478, 2), dtype=np.float32)
    points[33] = [center_x - 25, center_y]
    points[133] = [center_x + 25, center_y]
    points[159] = [center_x, center_y - 10]
    points[145] = [center_x, center_y + 10]
    points[362] = [center_x + 55, center_y]
    points[263] = [center_x + 105, center_y]
    points[386] = [center_x + 80, center_y - 10]
    points[374] = [center_x + 80, center_y + 10]

    for indices, cx in ((LEFT_IRIS, center_x), (RIGHT_IRIS, center_x + 80)):
        points[indices[0]] = [cx, center_y]
        points[indices[1]] = [cx + radius, center_y]
        points[indices[2]] = [cx, center_y + radius]
        points[indices[3]] = [cx - radius, center_y]
        points[indices[4]] = [cx, center_y - radius]
    return points


def test_detector_follows_black_pupil_inside_eye():
    frame = np.full((80, 140, 3), 160, dtype=np.uint8)
    cv2.circle(frame, (40, 40), 10, (85, 85, 85), -1)
    cv2.circle(frame, (46, 40), 4, (15, 15, 15), -1)
    cv2.circle(frame, (120, 40), 10, (85, 85, 85), -1)
    cv2.circle(frame, (114, 40), 4, (15, 15, 15), -1)

    points = _points(40, 40)
    detector = PupilDetector()
    left, right = detector.detect_both(
        frame,
        points,
        LEFT_EYE,
        RIGHT_EYE,
        LEFT_IRIS,
        RIGHT_IRIS,
    )

    assert left is not None
    assert right is not None
    assert left.confidence >= 0.42
    assert right.confidence >= 0.42
    assert left.normalized[0] > 0.5
    assert right.normalized[0] < 0.5


def test_detector_rejects_missing_dark_pupil():
    frame = np.full((80, 140, 3), 160, dtype=np.uint8)
    cv2.circle(frame, (40, 40), 10, (135, 135, 135), -1)
    cv2.circle(frame, (120, 40), 10, (135, 135, 135), -1)

    points = _points(40, 40)
    detector = PupilDetector()
    left, right = detector.detect_both(
        frame,
        points,
        LEFT_EYE,
        RIGHT_EYE,
        LEFT_IRIS,
        RIGHT_IRIS,
    )

    assert left is None or left.confidence < 0.60
    assert right is None or right.confidence < 0.60
