import numpy as np

from calibration import CalibrationMapper


def test_calibration_grid_is_5x5():
    targets = CalibrationMapper.build_grid((1920, 1080))
    assert len(targets) == 25
    assert len({label for label, _, _ in targets}) == 25
    assert len({point[0] for _, point, _ in targets}) == 5
    assert len({point[1] for _, point, _ in targets}) == 5


def test_pupil_feature_model_maps_calibrated_points():
    mapper = CalibrationMapper((1920, 1080))
    targets = mapper.build_grid((1920, 1080))

    for index, (label, point, _) in enumerate(targets):
        x = point[0] / 1920.0
        y = point[1] / 1080.0
        vector = (0.5 + (x - 0.5) / 2.25, 0.5 + (y - 0.5) / 2.10)
        features = (x, y, x, y)
        mapper.add_sample(label, point, vector, features)

    assert mapper.is_complete()
    mapped = mapper.map_observation(
        (0.5, 0.5),
        (0.5, 0.5, 0.5, 0.5),
    )
    assert 0 <= mapped[0] < 1920
    assert 0 <= mapped[1] < 1080
