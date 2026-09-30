import json
import tempfile
import unittest
from pathlib import Path

from calibration import CalibrationMapper
from smoothing_filter import ExponentialSmoothingFilter


class CalibrationMapperTests(unittest.TestCase):
    def test_grid_is_a_unique_five_by_five_layout(self):
        grid = CalibrationMapper.build_grid((1920, 1080))
        self.assertEqual(len(grid), 25)
        self.assertEqual(len({label for label, _, _ in grid}), 25)
        self.assertEqual(len({point[0] for _, point, _ in grid}), 5)
        self.assertEqual(len({point[1] for _, point, _ in grid}), 5)

    def test_complete_model_round_trips_and_maps_known_points(self):
        mapper = CalibrationMapper((1000, 800))
        for label, point, _ in mapper.build_grid((1000, 800)):
            vector = (point[0] / 1000.0, point[1] / 800.0)
            mapper.add_sample(label, point, vector, (vector[0], vector[1]) * 2)

        self.assertTrue(mapper.is_complete())
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "calibration.json"
            mapper.save(path)
            restored = CalibrationMapper((1000, 800))
            self.assertTrue(restored.load(path))
            features = (0.5, 0.5) * 2
            self.assertLessEqual(abs(restored.map_observation((0.5, 0.5), features)[0] - 500), 5)
            self.assertLessEqual(abs(restored.map_observation((0.5, 0.5), features)[1] - 400), 5)

    def test_invalid_or_partial_saved_model_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "calibration.json"
            path.write_text(json.dumps({"model_version": 3, "screen_width": 1000, "screen_height": 800, "samples": []}))
            self.assertFalse(CalibrationMapper((1000, 800)).load(path))

    def test_previous_head_assisted_model_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "calibration.json"
            path.write_text(
                json.dumps(
                    {
                        "model_version": 2,
                        "screen_width": 1000,
                        "screen_height": 800,
                        "samples": [],
                    }
                )
            )
            self.assertFalse(CalibrationMapper((1000, 800)).load(path))

    def test_invalid_feature_length_is_rejected_before_mapping(self):
        mapper = CalibrationMapper((1000, 800))
        label, point, _ = mapper.build_grid((1000, 800))[0]
        with self.assertRaises(ValueError):
            mapper.add_sample(label, point, (0.2, 0.3), (0.2, 0.3))


class SmoothingFilterTests(unittest.TestCase):
    def test_freeze_holds_the_last_point(self):
        filter_ = ExponentialSmoothingFilter()
        first = filter_.update((100, 100), 1.0, 1.0)
        filter_.freeze(1.0)
        frozen = filter_.update((900, 700), 1.0, 1.01)
        self.assertEqual(first.point, frozen.point)


if __name__ == "__main__":
    unittest.main()
