import unittest

from ftms2pad.calibration import build_calibration


class CalibrationTests(unittest.TestCase):
    def test_builds_from_robust_statistics(self):
        result = build_calibration([0.0] * 10, [-0.5] * 10 + [-9.0], [0.5] * 10 + [9.0])
        self.assertEqual(result.neutral, 0.0)
        self.assertEqual(result.left, -0.5)
        self.assertEqual(result.right, 0.5)

    def test_does_not_synthesize_missing_side(self):
        with self.assertRaisesRegex(ValueError, "redo the left phase"):
            build_calibration([0.0] * 10, [], [0.5] * 10)

    def test_rejects_side_too_close_to_neutral(self):
        with self.assertRaisesRegex(ValueError, "move your torso farther left"):
            build_calibration([0.0] * 10, [-0.01] * 10, [0.5] * 10)

    def test_accepts_mirrored_camera_direction(self):
        result = build_calibration([0.0] * 10, [0.5] * 10, [-0.5] * 10)
        self.assertEqual(result.normalize(0.5), -1.0)
        self.assertEqual(result.normalize(-0.5), 1.0)


if __name__ == "__main__":
    unittest.main()
