import unittest

from ftms2pad.tracking import (
    anchor_gate_pass,
    debug_camera_key,
    debug_centroid_px,
    percentile,
    phase_sign_accepts,
    pose_conf_threshold,
    trim_side_outliers,
)
from ftms2pad.vision.mux import parse_camera_arg


class TrackingHelperTests(unittest.TestCase):
    def test_parse_camera_arg(self):
        self.assertEqual(parse_camera_arg("0, 2"), ["0", "2"])
        self.assertEqual(parse_camera_arg(""), ["auto"])

    def test_percentile_uses_sorted_index(self):
        self.assertEqual(percentile([3.0, 1.0, 2.0], 0.0), 1.0)
        self.assertEqual(percentile([3.0, 1.0, 2.0], 0.5), 2.0)
        self.assertEqual(percentile([3.0, 1.0, 2.0], 1.0), 3.0)

    def test_pose_conf_thresholds(self):
        self.assertLess(pose_conf_threshold("camera-bike"), pose_conf_threshold("mediapipe"))
        self.assertEqual(pose_conf_threshold("camera-blob"), 0.1)

    def test_debug_centroid_mirrors_pixel_sources(self):
        debug = {"kind": "face", "centroid": (20, 10)}
        self.assertEqual(debug_centroid_px(debug, 100, 50, mirrored=False), (20, 10))
        self.assertEqual(debug_centroid_px(debug, 100, 50, mirrored=True), (80, 10))

    def test_anchor_gate_allows_non_face_missing_centroid(self):
        debug = {"kind": "bike_mask"}
        self.assertTrue(anchor_gate_pass("camera-bike", debug, (50, 50), 100, 100, mirrored=False))
        self.assertFalse(anchor_gate_pass("camera-face", debug, (50, 50), 100, 100, mirrored=False))

    def test_phase_sign_accepts_requires_side_separation(self):
        neutral = [0.0] * 20
        self.assertTrue(phase_sign_accepts("left", -0.05, neutral))
        self.assertFalse(phase_sign_accepts("left", 0.05, neutral))
        self.assertTrue(phase_sign_accepts("right", 0.05, neutral))
        self.assertFalse(phase_sign_accepts("right", -0.05, neutral))

    def test_trim_side_outliers_keeps_enough_samples(self):
        values = [0.2] * 20 + [1.0]
        trimmed, changed = trim_side_outliers(values)
        self.assertTrue(changed)
        self.assertEqual(trimmed, [0.2] * 20)

    def test_debug_camera_key_default(self):
        self.assertEqual(debug_camera_key({}), "default")
        self.assertEqual(debug_camera_key({"camera_idx": 2}), "2")


if __name__ == "__main__":
    unittest.main()
