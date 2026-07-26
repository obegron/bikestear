from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from ftms2pad.profiles import load_profile


PROFILE = """
name: test
vision: {camera: 0, width: 320, height: 180, fps: 20, min_confidence: 0.5, gesture: wrist_raise, gesture_hold_ms: 400, gesture_raise_margin: 0.08, gesture_stale_after_ms: 250}
x_axis: {gain: 1.0, deadzone: 0.05, smoothing: 0.2, invert: false, stale_after_ms: 250}
y_axis: {source: %s, min: %s, max: %s, deadzone: %s, smoothing: 0.2, curve: linear, invert: false}
uinput: {x_axis: ABS_X, y_axis: ABS_Y, accept_button: BTN_SOUTH}
"""


class ProfileValidationTests(unittest.TestCase):
    def load(self, source: str, minimum: float, maximum: float, deadzone: float):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "test.yaml"
            path.write_text(PROFILE % (source, minimum, maximum, deadzone), encoding="utf-8")
            return load_profile(str(path))

    def test_rejects_unknown_y_source(self):
        with self.assertRaisesRegex(ValueError, "Unknown y_axis.source 'heart_rate'"):
            self.load("heart_rate", 0, 100, 0)

    def test_rejects_reversed_range(self):
        with self.assertRaisesRegex(ValueError, "max must be greater"):
            self.load("watts", 100, 10, 0)

    def test_rejects_deadzone_that_consumes_range(self):
        with self.assertRaisesRegex(ValueError, "leave a usable range"):
            self.load("watts", 0, 100, 100)

    def test_rejects_non_numeric_range(self):
        with self.assertRaisesRegex(ValueError, "y_axis.min must be a finite number"):
            self.load("watts", "slow", 100, 0)

    def test_loads_wrist_gesture_and_accept_button(self):
        profile = self.load("speed_kph", 0, 40, 1)
        self.assertEqual(profile.vision.gesture, "wrist_raise")
        self.assertEqual(profile.vision.gesture_hold_ms, 400)
        self.assertEqual(profile.uinput.accept_button, "BTN_SOUTH")

    def test_rejects_unknown_gesture(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "test.yaml"
            source = PROFILE % ("speed_kph", 0, 40, 1)
            path.write_text(source.replace("wrist_raise", "wave"), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Unknown vision.gesture 'wave'"):
                load_profile(str(path))


if __name__ == "__main__":
    unittest.main()
