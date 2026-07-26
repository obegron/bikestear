from time import monotonic
import unittest

from ftms2pad.calibration import XCalibration
from ftms2pad.mapping import GestureMapper, XAxisMapper, YAxisMapper
from ftms2pad.profiles import VisionConfig, XAxisConfig, YAxisConfig
from ftms2pad.types import FtmsSample, VisionResult


def ftms(**overrides) -> FtmsSample:
    values = {
        "watts": 50.0,
        "cadence_rpm": 50.0,
        "speed_kph": 50.0,
        "resistance_level": 0.0,
        "connected": True,
        "ts": monotonic(),
    }
    values.update(overrides)
    return FtmsSample(**values)


class XAxisTests(unittest.TestCase):
    def mapper(
        self,
        *,
        invert: bool = False,
        smoothing: float = 1.0,
        stale_ms: float = 250.0,
        calibration: XCalibration | None = None,
    ):
        return XAxisMapper(
            XAxisConfig(gain=1.0, deadzone=0.1, smoothing=smoothing, invert=invert, stale_after_ms=stale_ms),
            VisionConfig(min_confidence=0.5),
            calibration or XCalibration(neutral=0.0, left=-1.0, right=1.0),
        )

    def test_neutral_and_deadzone(self):
        mapper = self.mapper()
        self.assertEqual(mapper.update(VisionResult(0.0, 0.0, 1.0), 0.0).mapped, 0.0)
        self.assertEqual(mapper.update(VisionResult(0.0, 0.05, 1.0), 0.0).mapped, 0.0)

    def test_left_and_right_reach_full_range(self):
        mapper = self.mapper()
        self.assertEqual(mapper.update(VisionResult(0.0, -1.0, 1.0), 0.0).mapped, -1.0)
        self.assertEqual(mapper.update(VisionResult(0.0, 1.0, 1.0), 0.0).mapped, 1.0)

    def test_inversion(self):
        mapper = self.mapper(invert=True)
        self.assertEqual(mapper.update(VisionResult(0.0, 1.0, 1.0), 0.0).mapped, -1.0)

    def test_camera_coordinate_direction_does_not_swap_left_and_right(self):
        mapper = self.mapper(calibration=XCalibration(neutral=0.0, left=1.0, right=-1.0))
        self.assertEqual(mapper.update(VisionResult(0.0, 1.0, 1.0), 0.0).mapped, -1.0)
        self.assertEqual(mapper.update(VisionResult(0.0, -1.0, 1.0), 0.0).mapped, 1.0)

    def test_stale_vision_decays_toward_neutral(self):
        mapper = self.mapper(smoothing=0.5, stale_ms=100.0)
        sample = VisionResult(0.0, 1.0, 1.0)
        self.assertEqual(mapper.update(sample, 0.0).mapped, 0.5)
        self.assertEqual(mapper.update(sample, 0.05).mapped, 0.75)
        stale = mapper.update(sample, 0.2)
        self.assertTrue(stale.stale)
        self.assertEqual(stale.mapped, 0.375)


class YAxisTests(unittest.TestCase):
    def test_each_supported_source(self):
        for source in ("speed_kph", "watts", "cadence_rpm"):
            with self.subTest(source=source):
                mapper = YAxisMapper(YAxisConfig(source=source, min=0, max=100, deadzone=0, smoothing=1, invert=False))
                self.assertEqual(mapper.update(ftms()).mapped, 0.5)

    def test_min_max_deadzone_and_clamping(self):
        config = YAxisConfig(source="watts", min=10, max=110, deadzone=10, smoothing=1, invert=False)
        mapper = YAxisMapper(config)
        self.assertEqual(mapper.update(ftms(watts=20)).mapped, 0.0)
        self.assertEqual(mapper.update(ftms(watts=65)).mapped, 0.5)
        self.assertEqual(mapper.update(ftms(watts=200)).mapped, 1.0)
        self.assertEqual(mapper.update(ftms(watts=-20)).mapped, 0.0)

    def test_inversion(self):
        mapper = YAxisMapper(YAxisConfig(source="watts", min=0, max=100, deadzone=0, smoothing=1, invert=True))
        self.assertAlmostEqual(mapper.update(ftms(watts=20)).mapped, 0.8)


class GestureMapperTests(unittest.TestCase):
    def mapper(self) -> GestureMapper:
        return GestureMapper(VisionConfig(
            min_confidence=0.5,
            gesture="wrist_raise",
            gesture_hold_ms=400,
            gesture_stale_after_ms=250,
        ))

    def sample(self, ts: float, candidate: bool = True, confidence: float = 0.9) -> VisionResult:
        return VisionResult(
            ts=ts,
            torso_x=0.0,
            confidence=1.0,
            gesture_candidate=candidate,
            gesture_confidence=confidence,
        )

    def test_requires_configured_hold_before_pressing(self):
        mapper = self.mapper()
        self.assertFalse(mapper.update(self.sample(0.0), 0.0).active)
        waiting = mapper.update(self.sample(0.35), 0.35)
        self.assertFalse(waiting.active)
        self.assertAlmostEqual(waiting.held_ms, 350.0)
        self.assertTrue(mapper.update(self.sample(0.4), 0.4).active)

    def test_release_resets_hold_before_reacquiring(self):
        mapper = self.mapper()
        mapper.update(self.sample(0.0), 0.0)
        self.assertTrue(mapper.update(self.sample(0.4), 0.4).active)
        self.assertFalse(mapper.update(self.sample(0.41, candidate=False), 0.41).active)
        self.assertFalse(mapper.update(self.sample(0.42), 0.42).active)

    def test_stale_tracking_releases_button(self):
        mapper = self.mapper()
        mapper.update(self.sample(0.0), 0.0)
        self.assertTrue(mapper.update(self.sample(0.4), 0.4).active)
        stale = mapper.update(self.sample(0.4), 0.7)
        self.assertTrue(stale.stale)
        self.assertFalse(stale.active)


if __name__ == "__main__":
    unittest.main()
