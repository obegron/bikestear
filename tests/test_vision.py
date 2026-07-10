from dataclasses import dataclass
import unittest

from ftms2pad.vision.tracker import estimate_torso


@dataclass
class FakeLandmark:
    x: float = 0.0
    y: float = 0.0
    visibility: float = 1.0


def landmarks() -> list[FakeLandmark]:
    return [FakeLandmark() for _ in range(25)]


class TorsoEstimateTests(unittest.TestCase):
    def test_torso_centre_uses_shoulders_and_hips(self):
        values = landmarks()
        values[11] = FakeLandmark(0.2, 0.3, 0.9)
        values[12] = FakeLandmark(0.6, 0.3, 0.8)
        values[23] = FakeLandmark(0.4, 0.7, 0.95)
        values[24] = FakeLandmark(0.6, 0.7, 0.85)

        result = estimate_torso(values, min_confidence=0.5)

        self.assertIsNotNone(result)
        self.assertAlmostEqual(result.torso_x, -0.175)
        self.assertAlmostEqual(result.confidence, 0.8)
        self.assertIsNotNone(result.hips)

    def test_shoulders_are_used_when_hips_are_unreliable(self):
        values = landmarks()
        values[11] = FakeLandmark(0.2, 0.3, 0.9)
        values[12] = FakeLandmark(0.6, 0.3, 0.8)
        values[23] = FakeLandmark(0.4, 0.7, 0.2)
        values[24] = FakeLandmark(0.6, 0.7, 0.9)

        result = estimate_torso(values, min_confidence=0.5)

        self.assertIsNotNone(result)
        self.assertAlmostEqual(result.torso_x, -0.25)
        self.assertAlmostEqual(result.confidence, 0.8)
        self.assertIsNone(result.hips)

    def test_rejects_insufficient_shoulder_confidence(self):
        values = landmarks()
        values[11] = FakeLandmark(0.2, 0.3, 0.4)
        values[12] = FakeLandmark(0.6, 0.3, 0.9)

        self.assertIsNone(estimate_torso(values, min_confidence=0.5))


if __name__ == "__main__":
    unittest.main()
