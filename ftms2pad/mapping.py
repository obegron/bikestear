from __future__ import annotations

from dataclasses import dataclass

from ftms2pad.calibration import XCalibration
from ftms2pad.profiles import VisionConfig, XAxisConfig, YAxisConfig
from ftms2pad.types import FtmsSample, VisionResult


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def _smooth(previous: float, target: float, alpha: float) -> float:
    return previous + alpha * (target - previous)


def _signed_deadzone(value: float, deadzone: float) -> float:
    if abs(value) <= deadzone:
        return 0.0
    return (abs(value) - deadzone) / (1.0 - deadzone) * (1.0 if value > 0.0 else -1.0)


@dataclass(frozen=True, slots=True)
class AxisValue:
    raw: float | None
    mapped: float
    stale: bool = False


class XAxisMapper:
    def __init__(self, config: XAxisConfig, vision: VisionConfig, calibration: XCalibration) -> None:
        self.config = config
        self.vision = vision
        self.calibration = calibration
        self._previous = 0.0

    def update(self, sample: VisionResult | None, now: float) -> AxisValue:
        raw = sample.torso_x if sample is not None else None
        stale = (
            sample is None
            or sample.torso_x is None
            or sample.confidence < self.vision.min_confidence
            or (now - sample.ts) * 1000.0 > self.config.stale_after_ms
        )
        target = 0.0
        if not stale and sample is not None and sample.torso_x is not None:
            target = self.calibration.normalize(sample.torso_x) * self.config.gain
            target = _clamp(target, -1.0, 1.0)
            target = _signed_deadzone(target, self.config.deadzone)
            if self.config.invert:
                target = -target
        self._previous = _smooth(self._previous, target, self.config.smoothing)
        if abs(self._previous) < 1e-6:
            self._previous = 0.0
        return AxisValue(raw=raw, mapped=self._previous, stale=stale)


class YAxisMapper:
    def __init__(self, config: YAxisConfig) -> None:
        self.config = config
        self._previous = 1.0 if config.invert else 0.0

    def update(self, sample: FtmsSample) -> AxisValue:
        raw = float(getattr(sample, self.config.source)) if sample.connected else 0.0
        floor = self.config.min + self.config.deadzone
        if raw <= floor:
            target = 0.0
        else:
            target = (raw - floor) / (self.config.max - floor)
        target = _clamp(target, 0.0, 1.0)
        if self.config.invert:
            target = 1.0 - target
        self._previous = _smooth(self._previous, target, self.config.smoothing)
        return AxisValue(raw=raw, mapped=_clamp(self._previous, 0.0, 1.0))
