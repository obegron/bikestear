from __future__ import annotations

from ftms2pad.calibration import Calibrator
from ftms2pad.profiles.loader import Profile


def _clamp(value: float, lo: float = -1.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, value))


def _apply_deadzone(value: float, dz: float) -> float:
    if abs(value) <= dz:
        return 0.0
    return (abs(value) - dz) / (1.0 - dz) * (1 if value >= 0 else -1)


def _lpf(prev: float, current: float, alpha: float) -> float:
    alpha = _clamp(alpha, 0.01, 1.0)
    return prev * (1.0 - alpha) + current * alpha


class FusionPipeline:
    def __init__(self, profile: Profile, calibrator: Calibrator | None = None) -> None:
        self.profile = profile
        self.calibrator = calibrator or Calibrator()
        self._steer_prev = 0.0
        self._throttle_prev = 0.0

    def steer(self, steer_raw: float, pose_ok: bool = True) -> float:
        if not pose_ok:
            self._steer_prev = _lpf(self._steer_prev, 0.0, 0.2)
            return self._steer_prev

        steer = self.calibrator.normalize(steer_raw)
        near_center = abs(steer) <= max(self.profile.steering.deadzone * 1.6, 0.12)
        if self.profile.steering.invert:
            steer *= -1.0
        steer *= self.profile.steering.gain
        steer = _clamp(steer)
        steer = _apply_deadzone(steer, self.profile.steering.deadzone)
        alpha = self.profile.steering.smoothing
        if near_center:
            alpha = min(1.0, max(alpha, 0.28))
        self._steer_prev = _lpf(self._steer_prev, steer, alpha)
        return self._steer_prev

    def throttle(self, watts: float, connected: bool = True) -> float:
        if not connected:
            self._throttle_prev = _lpf(self._throttle_prev, 0.0, 0.3)
            return self._throttle_prev

        if watts <= self.profile.throttle.deadzone_watts:
            target = 0.0
        else:
            if self.profile.throttle.curve == "sigmoid":
                target = 2.0 / (1.0 + (2.71828 ** (-watts * self.profile.throttle.gain))) - 1.0
            else:
                target = watts * self.profile.throttle.gain
        target = _clamp(target, 0.0, 1.0)
        self._throttle_prev = _lpf(self._throttle_prev, target, self.profile.throttle.smoothing)
        return self._throttle_prev
