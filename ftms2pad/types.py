from __future__ import annotations

from dataclasses import dataclass
from time import monotonic


@dataclass(frozen=True, slots=True)
class VisionResult:
    ts: float
    torso_x: float | None
    confidence: float
    actual_fps: float = 0.0
    inference_ms: float = 0.0
    gesture_candidate: bool = False
    gesture_left_raised: bool = False
    gesture_right_raised: bool = False
    gesture_confidence: float = 0.0
    gesture_ms: float = 0.0


@dataclass(frozen=True, slots=True)
class FtmsSample:
    watts: float
    cadence_rpm: float
    speed_kph: float
    resistance_level: float
    connected: bool
    ts: float
    raw_hex: str = ""
    control_point_hex: str = ""

    @classmethod
    def disconnected(cls) -> "FtmsSample":
        return cls(
            watts=0.0,
            cadence_rpm=0.0,
            speed_kph=0.0,
            resistance_level=0.0,
            connected=False,
            ts=monotonic(),
        )
