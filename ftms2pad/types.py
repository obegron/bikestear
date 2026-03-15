from __future__ import annotations

from dataclasses import dataclass
from time import monotonic


@dataclass(slots=True)
class PoseSample:
    steer_raw: float
    confidence: float
    source: str
    ts: float


@dataclass(slots=True)
class FtmsSample:
    watts: float
    cadence_rpm: float
    speed_kph: float
    resistance_level: float
    connected: bool
    ts: float

    @classmethod
    def disconnected(cls) -> "FtmsSample":
        return cls(watts=0.0, cadence_rpm=0.0, speed_kph=0.0, resistance_level=0.0, connected=False, ts=monotonic())
