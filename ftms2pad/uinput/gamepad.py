from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

try:
    from evdev import UInput, ecodes as e
except Exception:  # pragma: no cover
    UInput = None
    e = None


def _to_axis(v: float) -> int:
    v = max(-1.0, min(1.0, v))
    return int(v * 32767)


@dataclass(slots=True)
class VirtualGamepad:
    steer_axis: str = "ABS_X"
    throttle_axis: str = "ABS_Y"
    invert_throttle: bool = True
    enabled: bool = field(init=False, default=False)
    ui: Any = field(init=False, default=None)

    def __post_init__(self) -> None:
        self.enabled = UInput is not None and e is not None
        self.ui = None
        if not self.enabled:
            return

        absinfo = (-32768, 32767, 0, 0)
        capabilities = {
            e.EV_ABS: [
                (getattr(e, self.steer_axis), absinfo),
                (getattr(e, self.throttle_axis), absinfo),
            ],
            e.EV_KEY: [e.BTN_A, e.BTN_B, e.BTN_X, e.BTN_Y],
        }

        try:
            self.ui = UInput(events=capabilities, name="ftms2pad", version=0x3)
        except Exception:
            self.enabled = False

    def emit(self, steer: float, throttle: float) -> None:
        if self.invert_throttle:
            throttle = -throttle
        if not self.enabled or self.ui is None:
            return

        self.ui.write(e.EV_ABS, getattr(e, self.steer_axis), _to_axis(steer))
        self.ui.write(e.EV_ABS, getattr(e, self.throttle_axis), _to_axis(throttle))
        self.ui.syn()

    def close(self) -> None:
        if self.ui is not None:
            self.ui.close()
