from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

try:
    from evdev import AbsInfo, UInput, ecodes as e
except Exception:  # pragma: no cover
    AbsInfo = None
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
    error: str = field(init=False, default="")

    def __post_init__(self) -> None:
        self.enabled = UInput is not None and e is not None
        self.ui = None
        self.error = ""
        if not self.enabled:
            self.error = "evdev UInput not available"
            return

        # evdev expects AbsInfo(value, min, max, fuzz, flat, resolution).
        # Using a 4-tuple here can be misinterpreted and cause EINVAL from uinput.
        absinfo = AbsInfo(value=0, min=-32768, max=32767, fuzz=0, flat=0, resolution=0)
        pair = {
            "ABS_X": "ABS_Y",
            "ABS_Y": "ABS_X",
            "ABS_RX": "ABS_RY",
            "ABS_RY": "ABS_RX",
            "ABS_Z": "ABS_RZ",
            "ABS_RZ": "ABS_Z",
        }
        axis_names = {self.steer_axis, self.throttle_axis}
        # Expose stick pairs so SDL/games map axes consistently.
        for name in list(axis_names):
            other = pair.get(name)
            if other is not None:
                axis_names.add(other)
        abs_caps = [(getattr(e, name), absinfo) for name in sorted(axis_names)]

        capabilities = {
            e.EV_ABS: abs_caps,
            e.EV_KEY: [e.BTN_A, e.BTN_B, e.BTN_X, e.BTN_Y],
        }

        try:
            self.ui = UInput(events=capabilities, name="ftms2pad", version=0x3)
        except Exception as exc:
            self.enabled = False
            self.error = f"{type(exc).__name__}: {exc}"

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
