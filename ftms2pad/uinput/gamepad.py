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
    _pressed_buttons: set[int] = field(init=False, default_factory=set)

    def __post_init__(self) -> None:
        self.enabled = UInput is not None and e is not None
        self.ui = None
        self.error = ""
        self._pressed_buttons = set()
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

        button_names = [
            "BTN_A",
            "BTN_B",
            "BTN_X",
            "BTN_Y",
            "BTN_TL",
            "BTN_TR",
            "BTN_SELECT",
            "BTN_START",
            "BTN_THUMBL",
            "BTN_THUMBR",
        ]
        key_caps = [getattr(e, name) for name in button_names if hasattr(e, name)]

        capabilities = {
            e.EV_ABS: abs_caps,
            e.EV_KEY: key_caps,
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

    def emit_button(self, button_name: str, pressed: bool) -> None:
        if not self.enabled or self.ui is None or e is None:
            return
        code = getattr(e, button_name, None)
        if code is None:
            return
        if pressed:
            if code in self._pressed_buttons:
                return
            self._pressed_buttons.add(code)
            self.ui.write(e.EV_KEY, code, 1)
            self.ui.syn()
            return
        if code not in self._pressed_buttons:
            return
        self._pressed_buttons.remove(code)
        self.ui.write(e.EV_KEY, code, 0)
        self.ui.syn()

    def tap_button(self, button_name: str) -> None:
        self.emit_button(button_name, True)
        self.emit_button(button_name, False)

    def close(self) -> None:
        if self.ui is not None:
            for code in list(self._pressed_buttons):
                try:
                    self.ui.write(e.EV_KEY, code, 0)
                except Exception:
                    pass
            try:
                self.ui.syn()
            except Exception:
                pass
            self.ui.close()
