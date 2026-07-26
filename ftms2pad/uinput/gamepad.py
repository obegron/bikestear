from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

try:
    from evdev import AbsInfo, UInput, ecodes
except ImportError:  # pragma: no cover
    AbsInfo = None
    UInput = None
    ecodes = None


def _signed_axis(value: float) -> int:
    return int(max(-1.0, min(1.0, value)) * 32767)


def _unsigned_axis(value: float) -> int:
    return _signed_axis(max(0.0, min(1.0, value)) * 2.0 - 1.0)


@dataclass(slots=True)
class VirtualGamepad:
    x_axis: str = "ABS_X"
    y_axis: str = "ABS_Y"
    accept_button: str = "BTN_SOUTH"
    decline_button: str = "BTN_WEST"
    menu_button: str = "BTN_START"
    enabled: bool = field(init=False, default=False)
    error: str = field(init=False, default="")
    _device: Any = field(init=False, default=None)
    _button_codes: dict[str, int] = field(init=False, default_factory=dict)
    _pressed: dict[str, bool] = field(init=False, default_factory=dict)

    def __post_init__(self) -> None:
        if UInput is None or AbsInfo is None or ecodes is None:
            self.error = "evdev uinput support is unavailable"
            return
        try:
            x_code = getattr(ecodes, self.x_axis)
            y_code = getattr(ecodes, self.y_axis)
            self._button_codes = {
                "accept": getattr(ecodes, self.accept_button),
                "decline": getattr(ecodes, self.decline_button),
                "menu": getattr(ecodes, self.menu_button),
            }
        except AttributeError as exc:
            self.error = f"Unknown uinput axis: {exc}"
            return

        absinfo = AbsInfo(value=0, min=-32768, max=32767, fuzz=0, flat=0, resolution=0)
        pair = {
            "ABS_X": "ABS_Y",
            "ABS_Y": "ABS_X",
            "ABS_RX": "ABS_RY",
            "ABS_RY": "ABS_RX",
            "ABS_Z": "ABS_RZ",
            "ABS_RZ": "ABS_Z",
        }
        names = {self.x_axis, self.y_axis}
        for name in tuple(names):
            if name in pair:
                names.add(pair[name])
        capabilities = {
            ecodes.EV_ABS: [(getattr(ecodes, name), absinfo) for name in sorted(names)],
            ecodes.EV_KEY: list(self._button_codes.values()),
        }
        try:
            self._device = UInput(events=capabilities, name="ftms2pad", version=0x4)
            self.enabled = True
        except Exception as exc:
            self.error = f"{type(exc).__name__}: {exc}"

    def emit(
        self,
        x: float,
        y: float,
        accept: bool = False,
        decline: bool = False,
        menu: bool = False,
    ) -> None:
        if not self.enabled or self._device is None:
            return
        self._device.write(ecodes.EV_ABS, getattr(ecodes, self.x_axis), _signed_axis(x))
        self._device.write(ecodes.EV_ABS, getattr(ecodes, self.y_axis), _unsigned_axis(y))
        for action, pressed in (("accept", accept), ("decline", decline), ("menu", menu)):
            if pressed != self._pressed.get(action, False):
                self._device.write(ecodes.EV_KEY, self._button_codes[action], 1 if pressed else 0)
                self._pressed[action] = pressed
        self._device.syn()

    def close(self) -> None:
        if self._device is None:
            return
        released = False
        for action, pressed in tuple(self._pressed.items()):
            if pressed:
                self._device.write(ecodes.EV_KEY, self._button_codes[action], 0)
                self._pressed[action] = False
                released = True
        if released:
            self._device.syn()
        self._device.close()
        self._device = None
        self.enabled = False
