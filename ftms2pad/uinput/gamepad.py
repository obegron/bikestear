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
    enabled: bool = field(init=False, default=False)
    error: str = field(init=False, default="")
    _device: Any = field(init=False, default=None)
    _accept_code: int | None = field(init=False, default=None)
    _accept_pressed: bool = field(init=False, default=False)

    def __post_init__(self) -> None:
        if UInput is None or AbsInfo is None or ecodes is None:
            self.error = "evdev uinput support is unavailable"
            return
        try:
            x_code = getattr(ecodes, self.x_axis)
            y_code = getattr(ecodes, self.y_axis)
            self._accept_code = getattr(ecodes, self.accept_button)
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
            ecodes.EV_KEY: [self._accept_code],
        }
        try:
            self._device = UInput(events=capabilities, name="ftms2pad", version=0x4)
            self.enabled = True
        except Exception as exc:
            self.error = f"{type(exc).__name__}: {exc}"

    def emit(self, x: float, y: float, accept: bool = False) -> None:
        if not self.enabled or self._device is None:
            return
        self._device.write(ecodes.EV_ABS, getattr(ecodes, self.x_axis), _signed_axis(x))
        self._device.write(ecodes.EV_ABS, getattr(ecodes, self.y_axis), _unsigned_axis(y))
        if self._accept_code is not None and accept != self._accept_pressed:
            self._device.write(ecodes.EV_KEY, self._accept_code, 1 if accept else 0)
            self._accept_pressed = accept
        self._device.syn()

    def close(self) -> None:
        if self._device is None:
            return
        if self._accept_code is not None and self._accept_pressed:
            self._device.write(ecodes.EV_KEY, self._accept_code, 0)
            self._device.syn()
            self._accept_pressed = False
        self._device.close()
        self._device = None
        self.enabled = False
