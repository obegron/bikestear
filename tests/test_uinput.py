import unittest
from unittest.mock import patch

from ftms2pad.uinput import gamepad


class FakeCodes:
    EV_ABS = 3
    EV_KEY = 1
    ABS_X = 0
    ABS_Y = 1
    BTN_SOUTH = 304


class FakeDevice:
    def __init__(self, *, events, name, version):
        self.events = events
        self.writes = []
        self.closed = False

    def write(self, event_type, code, value):
        self.writes.append((event_type, code, value))

    def syn(self):
        pass

    def close(self):
        self.closed = True


class VirtualGamepadTests(unittest.TestCase):
    def test_registers_and_edges_accept_button(self):
        with (
            patch.object(gamepad, "ecodes", FakeCodes),
            patch.object(gamepad, "AbsInfo", lambda **values: values),
            patch.object(gamepad, "UInput", FakeDevice),
        ):
            device = gamepad.VirtualGamepad("ABS_X", "ABS_Y", "BTN_SOUTH")
            self.assertEqual(device._device.events[FakeCodes.EV_KEY], [FakeCodes.BTN_SOUTH])

            device.emit(0.0, 0.0, True)
            device.emit(0.0, 0.0, True)
            device.emit(0.0, 0.0, False)

            button_writes = [value for event_type, code, value in device._device.writes if event_type == 1]
            self.assertEqual(button_writes, [1, 0])
            device.close()


if __name__ == "__main__":
    unittest.main()
