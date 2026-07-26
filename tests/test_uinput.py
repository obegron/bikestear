import unittest
from unittest.mock import patch

from ftms2pad.uinput import gamepad


class FakeCodes:
    EV_ABS = 3
    EV_KEY = 1
    ABS_X = 0
    ABS_Y = 1
    BTN_SOUTH = 304
    BTN_WEST = 307
    BTN_START = 315


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
            device = gamepad.VirtualGamepad("ABS_X", "ABS_Y", "BTN_SOUTH", "BTN_WEST", "BTN_START")
            backend = device._device
            self.assertEqual(
                backend.events[FakeCodes.EV_KEY],
                [FakeCodes.BTN_SOUTH, FakeCodes.BTN_WEST, FakeCodes.BTN_START],
            )

            device.emit(0.0, 0.0, accept=True)
            device.emit(0.0, 0.0, accept=True)
            device.emit(0.0, 0.0, decline=True)
            device.emit(0.0, 0.0, menu=True)
            device.emit(0.0, 0.0)

            button_writes = [(code, value) for event_type, code, value in backend.writes if event_type == 1]
            self.assertEqual(button_writes, [
                (FakeCodes.BTN_SOUTH, 1),
                (FakeCodes.BTN_SOUTH, 0),
                (FakeCodes.BTN_WEST, 1),
                (FakeCodes.BTN_WEST, 0),
                (FakeCodes.BTN_START, 1),
                (FakeCodes.BTN_START, 0),
            ])
            device.close()


if __name__ == "__main__":
    unittest.main()
