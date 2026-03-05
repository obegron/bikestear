import unittest

from ftms2pad.ftms.source import parse_indoor_bike_data


class FtmsParserTests(unittest.TestCase):
    def test_parse_speed_cadence_power(self):
        # flags: cadence present (bit2), power present (bit6), speed present (bit0=0)
        payload = bytes([
            0x44, 0x00,
            0xB8, 0x0B,  # speed 3000 -> 30.00 km/h
            0xB4, 0x00,  # cadence 180 -> 90.0 rpm
            0xFA, 0x00,  # power 250 W
        ])
        watts, cadence, speed = parse_indoor_bike_data(payload)
        self.assertEqual(watts, 250.0)
        self.assertEqual(cadence, 90.0)
        self.assertEqual(speed, 30.0)

    def test_parse_more_data_flag_skips_speed(self):
        # flags: more data (bit0), power present (bit6)
        payload = bytes([
            0x41, 0x00,
            0xC8, 0x00,  # power 200 W
        ])
        watts, cadence, speed = parse_indoor_bike_data(payload)
        self.assertEqual(watts, 200.0)
        self.assertEqual(cadence, 0.0)
        self.assertEqual(speed, 0.0)


if __name__ == "__main__":
    unittest.main()
