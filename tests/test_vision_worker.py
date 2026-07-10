import unittest

from ftms2pad.profiles import VisionConfig
from ftms2pad.types import VisionResult
from ftms2pad.vision import VisionPacket, VisionWorker


class FakeSource:
    def __init__(self) -> None:
        self.values = [1.0, 2.0, 3.0]
        self.closed = False

    def read(self) -> VisionPacket:
        if not self.values:
            raise StopIteration
        value = self.values.pop(0)
        return VisionPacket(VisionResult(ts=value, torso_x=value, confidence=1.0))

    def close(self) -> None:
        self.closed = True


class VisionWorkerTests(unittest.TestCase):
    def test_only_latest_result_is_retained(self):
        source = FakeSource()
        worker = VisionWorker(VisionConfig(), source_factory=lambda _: source)
        worker.start()
        self.assertTrue(worker.wait_until_stopped(1.0))

        packet, generation = worker.latest()

        self.assertIsNotNone(packet)
        self.assertEqual(packet.result.torso_x, 3.0)
        self.assertEqual(generation, 3)
        self.assertTrue(source.closed)
        worker.close()


if __name__ == "__main__":
    unittest.main()
