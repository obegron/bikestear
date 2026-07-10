from __future__ import annotations

from collections.abc import Callable
from threading import Event, Lock, Thread
from typing import Protocol

from ftms2pad.profiles import VisionConfig

from .tracker import VisionPacket, VisionTracker


class VisionSource(Protocol):
    def read(self) -> VisionPacket: ...

    def close(self) -> None: ...


class LatestVisionPacket:
    """An overwrite-only slot shared by the vision and runtime threads."""

    def __init__(self) -> None:
        self._lock = Lock()
        self._packet: VisionPacket | None = None
        self._generation = 0

    def publish(self, packet: VisionPacket) -> None:
        with self._lock:
            self._packet = packet
            self._generation += 1

    def snapshot(self) -> tuple[VisionPacket | None, int]:
        with self._lock:
            return self._packet, self._generation


class VisionWorker:
    def __init__(
        self,
        config: VisionConfig,
        source_factory: Callable[[VisionConfig], VisionSource] = VisionTracker,
    ) -> None:
        self.config = config
        self._source_factory = source_factory
        self._latest = LatestVisionPacket()
        self._stop = Event()
        self._stopped = Event()
        self._thread: Thread | None = None
        self.error: Exception | None = None

    def start(self) -> None:
        if self._thread is not None:
            raise RuntimeError("Vision worker already started")
        self._thread = Thread(target=self._run, name="ftms2pad-vision")
        self._thread.start()

    def _run(self) -> None:
        source: VisionSource | None = None
        try:
            source = self._source_factory(self.config)
            while not self._stop.is_set():
                try:
                    packet = source.read()
                except StopIteration:
                    break
                self._latest.publish(packet)
        except Exception as exc:
            self.error = exc
        finally:
            if source is not None:
                try:
                    source.close()
                except Exception as exc:
                    if self.error is None:
                        self.error = exc
            self._stopped.set()

    def latest(self) -> tuple[VisionPacket | None, int]:
        return self._latest.snapshot()

    def wait_until_stopped(self, timeout: float | None = None) -> bool:
        return self._stopped.wait(timeout)

    def close(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join()
