from __future__ import annotations

import asyncio
import re
from math import sin
from time import monotonic

from ftms2pad.types import FtmsSample

try:
    from bleak import BleakClient, BleakScanner
except Exception:  # pragma: no cover
    BleakClient = None
    BleakScanner = None

FTMS_SERVICE_UUID = "00001826-0000-1000-8000-00805f9b34fb"
INDOOR_BIKE_DATA_CHAR_UUID = "00002ad2-0000-1000-8000-00805f9b34fb"
_MAC_RE = re.compile(r"^([0-9A-Fa-f]{2}:){5}[0-9A-Fa-f]{2}$")


def _u16(data: bytes, off: int) -> tuple[int, int]:
    return int.from_bytes(data[off : off + 2], "little"), off + 2


def _s16(data: bytes, off: int) -> tuple[int, int]:
    value = int.from_bytes(data[off : off + 2], "little", signed=True)
    return value, off + 2


def parse_indoor_bike_data(payload: bytes) -> tuple[float, float, float]:
    if len(payload) < 2:
        return 0.0, 0.0, 0.0

    flags = int.from_bytes(payload[0:2], "little")
    off = 2

    # FTMS Indoor Bike Data generally includes instantaneous speed unless bit0 indicates "more data".
    speed_kph = 0.0
    if (flags & (1 << 0)) == 0 and off + 2 <= len(payload):
        speed_raw, off = _u16(payload, off)  # 0.01 km/h
        speed_kph = speed_raw / 100.0

    cadence_rpm = 0.0
    if (flags & (1 << 2)) and off + 2 <= len(payload):
        cadence_raw, off = _u16(payload, off)  # 0.5 rpm
        cadence_rpm = cadence_raw / 2.0

    if (flags & (1 << 3)) and off + 2 <= len(payload):
        _, off = _u16(payload, off)  # average cadence
    if (flags & (1 << 4)) and off + 3 <= len(payload):
        off += 3  # total distance
    if (flags & (1 << 5)) and off + 2 <= len(payload):
        _, off = _s16(payload, off)  # resistance level

    watts = 0.0
    if (flags & (1 << 6)) and off + 2 <= len(payload):
        power_raw, off = _s16(payload, off)
        watts = float(max(0, power_raw))

    return watts, cadence_rpm, speed_kph


async def list_ble_devices(timeout: float = 4.0) -> list[tuple[str, str, bool]]:
    if BleakScanner is None:
        return []
    devices = await BleakScanner.discover(timeout=timeout)
    rows: list[tuple[str, str, bool]] = []
    for d in devices:
        uuids = (d.metadata or {}).get("uuids") or []
        ftms = any(str(u).lower() == FTMS_SERVICE_UUID for u in uuids)
        rows.append((d.name or "(unnamed)", d.address, ftms))
    return sorted(rows, key=lambda r: (not r[2], r[0].lower(), r[1]))


class FtmsSource:
    def __init__(self, bike: str = "sim") -> None:
        self.bike = bike
        self._t = 0.0
        self._latest = FtmsSample.disconnected()
        self._client = None
        self._connecting = asyncio.Lock()
        self._last_attempt = 0.0
        self._backoff_s = 1.0
        self._connected_evt = asyncio.Event()

    async def next(self) -> FtmsSample:
        if self.bike == "sim":
            await asyncio.sleep(0.02)
            self._t += 0.02
            watts = max(0.0, 180.0 + 120.0 * sin(self._t * 0.6))
            cadence = max(0.0, 78.0 + 12.0 * sin(self._t * 0.45 + 0.2))
            speed = max(0.0, 28.0 + 8.0 * sin(self._t * 0.5 - 0.1))
            return FtmsSample(watts=watts, cadence_rpm=cadence, speed_kph=speed, connected=True, ts=monotonic())

        await self._ensure_connected()
        await asyncio.sleep(0.02)

        if monotonic() - self._latest.ts > 2.0:
            return FtmsSample.disconnected()
        return self._latest

    async def _ensure_connected(self) -> None:
        if BleakClient is None or BleakScanner is None:
            return
        if self._client is not None and getattr(self._client, "is_connected", False):
            return

        now = monotonic()
        if now - self._last_attempt < self._backoff_s:
            return

        async with self._connecting:
            if self._client is not None and getattr(self._client, "is_connected", False):
                return
            self._last_attempt = monotonic()
            await self._connect_once()

    async def _connect_once(self) -> None:
        device = await self._resolve_device()
        if device is None:
            self._backoff_s = min(20.0, max(1.0, self._backoff_s * 1.5))
            return
        client = BleakClient(device, disconnected_callback=self._on_disconnected)
        try:
            await client.connect()
            await client.start_notify(INDOOR_BIKE_DATA_CHAR_UUID, self._on_indoor_bike_data)
        except Exception:
            try:
                await client.disconnect()
            except Exception:
                pass
            self._client = None
            self._backoff_s = min(20.0, max(1.0, self._backoff_s * 1.8))
            return

        self._client = client
        self._backoff_s = 1.0
        self._connected_evt.set()

    async def _resolve_device(self):
        if _MAC_RE.match(self.bike):
            try:
                return await BleakScanner.find_device_by_address(self.bike, timeout=6.0)
            except Exception:
                return None
        try:
            devices = await BleakScanner.discover(timeout=6.0)
        except Exception:
            return None
        needle = self.bike.strip().lower()
        for d in devices:
            if (d.name or "").strip().lower() == needle:
                return d
        for d in devices:
            if needle and needle in (d.name or "").lower():
                return d
        return None

    def _on_indoor_bike_data(self, _sender, payload: bytearray) -> None:
        watts, cadence_rpm, speed_kph = parse_indoor_bike_data(bytes(payload))
        self._latest = FtmsSample(
            watts=watts,
            cadence_rpm=cadence_rpm,
            speed_kph=speed_kph,
            connected=True,
            ts=monotonic(),
        )

    def _on_disconnected(self, _client) -> None:
        self._client = None
        self._connected_evt.clear()
        self._latest = FtmsSample.disconnected()

    async def close(self) -> None:
        if self._client is None:
            return
        try:
            if getattr(self._client, "is_connected", False):
                await self._client.stop_notify(INDOOR_BIKE_DATA_CHAR_UUID)
                await self._client.disconnect()
        except Exception:
            pass
        finally:
            self._client = None
