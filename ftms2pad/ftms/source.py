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
FITNESS_MACHINE_CONTROL_POINT_CHAR_UUID = "00002ad9-0000-1000-8000-00805f9b34fb"
_FTMS_REQUEST_CONTROL = b"\x00"
_FTMS_START_OR_RESUME = b"\x07"
_FTMS_SET_TARGET_RESISTANCE_LEVEL = 0x04
_MAC_RE = re.compile(r"^([0-9A-Fa-f]{2}:){5}[0-9A-Fa-f]{2}$")


def _u16(data: bytes, off: int) -> tuple[int, int]:
    return int.from_bytes(data[off : off + 2], "little"), off + 2


def _s16(data: bytes, off: int) -> tuple[int, int]:
    value = int.from_bytes(data[off : off + 2], "little", signed=True)
    return value, off + 2


def parse_indoor_bike_data(payload: bytes) -> tuple[float, float, float, float]:
    if len(payload) < 2:
        return 0.0, 0.0, 0.0, 0.0

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
        resistance_raw, off = _s16(payload, off)  # resistance level, 0.1 units in FTMS
        resistance_level = resistance_raw / 10.0
    else:
        resistance_level = 0.0

    watts = 0.0
    if (flags & (1 << 6)) and off + 2 <= len(payload):
        power_raw, off = _s16(payload, off)
        watts = float(max(0, power_raw))

    return watts, cadence_rpm, speed_kph, resistance_level


async def list_ble_devices(timeout: float = 4.0) -> list[tuple[str, str, bool]]:
    if BleakScanner is None:
        return []
    rows: list[tuple[str, str, bool]] = []
    try:
        discovered = await BleakScanner.discover(timeout=timeout, return_adv=True)
    except TypeError:
        discovered = await BleakScanner.discover(timeout=timeout)

    if isinstance(discovered, dict):
        for d, adv in discovered.values():
            uuids = getattr(adv, "service_uuids", None) or []
            ftms = any(str(u).lower() == FTMS_SERVICE_UUID for u in uuids)
            rows.append((d.name or getattr(adv, "local_name", None) or "(unnamed)", d.address, ftms))
        return sorted(rows, key=lambda r: (not r[2], r[0].lower(), r[1]))

    for d in discovered:
        metadata = getattr(d, "metadata", None) or {}
        uuids = metadata.get("uuids") or []
        ftms = any(str(u).lower() == FTMS_SERVICE_UUID for u in uuids)
        rows.append((d.name or "(unnamed)", d.address, ftms))
    return sorted(rows, key=lambda r: (not r[2], r[0].lower(), r[1]))


class FtmsSource:
    def __init__(self, bike: str = "sim", verbose: bool = False) -> None:
        self.bike = bike
        self.verbose = verbose
        self._t = 0.0
        self._latest = FtmsSample.disconnected()
        self._last_control_point_hex = ""
        self._client = None
        self._connecting = asyncio.Lock()
        self._last_attempt = 0.0
        self._backoff_s = 1.0
        self._connected_evt = asyncio.Event()

    def _log(self, message: str) -> None:
        if self.verbose:
            print(f"[ftms] {message}")

    async def next(self) -> FtmsSample:
        if self.bike == "sim":
            await asyncio.sleep(0.02)
            self._t += 0.02
            watts = max(0.0, 180.0 + 120.0 * sin(self._t * 0.6))
            cadence = max(0.0, 78.0 + 12.0 * sin(self._t * 0.45 + 0.2))
            speed = max(0.0, 28.0 + 8.0 * sin(self._t * 0.5 - 0.1))
            return FtmsSample(
                watts=watts,
                cadence_rpm=cadence,
                speed_kph=speed,
                resistance_level=0.0,
                connected=True,
                ts=monotonic(),
                raw_hex="",
                control_point_hex="",
            )

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
            if _MAC_RE.match(self.bike):
                # On BlueZ, a known BLE peripheral can sometimes be connected by
                # address even when a short scan did not return it.
                self._log(f"scan miss for {self.bike}; attempting direct connect")
                device = self.bike
            else:
                self._log(f"device not found: {self.bike}")
                self._backoff_s = min(20.0, max(1.0, self._backoff_s * 1.5))
                return
        self._log(f"connecting to {self.bike}")
        client = BleakClient(device, disconnected_callback=self._on_disconnected)
        try:
            await client.connect()
            self._log("connected")
            await self._initialize_ftms_session(client)
            await client.start_notify(INDOOR_BIKE_DATA_CHAR_UUID, self._on_indoor_bike_data)
            self._log("subscribed to indoor bike data")
        except Exception:
            self._log("connect/setup failed")
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

    async def _initialize_ftms_session(self, client) -> None:
        # Some trainers expose FTMS data only after a control-point handshake.
        try:
            await client.start_notify(FITNESS_MACHINE_CONTROL_POINT_CHAR_UUID, self._on_control_point)
            self._log("subscribed to control point")
        except Exception:
            self._log("control point notify unavailable")
            pass

        for command, label in (
            (_FTMS_REQUEST_CONTROL, "request_control"),
            (_FTMS_START_OR_RESUME, "start_or_resume"),
        ):
            try:
                await client.write_gatt_char(FITNESS_MACHINE_CONTROL_POINT_CHAR_UUID, command, response=True)
                self._log(f"sent {label}")
            except Exception:
                self._log(f"control point write failed: {label}")
                return

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
        payload_bytes = bytes(payload)
        watts, cadence_rpm, speed_kph, resistance_level = parse_indoor_bike_data(payload_bytes)
        self._latest = FtmsSample(
            watts=watts,
            cadence_rpm=cadence_rpm,
            speed_kph=speed_kph,
            resistance_level=resistance_level,
            connected=True,
            ts=monotonic(),
            raw_hex=payload_bytes.hex(),
            control_point_hex=self._last_control_point_hex,
        )

    async def set_target_resistance(self, level: float) -> bool:
        if self.bike == "sim":
            return False
        await self._ensure_connected()
        client = self._client
        if client is None or not getattr(client, "is_connected", False):
            return False
        raw = int(round(float(level) * 10.0))
        payload = bytes([_FTMS_SET_TARGET_RESISTANCE_LEVEL]) + int(raw).to_bytes(2, "little", signed=True)
        try:
            await client.write_gatt_char(FITNESS_MACHINE_CONTROL_POINT_CHAR_UUID, payload, response=True)
            self._log(f"sent set_target_resistance {level:.1f}")
            return True
        except Exception:
            self._log(f"set_target_resistance failed: {level:.1f}")
            return False

    def _on_disconnected(self, _client) -> None:
        self._client = None
        self._connected_evt.clear()
        self._latest = FtmsSample.disconnected()

    def _on_control_point(self, _sender, payload: bytearray) -> None:
        self._last_control_point_hex = bytes(payload).hex()
        self._log(f"control point indication: {self._last_control_point_hex}")

    async def close(self) -> None:
        if self._client is None:
            return
        try:
            if getattr(self._client, "is_connected", False):
                try:
                    await self._client.stop_notify(FITNESS_MACHINE_CONTROL_POINT_CHAR_UUID)
                except Exception:
                    pass
                await self._client.stop_notify(INDOOR_BIKE_DATA_CHAR_UUID)
                await self._client.disconnect()
                self._log("disconnected")
        except Exception:
            pass
        finally:
            self._client = None
