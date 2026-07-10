from __future__ import annotations

import asyncio
from dataclasses import dataclass, replace
from math import inf
from time import monotonic

from ftms2pad.calibration import XCalibration
from ftms2pad.ftms import FtmsSource
from ftms2pad.mapping import AxisValue, XAxisMapper, YAxisMapper
from ftms2pad.profiles import Profile, VisionConfig
from ftms2pad.types import FtmsSample
from ftms2pad.uinput import VirtualGamepad
from ftms2pad.vision import VisionPacket, VisionWorker


@dataclass(slots=True)
class RuntimeState:
    ftms: FtmsSample
    vision: VisionPacket | None = None
    x: AxisValue = AxisValue(raw=None, mapped=0.0, stale=True)
    y: AxisValue = AxisValue(raw=0.0, mapped=0.0)


def selected_vision_config(profile: Profile, camera: int | None) -> VisionConfig:
    return profile.vision if camera is None else replace(profile.vision, camera=camera)


async def _read_ftms(source: FtmsSource, state: RuntimeState) -> None:
    while True:
        try:
            state.ftms = await source.next()
        except asyncio.CancelledError:
            raise
        except Exception:
            state.ftms = FtmsSample.disconnected()
            await asyncio.sleep(0.25)


def _advance_mappers(
    state: RuntimeState,
    x_mapper: XAxisMapper,
    y_mapper: YAxisMapper,
    packet: VisionPacket | None,
    now: float,
) -> None:
    state.vision = packet
    state.x = x_mapper.update(packet.result if packet is not None else None, now)
    state.y = y_mapper.update(state.ftms)


def _status_line(state: RuntimeState, source: str, now: float) -> str:
    result = state.vision.result if state.vision is not None else None
    confidence = result.confidence if result is not None else 0.0
    age_ms = (now - result.ts) * 1000.0 if result is not None else inf
    raw_x = "  n/a" if state.x.raw is None else f"{state.x.raw:+.3f}"
    raw_y = float(state.y.raw or 0.0)
    return (
        f"vision conf={confidence:.2f} age={age_ms:5.0f}ms raw_x={raw_x} x={state.x.mapped:+.3f} "
        f"{source}={raw_y:6.1f} y={state.y.mapped:.3f}"
    )


async def run_controller(
    profile: Profile,
    calibration: XCalibration,
    *,
    bike: str,
    camera: int | None,
    hz: float = 60.0,
    dry_run: bool = False,
    verbose: bool = False,
    duration_seconds: float = 0.0,
) -> int:
    vision_config = selected_vision_config(profile, camera)
    worker = VisionWorker(vision_config)
    source = FtmsSource(bike, verbose=verbose)
    state = RuntimeState(ftms=FtmsSample.disconnected())
    x_mapper = XAxisMapper(profile.x_axis, vision_config, calibration)
    y_mapper = YAxisMapper(profile.y_axis)
    gamepad = None if dry_run else VirtualGamepad(profile.uinput.x_axis, profile.uinput.y_axis)
    if gamepad is not None and not gamepad.enabled:
        raise RuntimeError(f"Could not create virtual gamepad ({gamepad.error}). Check /dev/uinput permissions.")

    ftms_task: asyncio.Task[None] | None = None
    warned_vision = False
    started = monotonic()
    next_status = started
    period = 1.0 / hz
    deadline = asyncio.get_running_loop().time()
    try:
        worker.start()
        ftms_task = asyncio.create_task(_read_ftms(source, state), name="ftms-reader")
        while duration_seconds <= 0.0 or monotonic() - started < duration_seconds:
            now = monotonic()
            packet, _ = worker.latest()
            _advance_mappers(state, x_mapper, y_mapper, packet, now)
            if gamepad is not None:
                gamepad.emit(state.x.mapped, state.y.mapped)
            if worker.error is not None and not warned_vision:
                print(f"\nVision worker stopped: {worker.error}. X will return to neutral; Y remains active.")
                warned_vision = True
            if now >= next_status:
                print(_status_line(state, profile.y_axis.source, now), end="\r", flush=True)
                next_status = now + 0.5
            deadline += period
            delay = deadline - asyncio.get_running_loop().time()
            if delay < -period:
                deadline = asyncio.get_running_loop().time()
                delay = 0.0
            await asyncio.sleep(max(0.0, delay))
        print()
        return 0
    finally:
        if ftms_task is not None:
            ftms_task.cancel()
            await asyncio.gather(ftms_task, return_exceptions=True)
        worker.close()
        if gamepad is not None:
            gamepad.close()
        await source.close()


def draw_monitor_frame(frame, state: RuntimeState, source: str, now: float, mirror: bool):
    import cv2

    display = cv2.flip(frame, 1) if mirror else frame.copy()
    height, width = display.shape[:2]
    torso = state.vision.torso if state.vision is not None else None
    if torso is not None:
        def point(value: tuple[float, float]) -> tuple[int, int]:
            x = 1.0 - value[0] if mirror else value[0]
            return int(x * width), int(value[1] * height)

        shoulders = tuple(point(value) for value in torso.shoulders)
        cv2.line(display, shoulders[0], shoulders[1], (70, 220, 120), 2)
        for value in shoulders:
            cv2.circle(display, value, 4, (70, 220, 120), -1)
        if torso.hips is not None:
            hips = tuple(point(value) for value in torso.hips)
            cv2.line(display, hips[0], hips[1], (80, 190, 255), 2)
            for value in hips:
                cv2.circle(display, value, 4, (80, 190, 255), -1)

    result = state.vision.result if state.vision is not None else None
    confidence = result.confidence if result is not None else 0.0
    age_ms = (now - result.ts) * 1000.0 if result is not None else inf
    fps = result.actual_fps if result is not None else 0.0
    inference_ms = result.inference_ms if result is not None else 0.0
    raw_x = "n/a" if state.x.raw is None else f"{state.x.raw:+.3f}"
    lines = (
        f"torso confidence {confidence:.2f}   age {age_ms:.0f} ms",
        f"X raw {raw_x}   mapped {state.x.mapped:+.3f}",
        f"Y {source} raw {float(state.y.raw or 0.0):.1f}   mapped {state.y.mapped:.3f}",
        f"vision {fps:.1f} FPS   inference {inference_ms:.1f} ms",
    )
    for index, line in enumerate(lines):
        y = 28 + index * 26
        cv2.putText(display, line, (14, y), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(display, line, (14, y), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (245, 245, 245), 1, cv2.LINE_AA)
    return display


async def monitor_controller(
    profile: Profile,
    calibration: XCalibration,
    *,
    bike: str,
    camera: int | None,
    hz: float = 60.0,
    preview: bool = True,
    mirror: bool = True,
    verbose: bool = False,
) -> int:
    vision_config = selected_vision_config(profile, camera)
    worker = VisionWorker(vision_config)
    source = FtmsSource(bike, verbose=verbose)
    state = RuntimeState(ftms=FtmsSample.disconnected())
    x_mapper = XAxisMapper(profile.x_axis, vision_config, calibration)
    y_mapper = YAxisMapper(profile.y_axis)
    ftms_task: asyncio.Task[None] | None = None
    cv2 = None
    window = "ftms2pad monitor"
    if preview:
        import cv2 as cv2_module

        cv2 = cv2_module
        cv2.namedWindow(window, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window, 960, 540)

    period = 1.0 / hz
    deadline = asyncio.get_running_loop().time()
    next_status = monotonic()
    displayed_generation = -1
    warned_vision = False
    try:
        worker.start()
        ftms_task = asyncio.create_task(_read_ftms(source, state), name="ftms-reader")
        while True:
            now = monotonic()
            packet, generation = worker.latest()
            _advance_mappers(state, x_mapper, y_mapper, packet, now)
            if worker.error is not None and not warned_vision:
                print(f"\nVision worker stopped: {worker.error}. X will return to neutral; Y remains active.")
                warned_vision = True
            if now >= next_status:
                print(_status_line(state, profile.y_axis.source, now), end="\r", flush=True)
                next_status = now + 0.2
            if cv2 is not None:
                if packet is not None and packet.frame is not None and generation != displayed_generation:
                    cv2.imshow(window, draw_monitor_frame(packet.frame, state, profile.y_axis.source, now, mirror))
                    displayed_generation = generation
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    print()
                    return 0
            deadline += period
            delay = deadline - asyncio.get_running_loop().time()
            if delay < -period:
                deadline = asyncio.get_running_loop().time()
                delay = 0.0
            await asyncio.sleep(max(0.0, delay))
    finally:
        if ftms_task is not None:
            ftms_task.cancel()
            await asyncio.gather(ftms_task, return_exceptions=True)
        worker.close()
        await source.close()
        if cv2 is not None:
            cv2.destroyWindow(window)
