from __future__ import annotations

import asyncio
from time import monotonic

from ftms2pad.calibration import build_calibration, calibration_path, save_calibration
from ftms2pad.profiles import Profile
from ftms2pad.runtime import selected_vision_config
from ftms2pad.vision import VisionPacket, VisionWorker


def _draw_calibration_frame(frame, packet: VisionPacket, title: str, mirror: bool):
    import cv2

    display = cv2.flip(frame, 1) if mirror else frame.copy()
    height, width = display.shape[:2]
    torso = packet.torso
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
    text = f"{title}   confidence {packet.result.confidence:.2f}"
    cv2.putText(display, text, (16, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(display, text, (16, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (245, 245, 245), 1, cv2.LINE_AA)
    return display


async def calibrate_vision(
    profile: Profile,
    *,
    camera: int | None,
    prep_seconds: float = 2.0,
    phase_seconds: float = 3.0,
    preview: bool = True,
    mirror: bool = True,
) -> int:
    config = selected_vision_config(profile, camera)
    worker = VisionWorker(config)
    buckets: dict[str, list[float]] = {"neutral": [], "left": [], "right": []}
    phases = (
        ("neutral", "Sit in your neutral riding position"),
        ("left", "Move your torso left and hold"),
        ("right", "Move your torso right and hold"),
    )
    cv2 = None
    window = "ftms2pad calibrate"
    if preview:
        import cv2 as cv2_module

        cv2 = cv2_module
        cv2.namedWindow(window, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window, 960, 540)

    last_generation = -1
    try:
        worker.start()
        for phase, instruction in phases:
            print(f"\n{phase.upper()}: {instruction}. Starting in {prep_seconds:.1f}s.")
            prep_until = monotonic() + prep_seconds
            while monotonic() < prep_until:
                packet, generation = worker.latest()
                if worker.error is not None:
                    raise RuntimeError(f"Vision worker failed: {worker.error}")
                if cv2 is not None and packet is not None and packet.frame is not None and generation != last_generation:
                    cv2.imshow(window, _draw_calibration_frame(packet.frame, packet, f"GET READY: {phase.upper()}", mirror))
                    last_generation = generation
                if cv2 is not None and cv2.waitKey(1) & 0xFF == ord("q"):
                    print("Calibration cancelled.")
                    return 1
                await asyncio.sleep(1.0 / 60.0)

            print(f"Collecting {phase} samples...")
            collect_until = monotonic() + phase_seconds
            while monotonic() < collect_until:
                packet, generation = worker.latest()
                if worker.error is not None:
                    raise RuntimeError(f"Vision worker failed: {worker.error}")
                if packet is not None and generation != last_generation:
                    result = packet.result
                    if result.torso_x is not None and result.confidence >= config.min_confidence:
                        buckets[phase].append(result.torso_x)
                    if cv2 is not None and packet.frame is not None:
                        cv2.imshow(window, _draw_calibration_frame(packet.frame, packet, f"HOLD {phase.upper()}", mirror))
                    last_generation = generation
                if cv2 is not None and cv2.waitKey(1) & 0xFF == ord("q"):
                    print("Calibration cancelled.")
                    return 1
                await asyncio.sleep(1.0 / 60.0)
            print(f"Accepted {len(buckets[phase])} reliable {phase} samples.")

        calibration = build_calibration(buckets["neutral"], buckets["left"], buckets["right"])
        path = calibration_path(profile.path)
        save_calibration(path, calibration)
        print(
            f"Saved {path}: neutral={calibration.neutral:+.3f}, "
            f"left={calibration.left:+.3f}, right={calibration.right:+.3f}"
        )
        return 0
    finally:
        worker.close()
        if cv2 is not None:
            cv2.destroyWindow(window)
