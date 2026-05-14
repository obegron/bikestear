from __future__ import annotations

import argparse
import asyncio
import inspect
from pathlib import Path
from statistics import median
from time import monotonic

from ftms2pad.calibration import load_calibration, save_calibration
from ftms2pad.debuglog import DebugLogger
from ftms2pad.ftms import FtmsSource, list_ble_devices
from ftms2pad.fusion import Calibrator, FusionPipeline
from ftms2pad.overlay import (
    draw_calibration_frame as _draw_calibration_frame,
    draw_monitor_frame as _draw_monitor_frame,
    draw_tracking_overlay as _draw_tracking_overlay,
    safe_destroy_window as _safe_destroy_window,
)
from ftms2pad.profiles import load_profile
from ftms2pad.replay import cmd_replay  # Re-exported for ftms2pad.devtool.
from ftms2pad.tracking import (
    accept_neutral_sample as _accept_neutral_sample,
    accept_side_sample as _accept_side_sample,
    anchor_gate_pass as _anchor_gate_pass,
    debug_camera_key as _debug_camera_key,
    debug_centroid_px as _debug_centroid_px,
    percentile as _percentile,
    phase_sign_accepts as _phase_sign_accepts,
    phase_target_count as _phase_target_count,
    pose_conf_threshold as _pose_conf_threshold,
    stable_for_anchor as _stable_for_anchor,
    trim_side_outliers as _trim_side_outliers,
)
from ftms2pad.uinput import VirtualGamepad
from ftms2pad.vision import camera_name, list_cameras
from ftms2pad.vision.mux import VisionMux


def _calibration_path(profile: str) -> Path:
    return Path("profiles") / f"{profile}.calibration.json"


def _pedaling_started(f, cadence_threshold: float, watts_threshold: float) -> bool:
    return bool(getattr(f, "connected", False)) and (
        float(getattr(f, "cadence_rpm", 0.0)) >= cadence_threshold
        or float(getattr(f, "watts", 0.0)) >= watts_threshold
    )


async def _wait_for_pedaling(
    *,
    args: argparse.Namespace,
    vision,
    dbg: DebugLogger,
    mirror_preview: bool,
) -> int:
    ftms = FtmsSource(args.bike, verbose=getattr(args, "verbose", False))
    cv2 = None
    use_gui = not args.no_gui
    if use_gui:
        try:
            import cv2 as _cv2

            cv2 = _cv2
        except Exception:
            use_gui = False
    try:
        if use_gui and cv2 is not None:
            win = "ftms2pad calibrate"
            cv2.namedWindow(win, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(win, 960, 540)
            while True:
                f = await ftms.next()
                p, frame, debug = vision.next_with_frame()
                if frame is None:
                    await asyncio.sleep(1 / 30)
                    continue
                display = frame.copy()
                if mirror_preview:
                    display[:] = cv2.flip(display, 1)
                h, w = display.shape[:2]
                cv2.putText(display, "Start pedaling to begin calibration", (20, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.82, (60, 220, 255), 2, cv2.LINE_AA)
                cv2.putText(
                    display,
                    f"cadence={f.cadence_rpm:0.1f} rpm watts={f.watts:0.0f}",
                    (20, 68),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (235, 235, 235),
                    2,
                    cv2.LINE_AA,
                )
                cv2.putText(display, "Pedal gently and sway once to lock center", (20, 98), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1, cv2.LINE_AA)
                cv2.putText(display, "Press q to cancel", (20, 124), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (220, 220, 220), 1, cv2.LINE_AA)
                _draw_tracking_overlay(cv2, display, debug, mirrored=mirror_preview)
                dbg.log(
                    p=p,
                    f=f,
                    debug=debug,
                    frame=display,
                    extra={"state": "wait_pedaling"},
                )
                cv2.imshow(win, display)
                keycode = cv2.waitKey(1) & 0xFF
                if keycode == ord("q"):
                    print("Calibration canceled.")
                    return 1
                if _pedaling_started(f, args.start_cadence_rpm, args.start_watts):
                    return 0
                await asyncio.sleep(1 / 30)
        else:
            print("Waiting for pedaling to start calibration...")
            while True:
                f = await ftms.next()
                p = vision.next()
                dbg.log(p=p, f=f, extra={"state": "wait_pedaling"})
                if _pedaling_started(f, args.start_cadence_rpm, args.start_watts):
                    return 0
                await asyncio.sleep(1 / 30)
    except (KeyboardInterrupt, asyncio.CancelledError):
        print("Calibration canceled.")
        return 1
    finally:
        await ftms.close()


async def cmd_list_bikes(_: argparse.Namespace) -> int:
    devices = await list_ble_devices()
    if not devices:
        print("No BLE devices found (or bleak not installed).")
        return 0
    for name, addr, ftms in devices:
        tag = "FTMS" if ftms else "----"
        print(f"{tag} {name:30s} {addr}")
    return 0


def cmd_list_cameras(_: argparse.Namespace) -> int:
    cameras = list_cameras()
    if not cameras:
        print("No webcams detected.")
        return 1
    for idx in cameras:
        print(f"{idx}\t{camera_name(idx)}")
    return 0


async def cmd_calibrate(args: argparse.Namespace) -> int:
    profile = load_profile(args.profile)
    vision = VisionMux(
        profile.steering.mode,
        camera_arg=args.camera,
        width=args.vision_width,
        height=args.vision_height,
        idle_hz=args.mux_idle_hz,
    )
    dbg = DebugLogger(args.debug_log, "calibrate", args.debug_fps, args.debug_width, args.debug_height)
    ftms = FtmsSource(args.bike, verbose=getattr(args, "verbose", False))
    if dbg.enabled and dbg.session_dir is not None:
        print(f"debug log: {dbg.session_dir}")
    try:
        if not getattr(args, "no_wait_pedal", False):
            rc = await _wait_for_pedaling(
                args=args,
                vision=vision,
                dbg=dbg,
                mirror_preview=not args.no_mirror,
            )
            if rc != 0:
                return rc
            vision.reset_tracking()
        use_gui = not args.no_gui
        cv2 = None
        if use_gui:
            try:
                import cv2 as _cv2

                cv2 = _cv2
            except Exception:
                use_gui = False

        phases = [
            ("NEUTRAL", "neutral", "Stay centered"),
            ("LEFT", "left", "Lean left with torso"),
            ("NEUTRAL", "neutral", "Return to center"),
            ("RIGHT", "right", "Lean right with torso"),
        ]
        buckets: dict[str, list[float]] = {"neutral": [], "left": [], "right": []}
        prep_seconds = max(0.5, float(args.prep_seconds))
        phase_seconds = max(1.0, float(args.phase_seconds))
        anchors: dict[str, tuple[int, int]] = {}
        anchor_samples: dict[str, list[tuple[int, int]]] = {}
        anchor_locked: set[str] = set()
        anchor_frames: dict[str, tuple[int, int]] = {}

        def _anchor_from_samples(cam_key: str) -> tuple[int, int] | None:
            samples = anchor_samples.get(cam_key, [])
            if not samples:
                return anchors.get(cam_key)
            xs = [p[0] for p in samples]
            ys = [p[1] for p in samples]
            return int(median(xs)), int(median(ys))

        gui_completed = False
        if use_gui and cv2 is not None:
            try:
                win = "ftms2pad calibrate"
                cv2.namedWindow(win, cv2.WINDOW_NORMAL)
                cv2.resizeWindow(win, 960, 540)
                if float(getattr(args, "resistance_calibration_level", 0.0) or 0.0) > 0.0:
                    pedal_idle_started: float | None = None
                    target_level = float(getattr(args, "resistance_calibration_level", 5.0))
                    target_applied = False
                    phase_order = ["neutral", "left", "neutral", "right", "neutral"]
                    phase_titles = {
                        "neutral": "CENTER",
                        "left": "LEFT",
                        "right": "RIGHT",
                    }
                    phase_idx = 0
                    while True:
                        f = await ftms.next()
                        if not target_applied and bool(getattr(f, "connected", False)):
                            target_applied = await ftms.set_target_resistance(target_level)
                            if target_applied:
                                await asyncio.sleep(0.25)
                                f = await ftms.next()
                        p, frame, debug = vision.next_with_frame()
                        if frame is None:
                            await asyncio.sleep(1 / 30)
                            continue
                        h, w = frame.shape[:2]
                        cam_key = _debug_camera_key(debug)
                        anchor_frames[cam_key] = (w, h)
                        cent = _debug_centroid_px(debug, w, h, mirrored=not args.no_mirror)
                        anchor = _anchor_from_samples(cam_key)
                        if _stable_for_anchor(p, debug, cent):
                            samples = anchor_samples.setdefault(cam_key, [])
                            samples.append(cent)
                            if len(samples) > 24:
                                del samples[:-24]
                            anchor = _anchor_from_samples(cam_key)
                            if anchor is not None:
                                anchors[cam_key] = anchor
                            if len(samples) >= 8:
                                anchor_locked.add(cam_key)
                        center_locked = anchor is not None and cam_key in anchor_locked
                        phase_key = phase_order[min(phase_idx, len(phase_order) - 1)]
                        accepted = False
                        if phase_idx == 0 and not center_locked:
                            accepted = False
                        elif phase_key == "neutral":
                            if _accept_neutral_sample(p, debug):
                                buckets["neutral"].append(p.steer_raw)
                                accepted = True
                        else:
                            if _accept_side_sample(p, debug, anchor, frame, mirrored=not args.no_mirror) and _phase_sign_accepts(
                                phase_key, p.steer_raw, buckets["neutral"]
                            ):
                                buckets[phase_key].append(p.steer_raw)
                                accepted = True
                        pedaling_now = _pedaling_started(
                            f,
                            float(getattr(args, "start_cadence_rpm", 20.0)),
                            float(getattr(args, "start_watts", 35.0)),
                        )
                        ready_to_finish = bool(
                            len(buckets["neutral"]) >= 40 and len(buckets["left"]) >= 30 and len(buckets["right"]) >= 30
                        )
                        phase_done = len(buckets[phase_key]) >= _phase_target_count(phase_key)
                        if phase_idx == 0 and not center_locked:
                            phase_done = False
                        if not pedaling_now:
                            if pedal_idle_started is None:
                                pedal_idle_started = monotonic()
                            idle_elapsed = monotonic() - pedal_idle_started
                            if phase_idx >= len(phase_order) - 1:
                                if ready_to_finish and idle_elapsed >= 1.2:
                                    _safe_destroy_window(cv2, win)
                                    break
                            elif phase_done and idle_elapsed >= 0.8:
                                phase_idx += 1
                                pedal_idle_started = None
                        else:
                            pedal_idle_started = None
                        title = f"PEDAL CAL: {phase_titles[phase_key]}"
                        if phase_key == "neutral":
                            hint = "Pedal straight and stay centered. Stop pedaling briefly to advance."
                        elif phase_key == "left":
                            hint = "Lean left while pedaling. Stop pedaling briefly when done."
                        else:
                            hint = "Lean right while pedaling. Stop pedaling briefly when done."
                        display = frame.copy()
                        _draw_calibration_frame(
                            cv2,
                            display,
                            p,
                            debug,
                            title,
                            hint,
                            0.0,
                            mirror_preview=not args.no_mirror,
                            collecting=True,
                            anchor=anchor,
                        )
                        cv2.putText(
                            display,
                            (
                                f"phase {phase_idx + 1}/{len(phase_order)}  target-res={target_level:0.1f}  "
                                f"samples center={len(buckets['neutral'])} left={len(buckets['left'])} right={len(buckets['right'])}"
                            ),
                            (20, 122),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.50,
                            (235, 235, 235),
                            1,
                            cv2.LINE_AA,
                        )
                        if phase_idx >= len(phase_order) - 1 and ready_to_finish:
                            status = "Stop pedaling to save"
                            if pedal_idle_started is not None:
                                remaining = max(0.0, 1.2 - (monotonic() - pedal_idle_started))
                                status = f"Saving in {remaining:0.1f}s"
                        else:
                            need = max(0, _phase_target_count(phase_key) - len(buckets[phase_key]))
                            if phase_idx == 0 and not center_locked:
                                status = "Locking center before calibration starts"
                            elif phase_done:
                                status = "Stop pedaling briefly to advance"
                                if pedal_idle_started is not None:
                                    remaining = max(0.0, 0.8 - (monotonic() - pedal_idle_started))
                                    status = f"Advancing in {remaining:0.1f}s"
                            else:
                                status = f"Collecting {phase_titles[phase_key]}: need {need} more samples"
                        cv2.putText(
                            display,
                            status,
                            (20, 148),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.54,
                            (255, 210, 80),
                            1,
                            cv2.LINE_AA,
                        )
                        dbg.log(
                            p=p,
                            f=f,
                            debug=debug,
                            anchor=anchor,
                            frame=display,
                            extra={
                                "state": "resistance_calibration",
                                "phase_key": phase_key,
                                "phase_mode": "pedal",
                                "phase_idx": phase_idx,
                                "accepted": accepted,
                                "ready_to_finish": ready_to_finish,
                                "target_level": target_level,
                            },
                        )
                        cv2.imshow(win, display)
                        keycode = cv2.waitKey(1) & 0xFF
                        if keycode == ord("q"):
                            print("Calibration canceled.")
                            _safe_destroy_window(cv2, win)
                            return 1
                        if keycode == ord("r"):
                            anchors.clear()
                            anchor_samples.clear()
                            anchor_locked.clear()
                            buckets = {"neutral": [], "left": [], "right": []}
                            pedal_idle_started = None
                            phase_idx = 0
                            target_applied = False
                            vision.reset_tracking()
                        await asyncio.sleep(1 / 30)
                elif getattr(args, "manual_calibration", False):
                    active_key: str | None = None
                    pedal_idle_started: float | None = None
                    keycheck_stage = "minus"
                    while True:
                        p, frame, debug = vision.next_with_frame()
                        if frame is None:
                            await asyncio.sleep(1 / 30)
                            continue
                        h, w = frame.shape[:2]
                        cam_key = _debug_camera_key(debug)
                        anchor_frames[cam_key] = (w, h)
                        cent = _debug_centroid_px(debug, w, h, mirrored=not args.no_mirror)
                        anchor = _anchor_from_samples(cam_key)
                        if active_key is None and _stable_for_anchor(p, debug, cent):
                            samples = anchor_samples.setdefault(cam_key, [])
                            samples.append(cent)
                            if len(samples) > 24:
                                del samples[:-24]
                            anchor = _anchor_from_samples(cam_key)
                            if anchor is not None:
                                anchors[cam_key] = anchor
                        accepted = False
                        if active_key is None:
                            if _accept_neutral_sample(p, debug):
                                buckets["neutral"].append(p.steer_raw)
                                accepted = True
                        else:
                            if _accept_side_sample(p, debug, anchor, frame, mirrored=not args.no_mirror):
                                buckets[active_key].append(p.steer_raw)
                                accepted = True

                        title = "MANUAL CENTER" if active_key is None else f"CAPTURE {active_key.upper()}"
                        if keycheck_stage == "minus":
                            hint = "Key test: press -"
                        elif keycheck_stage == "plus":
                            hint = "Key test: press +"
                        else:
                            hint = (
                                "Pedal straight. Press - for left capture, + for right capture. Stop pedaling to save"
                                if active_key is None
                                else f"Move through {active_key}. Press {'-' if active_key == 'left' else '+'} again to stop"
                            )
                        pedaling_now = _pedaling_started(
                            await ftms.next(),
                            float(getattr(args, "start_cadence_rpm", 20.0)),
                            float(getattr(args, "start_watts", 35.0)),
                        )
                        ready_to_finish = bool(buckets["neutral"] and buckets["left"] and buckets["right"] and active_key is None)
                        if ready_to_finish and not pedaling_now:
                            if pedal_idle_started is None:
                                pedal_idle_started = monotonic()
                        else:
                            pedal_idle_started = None
                        display = frame.copy()
                        _draw_calibration_frame(
                            cv2,
                            display,
                            p,
                            debug,
                            title,
                            hint,
                            0.0,
                            mirror_preview=not args.no_mirror,
                            collecting=active_key is not None,
                            anchor=anchor,
                        )
                        cv2.putText(
                            display,
                            f"samples center={len(buckets['neutral'])} left={len(buckets['left'])} right={len(buckets['right'])}",
                            (20, 122),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.58,
                            (235, 235, 235),
                            1,
                            cv2.LINE_AA,
                        )
                        if ready_to_finish and keycheck_stage == "done":
                            status = "Stop pedaling to save"
                            if pedal_idle_started is not None:
                                remaining = max(0.0, 1.2 - (monotonic() - pedal_idle_started))
                                status = f"Saving in {remaining:0.1f}s"
                            cv2.putText(
                                display,
                                status,
                                (20, 148),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.58,
                                (255, 210, 80),
                                1,
                                cv2.LINE_AA,
                            )
                        dbg.log(
                            p=p,
                            debug=debug,
                            anchor=anchor,
                            frame=display,
                            extra={"state": "manual", "active": active_key or "neutral", "accepted": accepted, "ready_to_finish": ready_to_finish, "pedaling": pedaling_now, "keycheck_stage": keycheck_stage},
                        )
                        cv2.imshow(win, display)
                        keycode = cv2.waitKey(1) & 0xFF
                        if keycode == ord("q"):
                            print("Calibration canceled.")
                            _safe_destroy_window(cv2, win)
                            return 1
                        if keycode == ord("r"):
                            anchors.clear()
                            anchor_samples.clear()
                            anchor_locked.clear()
                            buckets = {"neutral": [], "left": [], "right": []}
                            active_key = None
                            pedal_idle_started = None
                            keycheck_stage = "minus"
                            vision.reset_tracking()
                        if keycheck_stage == "minus":
                            if keycode == ord("-"):
                                keycheck_stage = "plus"
                        elif keycheck_stage == "plus":
                            if keycode in (ord("+"), ord("=")):
                                keycheck_stage = "done"
                        else:
                            if keycode == ord("-"):
                                active_key = None if active_key == "left" else "left"
                            if keycode in (ord("+"), ord("=")):
                                active_key = None if active_key == "right" else "right"
                        if keycheck_stage == "done" and pedal_idle_started is not None and (monotonic() - pedal_idle_started) >= 1.2:
                            _safe_destroy_window(cv2, win)
                            break
                        await asyncio.sleep(1 / 30)
                else:
                    for title, key, hint in phases:
                        prep_end = asyncio.get_running_loop().time() + prep_seconds
                        while asyncio.get_running_loop().time() < prep_end:
                            p, frame, debug = vision.next_with_frame()
                            if frame is None:
                                await asyncio.sleep(1 / 30)
                                continue
                            h, w = frame.shape[:2]
                            cam_key = _debug_camera_key(debug)
                            anchor_frames[cam_key] = (w, h)
                            cent = _debug_centroid_px(debug, w, h, mirrored=not args.no_mirror)
                            if _stable_for_anchor(p, debug, cent) and key == "neutral" and cam_key not in anchor_locked:
                                samples = anchor_samples.setdefault(cam_key, [])
                                samples.append(cent)
                                if len(samples) > 24:
                                    del samples[:-24]
                                anchor = _anchor_from_samples(cam_key)
                                if anchor is not None:
                                    anchors[cam_key] = anchor
                            else:
                                anchor = _anchor_from_samples(cam_key)
                            display = frame.copy()
                            _draw_calibration_frame(
                                cv2,
                                display,
                                p,
                                debug,
                                f"GET READY: {title}",
                                hint,
                                prep_end - asyncio.get_running_loop().time(),
                                mirror_preview=not args.no_mirror,
                                collecting=False,
                                anchor=anchor,
                            )
                            dbg.log(
                                p=p,
                                debug=debug,
                                anchor=anchor,
                                frame=display,
                                extra={"phase": title, "phase_key": key, "state": "prep"},
                            )
                            cv2.imshow(win, display)
                            keycode = cv2.waitKey(1) & 0xFF
                            if keycode == ord("q"):
                                print("Calibration canceled.")
                                _safe_destroy_window(cv2, win)
                                return 1
                            if keycode == ord("r"):
                                anchors.clear()
                                anchor_samples.clear()
                                anchor_locked.clear()
                                vision.reset_tracking()
                            await asyncio.sleep(1 / 30)

                        end_at = asyncio.get_running_loop().time() + phase_seconds
                        while asyncio.get_running_loop().time() < end_at:
                            p, frame, debug = vision.next_with_frame()
                            if frame is None:
                                await asyncio.sleep(1 / 30)
                                continue
                            cam_key = _debug_camera_key(debug)
                            anchor = _anchor_from_samples(cam_key)
                            pass_anchor = _anchor_gate_pass(
                                p.source, debug, anchor, frame.shape[1], frame.shape[0], mirrored=not args.no_mirror
                            )
                            accept_sample = _accept_side_sample(p, debug, anchor, frame, mirrored=not args.no_mirror)
                            if key == "neutral":
                                accept_sample = _accept_neutral_sample(p, debug)
                            if accept_sample:
                                buckets[key].append(p.steer_raw)
                            h, w = frame.shape[:2]
                            anchor_frames[cam_key] = (w, h)
                            cent = _debug_centroid_px(debug, w, h, mirrored=not args.no_mirror)
                            if _stable_for_anchor(p, debug, cent) and key == "neutral" and cam_key not in anchor_locked:
                                samples = anchor_samples.setdefault(cam_key, [])
                                samples.append(cent)
                                if len(samples) > 24:
                                    del samples[:-24]
                                anchor = _anchor_from_samples(cam_key)
                                if anchor is not None:
                                    anchors[cam_key] = anchor
                                if len(samples) >= 8:
                                    anchor_locked.add(cam_key)

                            display = frame.copy()
                            _draw_calibration_frame(
                                cv2,
                                display,
                                p,
                                debug,
                                f"COLLECT: {title}",
                                hint,
                                end_at - asyncio.get_running_loop().time(),
                                mirror_preview=not args.no_mirror,
                                collecting=True,
                                anchor=anchor,
                            )
                            dbg.log(
                                p=p,
                                debug=debug,
                                anchor=anchor,
                                frame=display,
                                extra={"phase": title, "phase_key": key, "state": "collect", "pass_anchor": pass_anchor, "accepted": accept_sample},
                            )
                            cv2.imshow(win, display)
                            keycode = cv2.waitKey(1) & 0xFF
                            if keycode == ord("q"):
                                print("Calibration canceled.")
                                _safe_destroy_window(cv2, win)
                                return 1
                            if keycode == ord("r"):
                                anchors.clear()
                                anchor_samples.clear()
                                anchor_locked.clear()
                                vision.reset_tracking()
                            await asyncio.sleep(1 / 30)
                _safe_destroy_window(cv2, win)
                gui_completed = True
            except Exception:
                use_gui = False

        if not gui_completed and not use_gui:
            print("GUI unavailable; using text calibration mode.")
            if float(getattr(args, "resistance_calibration_level", 0.0) or 0.0) > 0.0:
                phase_order = ["neutral", "left", "neutral", "right", "neutral"]
                phase_titles = {"neutral": "CENTER", "left": "LEFT", "right": "RIGHT"}
                phase_idx = 0
                pedal_idle_started: float | None = None
                target_level = float(getattr(args, "resistance_calibration_level", 5.0))
                target_applied = False
                while True:
                    f = await ftms.next()
                    if not target_applied and bool(getattr(f, "connected", False)):
                        target_applied = await ftms.set_target_resistance(target_level)
                    p, frame, debug = vision.next_with_frame()
                    if frame is None:
                        await asyncio.sleep(1 / 30)
                        continue
                    h, w = frame.shape[:2]
                    cam_key = _debug_camera_key(debug)
                    anchor_frames[cam_key] = (w, h)
                    cent = _debug_centroid_px(debug, w, h, mirrored=not args.no_mirror)
                    anchor = _anchor_from_samples(cam_key)
                    if _stable_for_anchor(p, debug, cent):
                        samples = anchor_samples.setdefault(cam_key, [])
                        samples.append(cent)
                        if len(samples) > 24:
                            del samples[:-24]
                        anchor = _anchor_from_samples(cam_key)
                        if anchor is not None:
                            anchors[cam_key] = anchor
                        if len(samples) >= 8:
                            anchor_locked.add(cam_key)
                    center_locked = anchor is not None and cam_key in anchor_locked
                    phase_key = phase_order[min(phase_idx, len(phase_order) - 1)]
                    accept_sample = False
                    if phase_idx == 0 and not center_locked:
                        accept_sample = False
                    elif phase_key == "neutral":
                        if _accept_neutral_sample(p, debug):
                            buckets["neutral"].append(p.steer_raw)
                            accept_sample = True
                    else:
                        if _accept_side_sample(p, debug, anchor, frame, mirrored=not args.no_mirror) and _phase_sign_accepts(
                            phase_key, p.steer_raw, buckets["neutral"]
                        ):
                            buckets[phase_key].append(p.steer_raw)
                            accept_sample = True
                    pedaling_now = _pedaling_started(
                        f,
                        float(getattr(args, "start_cadence_rpm", 20.0)),
                        float(getattr(args, "start_watts", 35.0)),
                    )
                    ready_to_finish = bool(
                        len(buckets["neutral"]) >= 40 and len(buckets["left"]) >= 30 and len(buckets["right"]) >= 30
                    )
                    phase_done = len(buckets[phase_key]) >= _phase_target_count(phase_key)
                    if phase_idx == 0 and not center_locked:
                        phase_done = False
                    if not pedaling_now:
                        if pedal_idle_started is None:
                            pedal_idle_started = monotonic()
                        idle_elapsed = monotonic() - pedal_idle_started
                        if phase_idx >= len(phase_order) - 1:
                            if ready_to_finish and idle_elapsed >= 1.2:
                                break
                        elif phase_done and idle_elapsed >= 0.8:
                            phase_idx += 1
                            pedal_idle_started = None
                            phase_key = phase_order[min(phase_idx, len(phase_order) - 1)]
                            print(f"Advance: {phase_titles[phase_key]}")
                    else:
                        pedal_idle_started = None
                    dbg.log(
                        p=p,
                        f=f,
                        debug=debug,
                        anchor=anchor,
                        extra={
                            "phase": phase_titles[phase_key],
                            "phase_key": phase_key,
                            "phase_mode": "pedal",
                            "phase_idx": phase_idx,
                            "state": "collect-text",
                            "accepted": accept_sample,
                            "ready_to_finish": ready_to_finish,
                        },
                    )
                    await asyncio.sleep(1 / 30)
            else:
                phase_frames = max(30, int(float(args.phase_seconds) * 30))
                for title, key, hint in phases:
                    print(f"Get ready ({prep_seconds:.1f}s): {title} - {hint}")
                    await asyncio.sleep(prep_seconds)
                    print(f"Collecting {title} for {phase_seconds:.1f}s...")
                    for _ in range(phase_frames):
                        p = vision.next()
                        if p.confidence >= _pose_conf_threshold(p.source):
                            buckets[key].append(p.steer_raw)
                        dbg.log(
                            p=p,
                            extra={"phase": title, "phase_key": key, "state": "collect-text"},
                        )
                        await asyncio.sleep(1 / 30)

        neutral_samples = buckets["neutral"]
        left_vals = buckets["left"]
        right_vals = buckets["right"]
        neutral = _percentile(neutral_samples, 0.50) if neutral_samples else 0.0
        corrections: list[str] = []
        left_vals_used, left_trimmed = _trim_side_outliers(left_vals)
        right_vals_used, right_trimmed = _trim_side_outliers(right_vals)
        if left_trimmed:
            corrections.append("left_outliers_trimmed")
        if right_trimmed:
            corrections.append("right_outliers_trimmed")
        left_peak = _percentile(left_vals_used, 0.12) if left_vals_used else -0.7
        right_peak = _percentile(right_vals_used, 0.88) if right_vals_used else 0.7
        flip_sign = False
        anchor_x_norm = None
        anchor_y_norm = None
        if anchors:
            norm_xs: list[float] = []
            norm_ys: list[float] = []
            for cam_key, (ax, ay) in anchors.items():
                fw, fh = anchor_frames.get(cam_key, (args.vision_width, args.vision_height))
                norm_xs.append(ax / max(1.0, float(fw)))
                norm_ys.append(ay / max(1.0, float(fh)))
            norm_xs.sort()
            norm_ys.sort()
            anchor_x_norm = norm_xs[len(norm_xs) // 2]
            anchor_y_norm = norm_ys[len(norm_ys) // 2]

        left_med = _percentile(left_vals_used, 0.50) if left_vals_used else neutral
        right_med = _percentile(right_vals_used, 0.50) if right_vals_used else neutral
        left_delta = left_med - neutral
        right_delta = right_med - neutral
        sign_margin = 0.03
        if left_delta > sign_margin and right_delta < -sign_margin:
            flip_sign = True
            left_vals_used = [2.0 * neutral - v for v in left_vals_used]
            right_vals_used = [2.0 * neutral - v for v in right_vals_used]
            left_peak = _percentile(left_vals_used, 0.12) if left_vals_used else -0.7
            right_peak = _percentile(right_vals_used, 0.88) if right_vals_used else 0.7
            corrections.append("mirror_auto_flip")

        # Calibration safety: if one side never crosses neutral (common when tracking blinks during a phase),
        # synthesize the missing side from the opposite span so steering stays usable.
        min_span = 0.08
        if right_peak <= neutral:
            fallback = max(abs(neutral - left_peak), min_span)
            right_peak = neutral + fallback
            corrections.append("right_peak_auto_fixed")
        if left_peak >= neutral:
            fallback = max(abs(right_peak - neutral), min_span)
            left_peak = neutral - fallback
            corrections.append("left_peak_auto_fixed")

        # Ensure both sides keep enough dynamic range after corrections.
        if abs(neutral - left_peak) < min_span:
            left_peak = neutral - min_span
            corrections.append("left_span_min_applied")
        if abs(right_peak - neutral) < min_span:
            right_peak = neutral + min_span
            corrections.append("right_span_min_applied")

        calib = Calibrator(
            neutral=neutral,
            left_peak=left_peak,
            right_peak=right_peak,
            flip_sign=flip_sign,
            anchor_x_norm=anchor_x_norm,
            anchor_y_norm=anchor_y_norm,
        )
        out = _calibration_path(args.profile)
        save_calibration(out, calib)
        print(f"Saved calibration: {out}")
        print(
            f"samples neutral={len(neutral_samples)} "
            f"left={len(left_vals_used)}/{len(left_vals)} right={len(right_vals_used)}/{len(right_vals)}"
        )
        if len(left_vals_used) < 20 or len(right_vals_used) < 20:
            print("Warning: low valid samples. Try more light, visible camera (0), or longer phase seconds.")
        if corrections:
            print(f"Calibration correction: {', '.join(corrections)}")
        print(
            f"neutral={neutral:.4f} left_peak={left_peak:.4f} right_peak={right_peak:.4f} "
            f"flip_sign={flip_sign} anchor=({anchor_x_norm},{anchor_y_norm})"
        )
        return 0
    except (KeyboardInterrupt, asyncio.CancelledError):
        print("Calibration canceled.")
        return 1
    finally:
        await ftms.close()
        dbg.close()
        vision.close()


async def _run_loop(args: argparse.Namespace, monitor_only: bool, record_only: bool = False) -> int:
    profile = load_profile(args.profile)
    calib = load_calibration(_calibration_path(args.profile))
    fusion = FusionPipeline(profile, calibrator=calib)
    ftms = FtmsSource(args.bike, verbose=getattr(args, "verbose", False))
    vision = VisionMux(
        profile.steering.mode,
        camera_arg=args.camera,
        width=args.vision_width,
        height=args.vision_height,
        idle_hz=args.mux_idle_hz,
    )
    pad = None
    if not monitor_only and not record_only:
        pad = VirtualGamepad(
            steer_axis=profile.uinput.steer_axis,
            throttle_axis=profile.uinput.throttle_axis,
            invert_throttle=profile.uinput.invert_throttle,
        )
    mode = "record" if record_only else ("monitor" if monitor_only else "run")
    dbg = DebugLogger(args.debug_log, mode, args.debug_fps, args.debug_width, args.debug_height)

    try:
        print(f"camera={vision.camera_idx} bike={args.bike} profile={profile.name}")
        if dbg.enabled and dbg.session_dir is not None:
            print(f"debug log: {dbg.session_dir}")
        if not monitor_only and not record_only and (pad is None or not pad.enabled):
            reason = ""
            if pad is not None:
                reason = getattr(pad, "error", "") or ""
            if reason:
                print(f"uinput unavailable ({reason}). Running monitor-only output.")
            else:
                print("uinput unavailable. Running monitor-only output.")

        hz = max(20, min(120, args.hz))
        dt = 1.0 / hz
        mirror_preview = not getattr(args, "no_mirror", False)
        gui_enabled = (monitor_only or record_only) and not getattr(args, "no_gui", False)
        cv2 = None
        win = None
        anchors: dict[str, tuple[int, int]] = {}
        anchor_relock_until: dict[str, float] = {}
        anchor_relock_samples: dict[str, list[tuple[int, int]]] = {}
        if gui_enabled:
            try:
                import cv2 as _cv2

                cv2 = _cv2
                win = "ftms2pad monitor"
                cv2.namedWindow(win, cv2.WINDOW_NORMAL)
                cv2.resizeWindow(win, 1100, 650)
            except Exception:
                gui_enabled = False
        last_stand_press = 0.0
        button_releases: dict[str, float] = {}
        stand_active = False
        resistance_target = float(getattr(args, "resistance_start", 0.0) or 0.0)
        resistance_step = max(0.1, float(getattr(args, "resistance_step", 1.0)))
        resistance_applied = False
        pedal_ready = False

        started_at = monotonic()
        record_duration_s = max(0.0, float(getattr(args, "duration_seconds", 0.0) or 0.0))
        while True:
            f = await ftms.next()
            pedaling_now = _pedaling_started(
                f,
                float(getattr(args, "start_cadence_rpm", 20.0)),
                float(getattr(args, "start_watts", 35.0)),
            )
            if pedaling_now and not pedal_ready:
                pedal_ready = True
                anchors.clear()
                anchor_relock_until.clear()
                anchor_relock_samples.clear()
                stand_active = False
                vision.reset_tracking()
            if (
                not resistance_applied
                and resistance_target > 0.0
                and bool(getattr(f, "connected", False))
            ):
                resistance_applied = await ftms.set_target_resistance(resistance_target)
            if pad is not None:
                now = monotonic()
                for button_name, release_at in list(button_releases.items()):
                    if now >= release_at:
                        pad.emit_button(button_name, False)
                        del button_releases[button_name]
            p, frame, debug = vision.next_with_frame()
            cam_key = _debug_camera_key(debug)
            anchor = anchors.get(cam_key)
            if pedal_ready and frame is not None:
                h, w = frame.shape[:2]
                if (
                    anchor is None
                    and calib.anchor_x_norm is not None
                    and calib.anchor_y_norm is not None
                ):
                    anchor = (
                        int(max(0, min(w - 1, round(calib.anchor_x_norm * w)))),
                        int(max(0, min(h - 1, round(calib.anchor_y_norm * h)))),
                    )
                    anchors[cam_key] = anchor
                    anchor_relock_until[cam_key] = monotonic() + 1.0
                cent = _debug_centroid_px(debug, w, h, mirrored=mirror_preview)
                if (
                    cent is not None
                    and anchor is not None
                    and cam_key in anchor_relock_until
                    and (abs(cent[0] - anchor[0]) > int(w * 0.22) or abs(cent[1] - anchor[1]) > int(h * 0.18))
                ):
                    anchor = cent
                    anchors[cam_key] = anchor
                if cent is not None and anchor is None:
                    anchors[cam_key] = cent
                    anchor = cent
                elif cent is not None and anchor is not None and p.source == "camera-bike":
                    relock_deadline = anchor_relock_until.get(cam_key, 0.0)
                    if monotonic() < relock_deadline and p.confidence >= _pose_conf_threshold(p.source):
                        samples = anchor_relock_samples.setdefault(cam_key, [])
                        samples.append(cent)
                        if len(samples) > 24:
                            del samples[:-24]
                    elif cam_key in anchor_relock_until:
                        samples = anchor_relock_samples.get(cam_key, [])
                        if samples:
                            xs = sorted(p[0] for p in samples)
                            ys = sorted(p[1] for p in samples)
                            anchor = (xs[len(xs) // 2], ys[len(ys) // 2])
                            anchors[cam_key] = anchor
                        anchor_relock_until.pop(cam_key, None)
                        anchor_relock_samples.pop(cam_key, None)
            pass_anchor = _anchor_gate_pass(
                p.source,
                debug,
                anchor,
                frame.shape[1] if frame is not None else None,
                frame.shape[0] if frame is not None else None,
                mirrored=mirror_preview,
            )
            pose_ok = pedal_ready and p.confidence >= _pose_conf_threshold(p.source) and pass_anchor
            steer = fusion.steer(p.steer_raw, pose_ok=pose_ok)
            throttle = fusion.throttle(f.watts, connected=f.connected)

            # Optional stand-to-button mapping (e.g. jump) based on head rise from anchor.
            stand_button = str(getattr(args, "stand_button", "") or "").strip().upper()
            if stand_button and frame is not None:
                h, w = frame.shape[:2]
                cent = _debug_centroid_px(debug, w, h, mirrored=mirror_preview)
                if cent is not None and anchor is not None and pose_ok:
                    stand_px = int(max(0.05, min(0.4, float(getattr(args, "stand_threshold", 0.14)))) * h)
                    is_standing = cent[1] <= int(anchor[1]) - stand_px
                    now = monotonic()
                    cooldown = max(0.05, float(getattr(args, "stand_cooldown", 0.35)))
                    if is_standing and not stand_active and (now - last_stand_press) >= cooldown and not monitor_only and not record_only and pad is not None:
                        pad.emit_button(stand_button, True)
                        button_releases[stand_button] = now + 0.05
                        last_stand_press = now
                    stand_active = is_standing
                else:
                    stand_active = False

            if not monitor_only and not record_only and pad is not None:
                pad.emit(steer=steer, throttle=throttle)

            print(
                f"w={f.watts:6.1f} cad={f.cadence_rpm:5.1f} res={f.resistance_level:4.1f} "
                f"pose={p.confidence:0.2f} steer={steer:+0.3f} thr={throttle:0.3f}",
                end="\r",
                flush=True,
            )

            if gui_enabled and cv2 is not None and frame is not None and win is not None:
                h, w = frame.shape[:2]
                cent = _debug_centroid_px(debug, w, h, mirrored=mirror_preview)
                _draw_monitor_frame(
                    cv2,
                    frame,
                    p,
                    f,
                    steer,
                    throttle,
                    mirror_preview=mirror_preview,
                    debug=debug,
                    anchor=anchor,
                )
                if not pedal_ready:
                    cv2.putText(frame, "Start pedaling to lock center", (20, 114), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (60, 220, 255), 2, cv2.LINE_AA)
                dbg.log(
                    p=p,
                    f=f,
                    steer=steer,
                    throttle=throttle,
                    debug=debug,
                    anchor=anchor,
                    frame=frame,
                    extra={"pose_ok": pose_ok, "pass_anchor": pass_anchor, "pedal_ready": pedal_ready},
                )
                cv2.imshow(win, frame)
                keycode = cv2.waitKey(1) & 0xFF
                if keycode == ord("q"):
                    print("\nStopped.")
                    return 0
                if keycode == ord("r"):
                    anchors.clear()
                    stand_active = False
                    vision.reset_tracking()
                if keycode in (ord("+"), ord("="), 61, 43):
                    base = f.resistance_level if f.connected and f.resistance_level > 0 else resistance_target
                    resistance_target = max(0.0, base + resistance_step)
                    resistance_applied = await ftms.set_target_resistance(resistance_target)
                if keycode in (ord("-"), ord("_"), 45, 95):
                    base = f.resistance_level if f.connected and f.resistance_level > 0 else resistance_target
                    resistance_target = max(0.0, base - resistance_step)
                    resistance_applied = await ftms.set_target_resistance(resistance_target)
            else:
                dbg.log(
                    p=p,
                    f=f,
                    steer=steer,
                    throttle=throttle,
                    debug=debug,
                    anchor=anchor,
                    extra={"pose_ok": pose_ok, "pass_anchor": pass_anchor, "pedal_ready": pedal_ready},
                )
            if record_duration_s > 0.0 and (monotonic() - started_at) >= record_duration_s:
                print("\nCapture complete.")
                return 0
            await asyncio.sleep(dt)
    except (KeyboardInterrupt, asyncio.CancelledError):
        print("\nStopped.")
        return 0
    finally:
        if monitor_only and not getattr(args, "no_gui", False):
            try:
                import cv2 as _cv2

                _cv2.destroyAllWindows()
            except Exception:
                pass
        vision.close()
        dbg.close()
        if pad is not None:
            pad.close()
        await ftms.close()


async def cmd_run(args: argparse.Namespace) -> int:
    return await _run_loop(args, monitor_only=False)


async def cmd_monitor(args: argparse.Namespace) -> int:
    return await _run_loop(args, monitor_only=True)


async def cmd_record(args: argparse.Namespace) -> int:
    if not getattr(args, "debug_log", ""):
        args.debug_log = str(getattr(args, "out_dir", "sessions"))
    return await _run_loop(args, monitor_only=False, record_only=True)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="ftms2pad")
    sub = p.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("list-bikes", help="Scan BLE devices")
    c.set_defaults(fn=cmd_list_bikes)

    c = sub.add_parser("list-cameras", help="List webcam device indexes")
    c.set_defaults(fn=cmd_list_cameras)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--profile", default="supertuxkart")
    common.add_argument("--bike", default="sim", help="BLE addr/name, or sim")
    common.add_argument("--verbose", action="store_true", help="Print FTMS BLE connection details")
    common.add_argument("--camera", default="auto", help="camera index, auto, or comma list (e.g. 0,2)")
    common.add_argument("--hz", type=int, default=60, help="main loop frequency")
    common.add_argument("--debug-log", default="", help="directory to write debug bundle")
    common.add_argument("--debug-fps", type=float, default=10.0, help="debug video FPS")
    common.add_argument("--debug-width", type=int, default=640, help="debug video width")
    common.add_argument("--debug-height", type=int, default=360, help="debug video height")
    common.add_argument("--vision-width", type=int, default=640, help="camera capture width")
    common.add_argument("--vision-height", type=int, default=360, help="camera capture height")
    common.add_argument("--mux-idle-hz", type=float, default=8.0, help="poll rate for inactive cameras in multi-camera mode")
    common.add_argument("--stand-button", default="", help="button to tap when standing is detected (e.g. BTN_A)")
    common.add_argument("--stand-threshold", type=float, default=0.14, help="fraction of frame height above anchor to count as stand")
    common.add_argument("--stand-cooldown", type=float, default=0.35, help="seconds between stand-triggered taps")
    common.add_argument("--resistance-start", type=float, default=0.0, help="set trainer target resistance after connecting")
    common.add_argument("--resistance-step", type=float, default=1.0, help="GUI +/- resistance step")

    c = sub.add_parser("run", parents=[common], help="Emit virtual gamepad")
    c.set_defaults(fn=cmd_run)

    c = sub.add_parser("monitor", parents=[common], help="No uinput output, only live stats")
    c.add_argument("--no-gui", action="store_true", help="Use text-only monitor")
    c.add_argument("--no-mirror", action="store_true", help="Do not mirror preview window")
    c.set_defaults(fn=cmd_monitor)

    c = sub.add_parser("calibrate", parents=[common], help="Capture neutral and lean range")
    c.add_argument("--no-gui", action="store_true", help="Use text-only calibration")
    c.add_argument("--no-mirror", action="store_true", help="Do not mirror preview window")
    c.add_argument("--manual-calibration", action="store_true", help="Use manual toggle capture: '-' left, '+' right, 's' save")
    c.add_argument("--resistance-calibration-level", type=float, default=0.0, help="Use trainer resistance as calibration selector; target level is center")
    c.add_argument("--resistance-calibration-band", type=float, default=0.5, help="Center band around resistance calibration level")
    c.add_argument("--no-wait-pedal", action="store_true", help="Start calibration immediately instead of waiting for pedaling")
    c.add_argument("--start-cadence-rpm", type=float, default=20.0, help="cadence that starts calibration")
    c.add_argument("--start-watts", type=float, default=35.0, help="power that starts calibration")
    c.add_argument("--prep-seconds", type=float, default=2.0, help="Countdown seconds before each phase")
    c.add_argument("--phase-seconds", type=float, default=4.0, help="Seconds per calibration phase")
    c.set_defaults(fn=cmd_calibrate)

    return p


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    fn = args.fn
    try:
        if inspect.iscoroutinefunction(fn):
            raise SystemExit(asyncio.run(fn(args)))
        raise SystemExit(fn(args))
    except KeyboardInterrupt:
        print("\nStopped.")
        raise SystemExit(130)


if __name__ == "__main__":
    main()
