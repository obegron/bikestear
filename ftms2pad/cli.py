from __future__ import annotations

import argparse
import asyncio
import inspect
import json
from pathlib import Path
from time import monotonic, time

from ftms2pad.calibration import load_calibration, save_calibration
from ftms2pad.ftms import FtmsSource, list_ble_devices
from ftms2pad.fusion import Calibrator, FusionPipeline
from ftms2pad.profiles import load_profile
from ftms2pad.uinput import VirtualGamepad
from ftms2pad.vision import VisionTracker, camera_name, list_cameras


def _calibration_path(profile: str) -> Path:
    return Path("profiles") / f"{profile}.calibration.json"


class DebugLogger:
    def __init__(
        self,
        base_dir: str | None,
        mode: str,
        debug_fps: float = 10.0,
        width: int = 640,
        height: int = 360,
    ) -> None:
        self.enabled = bool(base_dir)
        self.session_dir: Path | None = None
        self.events_fp = None
        self.writer = None
        self._cv2 = None
        self._t_last_frame = 0.0
        self.debug_fps = max(1.0, float(debug_fps))
        self.size = (max(160, int(width)), max(120, int(height)))
        self.frame_idx = 0
        self.mode = mode

        if not self.enabled:
            return
        ts = time()
        root = Path(str(base_dir))
        self.session_dir = root / f"{mode}-{int(ts)}"
        self.session_dir.mkdir(parents=True, exist_ok=True)
        self.events_fp = (self.session_dir / "events.jsonl").open("w", encoding="utf-8", buffering=1)
        (self.session_dir / "meta.json").write_text(
            json.dumps(
                {
                    "mode": mode,
                    "created_epoch_s": ts,
                    "debug_fps": self.debug_fps,
                    "frame_size": {"w": self.size[0], "h": self.size[1]},
                },
                indent=2,
            )
        )

    def _ensure_writer(self, frame) -> None:
        if not self.enabled or self.session_dir is None or self.writer is not None:
            return
        try:
            import cv2

            self._cv2 = cv2
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            self.writer = cv2.VideoWriter(str(self.session_dir / "debug.mp4"), fourcc, self.debug_fps, self.size)
        except Exception:
            self.writer = None

    def log(
        self,
        *,
        p=None,
        f=None,
        steer: float | None = None,
        throttle: float | None = None,
        debug: dict | None = None,
        anchor: tuple[int, int] | None = None,
        frame=None,
        extra: dict | None = None,
    ) -> None:
        if not self.enabled:
            return
        event = {
            "t": monotonic(),
            "mode": self.mode,
            "pose": {
                "source": getattr(p, "source", ""),
                "confidence": float(getattr(p, "confidence", 0.0)),
                "steer_raw": float(getattr(p, "steer_raw", 0.0)),
            },
            "ftms": {
                "watts": float(getattr(f, "watts", 0.0)) if f is not None else 0.0,
                "cadence_rpm": float(getattr(f, "cadence_rpm", 0.0)) if f is not None else 0.0,
                "speed_kph": float(getattr(f, "speed_kph", 0.0)) if f is not None else 0.0,
                "connected": bool(getattr(f, "connected", False)) if f is not None else False,
            },
            "control": {
                "steer": float(steer if steer is not None else 0.0),
                "throttle": float(throttle if throttle is not None else 0.0),
            },
            "debug": debug or {},
            "anchor": {"x": anchor[0], "y": anchor[1]} if anchor is not None else None,
            "extra": extra or {},
            "frame_idx": None,
        }
        if self.events_fp is not None:
            self.events_fp.write(json.dumps(event) + "\n")
            self.events_fp.flush()

        if frame is None:
            return
        now = monotonic()
        if now - self._t_last_frame < (1.0 / self.debug_fps):
            return
        self._t_last_frame = now
        self._ensure_writer(frame)
        if self.writer is None or self._cv2 is None:
            return
        out = self._cv2.resize(frame, self.size, interpolation=self._cv2.INTER_AREA)
        self.writer.write(out)
        self.frame_idx += 1

    def close(self) -> None:
        if self.events_fp is not None:
            self.events_fp.close()
            self.events_fp = None
        if self.writer is not None:
            self.writer.release()
            self.writer = None


def _pose_conf_threshold(source: str) -> float:
    if source == "camera-face":
        return 0.18
    if source in ("camera-hog", "camera-blob"):
        return 0.1
    return 0.3


def _percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    vals = sorted(values)
    idx = int((len(vals) - 1) * p)
    return vals[max(0, min(len(vals) - 1, idx))]


def _debug_centroid_px(debug: dict, w: int, h: int, mirrored: bool) -> tuple[int, int] | None:
    kind = str(debug.get("kind", ""))
    if kind in ("face", "blob"):
        centroid = debug.get("centroid")
        if isinstance(centroid, tuple) and len(centroid) == 2:
            x, y = int(centroid[0]), int(centroid[1])
            if mirrored:
                x = w - x
            return x, y
    if kind == "hog":
        bbox_norm = debug.get("bbox_norm")
        if isinstance(bbox_norm, tuple) and len(bbox_norm) == 4:
            nx, ny, nw, nh = [float(v) for v in bbox_norm]
            if mirrored:
                nx = 1.0 - (nx + nw)
            x = int((nx + nw * 0.5) * w)
            y = int((ny + nh * 0.5) * h)
            return x, y
    if kind == "mediapipe":
        points = debug.get("points_norm")
        if isinstance(points, list) and points:
            px, py = points[0]
            x = int(float(px) * w)
            y = int(float(py) * h)
            if mirrored:
                x = w - x
            return x, y
    return None


def _anchor_gate_pass(
    source: str,
    debug: dict,
    anchor: tuple[int, int] | None,
    w: int | None,
    h: int | None,
    mirrored: bool,
) -> bool:
    if anchor is None or w is None or h is None:
        return True
    cent = _debug_centroid_px(debug, w, h, mirrored=mirrored)
    if cent is None:
        return source != "camera-face"
    ax, ay = anchor
    dx = abs(cent[0] - ax)
    dy = abs(cent[1] - ay)
    if source == "camera-face":
        return dx <= int(w * 0.28) and dy <= int(h * 0.30)
    if source in ("camera-hog", "camera-blob"):
        return dx <= int(w * 0.45)
    return True


def _draw_tracking_overlay(cv2, frame, debug: dict, mirrored: bool, color=(120, 220, 120)) -> None:
    h, w = frame.shape[:2]
    kind = str(debug.get("kind", ""))
    if kind == "hog":
        bbox_norm = debug.get("bbox_norm")
        if isinstance(bbox_norm, tuple) and len(bbox_norm) == 4:
            nx, ny, nw, nh = [float(v) for v in bbox_norm]
            if mirrored:
                nx = 1.0 - (nx + nw)
            x = int(nx * w)
            y = int(float(ny) * h)
            bw = int(nw * w)
            bh = int(nh * h)
            cv2.rectangle(frame, (x, y), (x + bw, y + bh), color, 2)
            cv2.putText(frame, "track: person(hog)", (x, max(20, y - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    elif kind == "blob":
        bbox = debug.get("bbox")
        centroid = debug.get("centroid")
        if isinstance(bbox, tuple) and len(bbox) == 4:
            x, y, bw, bh = [int(v) for v in bbox]
            if mirrored:
                x = w - (x + bw)
            cv2.rectangle(frame, (x, y), (x + bw, y + bh), color, 2)
            cv2.putText(frame, "track: motion blob", (x, max(20, y - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
        if isinstance(centroid, tuple) and len(centroid) == 2:
            cx, cy = [int(v) for v in centroid]
            if mirrored:
                cx = w - cx
            cv2.circle(frame, (cx, cy), 6, color, -1)
    elif kind == "mediapipe":
        points = debug.get("points_norm")
        if isinstance(points, list):
            for pt in points:
                if isinstance(pt, tuple) and len(pt) == 2:
                    px = float(pt[0])
                    if mirrored:
                        px = 1.0 - px
                    x = int(px * w)
                    y = int(float(pt[1]) * h)
                    cv2.circle(frame, (x, y), 5, color, -1)
            if points:
                cv2.putText(frame, "track: pose points", (20, 112), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    elif kind == "face":
        bbox = debug.get("bbox")
        centroid = debug.get("centroid")
        detector = str(debug.get("detector", "face"))
        if isinstance(bbox, tuple) and len(bbox) == 4:
            x, y, bw, bh = [int(v) for v in bbox]
            if mirrored:
                x = w - (x + bw)
            cv2.rectangle(frame, (x, y), (x + bw, y + bh), color, 2)
            cv2.putText(frame, f"track: {detector}", (x, max(20, y - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
        if isinstance(centroid, tuple) and len(centroid) == 2:
            cx, cy = [int(v) for v in centroid]
            if mirrored:
                cx = w - cx
            cv2.circle(frame, (cx, cy), 6, color, -1)


def _draw_monitor_frame(
    cv2,
    frame,
    p,
    f,
    steer: float,
    throttle: float,
    mirror_preview: bool,
    debug: dict,
    anchor: tuple[int, int] | None,
) -> None:
    if mirror_preview:
        frame[:] = cv2.flip(frame, 1)
    h, w = frame.shape[:2]
    if anchor is None:
        cx, cy = w // 2, h // 2
    else:
        cx = max(80, min(w - 80, int(anchor[0])))
        cy = max(80, min(h - 80, int(anchor[1])))
    # Head target zone: stay roughly here for stable detection.
    if anchor is not None:
        head_w = int(w * 0.34)
        head_h = int(h * 0.34)
        hx1 = cx - head_w // 2
        hy1 = cy - head_h // 2
        hx1 = max(0, min(w - head_w, hx1))
        hy1 = max(0, min(h - head_h, hy1))
        cv2.rectangle(frame, (hx1, hy1), (hx1 + head_w, hy1 + head_h), (90, 160, 255), 2)
        cv2.putText(frame, "Head zone", (hx1 + 4, hy1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (90, 160, 255), 1, cv2.LINE_AA)

    cv2.line(frame, (cx - 80, cy), (cx + 80, cy), (0, 255, 0), 2)
    cv2.line(frame, (cx, cy - 80), (cx, cy + 80), (0, 255, 0), 2)

    steer_dir = 1.0 if mirror_preview else -1.0
    dot_x = int(cx + steer_dir * max(-1.0, min(1.0, steer)) * int(w * 0.35))
    cv2.circle(frame, (dot_x, cy), 10, (0, 180, 255), -1)

    bar_w = int(w * 0.35)
    bar_h = 18
    sx, sy = 24, h - 70
    tx, ty = 24, h - 35
    cv2.rectangle(frame, (sx, sy), (sx + bar_w, sy + bar_h), (120, 120, 120), 1)
    cv2.rectangle(frame, (tx, ty), (tx + bar_w, ty + bar_h), (120, 120, 120), 1)

    steer_fill = int((steer + 1.0) * 0.5 * bar_w)
    cv2.rectangle(frame, (sx, sy), (sx + steer_fill, sy + bar_h), (0, 180, 255), -1)
    throttle_fill = int(max(0.0, min(1.0, throttle)) * bar_w)
    cv2.rectangle(frame, (tx, ty), (tx + throttle_fill, ty + bar_h), (255, 180, 0), -1)

    cv2.putText(frame, "steer", (sx, sy - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (230, 230, 230), 1, cv2.LINE_AA)
    cv2.putText(frame, "throttle", (tx, ty - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (230, 230, 230), 1, cv2.LINE_AA)
    cv2.putText(
        frame,
        f"w={f.watts:6.1f}  cad={f.cadence_rpm:5.1f}  speed={f.speed_kph:5.1f}  pose={p.confidence:0.2f}",
        (20, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        f"raw={p.steer_raw:+.3f}  steer={steer:+.3f}  thr={throttle:0.3f}  src={p.source}",
        (20, 58),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (230, 230, 230),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(frame, "Press q to stop, r to relock center", (20, 86), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1, cv2.LINE_AA)
    _draw_tracking_overlay(cv2, frame, debug, mirrored=mirror_preview)


def _draw_calibration_frame(
    cv2,
    frame,
    p,
    debug: dict,
    title: str,
    hint: str,
    seconds_left: float,
    mirror_preview: bool,
    collecting: bool,
    anchor: tuple[int, int] | None,
) -> None:
    if mirror_preview:
        frame[:] = cv2.flip(frame, 1)
    h, w = frame.shape[:2]
    if anchor is None:
        cx, cy = w // 2, h // 2
    else:
        cx = max(80, min(w - 80, int(anchor[0])))
        cy = max(80, min(h - 80, int(anchor[1])))
    if anchor is not None:
        head_w = int(w * 0.34)
        head_h = int(h * 0.34)
        hx1 = cx - head_w // 2
        hy1 = cy - head_h // 2
        hx1 = max(0, min(w - head_w, hx1))
        hy1 = max(0, min(h - head_h, hy1))
        cv2.rectangle(frame, (hx1, hy1), (hx1 + head_w, hy1 + head_h), (90, 160, 255), 2)
        cv2.putText(frame, "Keep head in this box", (hx1 + 4, hy1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (90, 160, 255), 1, cv2.LINE_AA)
    else:
        cv2.putText(frame, "Locking neutral head position...", (20, 126), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (90, 160, 255), 1, cv2.LINE_AA)

    cv2.line(frame, (cx - 70, cy), (cx + 70, cy), (0, 255, 0), 2)
    cv2.line(frame, (cx, cy - 70), (cx, cy + 70), (0, 255, 0), 2)

    steer_dir = 1.0 if mirror_preview else -1.0
    dot_x = int(cx + steer_dir * max(-1.0, min(1.0, p.steer_raw)) * (w * 0.35))
    cv2.circle(frame, (dot_x, cy), 10, (0, 180, 255), -1)

    if "LEFT" in title:
        cv2.arrowedLine(frame, (cx + 120, cy), (cx - 140, cy), (255, 180, 0), 6, tipLength=0.22)
    elif "RIGHT" in title:
        cv2.arrowedLine(frame, (cx - 120, cy), (cx + 140, cy), (255, 180, 0), 6, tipLength=0.22)
    # Tilt amount cue: aim dot into this target ring.
    if "LEFT" in title or "RIGHT" in title:
        target_sign = -1.0 if "LEFT" in title else 1.0
        tx = int(cx + steer_dir * target_sign * (w * 0.22))
        cv2.circle(frame, (tx, cy), 14, (255, 200, 80), 2)
        cv2.putText(frame, "aim here", (tx - 30, cy - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 200, 80), 1, cv2.LINE_AA)

    phase_color = (50, 200, 50) if collecting else (0, 200, 255)
    cv2.putText(frame, title, (20, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.95, phase_color, 2, cv2.LINE_AA)
    cv2.putText(frame, hint, (20, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.68, (245, 245, 245), 2, cv2.LINE_AA)
    cv2.putText(frame, f"starts/ends in {seconds_left:0.1f}s", (20, 96), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (225, 225, 225), 1, cv2.LINE_AA)
    cv2.putText(
        frame,
        f"conf={p.confidence:.2f} raw={p.steer_raw:+.3f} src={p.source}",
        (20, h - 18),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (230, 230, 230),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(frame, "Press q to cancel, r to relock", (w - 290, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (225, 225, 225), 1, cv2.LINE_AA)
    _draw_tracking_overlay(cv2, frame, debug, mirrored=mirror_preview)


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
    vision = VisionTracker(profile.steering.mode, camera=args.camera)
    dbg = DebugLogger(args.debug_log, "calibrate", args.debug_fps, args.debug_width, args.debug_height)
    if dbg.enabled and dbg.session_dir is not None:
        print(f"debug log: {dbg.session_dir}")
    try:
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
        anchor: tuple[int, int] | None = None
        anchor_locked = False

        if use_gui and cv2 is not None:
            try:
                win = "ftms2pad calibrate"
                cv2.namedWindow(win, cv2.WINDOW_NORMAL)
                cv2.resizeWindow(win, 960, 540)
                for title, key, hint in phases:
                    prep_end = asyncio.get_running_loop().time() + prep_seconds
                    while asyncio.get_running_loop().time() < prep_end:
                        p, frame, debug = vision.next_with_frame()
                        if frame is None:
                            await asyncio.sleep(1 / 30)
                            continue
                        h, w = frame.shape[:2]
                        cent = _debug_centroid_px(debug, w, h, mirrored=not args.no_mirror)
                        if cent is not None and key == "neutral" and not anchor_locked:
                            if anchor is None:
                                anchor = cent
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
                            cv2.destroyWindow(win)
                            return 1
                        if keycode == ord("r"):
                            anchor = None
                            anchor_locked = False
                            vision.reset_tracking()
                        await asyncio.sleep(1 / 30)

                    end_at = asyncio.get_running_loop().time() + phase_seconds
                    while asyncio.get_running_loop().time() < end_at:
                        p, frame, debug = vision.next_with_frame()
                        if frame is None:
                            await asyncio.sleep(1 / 30)
                            continue
                        conf_th = _pose_conf_threshold(p.source)
                        pass_anchor = _anchor_gate_pass(
                            p.source, debug, anchor, frame.shape[1], frame.shape[0], mirrored=not args.no_mirror
                        )
                        if p.confidence >= conf_th and pass_anchor:
                            buckets[key].append(p.steer_raw)
                        h, w = frame.shape[:2]
                        cent = _debug_centroid_px(debug, w, h, mirrored=not args.no_mirror)
                        if cent is not None and key == "neutral" and not anchor_locked:
                            if anchor is None:
                                anchor = cent
                            anchor_locked = True

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
                            extra={"phase": title, "phase_key": key, "state": "collect", "pass_anchor": pass_anchor},
                        )
                        cv2.imshow(win, display)
                        keycode = cv2.waitKey(1) & 0xFF
                        if keycode == ord("q"):
                            print("Calibration canceled.")
                            cv2.destroyWindow(win)
                            return 1
                        if keycode == ord("r"):
                            anchor = None
                            anchor_locked = False
                            vision.reset_tracking()
                        await asyncio.sleep(1 / 30)
                cv2.destroyWindow(win)
            except Exception:
                use_gui = False

        if not use_gui:
            phase_frames = max(30, int(float(args.phase_seconds) * 30))
            print("GUI unavailable; using text calibration mode.")
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
        neutral = sum(neutral_samples) / max(len(neutral_samples), 1)
        left_peak = _percentile(left_vals, 0.10) if left_vals else -0.7
        right_peak = _percentile(right_vals, 0.90) if right_vals else 0.7
        corrections: list[str] = []

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

        calib = Calibrator(neutral=neutral, left_peak=left_peak, right_peak=right_peak)
        out = _calibration_path(args.profile)
        save_calibration(out, calib)
        print(f"Saved calibration: {out}")
        print(f"samples neutral={len(neutral_samples)} left={len(left_vals)} right={len(right_vals)}")
        if len(left_vals) < 20 or len(right_vals) < 20:
            print("Warning: low valid samples. Try more light, visible camera (0), or longer phase seconds.")
        if corrections:
            print(f"Calibration correction: {', '.join(corrections)}")
        print(f"neutral={neutral:.4f} left_peak={left_peak:.4f} right_peak={right_peak:.4f}")
        return 0
    finally:
        dbg.close()
        vision.close()


async def _run_loop(args: argparse.Namespace, monitor_only: bool) -> int:
    profile = load_profile(args.profile)
    calib = load_calibration(_calibration_path(args.profile))
    fusion = FusionPipeline(profile, calibrator=calib)
    ftms = FtmsSource(args.bike)
    vision = VisionTracker(profile.steering.mode, camera=args.camera)
    pad = None
    if not monitor_only:
        pad = VirtualGamepad(
            steer_axis=profile.uinput.steer_axis,
            throttle_axis=profile.uinput.throttle_axis,
            invert_throttle=profile.uinput.invert_throttle,
        )
    mode = "monitor" if monitor_only else "run"
    dbg = DebugLogger(args.debug_log, mode, args.debug_fps, args.debug_width, args.debug_height)

    try:
        print(f"camera={vision.camera_idx} bike={args.bike} profile={profile.name}")
        if dbg.enabled and dbg.session_dir is not None:
            print(f"debug log: {dbg.session_dir}")
        if not monitor_only and (pad is None or not pad.enabled):
            print("uinput unavailable. Running monitor-only output.")

        hz = max(20, min(120, args.hz))
        dt = 1.0 / hz
        gui_enabled = monitor_only and not getattr(args, "no_gui", False)
        cv2 = None
        win = None
        anchor: tuple[int, int] | None = None
        anchor_locked = False
        if gui_enabled:
            try:
                import cv2 as _cv2

                cv2 = _cv2
                win = "ftms2pad monitor"
                cv2.namedWindow(win, cv2.WINDOW_NORMAL)
                cv2.resizeWindow(win, 1100, 650)
            except Exception:
                gui_enabled = False

        while True:
            f = await ftms.next()
            if gui_enabled:
                p, frame, debug = vision.next_with_frame()
            else:
                p = vision.next()
                frame = None
                debug = {}
            pass_anchor = _anchor_gate_pass(
                p.source,
                debug,
                anchor,
                frame.shape[1] if frame is not None else None,
                frame.shape[0] if frame is not None else None,
                mirrored=not args.no_mirror,
            )
            pose_ok = p.confidence >= _pose_conf_threshold(p.source) and pass_anchor
            steer = fusion.steer(p.steer_raw, pose_ok=pose_ok)
            throttle = fusion.throttle(f.watts, connected=f.connected)

            if not monitor_only and pad is not None:
                pad.emit(steer=steer, throttle=throttle)

            print(
                f"w={f.watts:6.1f} cad={f.cadence_rpm:5.1f} "
                f"pose={p.confidence:0.2f} steer={steer:+0.3f} thr={throttle:0.3f}",
                end="\r",
                flush=True,
            )

            if gui_enabled and cv2 is not None and frame is not None and win is not None:
                h, w = frame.shape[:2]
                cent = _debug_centroid_px(debug, w, h, mirrored=not args.no_mirror)
                if cent is not None and not anchor_locked:
                    if anchor is None:
                        anchor = cent
                        anchor_locked = True
                _draw_monitor_frame(
                    cv2,
                    frame,
                    p,
                    f,
                    steer,
                    throttle,
                    mirror_preview=not args.no_mirror,
                    debug=debug,
                    anchor=anchor,
                )
                dbg.log(
                    p=p,
                    f=f,
                    steer=steer,
                    throttle=throttle,
                    debug=debug,
                    anchor=anchor,
                    frame=frame,
                    extra={"pose_ok": pose_ok, "pass_anchor": pass_anchor},
                )
                cv2.imshow(win, frame)
                keycode = cv2.waitKey(1) & 0xFF
                if keycode == ord("q"):
                    print("\nStopped.")
                    return 0
                if keycode == ord("r"):
                    anchor = None
                    anchor_locked = False
                    vision.reset_tracking()
            else:
                dbg.log(
                    p=p,
                    f=f,
                    steer=steer,
                    throttle=throttle,
                    debug=debug,
                    anchor=anchor,
                    extra={"pose_ok": pose_ok, "pass_anchor": pass_anchor},
                )
            await asyncio.sleep(dt)
    except KeyboardInterrupt:
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
    common.add_argument("--camera", default="auto", help="camera index or auto")
    common.add_argument("--hz", type=int, default=60, help="main loop frequency")
    common.add_argument("--debug-log", default="", help="directory to write debug bundle")
    common.add_argument("--debug-fps", type=float, default=10.0, help="debug video FPS")
    common.add_argument("--debug-width", type=int, default=640, help="debug video width")
    common.add_argument("--debug-height", type=int, default=360, help="debug video height")

    c = sub.add_parser("run", parents=[common], help="Emit virtual gamepad")
    c.set_defaults(fn=cmd_run)

    c = sub.add_parser("monitor", parents=[common], help="No uinput output, only live stats")
    c.add_argument("--no-gui", action="store_true", help="Use text-only monitor")
    c.add_argument("--no-mirror", action="store_true", help="Do not mirror preview window")
    c.set_defaults(fn=cmd_monitor)

    c = sub.add_parser("calibrate", parents=[common], help="Capture neutral and lean range")
    c.add_argument("--no-gui", action="store_true", help="Use text-only calibration")
    c.add_argument("--no-mirror", action="store_true", help="Do not mirror preview window")
    c.add_argument("--prep-seconds", type=float, default=2.0, help="Countdown seconds before each phase")
    c.add_argument("--phase-seconds", type=float, default=4.0, help="Seconds per calibration phase")
    c.set_defaults(fn=cmd_calibrate)

    return p


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    fn = args.fn
    if inspect.iscoroutinefunction(fn):
        raise SystemExit(asyncio.run(fn(args)))
    raise SystemExit(fn(args))


if __name__ == "__main__":
    main()
