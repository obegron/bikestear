from __future__ import annotations

import csv
import json
from pathlib import Path
from time import monotonic, time


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
        self.samples_fp = None
        self.samples_writer = None
        self.writer = None
        self._cv2 = None
        self._t_last_frame = 0.0
        self._t_last_snapshot = 0.0
        self._last_snapshot_key = ""
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
        (self.session_dir / "keyframes").mkdir(parents=True, exist_ok=True)
        self.events_fp = (self.session_dir / "events.jsonl").open("w", encoding="utf-8", buffering=1)
        self.samples_fp = (self.session_dir / "samples.csv").open("w", encoding="utf-8", newline="")
        self.samples_writer = csv.DictWriter(
            self.samples_fp,
            fieldnames=[
                "t",
                "mode",
                "frame_idx",
                "pose_source",
                "pose_confidence",
                "pose_steer_raw",
                "ftms_connected",
                "ftms_watts",
                "ftms_cadence_rpm",
                "ftms_speed_kph",
                "ftms_resistance_level",
                "control_steer",
                "control_throttle",
                "debug_kind",
                "centroid_x",
                "centroid_y",
                "bbox_x",
                "bbox_y",
                "bbox_w",
                "bbox_h",
                "anchor_x",
                "anchor_y",
                "center_error_x_px",
                "center_error_y_px",
                "pose_ok",
                "pass_anchor",
                "pedal_ready",
                "state",
                "phase_key",
            ],
        )
        self.samples_writer.writeheader()
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

    def _sample_row(self, event: dict[str, object]) -> dict[str, object]:
        pose = event.get("pose", {}) if isinstance(event.get("pose"), dict) else {}
        ftms = event.get("ftms", {}) if isinstance(event.get("ftms"), dict) else {}
        control = event.get("control", {}) if isinstance(event.get("control"), dict) else {}
        debug = event.get("debug", {}) if isinstance(event.get("debug"), dict) else {}
        extra = event.get("extra", {}) if isinstance(event.get("extra"), dict) else {}
        anchor = event.get("anchor", {}) if isinstance(event.get("anchor"), dict) else {}
        centroid = debug.get("centroid")
        bbox = debug.get("bbox")
        cx = int(centroid[0]) if isinstance(centroid, (tuple, list)) and len(centroid) == 2 else ""
        cy = int(centroid[1]) if isinstance(centroid, (tuple, list)) and len(centroid) == 2 else ""
        bx = by = bw = bh = ""
        if isinstance(bbox, (tuple, list)) and len(bbox) == 4:
            bx, by, bw, bh = [int(v) for v in bbox]
        ax = anchor.get("x", "") if isinstance(anchor, dict) else ""
        ay = anchor.get("y", "") if isinstance(anchor, dict) else ""
        err_x = err_y = ""
        if isinstance(ax, (int, float)) and isinstance(cx, int):
            err_x = float(cx) - float(ax)
        if isinstance(ay, (int, float)) and isinstance(cy, int):
            err_y = float(cy) - float(ay)
        return {
            "t": event.get("t", 0.0),
            "mode": event.get("mode", ""),
            "frame_idx": event.get("frame_idx", ""),
            "pose_source": pose.get("source", ""),
            "pose_confidence": pose.get("confidence", 0.0),
            "pose_steer_raw": pose.get("steer_raw", 0.0),
            "ftms_connected": ftms.get("connected", False),
            "ftms_watts": ftms.get("watts", 0.0),
            "ftms_cadence_rpm": ftms.get("cadence_rpm", 0.0),
            "ftms_speed_kph": ftms.get("speed_kph", 0.0),
            "ftms_resistance_level": ftms.get("resistance_level", 0.0),
            "control_steer": control.get("steer", 0.0),
            "control_throttle": control.get("throttle", 0.0),
            "debug_kind": debug.get("kind", ""),
            "centroid_x": cx,
            "centroid_y": cy,
            "bbox_x": bx,
            "bbox_y": by,
            "bbox_w": bw,
            "bbox_h": bh,
            "anchor_x": ax,
            "anchor_y": ay,
            "center_error_x_px": err_x,
            "center_error_y_px": err_y,
            "pose_ok": extra.get("pose_ok", ""),
            "pass_anchor": extra.get("pass_anchor", ""),
            "pedal_ready": extra.get("pedal_ready", ""),
            "state": extra.get("state", ""),
            "phase_key": extra.get("phase_key", ""),
        }

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
                "resistance_level": float(getattr(f, "resistance_level", 0.0)) if f is not None else 0.0,
                "connected": bool(getattr(f, "connected", False)) if f is not None else False,
                "raw_hex": str(getattr(f, "raw_hex", "")) if f is not None else "",
                "control_point_hex": str(getattr(f, "control_point_hex", "")) if f is not None else "",
            },
            "control": {
                "steer": float(steer if steer is not None else 0.0),
                "throttle": float(throttle if throttle is not None else 0.0),
            },
            "debug": debug or {},
            "anchor": {"x": anchor[0], "y": anchor[1]} if anchor is not None else None,
            "extra": extra or {},
            "frame_idx": self.frame_idx if frame is not None else None,
        }
        if self.events_fp is not None:
            self.events_fp.write(json.dumps(event) + "\n")
            self.events_fp.flush()
        if self.samples_writer is not None and self.samples_fp is not None:
            self.samples_writer.writerow(self._sample_row(event))
            self.samples_fp.flush()

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
        self._maybe_write_snapshot(out, extra or {})
        self.frame_idx += 1

    def _maybe_write_snapshot(self, frame, extra: dict[str, object]) -> None:
        if self.session_dir is None or self._cv2 is None:
            return
        now = monotonic()
        state = str(extra.get("state", ""))
        phase = str(extra.get("phase_key", ""))
        key = f"{state}:{phase}"
        should_write = False
        if key != self._last_snapshot_key:
            should_write = True
        elif now - self._t_last_snapshot >= 2.0:
            should_write = True
        if not should_write:
            return
        snapshot_name = f"{self.frame_idx:05d}-{state or 'frame'}"
        if phase:
            snapshot_name += f"-{phase}"
        snapshot_path = self.session_dir / "keyframes" / f"{snapshot_name}.jpg"
        try:
            self._cv2.imwrite(str(snapshot_path), frame)
            self._t_last_snapshot = now
            self._last_snapshot_key = key
        except Exception:
            pass

    def close(self) -> None:
        if self.events_fp is not None:
            self.events_fp.close()
            self.events_fp = None
        if self.samples_fp is not None:
            self.samples_fp.close()
            self.samples_fp = None
            self.samples_writer = None
        if self.writer is not None:
            self.writer.release()
            self.writer = None
