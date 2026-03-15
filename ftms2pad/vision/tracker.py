from __future__ import annotations

from dataclasses import dataclass
from math import atan2
from statistics import median
import os
from pathlib import Path
from time import monotonic
from typing import Any

os.environ.setdefault("OPENCV_LOG_LEVEL", "ERROR")
import cv2

from ftms2pad.types import PoseSample

try:
    import mediapipe as mp
except Exception:  # pragma: no cover
    mp = None

LS = 11
RS = 12
LH = 23
RH = 24


@dataclass(slots=True)
class CameraCandidate:
    idx: int
    score: float
    mode: str


def _video_nodes() -> list[int]:
    nodes: list[int] = []
    for path in sorted(Path("/dev").glob("video*")):
        suffix = path.name.replace("video", "", 1)
        if suffix.isdigit():
            nodes.append(int(suffix))
    return nodes


def camera_name(index: int) -> str:
    name_path = Path(f"/sys/class/video4linux/video{index}/name")
    try:
        return name_path.read_text().strip()
    except Exception:
        return f"video{index}"


def list_cameras() -> list[int]:
    found: list[int] = []
    for idx in _video_nodes():
        cap = cv2.VideoCapture(idx, cv2.CAP_V4L2)
        ok, _ = cap.read()
        cap.release()
        if ok:
            found.append(idx)
    return found


def _create_pose() -> Any:
    if mp is None:
        return None
    # Classic API path.
    try:
        if hasattr(mp, "solutions") and hasattr(mp.solutions, "pose"):
            return mp.solutions.pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
    except Exception:
        pass
    # Newer package layout fallback.
    try:
        from mediapipe.python.solutions.pose import Pose  # type: ignore

        return Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
    except Exception:
        return None


class VisionTracker:
    def __init__(self, steering_mode: str, camera: str = "auto", width: int = 640, height: int = 360) -> None:
        self.steering_mode = steering_mode
        self.camera_idx = self._pick_camera(camera)
        self._camera_label = camera_name(self.camera_idx).lower()
        self._is_ir_camera = "ir" in self._camera_label
        self._use_bike_mask = self.steering_mode == "bike_relative_torso"
        self._vision_width = max(160, int(width))
        self._vision_height = max(120, int(height))
        self.cap = cv2.VideoCapture(self.camera_idx)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._vision_width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._vision_height)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        self._t = 0.0
        self._pose = None if self._use_bike_mask else _create_pose()
        self._backend = "bike_mask" if self._use_bike_mask else ("mediapipe" if self._pose is not None else "blob")
        self._bg = cv2.createBackgroundSubtractorMOG2(history=120, varThreshold=32, detectShadows=False)
        self._bike_bg = None
        self._bike_last_bbox: tuple[int, int, int, int] | None = None
        self._bike_last_ts = 0.0
        self._bike_anchor_x: float | None = None
        self._bike_anchor_samples: list[float] = []
        self._bike_warmup_until = monotonic() + 1.0 if self._use_bike_mask else 0.0
        self._hog = cv2.HOGDescriptor()
        self._hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
        self._face = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        self._face_profile = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_profileface.xml")
        self._face_last_bbox: tuple[int, int, int, int] | None = None
        self._face_last_ts: float = 0.0
        self._face_last_hard_bbox: tuple[int, int, int, int] | None = None
        self._face_last_hard_ts: float = 0.0
        self._face_template = None
        self._tracker = None
        self._last_debug: dict[str, object] = {}

    def _create_tracker(self):
        try:
            if hasattr(cv2, "legacy"):
                legacy = cv2.legacy
                for name in ("TrackerCSRT_create", "TrackerKCF_create", "TrackerMOSSE_create"):
                    fn = getattr(legacy, name, None)
                    if callable(fn):
                        return fn()
            for name in ("TrackerCSRT_create", "TrackerKCF_create", "TrackerMOSSE_create"):
                fn = getattr(cv2, name, None)
                if callable(fn):
                    return fn()
        except Exception:
            return None
        return None

    def _tracker_init(self, frame, bbox: tuple[int, int, int, int]) -> None:
        tracker = self._create_tracker()
        if tracker is None:
            self._tracker = None
            return
        try:
            ok = tracker.init(frame, tuple(float(v) for v in bbox))
            self._tracker = tracker if ok is not False else None
        except Exception:
            self._tracker = None

    def _tracker_update_bbox(self, frame) -> tuple[int, int, int, int] | None:
        if self._tracker is None:
            return None
        try:
            ok, box = self._tracker.update(frame)
            if not ok:
                self._tracker = None
                return None
            x, y, w, h = [int(v) for v in box]
            if w <= 0 or h <= 0:
                self._tracker = None
                return None
            return x, y, w, h
        except Exception:
            self._tracker = None
            return None

    def _pick_camera(self, camera: str) -> int:
        if camera != "auto":
            return int(camera)
        cameras = list_cameras()
        if not cameras:
            return 0
        if mp is None:
            return cameras[0]

        candidates: list[CameraCandidate] = []
        for idx in cameras:
            cap = cv2.VideoCapture(idx)
            if not cap.isOpened():
                continue
            pose = _create_pose()
            if pose is None:
                cap.release()
                return cameras[0]
            total = 0.0
            frames = 0
            gray_like = 0
            for _ in range(8):
                ok, frame = cap.read()
                if not ok:
                    continue
                frames += 1
                b, g, r = cv2.split(frame)
                if (abs((b - g)).mean() + abs((g - r)).mean()) < 4.0:
                    gray_like += 1
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                res = pose.process(rgb)
                if res.pose_landmarks:
                    vis = [lm.visibility for lm in res.pose_landmarks.landmark]
                    total += sum(vis) / len(vis)
            pose.close()
            cap.release()
            if frames == 0:
                continue
            color_bonus = 0.1 if gray_like / frames < 0.8 else 0.0
            score = total / max(frames, 1) + color_bonus
            mode = "visible" if color_bonus > 0 else "ir_or_gray"
            candidates.append(CameraCandidate(idx=idx, score=score, mode=mode))

        if not candidates:
            return cameras[0]
        candidates.sort(key=lambda c: c.score, reverse=True)
        return candidates[0].idx

    def _torso_combo(self, landmarks) -> tuple[float, float]:
        ls = landmarks[LS]
        rs = landmarks[RS]
        lh = landmarks[LH]
        rh = landmarks[RH]

        roll = atan2(ls.y - rs.y, ls.x - rs.x)
        roll_norm = max(-1.0, min(1.0, roll / 0.6))

        mid_sh_x = (ls.x + rs.x) * 0.5
        mid_hip_x = (lh.x + rh.x) * 0.5
        shoulder_width = max(abs(ls.x - rs.x), 1e-4)
        shift = (mid_hip_x - mid_sh_x) / shoulder_width
        shift_norm = max(-1.0, min(1.0, shift * 0.7))

        raw = 0.6 * roll_norm + 0.4 * shift_norm
        conf = min(ls.visibility, rs.visibility, lh.visibility, rh.visibility)
        return raw, conf

    def _bike_relative_torso_lean(self, frame) -> tuple[float, float, dict[str, object]]:
        h, w = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (7, 7), 0)

        if self._bike_bg is None:
            self._bike_bg = gray.astype("float32")
            self._last_debug = {"kind": "bike_mask", "warmup": True}
            return 0.0, 0.0, {"kind": "bike_mask", "warmup": True}

        diff = cv2.absdiff(gray, cv2.convertScaleAbs(self._bike_bg))
        _, mask = cv2.threshold(diff, 18, 255, cv2.THRESH_BINARY)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.dilate(mask, kernel, iterations=1)

        body_top = int(h * 0.12)
        body_bottom = int(h * 0.82)
        torso = mask[body_top:body_bottom, :]
        roi_x0 = 0
        roi_y0 = body_top
        roi_x1 = w
        roi_y1 = body_bottom
        if self._bike_last_bbox is not None and (monotonic() - self._bike_last_ts) < 1.2:
            lx, ly, lw, lh = self._bike_last_bbox
            pad_x = max(int(lw * 1.2), int(w * 0.18))
            pad_y = max(int(lh * 0.8), int(h * 0.12))
            roi_x0 = max(0, lx - pad_x)
            roi_x1 = min(w, lx + lw + pad_x)
            roi_y0 = max(body_top, ly - pad_y)
            roi_y1 = min(body_bottom, ly + lh + pad_y)
            torso = mask[roi_y0:roi_y1, roi_x0:roi_x1]

        contours, _ = cv2.findContours(torso, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        best = None
        best_score = None
        best_area = 0.0
        for contour in contours:
            area = float(cv2.contourArea(contour))
            if area < (w * h) * 0.008:
                continue
            x, y, bw, bh = cv2.boundingRect(contour)
            x += roi_x0
            y += roi_y0
            m = cv2.moments(contour)
            if abs(m["m00"]) > 1e-5:
                cx = roi_x0 + (float(m["m10"]) / float(m["m00"]))
                cy = roi_y0 + (float(m["m01"]) / float(m["m00"]))
            else:
                cx = x + bw * 0.5
                cy = y + bh * 0.5
            if bw > int(w * 0.72) or bh > int(h * 0.72):
                continue
            if cy < h * 0.18 or cy > h * 0.78:
                continue
            continuity = 0.0
            if self._bike_last_bbox is not None:
                lx, ly, lw, lh = self._bike_last_bbox
                lcx = lx + lw * 0.5
                lcy = ly + lh * 0.5
                dist = ((cx - lcx) ** 2 + (cy - lcy) ** 2) ** 0.5
                continuity = 1.0 - min(1.0, dist / max(1.0, w * 0.35))
            anchor_bias = 0.0
            if self._bike_anchor_x is not None:
                anchor_bias = 1.0 - min(1.0, abs(cx - self._bike_anchor_x) / max(1.0, w * 0.35))
            area_score = min(1.0, area / ((w * h) * 0.09))
            score = (0.45 * area_score) + (0.40 * continuity) + (0.15 * anchor_bias)
            if best_score is None or score > best_score:
                best_score = score
                best_area = area
                best = (x, y, bw, bh, cx, cy, contour)

        # Slowly adapt background, excluding detected torso so the rider remains foreground.
        learn_mask = cv2.bitwise_not(mask)
        cv2.accumulateWeighted(gray, self._bike_bg, 0.015, mask=learn_mask)

        if best is None:
            if self._bike_last_bbox is not None and (monotonic() - self._bike_last_ts) < 0.45:
                x, y, bw, bh = self._bike_last_bbox
                cx = x + bw * 0.5
                anchor_x = self._bike_anchor_x if self._bike_anchor_x is not None else (w * 0.5)
                raw = ((anchor_x - cx) / max(w * 0.22, 1.0)) * 1.4
                debug = {
                    "kind": "bike_mask",
                    "bbox": (int(x), int(y), int(bw), int(bh)),
                    "centroid": (int(cx), int(y + bh * 0.5)),
                    "held": True,
                }
                return max(-1.0, min(1.0, raw)), 0.18, debug
            return 0.0, 0.0, {"kind": "bike_mask", "miss": True}

        x, y, bw, bh, cx, cy, contour = best
        # Shoulder/arm motion can pull the full-contour centroid sideways.
        # Recompute x from the lower torso slice so steering follows body mass better.
        try:
            contour_local = contour.copy()
            contour_local[:, 0, 0] = contour_local[:, 0, 0] + roi_x0
            contour_local[:, 0, 1] = contour_local[:, 0, 1] + roi_y0
            local_mask = gray.copy()
            local_mask[:] = 0
            cv2.drawContours(local_mask, [contour_local], -1, 255, thickness=-1)
            lower_y = int(y + bh * 0.42)
            lower_mask = local_mask[lower_y : y + bh, x : x + bw]
            if lower_mask.size > 0:
                m2 = cv2.moments(lower_mask, binaryImage=True)
                if abs(m2["m00"]) > 1e-5:
                    cx = x + (float(m2["m10"]) / float(m2["m00"]))
        except Exception:
            pass
        if monotonic() < self._bike_warmup_until:
            return 0.0, 0.0, {
                "kind": "bike_mask",
                "bbox": (int(x), int(y), int(bw), int(bh)),
                "centroid": (int(cx), int(cy)),
                "warmup": True,
            }
        self._bike_last_bbox = (x, y, bw, bh)
        self._bike_last_ts = monotonic()
        if self._bike_anchor_x is None:
            self._bike_anchor_samples.append(float(cx))
            if len(self._bike_anchor_samples) > 15:
                self._bike_anchor_samples = self._bike_anchor_samples[-15:]
            if len(self._bike_anchor_samples) < 6:
                return 0.0, 0.0, {
                    "kind": "bike_mask",
                    "bbox": (int(x), int(y), int(bw), int(bh)),
                    "centroid": (int(cx), int(cy)),
                    "warmup": True,
                    "anchor_pending": True,
                }
            self._bike_anchor_x = float(median(self._bike_anchor_samples))
        # Track torso shift relative to learned bike/rider neutral, not frame center.
        anchor_x = self._bike_anchor_x
        raw = ((anchor_x - cx) / max(w * 0.16, 1.0)) * 1.8
        conf = max(0.2, min(1.0, best_area / ((w * h) * 0.09)))
        debug = {
            "kind": "bike_mask",
            "bbox": (int(x), int(y), int(bw), int(bh)),
            "centroid": (int(cx), int(cy)),
            "area": best_area,
            "anchor_x": float(anchor_x),
            "roi": (int(roi_x0), int(roi_y0), int(max(1, roi_x1 - roi_x0)), int(max(1, roi_y1 - roi_y0))),
        }
        return max(-1.0, min(1.0, raw)), conf, debug

    def _face_lean(self, frame) -> tuple[float, float, dict[str, object]] | None:
        if self._face is None or self._face.empty():
            return None
        h, w = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        roi_h = max(1, int(h * 0.85))
        roi = gray[:roi_h, :]
        search_x0 = 0
        search_x1 = w
        if self._face_last_bbox is not None and (monotonic() - self._face_last_ts) < 3.0:
            lx, ly, lw, lh = self._face_last_bbox
            lc = lx + lw * 0.5
            half = max(int(w * 0.24), int(lw * 2.0))
            search_x0 = max(0, int(lc) - half)
            search_x1 = min(w, int(lc) + half)
            min_span = int(w * 0.42)
            if (search_x1 - search_x0) < min_span:
                cx = (search_x0 + search_x1) // 2
                search_x0 = max(0, cx - min_span // 2)
                search_x1 = min(w, cx + min_span // 2)
        roi_search = roi[:, search_x0:search_x1]
        frontal = self._face.detectMultiScale(
            roi_search,
            scaleFactor=1.1,
            minNeighbors=7,
            minSize=(max(36, w // 20), max(36, h // 20)),
        )
        candidates: list[tuple[int, int, int, int, str]] = []
        for (x_local, y, fw, fh) in frontal:
            x = int(x_local) + search_x0
            cy = y + fh * 0.5
            cx = x + fw * 0.5
            if cy < h * 0.18:
                continue
            if cy > h * 0.75:
                continue
            if fw * fh < (w * h) * 0.002:
                continue
            if not (w * 0.12 <= cx <= w * 0.88):
                continue
            ar = fw / max(1.0, float(fh))
            if not (0.65 <= ar <= 1.55):
                continue
            candidates.append((int(x), int(y), int(fw), int(fh), "frontal"))

        if self._face_profile is not None and not self._face_profile.empty():
            prof_l = self._face_profile.detectMultiScale(
                roi_search,
                scaleFactor=1.1,
                minNeighbors=4,
                minSize=(max(30, w // 24), max(30, h // 24)),
            )
            for (x_local, y, fw, fh) in prof_l:
                x = int(x_local) + search_x0
                cy = y + fh * 0.5
                cx = x + fw * 0.5
                if cy < h * 0.18:
                    continue
                if cy > h * 0.75:
                    continue
                if fw * fh < (w * h) * 0.0015:
                    continue
                if not (w * 0.10 <= cx <= w * 0.90):
                    continue
                ar = fw / max(1.0, float(fh))
                if not (0.45 <= ar <= 1.65):
                    continue
                candidates.append((int(x), int(y), int(fw), int(fh), "profile_l"))

            roi_flip = cv2.flip(roi_search, 1)
            prof_r_flip = self._face_profile.detectMultiScale(
                roi_flip,
                scaleFactor=1.1,
                minNeighbors=4,
                minSize=(max(30, w // 24), max(30, h // 24)),
            )
            for (xf, y, fw, fh) in prof_r_flip:
                roi_w = max(1, search_x1 - search_x0)
                x = search_x0 + (roi_w - (int(xf) + int(fw)))
                cy = y + fh * 0.5
                cx = x + fw * 0.5
                if cy < h * 0.18:
                    continue
                if cy > h * 0.75:
                    continue
                if fw * fh < (w * h) * 0.0015:
                    continue
                if not (w * 0.10 <= cx <= w * 0.90):
                    continue
                ar = fw / max(1.0, float(fh))
                if not (0.45 <= ar <= 1.65):
                    continue
                candidates.append((int(x), int(y), int(fw), int(fh), "profile_r"))

        if not candidates:
            # Relaxed local re-detection around last known box helps re-acquire during turns.
            if self._face_last_bbox is not None and (monotonic() - self._face_last_ts) < 2.5:
                x0, y0, w0, h0 = self._face_last_bbox
                rx0 = max(0, x0 - int(w0 * 1.8))
                ry0 = max(0, y0 - int(h0 * 1.6))
                rx1 = min(w, x0 + w0 + int(w0 * 1.8))
                ry1 = min(roi_h, y0 + h0 + int(h0 * 1.6))
                if rx1 - rx0 >= max(24, w0 // 2) and ry1 - ry0 >= max(24, h0 // 2):
                    local = roi[ry0:ry1, rx0:rx1]
                    try:
                        rel = self._face.detectMultiScale(
                            local,
                            scaleFactor=1.05,
                            minNeighbors=4,
                            minSize=(max(28, w // 26), max(28, h // 26)),
                        )
                    except Exception:
                        rel = []
                    for (lx, ly, fw, fh) in rel:
                        x = rx0 + int(lx)
                        y = ry0 + int(ly)
                        cy = y + fh * 0.5
                        cx = x + fw * 0.5
                        if cy < h * 0.16 or cy > h * 0.78:
                            continue
                        if fw * fh < (w * h) * 0.0014:
                            continue
                        if not (w * 0.08 <= cx <= w * 0.92):
                            continue
                        ar = fw / max(1.0, float(fh))
                        if not (0.55 <= ar <= 1.7):
                            continue
                        candidates.append((int(x), int(y), int(fw), int(fh), "frontal_relaxed"))

        if not candidates:
            # First try persistent tracker to bridge short face-detector misses while turning.
            tracked_box = self._tracker_update_bbox(frame)
            if tracked_box is not None and self._face_last_bbox is not None:
                x, y, fw, fh = tracked_box
                px, py, pw, ph = self._face_last_bbox
                cx_prev = px + pw * 0.5
                cy_prev = py + ph * 0.5
                cx_new = x + fw * 0.5
                cy_new = y + fh * 0.5
                dx = abs(cx_new - cx_prev)
                dy = abs(cy_new - cy_prev)
                area_ratio = (fw * fh) / max(1.0, float(pw * ph))
                jump_ok = dx <= max(w * 0.12, pw * 1.0) and dy <= max(h * 0.10, ph * 1.0)
                area_ok = 0.55 <= area_ratio <= 1.8
                top_ok = int(h * 0.18) <= cy_new <= int(h * 0.75)
                center_ok = (w * 0.12) <= cx_new <= (w * 0.88)
                min_area_ok = (fw * fh) >= (w * h) * 0.0015
                hard_ok = False
                hard_window_s = 1.2 if self._is_ir_camera else 2.0
                if self._face_last_hard_bbox is not None and (monotonic() - self._face_last_hard_ts) < hard_window_s:
                    hx, hy, hw, hh = self._face_last_hard_bbox
                    hcx = hx + hw * 0.5
                    hcy = hy + hh * 0.5
                    hard_dx = abs(cx_new - hcx)
                    hard_dy = abs(cy_new - hcy)
                    hard_dx_lim = max(w * (0.20 if self._is_ir_camera else 0.28), hw * (1.8 if self._is_ir_camera else 2.3))
                    hard_dy_lim = max(h * (0.16 if self._is_ir_camera else 0.22), hh * (1.8 if self._is_ir_camera else 2.3))
                    hard_ok = hard_dx <= hard_dx_lim and hard_dy <= hard_dy_lim
                if jump_ok and area_ok and top_ok and center_ok and min_area_ok and hard_ok:
                    self._face_last_bbox = (x, y, fw, fh)
                    self._face_last_ts = monotonic()
                    cx = x + fw * 0.5
                    raw = (0.5 - (cx / max(w, 1))) * 1.6
                    debug = {
                        "kind": "face",
                        "bbox": (int(x), int(y), int(fw), int(fh)),
                        "centroid": (int(cx), int(y + fh * 0.5)),
                        "detector": "tracker",
                    }
                    return max(-1.0, min(1.0, raw)), 0.21, debug
                self._tracker = None

            # Try template tracking around last bbox before declaring miss.
            tracked = None
            if self._face_last_bbox is not None and self._face_template is not None:
                x0, y0, w0, h0 = self._face_last_bbox
                sx0 = max(0, x0 - int(w0 * 0.9))
                sy0 = max(0, y0 - int(h0 * 0.9))
                sx1 = min(w, x0 + w0 + int(w0 * 0.9))
                sy1 = min(roi_h, y0 + h0 + int(h0 * 0.9))
                if sx1 - sx0 >= w0 and sy1 - sy0 >= h0:
                    search = roi[sy0:sy1, sx0:sx1]
                    tmpl = self._face_template
                    if tmpl is not None and search.shape[0] >= tmpl.shape[0] and search.shape[1] >= tmpl.shape[1]:
                        try:
                            res = cv2.matchTemplate(search, tmpl, cv2.TM_CCOEFF_NORMED)
                            _, max_val, _, max_loc = cv2.minMaxLoc(res)
                            if max_val >= 0.72:
                                tx = sx0 + int(max_loc[0])
                                ty = sy0 + int(max_loc[1])
                                tw, th = int(tmpl.shape[1]), int(tmpl.shape[0])
                                cx_prev = x0 + w0 * 0.5
                                cy_prev = y0 + h0 * 0.5
                                cx_new = tx + tw * 0.5
                                cy_new = ty + th * 0.5
                                dx = abs(cx_new - cx_prev)
                                dy = abs(cy_new - cy_prev)
                                top_ok = ty >= int(h * 0.08)
                                max_tmpl_jump_x = max(w * 0.10, w0 * 0.7)
                                max_tmpl_jump_y = max(h * 0.08, h0 * 0.65)
                                jump_ok = dx <= max_tmpl_jump_x and dy <= max_tmpl_jump_y
                                if top_ok and jump_ok:
                                    tracked = (tx, ty, tw, th, float(max_val))
                        except Exception:
                            tracked = None
            if tracked is not None:
                x, y, fw, fh, score = tracked
                self._face_last_bbox = (x, y, fw, fh)
                self._face_last_ts = monotonic()
                cx = x + fw * 0.5
                raw = (0.5 - (cx / max(w, 1))) * 1.6
                debug = {
                    "kind": "face",
                    "bbox": (int(x), int(y), int(fw), int(fh)),
                    "centroid": (int(cx), int(y + fh * 0.5)),
                    "detector": "template",
                    "score": score,
                }
                return max(-1.0, min(1.0, raw)), 0.22, debug
            # Short hold to reduce flicker on brief misses.
            if self._face_last_bbox is not None and (monotonic() - self._face_last_ts) < 1.2:
                x, y, fw, fh = self._face_last_bbox
                cx = x + fw * 0.5
                raw = (0.5 - (cx / max(w, 1))) * 1.6
                debug = {
                    "kind": "face",
                    "bbox": (int(x), int(y), int(fw), int(fh)),
                    "centroid": (int(cx), int(y + fh * 0.5)),
                    "held": True,
                }
                return max(-1.0, min(1.0, raw)), 0.2, debug
            return None

        prev_cx = None
        prev_cy = None
        if self._face_last_bbox is not None:
            px, py, pw, ph = self._face_last_bbox
            prev_cx = px + pw * 0.5
            prev_cy = py + ph * 0.5

        def _score(rect: tuple[int, int, int, int, str]) -> float:
            x, y, fw, fh, detector = rect
            cx = x + fw * 0.5
            cy = y + fh * 0.5
            area_score = min(1.0, (fw * fh) / ((w * h) * 0.06))
            center_score = 1.0 - min(1.0, abs(cx - (w * 0.5)) / (w * 0.5))
            upper_score = 1.0 - min(1.0, cy / max(1.0, h * 0.8))
            prev_score = 0.0
            if prev_cx is not None and prev_cy is not None:
                dist = ((cx - prev_cx) ** 2 + (cy - prev_cy) ** 2) ** 0.5
                prev_score = 1.0 - min(1.0, dist / (w * 0.35))
            det_bonus = 0.08 if detector == "frontal" else 0.0
            return (0.42 * area_score) + (0.24 * center_score) + (0.2 * prev_score) + (0.1 * upper_score) + det_bonus

        chosen: tuple[int, int, int, int, str] | None = None
        if prev_cx is not None and prev_cy is not None and self._face_last_bbox is not None:
            _, _, prev_w, prev_h = self._face_last_bbox
            prev_area = max(1.0, float(prev_w * prev_h))
            max_jump_x = max(w * 0.09, prev_w * 1.0)
            max_jump_y = max(h * 0.08, prev_h * 0.9)
            near = []
            far = []
            for rect in candidates:
                x, y, fw, fh, det = rect
                cx = x + fw * 0.5
                cy = y + fh * 0.5
                area_ratio = (fw * fh) / prev_area
                # Reject abrupt tiny frontal detections; these are usually wall/carpet false positives.
                if det == "frontal" and area_ratio < 0.6:
                    continue
                if abs(cx - prev_cx) <= max_jump_x and abs(cy - prev_cy) <= max_jump_y:
                    near.append(rect)
                else:
                    far.append(rect)

            if near:
                chosen = max(near, key=_score)
            else:
                # Ignore far-away re-detections while we still have a recent lock.
                if (monotonic() - self._face_last_ts) < 1.2:
                    candidates = []
                else:
                    chosen = max(far, key=_score) if far else None
        else:
            chosen = max(candidates, key=_score)

        if chosen is None:
            if self._face_last_bbox is not None and (monotonic() - self._face_last_ts) < 0.8:
                x, y, fw, fh = self._face_last_bbox
                cx = x + fw * 0.5
                raw = (0.5 - (cx / max(w, 1))) * 1.6
                debug = {
                    "kind": "face",
                    "bbox": (int(x), int(y), int(fw), int(fh)),
                    "centroid": (int(cx), int(y + fh * 0.5)),
                    "held": True,
                    "reason": "rejected_far_jump",
                }
                return max(-1.0, min(1.0, raw)), 0.2, debug
            return None

        x, y, fw, fh, detector = chosen
        self._face_last_bbox = (x, y, fw, fh)
        self._face_last_ts = monotonic()
        if detector in ("frontal", "frontal_relaxed", "profile_l", "profile_r"):
            self._face_last_hard_bbox = (x, y, fw, fh)
            self._face_last_hard_ts = self._face_last_ts
            self._tracker_init(frame, (x, y, fw, fh))
        if detector == "frontal":
            try:
                patch = roi[y : y + fh, x : x + fw]
                if patch.size > 0:
                    self._face_template = patch.copy()
            except Exception:
                pass
        cx = x + fw * 0.5
        raw = (0.5 - (cx / max(w, 1))) * 1.6
        area = float(fw * fh)
        conf = max(0.2, min(1.0, area / ((w * h) * 0.12)))
        debug = {
            "kind": "face",
            "bbox": (int(x), int(y), int(fw), int(fh)),
            "centroid": (int(cx), int(y + fh * 0.5)),
            "detector": detector,
        }
        return max(-1.0, min(1.0, raw)), conf, debug

    def _hog_lean(self, frame) -> tuple[float, float, dict[str, object]] | None:
        h, w = frame.shape[:2]
        scale = 320.0 / max(w, 1)
        small = cv2.resize(frame, (320, max(180, int(h * scale))))
        rects, weights = self._hog.detectMultiScale(
            small, winStride=(8, 8), padding=(8, 8), scale=1.05
        )
        if len(rects) == 0:
            return None
        best_i = max(range(len(rects)), key=lambda i: rects[i][2] * rects[i][3])
        x, y, rw, rh = rects[best_i]
        cx = (x + rw * 0.5) / max(small.shape[1], 1)
        raw = (0.5 - cx) * 2.0
        weight = float(weights[best_i]) if len(weights) > best_i else 0.0
        conf = max(0.2, min(1.0, (weight + 1.0) / 4.0))
        debug = {
            "kind": "hog",
            "bbox_norm": (
                x / max(small.shape[1], 1),
                y / max(small.shape[0], 1),
                rw / max(small.shape[1], 1),
                rh / max(small.shape[0], 1),
            ),
            "score": weight,
        }
        return max(-1.0, min(1.0, raw)), conf, debug

    def _blob_lean(self, frame) -> tuple[float, float, dict[str, object]]:
        h, w = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        fg = self._bg.apply(gray)
        _, fg = cv2.threshold(fg, 200, 255, cv2.THRESH_BINARY)
        fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
        fg = cv2.morphologyEx(fg, cv2.MORPH_DILATE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))

        contours, _ = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return 0.0, 0.0, {"kind": "blob", "miss": True}
        c = max(contours, key=cv2.contourArea)
        area = float(cv2.contourArea(c))
        if area > (w * h) * 0.55:
            return 0.0, 0.0, {"kind": "blob", "miss": True, "area": area, "reason": "too_large"}
        min_area = (w * h) * 0.015
        if area < min_area:
            return 0.0, 0.0, {"kind": "blob", "miss": True, "area": area}

        x, y, cw, ch = cv2.boundingRect(c)
        cx = x + cw * 0.5
        raw = (0.5 - (cx / max(w, 1))) * 2.0
        conf = min(1.0, area / ((w * h) * 0.25))
        debug = {
            "kind": "blob",
            "bbox": (int(x), int(y), int(cw), int(ch)),
            "centroid": (int(cx), int(y + ch * 0.5)),
            "area": area,
        }
        return max(-1.0, min(1.0, raw)), conf, debug

    def _sample_from_frame(self, frame) -> PoseSample:
        if self._use_bike_mask:
            raw, conf, debug = self._bike_relative_torso_lean(frame)
            self._last_debug = debug
            return PoseSample(steer_raw=raw, confidence=conf, source="camera-bike", ts=monotonic())
        if self._pose is None:
            face = self._face_lean(frame)
            if face is not None:
                raw, conf, debug = face
                self._last_debug = debug
                return PoseSample(steer_raw=raw, confidence=conf, source="camera-face", ts=monotonic())
            hog = self._hog_lean(frame)
            if hog is not None:
                raw, conf, debug = hog
                self._last_debug = debug
                return PoseSample(steer_raw=raw, confidence=conf, source="camera-hog", ts=monotonic())
            # In face/HOG fallback mode, motion-blob steering is too noisy and can cause random jumps.
            # Fail safe to neutral until a face/hog detection is reacquired.
            self._last_debug = {"kind": "face", "miss": True, "fallback": "neutral"}
            return PoseSample(steer_raw=0.0, confidence=0.0, source="camera-face", ts=monotonic())

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = self._pose.process(rgb)
        if not res.pose_landmarks:
            self._last_debug = {"kind": "mediapipe", "miss": True}
            return PoseSample(steer_raw=0.0, confidence=0.0, source="camera", ts=monotonic())

        lms = res.pose_landmarks.landmark
        if self.steering_mode in ("torso_combo", "shoulder_roll"):
            raw, conf = self._torso_combo(lms)
            if self.steering_mode == "shoulder_roll":
                raw *= 1.2
            pts = []
            for idx in (LS, RS, LH, RH):
                lm = lms[idx]
                pts.append((float(lm.x), float(lm.y)))
            self._last_debug = {"kind": "mediapipe", "points_norm": pts}
            return PoseSample(steer_raw=raw, confidence=conf, source="camera", ts=monotonic())

        nose = lms[0]
        self._last_debug = {"kind": "mediapipe", "points_norm": [(float(nose.x), float(nose.y))]}
        raw = (0.5 - nose.x) * 1.8
        return PoseSample(steer_raw=max(-1.0, min(1.0, raw)), confidence=nose.visibility, source="camera", ts=monotonic())

    def next_with_frame(self):
        ok, frame = self.cap.read()
        if not ok:
            return PoseSample(steer_raw=0.0, confidence=0.0, source="camera", ts=monotonic()), None, {}
        return self._sample_from_frame(frame), frame, dict(self._last_debug)

    def next(self) -> PoseSample:
        ok, frame = self.cap.read()
        if not ok:
            return PoseSample(steer_raw=0.0, confidence=0.0, source="camera", ts=monotonic())
        return self._sample_from_frame(frame)

    def close(self) -> None:
        if self.cap:
            self.cap.release()
        if self._pose:
            self._pose.close()

    def reset_tracking(self) -> None:
        self._face_last_bbox = None
        self._face_last_ts = 0.0
        self._face_last_hard_bbox = None
        self._face_last_hard_ts = 0.0
        self._face_template = None
        self._tracker = None
        self._bike_bg = None
        self._bike_last_bbox = None
        self._bike_last_ts = 0.0
        self._bike_anchor_x = None
        self._bike_anchor_samples = []
        self._bike_warmup_until = monotonic() + 1.0 if self._use_bike_mask else 0.0
