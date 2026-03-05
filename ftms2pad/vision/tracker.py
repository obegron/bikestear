from __future__ import annotations

from dataclasses import dataclass
from math import atan2
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
    def __init__(self, steering_mode: str, camera: str = "auto") -> None:
        self.steering_mode = steering_mode
        self.camera_idx = self._pick_camera(camera)
        self.cap = cv2.VideoCapture(self.camera_idx)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        self._t = 0.0
        self._pose = _create_pose()
        self._backend = "mediapipe" if self._pose is not None else "blob"
        self._bg = cv2.createBackgroundSubtractorMOG2(history=120, varThreshold=32, detectShadows=False)
        self._hog = cv2.HOGDescriptor()
        self._hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
        self._face = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        self._face_profile = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_profileface.xml")
        self._face_last_bbox: tuple[int, int, int, int] | None = None
        self._face_last_ts: float = 0.0
        self._face_template = None
        self._last_debug: dict[str, object] = {}

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

    def _face_lean(self, frame) -> tuple[float, float, dict[str, object]] | None:
        if self._face is None or self._face.empty():
            return None
        h, w = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        roi_h = max(1, int(h * 0.85))
        roi = gray[:roi_h, :]
        frontal = self._face.detectMultiScale(
            roi,
            scaleFactor=1.1,
            minNeighbors=7,
            minSize=(max(36, w // 20), max(36, h // 20)),
        )
        candidates: list[tuple[int, int, int, int, str]] = []
        for (x, y, fw, fh) in frontal:
            cy = y + fh * 0.5
            cx = x + fw * 0.5
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
                roi,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(max(30, w // 24), max(30, h // 24)),
            )
            for (x, y, fw, fh) in prof_l:
                cy = y + fh * 0.5
                cx = x + fw * 0.5
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

            roi_flip = cv2.flip(roi, 1)
            prof_r_flip = self._face_profile.detectMultiScale(
                roi_flip,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(max(30, w // 24), max(30, h // 24)),
            )
            for (xf, y, fw, fh) in prof_r_flip:
                x = w - (int(xf) + int(fw))
                cy = y + fh * 0.5
                cx = x + fw * 0.5
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
            # Try template tracking around last bbox before declaring miss.
            tracked = None
            if self._face_last_bbox is not None and self._face_template is not None:
                x0, y0, w0, h0 = self._face_last_bbox
                sx0 = max(0, x0 - int(w0 * 0.7))
                sy0 = max(0, y0 - int(h0 * 0.7))
                sx1 = min(w, x0 + w0 + int(w0 * 0.7))
                sy1 = min(roi_h, y0 + h0 + int(h0 * 0.7))
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
                                jump_ok = dx <= max(w * 0.11, w0 * 1.2) and dy <= max(h * 0.10, h0 * 1.2)
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
            if self._face_last_bbox is not None and (monotonic() - self._face_last_ts) < 2.2:
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
            if self._face_last_bbox is not None and (monotonic() - self._face_last_ts) < 1.4:
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
            raw, conf, debug = self._blob_lean(frame)
            self._last_debug = debug
            return PoseSample(steer_raw=raw, confidence=conf, source="camera-blob", ts=monotonic())

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
        self._face_template = None
