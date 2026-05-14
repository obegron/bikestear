from __future__ import annotations


def pose_conf_threshold(source: str) -> float:
    if source == "camera-face":
        return 0.18
    if source == "camera-bike":
        return 0.16
    if source in ("camera-hog", "camera-blob"):
        return 0.1
    return 0.3


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    vals = sorted(values)
    idx = int((len(vals) - 1) * p)
    return vals[max(0, min(len(vals) - 1, idx))]


def debug_centroid_px(debug: dict, w: int, h: int, mirrored: bool) -> tuple[int, int] | None:
    kind = str(debug.get("kind", ""))
    if kind in ("face", "blob", "bike_mask"):
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


def debug_camera_key(debug: dict) -> str:
    return str(debug.get("camera_idx", "default"))


def anchor_gate_pass(
    source: str,
    debug: dict,
    anchor: tuple[int, int] | None,
    w: int | None,
    h: int | None,
    mirrored: bool,
) -> bool:
    if anchor is None or w is None or h is None:
        return True
    cent = debug_centroid_px(debug, w, h, mirrored=mirrored)
    if cent is None:
        return source != "camera-face"
    ax, ay = anchor
    dx = abs(cent[0] - ax)
    dy = abs(cent[1] - ay)
    if source == "camera-face":
        detector = str(debug.get("detector", ""))
        if detector == "tracker":
            return dx <= int(w * 0.22) and dy <= int(h * 0.22)
        return dx <= int(w * 0.28) and dy <= int(h * 0.30)
    if source == "camera-bike":
        return dx <= int(w * 0.30) and dy <= int(h * 0.28)
    if source in ("camera-hog", "camera-blob"):
        return dx <= int(w * 0.45)
    return True


def stable_for_anchor(p, debug: dict, cent) -> bool:
    conf_th = pose_conf_threshold(p.source)
    return (
        cent is not None
        and p.confidence >= conf_th
        and not bool(debug.get("warmup", False))
        and not bool(debug.get("anchor_pending", False))
        and not bool(debug.get("held", False))
    )


def accept_neutral_sample(p, debug: dict) -> bool:
    conf_th = pose_conf_threshold(p.source)
    if p.source == "camera-bike":
        return (
            p.confidence >= conf_th
            and not bool(debug.get("warmup", False))
            and not bool(debug.get("anchor_pending", False))
        )
    return p.confidence >= conf_th


def accept_side_sample(p, debug: dict, anchor, frame, mirrored: bool) -> bool:
    conf_th = pose_conf_threshold(p.source)
    return (
        p.confidence >= conf_th
        and not bool(debug.get("warmup", False))
        and not bool(debug.get("anchor_pending", False))
        and not bool(debug.get("held", False))
        and anchor_gate_pass(p.source, debug, anchor, frame.shape[1], frame.shape[0], mirrored=mirrored)
    )


def phase_sign_accepts(phase_key: str, raw: float, neutral_vals: list[float]) -> bool:
    if phase_key not in ("left", "right") or len(neutral_vals) < 20:
        return True
    neutral_lo = percentile(neutral_vals, 0.15)
    neutral_hi = percentile(neutral_vals, 0.85)
    margin = max(0.015, (neutral_hi - neutral_lo) * 0.5)
    if phase_key == "left":
        return raw < (neutral_lo - margin)
    return raw > (neutral_hi + margin)


def trim_side_outliers(values: list[float]) -> tuple[list[float], bool]:
    if len(values) < 8:
        return list(values), False
    mid = percentile(values, 0.50)
    deviations = [abs(v - mid) for v in values]
    mad = percentile(deviations, 0.50)
    band = max(0.12, mad * 3.5)
    trimmed = [v for v in values if abs(v - mid) <= band]
    min_keep = max(12, int(len(values) * 0.5))
    if len(trimmed) < min_keep:
        return list(values), False
    return trimmed, len(trimmed) != len(values)


def phase_target_count(phase_key: str) -> int:
    return 40 if phase_key == "neutral" else 30
