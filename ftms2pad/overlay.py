from __future__ import annotations


def safe_destroy_window(cv2, name: str) -> None:
    try:
        cv2.destroyWindow(name)
    except Exception:
        pass


def draw_tracking_overlay(cv2, frame, debug: dict, mirrored: bool, color=(120, 220, 120)) -> None:
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
            cv2.putText(
                frame,
                "track: person(hog)",
                (x, max(20, y - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1,
                cv2.LINE_AA,
            )
    elif kind == "blob":
        bbox = debug.get("bbox")
        centroid = debug.get("centroid")
        if isinstance(bbox, tuple) and len(bbox) == 4:
            x, y, bw, bh = [int(v) for v in bbox]
            if mirrored:
                x = w - (x + bw)
            cv2.rectangle(frame, (x, y), (x + bw, y + bh), color, 2)
            cv2.putText(
                frame,
                "track: motion blob",
                (x, max(20, y - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1,
                cv2.LINE_AA,
            )
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
                cv2.putText(
                    frame,
                    "track: pose points",
                    (20, 112),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    1,
                    cv2.LINE_AA,
                )
    elif kind == "face":
        bbox = debug.get("bbox")
        centroid = debug.get("centroid")
        detector = str(debug.get("detector", "face"))
        if isinstance(bbox, tuple) and len(bbox) == 4:
            x, y, bw, bh = [int(v) for v in bbox]
            if mirrored:
                x = w - (x + bw)
            cv2.rectangle(frame, (x, y), (x + bw, y + bh), color, 2)
            cv2.putText(
                frame,
                f"track: {detector}",
                (x, max(20, y - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1,
                cv2.LINE_AA,
            )
        if isinstance(centroid, tuple) and len(centroid) == 2:
            cx, cy = [int(v) for v in centroid]
            if mirrored:
                cx = w - cx
            cv2.circle(frame, (cx, cy), 6, color, -1)
    elif kind == "bike_mask":
        bbox = debug.get("bbox")
        roi = debug.get("roi")
        centroid = debug.get("centroid")
        anchor_x = debug.get("anchor_x")
        if isinstance(roi, tuple) and len(roi) == 4:
            x, y, bw, bh = [int(v) for v in roi]
            if mirrored:
                x = w - (x + bw)
            cv2.rectangle(frame, (x, y), (x + bw, y + bh), (90, 90, 160), 1)
            cv2.putText(
                frame,
                "search roi",
                (x, max(20, y - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (90, 90, 160),
                1,
                cv2.LINE_AA,
            )
        if isinstance(bbox, tuple) and len(bbox) == 4:
            x, y, bw, bh = [int(v) for v in bbox]
            if mirrored:
                x = w - (x + bw)
            cv2.rectangle(frame, (x, y), (x + bw, y + bh), color, 2)
            cv2.putText(
                frame,
                "track: bike torso",
                (x, max(20, y - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1,
                cv2.LINE_AA,
            )
        if isinstance(anchor_x, (int, float)):
            ax = int(anchor_x)
            if mirrored:
                ax = w - ax
            cv2.line(frame, (ax, int(h * 0.14)), (ax, int(h * 0.80)), (255, 210, 80), 1)
        if isinstance(centroid, tuple) and len(centroid) == 2:
            cx, cy = [int(v) for v in centroid]
            if mirrored:
                cx = w - cx
            cv2.circle(frame, (cx, cy), 6, color, -1)


def draw_monitor_frame(
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
        f"w={f.watts:6.1f}  cad={f.cadence_rpm:5.1f}  speed={f.speed_kph:5.1f}  res={f.resistance_level:4.1f}  pose={p.confidence:0.2f}",
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
    cv2.putText(
        frame,
        "Press q to stop, r to relock center, +/- resistance",
        (20, 86),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (220, 220, 220),
        1,
        cv2.LINE_AA,
    )
    draw_tracking_overlay(cv2, frame, debug, mirrored=mirror_preview)


def draw_calibration_frame(
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
    is_bike = str(debug.get("kind", "")) == "bike_mask"
    if anchor is None:
        cx, cy = w // 2, h // 2
    else:
        cx = max(80, min(w - 80, int(anchor[0])))
        cy = max(80, min(h - 80, int(anchor[1])))
    if anchor is not None:
        head_w = int(w * (0.42 if is_bike else 0.34))
        head_h = int(h * (0.30 if is_bike else 0.34))
        hx1 = cx - head_w // 2
        hy1 = cy - head_h // 2
        hx1 = max(0, min(w - head_w, hx1))
        hy1 = max(0, min(h - head_h, hy1))
        cv2.rectangle(frame, (hx1, hy1), (hx1 + head_w, hy1 + head_h), (90, 160, 255), 2)
        label = "Keep torso in this box" if is_bike else "Keep head in this box"
        cv2.putText(frame, label, (hx1 + 4, hy1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (90, 160, 255), 1, cv2.LINE_AA)
    else:
        prompt = "Locking neutral torso position..." if is_bike else "Locking neutral head position..."
        cv2.putText(frame, prompt, (20, 126), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (90, 160, 255), 1, cv2.LINE_AA)

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
    if seconds_left > 0.05:
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
    cv2.putText(frame, "q cancel  r relock", (w - 190, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (225, 225, 225), 1, cv2.LINE_AA)
    draw_tracking_overlay(cv2, frame, debug, mirrored=mirror_preview)
