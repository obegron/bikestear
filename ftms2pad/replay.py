from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from ftms2pad.calibration import load_calibration
from ftms2pad.fusion import FusionPipeline
from ftms2pad.profiles import load_profile
from ftms2pad.uinput import VirtualGamepad


def _calibration_path(profile: str) -> Path:
    return Path("profiles") / f"{profile}.calibration.json"


def _pose_conf_threshold(source: str) -> float:
    if source == "camera-face":
        return 0.18
    if source == "camera-bike":
        return 0.16
    if source in ("camera-hog", "camera-blob"):
        return 0.1
    return 0.3


def _percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    vals = sorted(values)
    idx = int((len(vals) - 1) * p)
    return vals[max(0, min(len(vals) - 1, idx))]


def _session_events_path(session: str) -> Path:
    p = Path(session)
    if p.is_dir():
        return p / "events.jsonl"
    return p


def _replay_report_paths(events_path: Path, replay_mode: str, report_out: str = "") -> tuple[Path, Path]:
    if report_out:
        base = Path(report_out)
        if base.suffix:
            base = base.with_suffix("")
    elif events_path.parent.is_dir():
        base = events_path.parent / f"replay-report-{replay_mode}"
    else:
        base = events_path.with_suffix("")
    return base.with_suffix(".json"), base.with_suffix(".md")


def _series_stats(values: list[float]) -> dict[str, float]:
    if not values:
        return {"count": 0}
    vals = sorted(values)
    return {
        "count": len(vals),
        "min": vals[0],
        "p10": _percentile(vals, 0.10),
        "p50": _percentile(vals, 0.50),
        "p90": _percentile(vals, 0.90),
        "max": vals[-1],
        "mean": sum(vals) / len(vals),
    }


def _center_return_report(samples: list[dict[str, float]], value_key: str) -> dict[str, float | int | None]:
    if not samples:
        return {"episodes": 0, "completed": 0, "completion_rate": 0.0, "median_return_s": None}
    leave = 0.20
    center = 0.08
    hold_frames = 5
    episodes: list[float] = []
    armed = False
    start_t: float | None = None
    held = 0
    for sample in samples:
        value = abs(float(sample.get(value_key, 0.0)))
        t = float(sample.get("t", 0.0))
        if not armed:
            if value >= leave:
                armed = True
                start_t = t
                held = 0
            continue
        if value <= center:
            held += 1
            if held >= hold_frames and start_t is not None:
                episodes.append(max(0.0, t - start_t))
                armed = False
                start_t = None
                held = 0
        else:
            held = 0
    episode_count = len(episodes) + (1 if armed and start_t is not None else 0)
    return {
        "episodes": episode_count,
        "completed": len(episodes),
        "completion_rate": len(episodes) / max(1, episode_count),
        "median_return_s": _percentile(sorted(episodes), 0.50) if episodes else None,
        "p90_return_s": _percentile(sorted(episodes), 0.90) if episodes else None,
    }


def _write_replay_report(json_path: Path, md_path: Path, report: dict[str, object]) -> None:
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    lines = [
        "# Replay Report",
        "",
        f"- Session: `{report['session']}`",
        f"- Events: `{report['events_path']}`",
        f"- Replay mode: `{report['replay_mode']}`",
        f"- Samples: `{report['sample_count']}`",
        f"- Duration: `{report['duration_s']:.2f}s`",
        "",
        "## Center Return",
    ]
    cr = report["center_return"]
    lines.extend(
        [
            f"- Episodes: `{cr['episodes']}`",
            f"- Completed: `{cr['completed']}`",
            f"- Completion rate: `{cr['completion_rate']:.2%}`",
            f"- Median return: `{cr['median_return_s']}`",
            f"- P90 return: `{cr['p90_return_s']}`",
            "",
            "## Steering Stats",
        ]
    )
    for key in ("raw", "steer", "throttle"):
        stats = report["stats"][key]
        lines.append(
            f"- {key}: count={stats.get('count', 0)} min={stats.get('min')} p10={stats.get('p10')} "
            f"p50={stats.get('p50')} p90={stats.get('p90')} max={stats.get('max')} mean={stats.get('mean')}"
        )
    asym = report["asymmetry"]
    lines.extend(
        [
            "",
            "## Asymmetry",
            f"- Left steer mean: `{asym['left_mean']}`",
            f"- Right steer mean: `{asym['right_mean']}`",
            f"- Left count: `{asym['left_count']}`",
            f"- Right count: `{asym['right_count']}`",
            "",
            "## Files",
            f"- JSON: `{json_path}`",
            f"- Markdown: `{md_path}`",
        ]
    )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


async def cmd_replay(args: argparse.Namespace) -> int:
    events_path = _session_events_path(args.session)
    if not events_path.exists():
        print(f"Missing session events: {events_path}")
        return 1
    profile = load_profile(args.profile)
    calib = load_calibration(_calibration_path(args.profile))
    fusion = FusionPipeline(profile, calibrator=calib)
    pad = None
    if getattr(args, "emit", False):
        pad = VirtualGamepad(
            steer_axis=profile.uinput.steer_axis,
            throttle_axis=profile.uinput.throttle_axis,
            invert_throttle=profile.uinput.invert_throttle,
        )
        if not pad.enabled:
            reason = getattr(pad, "error", "") or "uinput unavailable"
            print(f"Replay emit unavailable ({reason}).")
            pad = None
    lines = events_path.read_text(encoding="utf-8").splitlines()
    prev_t: float | None = None
    count = 0
    samples: list[dict[str, float | bool | str]] = []
    try:
        for line in lines:
            if not line.strip():
                continue
            event = json.loads(line)
            pose = event.get("pose", {})
            ftms = event.get("ftms", {})
            control = event.get("control", {})
            extra = event.get("extra", {})
            t = float(event.get("t", 0.0))
            raw = float(pose.get("steer_raw", 0.0))
            conf = float(pose.get("confidence", 0.0))
            source = str(pose.get("source", ""))
            pose_ok = bool(extra.get("pose_ok", conf >= _pose_conf_threshold(source)))
            connected = bool(ftms.get("connected", False))
            watts = float(ftms.get("watts", 0.0))
            if args.replay_mode == "recorded":
                steer = float(control.get("steer", 0.0))
                throttle = float(control.get("throttle", 0.0))
            else:
                steer = fusion.steer(raw, pose_ok=pose_ok)
                throttle = fusion.throttle(watts, connected=connected)
            samples.append(
                {
                    "t": t,
                    "raw": raw,
                    "steer": steer,
                    "throttle": throttle,
                    "conf": conf,
                    "pose_ok": pose_ok,
                    "connected": connected,
                }
            )
            if pad is not None:
                pad.emit(steer=steer, throttle=throttle)
            if prev_t is not None and not getattr(args, "no_timing", False):
                delay = max(0.0, (t - prev_t) / max(0.01, float(args.speed)))
                if delay > 0:
                    await asyncio.sleep(min(delay, 0.25))
            prev_t = t
            count += 1
            if count % 30 == 0 or count == 1:
                print(
                    f"sample={count:5d} raw={raw:+0.3f} conf={conf:0.2f} "
                    f"steer={steer:+0.3f} thr={throttle:0.3f}",
                    end="\r",
                    flush=True,
                )
        raw_vals = [float(s["raw"]) for s in samples]
        steer_vals = [float(s["steer"]) for s in samples]
        throttle_vals = [float(s["throttle"]) for s in samples]
        left = [v for v in steer_vals if v < -0.05]
        right = [v for v in steer_vals if v > 0.05]
        duration_s = max(0.0, float(samples[-1]["t"]) - float(samples[0]["t"])) if len(samples) >= 2 else 0.0
        report = {
            "session": str(Path(args.session)),
            "events_path": str(events_path),
            "replay_mode": str(args.replay_mode),
            "sample_count": count,
            "duration_s": duration_s,
            "stats": {
                "raw": _series_stats(raw_vals),
                "steer": _series_stats(steer_vals),
                "throttle": _series_stats(throttle_vals),
            },
            "center_return": _center_return_report(samples, "steer"),
            "asymmetry": {
                "left_count": len(left),
                "right_count": len(right),
                "left_mean": (sum(left) / len(left)) if left else 0.0,
                "right_mean": (sum(right) / len(right)) if right else 0.0,
            },
        }
        report_json, report_md = _replay_report_paths(events_path, str(args.replay_mode), getattr(args, "report_out", ""))
        _write_replay_report(report_json, report_md, report)
        print(f"\nReplayed {count} samples from {events_path}")
        print(f"report json: {report_json}")
        print(f"report md: {report_md}")
        return 0
    finally:
        if pad is not None:
            pad.close()
