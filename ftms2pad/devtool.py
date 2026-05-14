from __future__ import annotations

import argparse
import asyncio
import inspect

from ftms2pad.cli import cmd_record, cmd_replay


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="ftms2pad-devtool")
    sub = p.add_subparsers(dest="cmd", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--profile", default="supertuxkart")
    common.add_argument("--bike", default="sim", help="BLE addr/name, or sim")
    common.add_argument("--verbose", action="store_true", help="Print FTMS BLE connection details")
    common.add_argument("--camera", default="auto", help="camera index, auto, or comma list (e.g. 0,2)")
    common.add_argument("--hz", type=int, default=60, help="main loop frequency")
    common.add_argument("--debug-log", default="", help="directory to write session bundle")
    common.add_argument("--debug-fps", type=float, default=10.0, help="recorded video FPS")
    common.add_argument("--debug-width", type=int, default=640, help="recorded video width")
    common.add_argument("--debug-height", type=int, default=360, help="recorded video height")
    common.add_argument("--vision-width", type=int, default=640, help="camera capture width")
    common.add_argument("--vision-height", type=int, default=360, help="camera capture height")
    common.add_argument("--mux-idle-hz", type=float, default=8.0, help="poll rate for inactive cameras in multi-camera mode")
    common.add_argument("--stand-button", default="", help="button to tap when standing is detected (e.g. BTN_A)")
    common.add_argument("--stand-threshold", type=float, default=0.14, help="fraction of frame height above anchor to count as stand")
    common.add_argument("--stand-cooldown", type=float, default=0.35, help="seconds between stand-triggered taps")
    common.add_argument("--resistance-start", type=float, default=0.0, help="set trainer target resistance after connecting")
    common.add_argument("--resistance-step", type=float, default=1.0, help="GUI +/- resistance step")

    c = sub.add_parser("record", parents=[common], help="Record video plus sampled control data for offline tuning")
    c.add_argument("--out-dir", default="sessions", help="directory to write recorded sessions when --debug-log is unset")
    c.add_argument("--duration-seconds", type=float, default=0.0, help="stop automatically after this many seconds (0 = until q/Ctrl+C)")
    c.add_argument("--no-gui", action="store_true", help="Use text-only recording")
    c.add_argument("--no-mirror", action="store_true", help="Do not mirror preview window")
    c.set_defaults(fn=cmd_record)

    c = sub.add_parser("replay", help="Replay recorded sampled control data from a session")
    c.add_argument("session", help="session directory or events.jsonl path")
    c.add_argument("--profile", default="supertuxkart")
    c.add_argument("--replay-mode", choices=("recomputed", "recorded"), default="recomputed", help="use current tuning on recorded raw samples, or replay recorded joystick outputs")
    c.add_argument("--emit", action="store_true", help="emit replayed joystick events via uinput")
    c.add_argument("--speed", type=float, default=1.0, help="timing multiplier for replay (1.0 = original timing)")
    c.add_argument("--no-timing", action="store_true", help="replay as fast as possible")
    c.add_argument("--report-out", default="", help="base path for replay report files (.json/.md)")
    c.set_defaults(fn=cmd_replay)

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
        raise SystemExit(130)


if __name__ == "__main__":
    main()
