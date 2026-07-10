from __future__ import annotations

import argparse
import asyncio
import inspect
import sys

from ftms2pad.calibrate import calibrate_vision
from ftms2pad.calibration import calibration_path, load_calibration
from ftms2pad.ftms import list_ble_devices
from ftms2pad.profiles import load_profile
from ftms2pad.runtime import monitor_controller, run_controller
from ftms2pad.vision import camera_name, list_cameras


def _camera_index(value: str) -> int:
    try:
        index = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("camera must be one numeric index") from exc
    if index < 0:
        raise argparse.ArgumentTypeError("camera index must be non-negative")
    return index


async def cmd_list_bikes(_: argparse.Namespace) -> int:
    devices = await list_ble_devices()
    if not devices:
        print("No BLE devices found.")
        return 0
    for name, address, is_ftms in devices:
        print(f"{'FTMS' if is_ftms else '----'} {name:30s} {address}")
    return 0


def cmd_list_cameras(_: argparse.Namespace) -> int:
    cameras = list_cameras()
    if not cameras:
        print("No camera devices found.")
        return 1
    for index in cameras:
        print(f"{index}\t{camera_name(index)}")
    return 0


async def cmd_calibrate(args: argparse.Namespace) -> int:
    profile = load_profile(args.profile)
    return await calibrate_vision(
        profile,
        camera=args.camera,
        prep_seconds=args.prep_seconds,
        phase_seconds=args.phase_seconds,
        preview=not args.no_preview,
        mirror=not args.no_mirror,
    )


async def cmd_monitor(args: argparse.Namespace) -> int:
    profile = load_profile(args.profile)
    calibration = load_calibration(calibration_path(profile.path))
    return await monitor_controller(
        profile,
        calibration,
        bike=args.bike,
        camera=args.camera,
        hz=args.hz,
        preview=not args.no_preview,
        mirror=not args.no_mirror,
        verbose=args.verbose,
    )


async def cmd_run(args: argparse.Namespace) -> int:
    profile = load_profile(args.profile)
    calibration = load_calibration(calibration_path(profile.path))
    return await run_controller(
        profile,
        calibration,
        bike=args.bike,
        camera=args.camera,
        hz=args.hz,
        dry_run=args.dry_run,
        verbose=args.verbose,
        duration_seconds=args.duration_seconds,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ftms2pad")
    commands = parser.add_subparsers(dest="command", required=True)

    command = commands.add_parser("list-bikes", help="scan for BLE bikes")
    command.set_defaults(fn=cmd_list_bikes)

    command = commands.add_parser("list-cameras", help="list camera indexes")
    command.set_defaults(fn=cmd_list_cameras)

    profile_parent = argparse.ArgumentParser(add_help=False)
    profile_parent.add_argument("--profile", default="supertuxkart", help="profile name or YAML path")
    profile_parent.add_argument("--camera", type=_camera_index, default=None, help="one camera index; overrides the profile")

    command = commands.add_parser(
        "calibrate",
        parents=[profile_parent],
        help="calibrate neutral, left, and right torso positions",
    )
    command.add_argument("--prep-seconds", type=float, default=2.0)
    command.add_argument("--phase-seconds", type=float, default=3.0)
    command.add_argument("--no-preview", action="store_true")
    command.add_argument("--no-mirror", action="store_true")
    command.set_defaults(fn=cmd_calibrate)

    runtime_parent = argparse.ArgumentParser(add_help=False, parents=[profile_parent])
    runtime_parent.add_argument("--bike", required=True, help="BLE address/name, or 'sim'")
    runtime_parent.add_argument("--hz", type=float, default=60.0, help="gamepad loop rate")
    runtime_parent.add_argument("--verbose", action="store_true", help="show BLE connection details")

    command = commands.add_parser("monitor", parents=[runtime_parent], help="show current vision, FTMS, and mapped values")
    command.add_argument("--no-preview", action="store_true")
    command.add_argument("--no-mirror", action="store_true")
    command.set_defaults(fn=cmd_monitor)

    command = commands.add_parser("run", parents=[runtime_parent], help="emit the Linux virtual gamepad")
    command.add_argument("--dry-run", action="store_true", help="run mappings without opening uinput")
    command.add_argument("--duration-seconds", type=float, default=0.0, help=argparse.SUPPRESS)
    command.set_defaults(fn=cmd_run)
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    if hasattr(args, "hz") and not 1.0 <= args.hz <= 240.0:
        raise ValueError("--hz must be between 1 and 240")
    if hasattr(args, "prep_seconds") and args.prep_seconds < 0.0:
        raise ValueError("--prep-seconds must be at least 0")
    if hasattr(args, "phase_seconds") and args.phase_seconds <= 0.0:
        raise ValueError("--phase-seconds must be greater than 0")
    if hasattr(args, "duration_seconds") and args.duration_seconds < 0.0:
        raise ValueError("--duration-seconds must be at least 0")


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    try:
        _validate_args(args)
        result = asyncio.run(args.fn(args)) if inspect.iscoroutinefunction(args.fn) else args.fn(args)
    except KeyboardInterrupt:
        print("\nStopped.")
        raise SystemExit(130)
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"ftms2pad: {exc}", file=sys.stderr)
        raise SystemExit(2)
    raise SystemExit(result)


if __name__ == "__main__":
    main()
