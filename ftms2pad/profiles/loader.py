from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(slots=True)
class SteeringConfig:
    mode: str = "torso_combo"
    gain: float = 1.4
    deadzone: float = 0.06
    smoothing: float = 0.18
    invert: bool = False


@dataclass(slots=True)
class ThrottleConfig:
    source: str = "watts"
    gain: float = 0.0025
    deadzone_watts: float = 25.0
    smoothing: float = 0.12
    curve: str = "linear"


@dataclass(slots=True)
class UInputConfig:
    steer_axis: str = "ABS_X"
    throttle_axis: str = "ABS_Y"
    invert_throttle: bool = True


@dataclass(slots=True)
class Profile:
    name: str
    steering: SteeringConfig
    throttle: ThrottleConfig
    uinput: UInputConfig
    buttons: dict[str, str]


def _section(data: dict[str, Any], key: str) -> dict[str, Any]:
    section = data.get(key, {})
    if not isinstance(section, dict):
        raise ValueError(f"Profile section '{key}' must be a mapping")
    return section


def load_profile(profile: str, search_dir: Path | None = None) -> Profile:
    search = search_dir or Path("profiles")
    path = search / f"{profile}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"Profile not found: {path}")

    data = yaml.safe_load(path.read_text()) or {}
    if not isinstance(data, dict):
        raise ValueError("Profile root must be a mapping")

    steering = SteeringConfig(**_section(data, "steering"))
    throttle = ThrottleConfig(**_section(data, "throttle"))
    uinput_cfg = UInputConfig(**_section(data, "uinput"))
    buttons = _section(data, "buttons")

    return Profile(
        name=str(data.get("name", profile)),
        steering=steering,
        throttle=throttle,
        uinput=uinput_cfg,
        buttons={str(k): str(v) for k, v in buttons.items()},
    )
