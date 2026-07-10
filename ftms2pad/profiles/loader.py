from __future__ import annotations

from dataclasses import dataclass, fields
from math import isfinite
from pathlib import Path
from typing import Any, TypeVar

import yaml

Y_SOURCES = frozenset({"speed_kph", "watts", "cadence_rpm"})


@dataclass(frozen=True, slots=True)
class VisionConfig:
    camera: int = 0
    width: int = 320
    height: int = 180
    fps: float = 20.0
    min_confidence: float = 0.5


@dataclass(frozen=True, slots=True)
class XAxisConfig:
    gain: float = 1.4
    deadzone: float = 0.06
    smoothing: float = 0.25
    invert: bool = False
    stale_after_ms: float = 250.0


@dataclass(frozen=True, slots=True)
class YAxisConfig:
    source: str = "speed_kph"
    min: float = 0.0
    max: float = 40.0
    deadzone: float = 1.0
    smoothing: float = 0.15
    curve: str = "linear"
    invert: bool = True


@dataclass(frozen=True, slots=True)
class UInputConfig:
    x_axis: str = "ABS_X"
    y_axis: str = "ABS_Y"


@dataclass(frozen=True, slots=True)
class Profile:
    name: str
    vision: VisionConfig
    x_axis: XAxisConfig
    y_axis: YAxisConfig
    uinput: UInputConfig
    path: Path


T = TypeVar("T")


def resolve_profile_path(profile: str, search_dir: Path | None = None) -> Path:
    candidate = Path(profile)
    if candidate.suffix in {".yaml", ".yml"} or candidate.parent != Path("."):
        return candidate
    return (search_dir or Path("profiles")) / f"{profile}.yaml"


def _section(data: dict[str, Any], key: str) -> dict[str, Any]:
    section = data.get(key)
    if not isinstance(section, dict):
        raise ValueError(f"Profile section '{key}' must be a mapping")
    return section


def _config(cls: type[T], data: dict[str, Any], section: str) -> T:
    allowed = {field.name for field in fields(cls)}
    unknown = sorted(set(data) - allowed)
    if unknown:
        raise ValueError(f"Unknown {section} setting(s): {', '.join(unknown)}")
    try:
        return cls(**data)
    except TypeError as exc:
        raise ValueError(f"Invalid {section} settings: {exc}") from exc


def _validate(profile: Profile) -> None:
    def number(value: Any, name: str) -> float:
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not isfinite(float(value)):
            raise ValueError(f"{name} must be a finite number")
        return float(value)

    def boolean(value: Any, name: str) -> None:
        if not isinstance(value, bool):
            raise ValueError(f"{name} must be true or false")

    if not profile.name.strip():
        raise ValueError("Profile name must not be empty")
    vision = profile.vision
    if isinstance(vision.camera, bool) or not isinstance(vision.camera, int) or vision.camera < 0:
        raise ValueError("vision.camera must be one non-negative camera index")
    if isinstance(vision.width, bool) or not isinstance(vision.width, int):
        raise ValueError("vision.width must be an integer")
    if isinstance(vision.height, bool) or not isinstance(vision.height, int):
        raise ValueError("vision.height must be an integer")
    if vision.width < 160 or vision.height < 120:
        raise ValueError("vision.width and vision.height must be at least 160x120")
    if not 1.0 <= number(vision.fps, "vision.fps") <= 60.0:
        raise ValueError("vision.fps must be between 1 and 60")
    if not 0.0 < number(vision.min_confidence, "vision.min_confidence") <= 1.0:
        raise ValueError("vision.min_confidence must be greater than 0 and at most 1")

    x_axis = profile.x_axis
    if number(x_axis.gain, "x_axis.gain") <= 0.0:
        raise ValueError("x_axis.gain must be greater than 0")
    if not 0.0 <= number(x_axis.deadzone, "x_axis.deadzone") < 1.0:
        raise ValueError("x_axis.deadzone must be at least 0 and less than 1")
    if not 0.0 < number(x_axis.smoothing, "x_axis.smoothing") <= 1.0:
        raise ValueError("x_axis.smoothing must be greater than 0 and at most 1")
    if number(x_axis.stale_after_ms, "x_axis.stale_after_ms") <= 0.0:
        raise ValueError("x_axis.stale_after_ms must be greater than 0")
    boolean(x_axis.invert, "x_axis.invert")

    y_axis = profile.y_axis
    if y_axis.source not in Y_SOURCES:
        choices = ", ".join(sorted(Y_SOURCES))
        raise ValueError(f"Unknown y_axis.source '{y_axis.source}'; choose one of: {choices}")
    minimum = number(y_axis.min, "y_axis.min")
    maximum = number(y_axis.max, "y_axis.max")
    deadzone = number(y_axis.deadzone, "y_axis.deadzone")
    if maximum <= minimum:
        raise ValueError("y_axis.max must be greater than y_axis.min")
    if deadzone < 0.0:
        raise ValueError("y_axis.deadzone must be at least 0")
    if minimum + deadzone >= maximum:
        raise ValueError("y_axis.deadzone must leave a usable range below y_axis.max")
    if not 0.0 < number(y_axis.smoothing, "y_axis.smoothing") <= 1.0:
        raise ValueError("y_axis.smoothing must be greater than 0 and at most 1")
    if y_axis.curve != "linear":
        raise ValueError("y_axis.curve currently supports only 'linear'")
    boolean(y_axis.invert, "y_axis.invert")

    for field_name, value in (("x_axis", profile.uinput.x_axis), ("y_axis", profile.uinput.y_axis)):
        if not isinstance(value, str) or not value.startswith("ABS_"):
            raise ValueError(f"uinput.{field_name} must be a Linux ABS_* axis name")
    if profile.uinput.x_axis == profile.uinput.y_axis:
        raise ValueError("uinput.x_axis and uinput.y_axis must be different")


def load_profile(profile: str, search_dir: Path | None = None) -> Profile:
    path = resolve_profile_path(profile, search_dir)
    if not path.exists():
        raise FileNotFoundError(f"Profile not found: {path}")
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError("Profile root must be a mapping")

    allowed = {"name", "vision", "x_axis", "y_axis", "uinput"}
    unknown = sorted(set(data) - allowed)
    if unknown:
        raise ValueError(f"Unknown profile section(s): {', '.join(unknown)}")

    loaded = Profile(
        name=str(data.get("name", path.stem)),
        vision=_config(VisionConfig, _section(data, "vision"), "vision"),
        x_axis=_config(XAxisConfig, _section(data, "x_axis"), "x_axis"),
        y_axis=_config(YAxisConfig, _section(data, "y_axis"), "y_axis"),
        uinput=_config(UInputConfig, _section(data, "uinput"), "uinput"),
        path=path,
    )
    _validate(loaded)
    return loaded
