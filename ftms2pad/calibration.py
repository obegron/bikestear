from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import median


@dataclass(frozen=True, slots=True)
class XCalibration:
    neutral: float
    left: float
    right: float

    def normalize(self, torso_x: float) -> float:
        delta = torso_x - self.neutral
        left_delta = self.left - self.neutral
        right_delta = self.right - self.neutral
        if delta * left_delta >= 0.0:
            normalized = -(delta / left_delta)
        else:
            normalized = delta / right_delta
        return max(-1.0, min(1.0, normalized))


def _percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        raise ValueError("Cannot calculate a percentile without samples")
    position = (len(ordered) - 1) * quantile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def build_calibration(
    neutral_samples: list[float],
    left_samples: list[float],
    right_samples: list[float],
    *,
    min_samples: int = 10,
    min_separation: float = 0.10,
) -> XCalibration:
    phases = (
        ("neutral", neutral_samples),
        ("left", left_samples),
        ("right", right_samples),
    )
    for phase, samples in phases:
        if len(samples) < min_samples:
            raise ValueError(
                f"Not enough reliable {phase} samples ({len(samples)}/{min_samples}); "
                f"redo the {phase} phase and keep both shoulders visible"
            )

    neutral = float(median(neutral_samples))
    left_median = float(median(left_samples))
    right_median = float(median(right_samples))
    left = _percentile(left_samples, 0.25 if left_median < neutral else 0.75)
    right = _percentile(right_samples, 0.25 if right_median < neutral else 0.75)
    if abs(neutral - left) < min_separation:
        raise ValueError("Left calibration is too close to neutral; redo calibration and move your torso farther left")
    if abs(right - neutral) < min_separation:
        raise ValueError("Right calibration is too close to neutral; redo calibration and move your torso farther right")
    if (left - neutral) * (right - neutral) >= 0.0:
        raise ValueError("Left and right calibration are on the same side of neutral; redo both side phases")
    return XCalibration(neutral=neutral, left=left, right=right)


def calibration_path(profile_path: Path) -> Path:
    return profile_path.with_suffix(".calibration.json")


def load_calibration(path: Path) -> XCalibration:
    if not path.exists():
        raise FileNotFoundError(f"Calibration not found: {path}. Run 'ftms2pad calibrate' for this profile first.")
    data = json.loads(path.read_text(encoding="utf-8"))
    try:
        calibration = XCalibration(
            neutral=float(data["neutral"]),
            left=float(data["left"]),
            right=float(data["right"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"Invalid calibration file {path}; redo calibration") from exc
    if (calibration.left - calibration.neutral) * (calibration.right - calibration.neutral) >= 0.0:
        raise ValueError(f"Invalid calibration ranges in {path}; redo calibration")
    return calibration


def save_calibration(path: Path, calibration: XCalibration) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(calibration), indent=2) + "\n", encoding="utf-8")
