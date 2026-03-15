from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(slots=True)
class Calibrator:
    neutral: float = 0.0
    left_peak: float = -0.7
    right_peak: float = 0.7
    flip_sign: bool = False
    anchor_x_norm: float | None = None
    anchor_y_norm: float | None = None

    def normalize(self, raw: float) -> float:
        if self.flip_sign:
            raw = (2.0 * self.neutral) - raw
        left_span = abs(self.neutral - self.left_peak)
        right_span = abs(self.right_peak - self.neutral)
        if raw >= self.neutral:
            span = max(right_span, 1e-4)
        else:
            span = max(left_span, 1e-4)
        return max(-1.0, min(1.0, (raw - self.neutral) / span))


def load_calibration(path: Path) -> Calibrator:
    if not path.exists():
        return Calibrator()
    data = json.loads(path.read_text())
    return Calibrator(
        neutral=float(data.get("neutral", 0.0)),
        left_peak=float(data.get("left_peak", -0.7)),
        right_peak=float(data.get("right_peak", 0.7)),
        flip_sign=bool(data.get("flip_sign", False)),
        anchor_x_norm=float(data["anchor_x_norm"]) if data.get("anchor_x_norm") is not None else None,
        anchor_y_norm=float(data["anchor_y_norm"]) if data.get("anchor_y_norm") is not None else None,
    )


def save_calibration(path: Path, calib: Calibrator) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(calib), indent=2))
