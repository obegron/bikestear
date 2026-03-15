from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from ftms2pad.fusion import Calibrator


def load_calibration(path: Path) -> Calibrator:
    if not path.exists():
        return Calibrator()
    data = json.loads(path.read_text())
    return Calibrator(
        neutral=float(data.get("neutral", 0.0)),
        left_peak=float(data.get("left_peak", -0.7)),
        right_peak=float(data.get("right_peak", 0.7)),
        flip_sign=bool(data.get("flip_sign", False)),
    )


def save_calibration(path: Path, calib: Calibrator) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(calib), indent=2))
