from __future__ import annotations

try:
    from .tracker import VisionTracker, camera_name, list_cameras
except ModuleNotFoundError as exc:  # pragma: no cover
    _import_error = exc

    def list_cameras(*args, **kwargs):
        raise RuntimeError("Vision dependencies missing. Install opencv-python and mediapipe.") from _import_error

    def camera_name(*args, **kwargs):
        raise RuntimeError("Vision dependencies missing. Install opencv-python and mediapipe.") from _import_error

    class VisionTracker:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs):
            raise RuntimeError("Vision dependencies missing. Install opencv-python and mediapipe.") from _import_error
