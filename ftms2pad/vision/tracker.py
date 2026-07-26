from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import sys
from tempfile import gettempdir
from time import monotonic, sleep
import types
from typing import Any, Protocol, Sequence

from ftms2pad.profiles import VisionConfig
from ftms2pad.types import VisionResult

LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12
LEFT_WRIST = 15
RIGHT_WRIST = 16
LEFT_HIP = 23
RIGHT_HIP = 24


class Landmark(Protocol):
    x: float
    y: float
    visibility: float


@dataclass(frozen=True, slots=True)
class TorsoEstimate:
    torso_x: float
    confidence: float
    shoulders: tuple[tuple[float, float], tuple[float, float]]
    hips: tuple[tuple[float, float], tuple[float, float]] | None


@dataclass(frozen=True, slots=True)
class WristGestureEstimate:
    left_raised: bool
    right_raised: bool
    confidence: float
    wrists: tuple[tuple[float, float], tuple[float, float]]

    @property
    def candidate(self) -> bool:
        return self.left_raised or self.right_raised


@dataclass(frozen=True, slots=True)
class VisionPacket:
    result: VisionResult
    frame: Any = None
    torso: TorsoEstimate | None = None
    gesture: WristGestureEstimate | None = None


def _confidence(landmark: Landmark) -> float:
    visibility = float(getattr(landmark, "visibility", 0.0))
    presence = float(getattr(landmark, "presence", 1.0))
    return max(0.0, min(1.0, visibility, presence))


def estimate_torso(landmarks: Sequence[Landmark], min_confidence: float = 0.5) -> TorsoEstimate | None:
    if len(landmarks) <= RIGHT_SHOULDER:
        return None
    left_shoulder = landmarks[LEFT_SHOULDER]
    right_shoulder = landmarks[RIGHT_SHOULDER]
    shoulder_confidence = min(_confidence(left_shoulder), _confidence(right_shoulder))
    shoulder_width = abs(float(left_shoulder.x) - float(right_shoulder.x))
    if shoulder_confidence < min_confidence or shoulder_width < 0.02:
        return None

    shoulders_x = (float(left_shoulder.x) + float(right_shoulder.x)) * 0.5
    hips = None
    confidence = shoulder_confidence
    torso_x = shoulders_x
    if len(landmarks) > RIGHT_HIP:
        left_hip = landmarks[LEFT_HIP]
        right_hip = landmarks[RIGHT_HIP]
        hip_confidence = min(_confidence(left_hip), _confidence(right_hip))
        if hip_confidence >= min_confidence:
            hips_x = (float(left_hip.x) + float(right_hip.x)) * 0.5
            torso_x = shoulders_x * 0.7 + hips_x * 0.3
            confidence = min(shoulder_confidence, hip_confidence)
            hips = ((float(left_hip.x), float(left_hip.y)), (float(right_hip.x), float(right_hip.y)))

    # Frame-centred position in shoulder-width units reduces sensitivity to camera distance.
    normalized_x = (torso_x - 0.5) / shoulder_width
    return TorsoEstimate(
        torso_x=normalized_x,
        confidence=confidence,
        shoulders=(
            (float(left_shoulder.x), float(left_shoulder.y)),
            (float(right_shoulder.x), float(right_shoulder.y)),
        ),
        hips=hips,
    )


def estimate_wrist_raise(
    landmarks: Sequence[Landmark],
    min_confidence: float = 0.5,
    raise_margin: float = 0.08,
) -> WristGestureEstimate | None:
    """Recognize either wrist held clearly above its matching shoulder.

    MediaPipe Pose already computes these landmarks for torso steering, so this
    classifier adds no model or image-processing pass. Comparing each wrist to
    its shoulder also keeps ordinary torso lean from becoming an accept action.
    """
    if len(landmarks) <= RIGHT_WRIST:
        return None
    left_shoulder = landmarks[LEFT_SHOULDER]
    right_shoulder = landmarks[RIGHT_SHOULDER]
    left_wrist = landmarks[LEFT_WRIST]
    right_wrist = landmarks[RIGHT_WRIST]
    left_confidence = min(_confidence(left_wrist), _confidence(left_shoulder))
    right_confidence = min(_confidence(right_wrist), _confidence(right_shoulder))
    left_reliable = left_confidence >= min_confidence
    right_reliable = right_confidence >= min_confidence
    reliable_confidences = [
        confidence
        for confidence, reliable in ((left_confidence, left_reliable), (right_confidence, right_reliable))
        if reliable
    ]
    if not reliable_confidences:
        return None
    left_raised = left_reliable and float(left_wrist.y) <= float(left_shoulder.y) - raise_margin
    right_raised = right_reliable and float(right_wrist.y) <= float(right_shoulder.y) - raise_margin
    raised_confidences = [
        confidence
        for confidence, raised in ((left_confidence, left_raised), (right_confidence, right_raised))
        if raised
    ]
    return WristGestureEstimate(
        left_raised=left_raised,
        right_raised=right_raised,
        confidence=max(raised_confidences or reliable_confidences),
        wrists=(
            (float(left_wrist.x), float(left_wrist.y)),
            (float(right_wrist.x), float(right_wrist.y)),
        ),
    )


def _video_nodes() -> list[int]:
    indexes: list[int] = []
    for path in sorted(Path("/dev").glob("video*")):
        suffix = path.name.removeprefix("video")
        if suffix.isdigit():
            indexes.append(int(suffix))
    return indexes


def camera_name(index: int) -> str:
    try:
        return Path(f"/sys/class/video4linux/video{index}/name").read_text().strip()
    except OSError:
        return f"video{index}"


def list_cameras() -> list[int]:
    try:
        import cv2
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("OpenCV is required to list cameras") from exc

    available: list[int] = []
    for index in _video_nodes():
        capture = cv2.VideoCapture(index, cv2.CAP_V4L2)
        if capture.isOpened():
            available.append(index)
        capture.release()
    return available


class VisionTracker:
    """Owns one camera and one lightweight MediaPipe Pose instance."""

    def __init__(self, config: VisionConfig) -> None:
        try:
            import cv2
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("Vision requires opencv-contrib-python and mediapipe") from exc

        self.config = config
        self._cv2 = cv2
        cv2.setNumThreads(1)
        self._capture = cv2.VideoCapture(config.camera, cv2.CAP_V4L2)
        self._capture.set(cv2.CAP_PROP_FRAME_WIDTH, config.width)
        self._capture.set(cv2.CAP_PROP_FRAME_HEIGHT, config.height)
        self._capture.set(cv2.CAP_PROP_FPS, config.fps)
        self._capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if not self._capture.isOpened():
            self._capture.release()
            raise RuntimeError(f"Could not open camera {config.camera}")

        # MediaPipe 0.10.21 eagerly imports its unrelated Tasks audio stack.
        # This process uses only the lightweight legacy Pose graph, so avoid
        # loading that stack (and probing audio devices) on worker startup.
        if "mediapipe" not in sys.modules:
            tasks_package = types.ModuleType("mediapipe.tasks")
            tasks_package.__path__ = []  # type: ignore[attr-defined]
            tasks_python = types.ModuleType("mediapipe.tasks.python")
            tasks_package.python = tasks_python  # type: ignore[attr-defined]
            sys.modules["mediapipe.tasks"] = tasks_package
            sys.modules["mediapipe.tasks.python"] = tasks_python
        matplotlib_cache = Path(gettempdir()) / f"ftms2pad-matplotlib-{os.getuid()}"
        matplotlib_cache.mkdir(exist_ok=True)
        os.environ.setdefault("MPLCONFIGDIR", str(matplotlib_cache))
        try:
            import mediapipe as mp
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("Vision requires mediapipe") from exc
        pose_module = mp.solutions.pose
        self._pose = pose_module.Pose(
            static_image_mode=False,
            model_complexity=0,
            enable_segmentation=False,
            min_detection_confidence=config.min_confidence,
            min_tracking_confidence=config.min_confidence,
        )
        self._last_capture_at = 0.0
        self._last_result_at = 0.0
        self._actual_fps = 0.0
        self._closed = False

    def read(self) -> VisionPacket:
        interval = 1.0 / self.config.fps
        remaining = interval - (monotonic() - self._last_capture_at)
        if remaining > 0.0:
            sleep(remaining)
        self._last_capture_at = monotonic()

        ok, frame = self._capture.read()
        captured_at = monotonic()
        if not ok:
            result = VisionResult(ts=captured_at, torso_x=None, confidence=0.0, actual_fps=self._actual_fps)
            return VisionPacket(result=result)

        rgb = self._cv2.cvtColor(frame, self._cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        inference_started = monotonic()
        processed = self._pose.process(rgb)
        inference_ms = (monotonic() - inference_started) * 1000.0

        if self._last_result_at > 0.0 and captured_at > self._last_result_at:
            instant_fps = 1.0 / (captured_at - self._last_result_at)
            self._actual_fps = instant_fps if self._actual_fps == 0.0 else self._actual_fps * 0.8 + instant_fps * 0.2
        self._last_result_at = captured_at

        torso = None
        gesture = None
        gesture_ms = 0.0
        if processed.pose_landmarks is not None:
            landmarks = processed.pose_landmarks.landmark
            torso = estimate_torso(landmarks, self.config.min_confidence)
            if self.config.gesture == "wrist_raise":
                gesture_started = monotonic()
                gesture = estimate_wrist_raise(
                    landmarks,
                    self.config.min_confidence,
                    self.config.gesture_raise_margin,
                )
                gesture_ms = (monotonic() - gesture_started) * 1000.0
        result = VisionResult(
            ts=captured_at,
            torso_x=torso.torso_x if torso is not None else None,
            confidence=torso.confidence if torso is not None else 0.0,
            actual_fps=self._actual_fps,
            inference_ms=inference_ms,
            gesture_candidate=gesture.candidate if gesture is not None else False,
            gesture_left_raised=gesture.left_raised if gesture is not None else False,
            gesture_right_raised=gesture.right_raised if gesture is not None else False,
            gesture_confidence=gesture.confidence if gesture is not None else 0.0,
            gesture_ms=gesture_ms,
        )
        return VisionPacket(result=result, frame=frame, torso=torso, gesture=gesture)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self._capture.release()
        finally:
            self._pose.close()
