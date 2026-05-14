from __future__ import annotations

from time import monotonic

from ftms2pad.vision import VisionTracker


def parse_camera_arg(camera_arg: str) -> list[str]:
    parts = [p.strip() for p in str(camera_arg).split(",") if p.strip()]
    return parts or ["auto"]


class VisionMux:
    def __init__(
        self,
        steering_mode: str,
        camera_arg: str,
        width: int = 640,
        height: int = 360,
        idle_hz: float = 8.0,
    ) -> None:
        cameras = parse_camera_arg(camera_arg)
        self._trackers = [VisionTracker(steering_mode, camera=c, width=width, height=height) for c in cameras]
        self.camera_idx = ",".join(str(t.camera_idx) for t in self._trackers)
        self._active = 0
        self._pending: int | None = None
        self._pending_count = 0
        self._idle_interval = 1.0 / max(1.0, float(idle_hz))
        self._cache: list[tuple[object, object | None, dict[str, object], float] | None] = [None] * len(self._trackers)

    def _score(self, p, debug: dict[str, object], idx: int) -> float:
        score = float(getattr(p, "confidence", 0.0))
        detector = str(debug.get("detector", ""))
        held = bool(debug.get("held", False))
        centroid = debug.get("centroid")
        if centroid is None:
            score -= 0.35
        if detector in ("frontal", "profile_l", "profile_r"):
            score += 0.07
        elif detector in ("template", "tracker"):
            score += 0.03
        if held:
            score -= 0.05
        if idx == self._active:
            score += 0.03
        return score

    def _select_index(self, scored: list[tuple[float, int]]) -> int:
        if len(scored) == 1:
            return scored[0][1]
        scored.sort(reverse=True)
        best_score, best_idx = scored[0]
        active_score = next((s for s, i in scored if i == self._active), -1.0)
        if best_idx == self._active:
            self._pending = None
            self._pending_count = 0
            return self._active
        # Hysteresis: require clear improvement for a few consecutive frames.
        if best_score < active_score + 0.07:
            self._pending = None
            self._pending_count = 0
            return self._active
        if self._pending == best_idx:
            self._pending_count += 1
        else:
            self._pending = best_idx
            self._pending_count = 1
        if self._pending_count >= 3:
            self._active = best_idx
            self._pending = None
            self._pending_count = 0
        return self._active

    def _sample_tracker(self, idx: int) -> tuple[object, object | None, dict[str, object]]:
        p, frame, debug = self._trackers[idx].next_with_frame()
        debug = dict(debug)
        debug["camera_idx"] = self._trackers[idx].camera_idx
        debug["mux"] = {"active": idx, "count": len(self._trackers)}
        self._cache[idx] = (p, frame, debug, monotonic())
        return p, frame, debug

    def _ensure_fresh_samples(self) -> None:
        now = monotonic()
        self._sample_tracker(self._active)

        for i in range(len(self._trackers)):
            if i == self._active:
                continue
            cached = self._cache[i]
            if cached is None:
                self._sample_tracker(i)
                continue
            _, _frame, _debug, ts = cached
            if (now - ts) >= self._idle_interval:
                self._sample_tracker(i)

    def _scored_indexes(self) -> list[tuple[float, int]]:
        now = monotonic()
        scored: list[tuple[float, int]] = []
        for i, cached in enumerate(self._cache):
            if cached is None:
                continue
            p, _frame, debug, ts = cached
            age = max(0.0, now - ts)
            score = self._score(p, debug, i) - min(0.35, age * 0.45)
            scored.append((score, i))
        return scored

    def next_with_frame(self):
        self._ensure_fresh_samples()
        scored = self._scored_indexes()
        if not scored:
            p, frame, debug = self._sample_tracker(self._active)
            return p, frame, debug
        chosen = self._select_index(scored)
        cached = self._cache[chosen]
        if cached is None:
            p, frame, debug = self._sample_tracker(chosen)
            return p, frame, debug
        p, frame, debug, _ts = cached
        debug = dict(debug)
        debug["mux"] = {"active": chosen, "count": len(self._trackers)}
        return p, frame, debug

    def next(self):
        self._ensure_fresh_samples()
        scored = self._scored_indexes()
        if not scored:
            p, _frame, _debug = self._sample_tracker(self._active)
            return p
        chosen = self._select_index(scored)
        cached = self._cache[chosen]
        if cached is None:
            p, _frame, _debug = self._sample_tracker(chosen)
            return p
        p, _frame, _debug, _ts = cached
        return p

    def reset_tracking(self) -> None:
        for t in self._trackers:
            t.reset_tracking()

    def close(self) -> None:
        for t in self._trackers:
            t.close()
