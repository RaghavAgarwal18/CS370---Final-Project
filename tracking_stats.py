from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from threading import RLock
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


@dataclass
class TrackingStats:
    """Shared posture session state that trackers can update and publish."""

    server_url: str | None = None
    tracker_name: str = "tracker"
    session_start: float = field(default_factory=time.time)
    last_update: float = field(default_factory=time.time)
    status: str = "calibrating"
    score: int = 0
    issues: list[str] = field(default_factory=list)
    face_detected: bool = False
    calibrating: bool = True
    pulse_active: bool = False
    bad_posture_remaining: float = 0.0
    cooldown_remaining: float = 0.0
    pulse_remaining: float = 0.0
    good_seconds: float = 0.0
    bad_seconds: float = 0.0
    shock_count: int = 0
    shock_log: list[dict[str, Any]] = field(default_factory=list)
    extra: dict[str, Any] = field(default_factory=dict)

    _lock: RLock = field(default_factory=RLock, init=False, repr=False, compare=False)

    def reset(self) -> None:
        with self._lock:
            now = time.time()
            self.session_start = now
            self.last_update = now
            self.status = "calibrating"
            self.score = 0
            self.issues = []
            self.face_detected = False
            self.calibrating = True
            self.pulse_active = False
            self.bad_posture_remaining = 0.0
            self.cooldown_remaining = 0.0
            self.pulse_remaining = 0.0
            self.good_seconds = 0.0
            self.bad_seconds = 0.0
            self.shock_count = 0
            self.shock_log = []
            self.extra = {}

    def _accumulate_time(self, now: float) -> None:
        delta = max(0.0, now - self.last_update)
        if self.status == "good":
            self.good_seconds += delta
        elif self.status == "bad":
            self.bad_seconds += delta
        self.last_update = now

    def set_state(
        self,
        *,
        status: str,
        score: int,
        issues: list[str] | None = None,
        face_detected: bool = False,
        calibrating: bool = False,
        pulse_active: bool = False,
        bad_posture_remaining: float = 0.0,
        cooldown_remaining: float = 0.0,
        pulse_remaining: float = 0.0,
        extra: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        with self._lock:
            now = time.time()
            self._accumulate_time(now)
            self.status = status
            self.score = int(max(0, min(100, score)))
            self.issues = list(issues or [])
            self.face_detected = face_detected
            self.calibrating = calibrating
            self.pulse_active = pulse_active
            self.bad_posture_remaining = max(0.0, float(bad_posture_remaining))
            self.cooldown_remaining = max(0.0, float(cooldown_remaining))
            self.pulse_remaining = max(0.0, float(pulse_remaining))
            self.extra = dict(extra or {})
            return self.snapshot()

    def record_shock(self, issues: list[str] | None = None) -> dict[str, Any]:
        with self._lock:
            entry = {
                "time": time.time(),
                "tracker": self.tracker_name,
                "issues": list(issues or []),
            }
            self.shock_count += 1
            self.shock_log.append(entry)
            self.extra = {**self.extra, "last_shock": entry}
            return entry

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            session_seconds = max(0.0, time.time() - self.session_start)
            return {
                "tracker_name": self.tracker_name,
                "status": self.status,
                "score": self.score,
                "issues": list(self.issues),
                "session_seconds": round(session_seconds, 1),
                "session_mins": round(session_seconds / 60.0, 1),
                "good_seconds": round(self.good_seconds, 1),
                "bad_seconds": round(self.bad_seconds, 1),
                "shock_count": self.shock_count,
                "shock_log": list(self.shock_log[-20:]),
                "face_detected": self.face_detected,
                "calibrating": self.calibrating,
                "pulse_active": self.pulse_active,
                "bad_posture_remaining": round(self.bad_posture_remaining, 1),
                "cooldown_remaining": round(self.cooldown_remaining, 1),
                "pulse_remaining": round(self.pulse_remaining, 1),
                "updated_at": time.time(),
                "extra": dict(self.extra),
            }

    def apply_snapshot(self, payload: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            now = time.time()
            self._accumulate_time(now)
            self.status = str(payload.get("status", self.status))
            self.score = int(payload.get("score", self.score) or 0)
            self.issues = list(payload.get("issues", self.issues) or [])
            self.face_detected = bool(payload.get("face_detected", self.face_detected))
            self.calibrating = bool(payload.get("calibrating", self.calibrating))
            self.pulse_active = bool(payload.get("pulse_active", self.pulse_active))
            self.bad_posture_remaining = float(payload.get("bad_posture_remaining", self.bad_posture_remaining) or 0.0)
            self.cooldown_remaining = float(payload.get("cooldown_remaining", self.cooldown_remaining) or 0.0)
            self.pulse_remaining = float(payload.get("pulse_remaining", self.pulse_remaining) or 0.0)
            self.extra = dict(payload.get("extra", self.extra) or {})

            if "good_seconds" in payload:
                self.good_seconds = float(payload["good_seconds"])
            if "bad_seconds" in payload:
                self.bad_seconds = float(payload["bad_seconds"])
            if "shock_count" in payload:
                self.shock_count = int(payload["shock_count"])
            if "shock_log" in payload:
                self.shock_log = list(payload["shock_log"])[-20:]
            return self.snapshot()

    def publish(self) -> bool:
        if not self.server_url:
            return False

        payload = json.dumps(self.snapshot()).encode("utf-8")
        request = Request(
            self.server_url.rstrip("/") + "/update",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urlopen(request, timeout=0.75) as response:
                response.read()
            return True
        except URLError:
            return False
