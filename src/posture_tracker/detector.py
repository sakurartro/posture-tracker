"""Pure posture-analysis logic: calibration, deviation math, hysteresis timer.

Kept independent of MediaPipe/OpenCV objects so it can be unit-tested with
plain landmark dataclasses.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from enum import Enum, auto

from posture_tracker.config import Settings


@dataclass(frozen=True)
class Landmark:
    x: float
    y: float
    visibility: float = 1.0


@dataclass(frozen=True)
class PoseLandmarks:
    """Subset of MediaPipe Pose landmarks this app cares about."""

    nose: Landmark
    left_ear: Landmark
    right_ear: Landmark
    left_shoulder: Landmark
    right_shoulder: Landmark

    def visible(self, min_visibility: float) -> bool:
        return all(
            lm.visibility >= min_visibility
            for lm in (
                self.nose,
                self.left_ear,
                self.right_ear,
                self.left_shoulder,
                self.right_shoulder,
            )
        )


@dataclass(frozen=True)
class Baseline:
    head_tilt_deg: float
    shoulder_tilt_deg: float
    nose_to_shoulder_line: float


@dataclass(frozen=True)
class Deviation:
    head_tilt_deg: float
    shoulder_tilt_deg: float
    slouch_pct: float

    def within(self, settings: Settings) -> bool:
        return (
            abs(self.head_tilt_deg) <= settings.head_tilt_threshold_deg
            and abs(self.shoulder_tilt_deg) <= settings.shoulder_tilt_threshold_deg
            and abs(self.slouch_pct) <= settings.slouch_threshold_pct
        )


def _tilt_angle_deg(left: Landmark, right: Landmark) -> float:
    """Angle of the line left->right relative to horizontal, in degrees."""
    return math.degrees(math.atan2(right.y - left.y, right.x - left.x))


def _normalize_angle_deg(angle: float) -> float:
    """Wraps an angle to (-180, 180] so deviations near the +/-180 seam
    (e.g. baseline=179, current=-179) don't read as a ~360 degree jump."""
    return (angle + 180.0) % 360.0 - 180.0


def _shoulder_midpoint(landmarks: PoseLandmarks) -> tuple[float, float]:
    mx = (landmarks.left_shoulder.x + landmarks.right_shoulder.x) / 2
    my = (landmarks.left_shoulder.y + landmarks.right_shoulder.y) / 2
    return mx, my


def _nose_to_shoulder_line_distance(landmarks: PoseLandmarks) -> float:
    mx, my = _shoulder_midpoint(landmarks)
    return math.hypot(landmarks.nose.x - mx, landmarks.nose.y - my)


def compute_baseline(samples: list[PoseLandmarks]) -> Baseline:
    """Average calibration samples into a reference baseline."""
    if not samples:
        raise ValueError("cannot compute baseline from zero samples")

    head_tilts = [_tilt_angle_deg(s.left_ear, s.right_ear) for s in samples]
    shoulder_tilts = [_tilt_angle_deg(s.left_shoulder, s.right_shoulder) for s in samples]
    distances = [_nose_to_shoulder_line_distance(s) for s in samples]

    return Baseline(
        head_tilt_deg=sum(head_tilts) / len(head_tilts),
        shoulder_tilt_deg=sum(shoulder_tilts) / len(shoulder_tilts),
        nose_to_shoulder_line=sum(distances) / len(distances),
    )


def compute_deviation(landmarks: PoseLandmarks, baseline: Baseline) -> Deviation:
    head_tilt = _normalize_angle_deg(
        _tilt_angle_deg(landmarks.left_ear, landmarks.right_ear) - baseline.head_tilt_deg
    )
    shoulder_tilt = _normalize_angle_deg(
        _tilt_angle_deg(landmarks.left_shoulder, landmarks.right_shoulder) - baseline.shoulder_tilt_deg
    )
    distance = _nose_to_shoulder_line_distance(landmarks)
    if baseline.nose_to_shoulder_line > 0:
        slouch_pct = (distance - baseline.nose_to_shoulder_line) / baseline.nose_to_shoulder_line * 100
    else:
        slouch_pct = 0.0

    return Deviation(head_tilt_deg=head_tilt, shoulder_tilt_deg=shoulder_tilt, slouch_pct=slouch_pct)


class Status(Enum):
    OK = auto()
    WARN = auto()
    ALERT = auto()
    PAUSED = auto()


@dataclass
class HysteresisResult:
    status: Status
    violation_seconds: float


class HysteresisTimer:
    """Tracks continuous-bad-posture duration with grace-period hysteresis.

    - Good posture -> timer resets to 0, status OK.
    - Bad posture -> timer accumulates; status WARN until grace period elapses,
      then ALERT.
    - Face lost -> timer resets to 0, status PAUSED, regardless of prior state.
    """

    def __init__(self, grace_period_seconds: float, clock=time.monotonic):
        self._grace_period = grace_period_seconds
        self._clock = clock
        self._violation_start: float | None = None
        self._last_status = Status.OK

    def update(self, posture_ok: bool | None) -> HysteresisResult:
        """posture_ok: True=good, False=bad, None=face not visible."""
        now = self._clock()

        if posture_ok is None:
            self._violation_start = None
            self._last_status = Status.PAUSED
            return HysteresisResult(status=Status.PAUSED, violation_seconds=0.0)

        if posture_ok:
            self._violation_start = None
            self._last_status = Status.OK
            return HysteresisResult(status=Status.OK, violation_seconds=0.0)

        if self._violation_start is None:
            self._violation_start = now

        elapsed = now - self._violation_start
        status = Status.ALERT if elapsed >= self._grace_period else Status.WARN
        self._last_status = status
        return HysteresisResult(status=status, violation_seconds=elapsed)
