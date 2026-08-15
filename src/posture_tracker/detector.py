"""Pure posture-analysis logic: calibration, deviation math, hysteresis timer.

Kept independent of MediaPipe/OpenCV objects so it can be unit-tested with
plain numbers.

Posture is judged from the head's 3D orientation, which MediaPipe's Face
Landmarker hands back as a transform matrix. The earlier approach derived
angles from Pose body landmarks instead, and measurements on a laptop webcam
killed it: Pose is a whole-body model, it never had a body to look at, and it
reported joints it could not see by extrapolating them. Sitting motionless,
its head roll drifted with 12 deg of noise and its forward-tilt proxy with
7 deg -- larger than the postural changes being looked for. Face Landmarker
only fires when a face is genuinely visible, which additionally makes
"can I see the user at all" an honest question rather than a guess.
"""

from __future__ import annotations

import math
import statistics
import time
from dataclasses import dataclass
from enum import Enum, auto

from posture_tracker.config import Settings


@dataclass(frozen=True)
class HeadPose:
    """Head orientation in degrees.

    roll  - tilting the head towards a shoulder
    pitch - dropping the chin towards the chest, or lifting it
    yaw   - turning to look left or right

    Absolute values depend on where the camera sits, so only differences
    against a calibrated baseline are ever used.
    """

    roll_deg: float
    pitch_deg: float
    yaw_deg: float


def head_pose_from_matrix(matrix) -> HeadPose:
    """Extracts Tait-Bryan angles from a 4x4 head transform matrix."""
    r = [[float(matrix[i][j]) for j in range(3)] for i in range(3)]
    pitch = math.degrees(math.atan2(r[2][1], r[2][2]))
    yaw = math.degrees(math.atan2(-r[2][0], math.hypot(r[2][1], r[2][2])))
    roll = math.degrees(math.atan2(r[1][0], r[0][0]))
    return HeadPose(roll_deg=roll, pitch_deg=pitch, yaw_deg=yaw)


@dataclass(frozen=True)
class Baseline:
    roll_deg: float
    pitch_deg: float


@dataclass(frozen=True)
class Deviation:
    roll_deg: float
    pitch_deg: float

    def within(self, settings: Settings) -> bool:
        return (
            abs(self.roll_deg) <= settings.head_tilt_threshold_deg
            and abs(self.pitch_deg) <= settings.head_pitch_threshold_deg
        )


def _normalize_angle_deg(diff: float) -> float:
    """Folds an angle difference into (-180, 180], so readings either side of
    the wraparound seam do not look like a full turn apart."""
    return (diff + 180.0) % 360.0 - 180.0


def compute_baseline(samples: list[HeadPose]) -> Baseline:
    """Reduce calibration samples to a reference posture.

    Uses the median rather than the mean: a few bad frames (the user still
    settling into the chair, a momentary mis-detection) would drag a mean far
    enough to bias every later comparison, and a skewed baseline shows up as
    permanent false "bad posture".
    """
    if not samples:
        raise ValueError("cannot compute baseline from zero samples")

    return Baseline(
        roll_deg=statistics.median(s.roll_deg for s in samples),
        pitch_deg=statistics.median(s.pitch_deg for s in samples),
    )


def compute_deviation(pose: HeadPose, baseline: Baseline) -> Deviation:
    return Deviation(
        roll_deg=_normalize_angle_deg(pose.roll_deg - baseline.roll_deg),
        pitch_deg=_normalize_angle_deg(pose.pitch_deg - baseline.pitch_deg),
    )


class DeviationSmoother:
    """Exponential moving average over successive deviations.

    Per-frame jitter is large enough to cross the thresholds on its own, which
    flickers the status and restarts the violation timer at random. Averaging
    over roughly the last second turns that into a stable reading while still
    tracking a genuine posture change quickly.
    """

    def __init__(self, alpha: float):
        if not 0.0 < alpha <= 1.0:
            raise ValueError("alpha must be in (0, 1]")
        self._alpha = alpha
        self._current: Deviation | None = None

    def reset(self) -> None:
        """Drops accumulated history, e.g. after the subject left the frame."""
        self._current = None

    def update(self, deviation: Deviation) -> Deviation:
        if self._current is None:
            self._current = deviation
            return deviation

        a = self._alpha
        prev = self._current
        self._current = Deviation(
            roll_deg=a * deviation.roll_deg + (1 - a) * prev.roll_deg,
            pitch_deg=a * deviation.pitch_deg + (1 - a) * prev.pitch_deg,
        )
        return self._current


class Status(Enum):
    OK = auto()
    WARN = auto()
    ALERT = auto()
    PAUSED = auto()


@dataclass
class HysteresisResult:
    status: Status
    violation_seconds: float
    should_notify: bool = False


class HysteresisTimer:
    """Tracks continuous-bad-posture duration with grace-period hysteresis.

    - Good posture -> timer resets to 0, status OK.
    - Bad posture -> timer accumulates; status WARN until grace period elapses,
      then ALERT (this is the threshold that triggers the fullscreen overlay).
    - Face lost -> timer resets to 0, status PAUSED, regardless of prior state.
    - `should_notify` fires exactly once per continuous violation, the first
      update where the violation has lasted at least `notify_after_seconds`
      (a separate, earlier threshold meant for a background desktop
      notification rather than the fullscreen overlay).
    """

    def __init__(
        self,
        grace_period_seconds: float,
        notify_after_seconds: float | None = None,
        clock=time.monotonic,
    ):
        self._grace_period = grace_period_seconds
        self._notify_after = notify_after_seconds
        self._clock = clock
        self._violation_start: float | None = None
        self._notified = False

    def update(self, posture_ok: bool | None) -> HysteresisResult:
        """posture_ok: True=good, False=bad, None=face not visible."""
        now = self._clock()

        if posture_ok is None:
            self._violation_start = None
            self._notified = False
            return HysteresisResult(status=Status.PAUSED, violation_seconds=0.0)

        if posture_ok:
            self._violation_start = None
            self._notified = False
            return HysteresisResult(status=Status.OK, violation_seconds=0.0)

        if self._violation_start is None:
            self._violation_start = now

        elapsed = now - self._violation_start
        status = Status.ALERT if elapsed >= self._grace_period else Status.WARN

        should_notify = False
        if (
            self._notify_after is not None
            and not self._notified
            and elapsed >= self._notify_after
        ):
            should_notify = True
            self._notified = True

        return HysteresisResult(status=status, violation_seconds=elapsed, should_notify=should_notify)
