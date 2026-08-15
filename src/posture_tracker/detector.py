"""Pure posture-analysis logic: calibration, deviation math, hysteresis timer.

Kept independent of MediaPipe/OpenCV objects so it can be unit-tested with
plain landmark dataclasses.
"""

from __future__ import annotations

import math
import statistics
import time
from dataclasses import dataclass
from enum import Enum, auto

from posture_tracker.config import Settings


@dataclass(frozen=True)
class Landmark:
    """A single body keypoint in *square* units.

    MediaPipe hands back coordinates normalized per axis (x by frame width,
    y by frame height), so on a 4:3 frame a vertical offset is stretched 1.33x
    relative to a horizontal one and an angle computed from them is not a real
    angle. Callers are expected to rescale x by the frame aspect ratio before
    building a Landmark, so everything in this module works in square units
    and a degree is an actual degree.
    """

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

    def face_visible(self, min_visibility: float) -> bool:
        """Whether there is a subject to judge at all.

        Only face landmarks count. They are the reliable ones -- measured at
        0.98+ visibility on a laptop webcam -- and requiring shoulders here
        used to blank out tracking exactly when the user slouched, which is
        the moment that matters most.
        """
        return all(
            lm.visibility >= min_visibility
            for lm in (self.nose, self.left_ear, self.right_ear)
        )

    def shoulders_usable(self, min_visibility: float) -> bool:
        """Whether the shoulders were actually seen, rather than guessed.

        MediaPipe reports coordinates for joints outside the frame too,
        extrapolated from the rest of the body and betrayed only by a lower
        visibility score and a y past the frame edge. Anything derived from
        those is noise, so the y bound is checked as well as the score.
        """
        return all(
            lm.visibility >= min_visibility and 0.0 <= lm.y <= 1.0
            for lm in (self.left_shoulder, self.right_shoulder)
        )


@dataclass(frozen=True)
class Baseline:
    head_tilt_deg: float
    head_pitch_deg: float
    # None when the shoulders were never properly in frame during calibration,
    # in which case shoulder tilt is not judged at all.
    shoulder_tilt_deg: float | None


@dataclass(frozen=True)
class Deviation:
    head_tilt_deg: float
    head_pitch_deg: float
    shoulder_tilt_deg: float | None = None

    def within(self, settings: Settings) -> bool:
        if abs(self.head_tilt_deg) > settings.head_tilt_threshold_deg:
            return False
        if abs(self.head_pitch_deg) > settings.head_pitch_threshold_deg:
            return False
        # An unmeasurable shoulder is not a crooked shoulder.
        if (
            self.shoulder_tilt_deg is not None
            and abs(self.shoulder_tilt_deg) > settings.shoulder_tilt_threshold_deg
        ):
            return False
        return True


def _tilt_angle_deg(left: Landmark, right: Landmark) -> float:
    """Tilt of the line through two landmarks, in degrees; 0 means level.

    A line has no direction: left->right and right->left describe the same
    tilt. That matters here because MediaPipe labels landmarks *anatomically*,
    so on a non-mirrored webcam the subject's "left" landmark sits at a larger
    x than the "right" one. A naive atan2(dy, dx) then lands near +/-180 deg,
    right on the wraparound seam, where sub-pixel jitter flips the result
    between +179 and -179 -- and averaging such samples (calibration) yields a
    meaningless baseline near 0.

    Forcing dx positive folds the result into (-90, 90], far away from any
    seam, so both averaging and comparison behave.
    """
    dx = right.x - left.x
    dy = right.y - left.y
    if dx < 0:
        dx, dy = -dx, -dy
    return math.degrees(math.atan2(dy, dx))


def _normalize_tilt_diff_deg(diff: float) -> float:
    """Folds a difference of two tilt angles into (-90, 90].

    Tilts live modulo 180 (see _tilt_angle_deg), so a baseline of +89 and a
    current reading of -89 differ by 2 degrees, not 178.
    """
    return (diff + 90.0) % 180.0 - 90.0


def _shoulder_midpoint(landmarks: PoseLandmarks) -> tuple[float, float]:
    mx = (landmarks.left_shoulder.x + landmarks.right_shoulder.x) / 2
    my = (landmarks.left_shoulder.y + landmarks.right_shoulder.y) / 2
    return mx, my


def _head_pitch_deg(landmarks: PoseLandmarks) -> float:
    """How far the nose has dropped below the line between the ears.

    This is the slouch signal. Letting the chin sink towards the chest --
    the classic desk slouch -- swings the nose downward relative to the ears;
    measured on a real session, roughly 2 deg sitting upright against 10 deg
    slouching.

    Expressed as an angle against the ear separation rather than a raw
    distance, which makes it immune to how far the chair is from the camera:
    both quantities scale together, so only the actual head geometry moves it.
    It uses face landmarks alone (visibility 0.98+) rather than the shoulders,
    which a laptop webcam typically cannot see at all.
    """
    ear_span = math.hypot(
        landmarks.right_ear.x - landmarks.left_ear.x,
        landmarks.right_ear.y - landmarks.left_ear.y,
    )
    if ear_span <= 0:
        return 0.0
    ear_mid_y = (landmarks.left_ear.y + landmarks.right_ear.y) / 2
    return math.degrees(math.atan2(landmarks.nose.y - ear_mid_y, ear_span))


def compute_baseline(
    samples: list[PoseLandmarks],
    shoulder_min_visibility: float | None = None,
) -> Baseline:
    """Reduce calibration samples to a reference posture.

    Uses the median rather than the mean: a few bad frames (the user still
    settling into the chair, a momentary mis-detection) would drag a mean far
    enough to bias every later comparison, and a skewed baseline shows up as
    permanent false "bad posture".

    A shoulder baseline is only established if the shoulders were genuinely
    in frame for most of calibration; otherwise it stays None and shoulder
    tilt is left unjudged rather than judged against a guess.
    """
    if not samples:
        raise ValueError("cannot compute baseline from zero samples")

    shoulder_tilt = None
    if shoulder_min_visibility is not None:
        usable = [s for s in samples if s.shoulders_usable(shoulder_min_visibility)]
        if len(usable) >= len(samples) / 2:
            shoulder_tilt = statistics.median(
                _tilt_angle_deg(s.left_shoulder, s.right_shoulder) for s in usable
            )

    return Baseline(
        head_tilt_deg=statistics.median(
            _tilt_angle_deg(s.left_ear, s.right_ear) for s in samples
        ),
        head_pitch_deg=statistics.median(_head_pitch_deg(s) for s in samples),
        shoulder_tilt_deg=shoulder_tilt,
    )


def compute_deviation(
    landmarks: PoseLandmarks,
    baseline: Baseline,
    shoulder_min_visibility: float | None = None,
) -> Deviation:
    head_tilt = _normalize_tilt_diff_deg(
        _tilt_angle_deg(landmarks.left_ear, landmarks.right_ear) - baseline.head_tilt_deg
    )
    head_pitch = _head_pitch_deg(landmarks) - baseline.head_pitch_deg

    shoulder_tilt = None
    if (
        baseline.shoulder_tilt_deg is not None
        and shoulder_min_visibility is not None
        and landmarks.shoulders_usable(shoulder_min_visibility)
    ):
        shoulder_tilt = _normalize_tilt_diff_deg(
            _tilt_angle_deg(landmarks.left_shoulder, landmarks.right_shoulder)
            - baseline.shoulder_tilt_deg
        )

    return Deviation(
        head_tilt_deg=head_tilt,
        head_pitch_deg=head_pitch,
        shoulder_tilt_deg=shoulder_tilt,
    )


class DeviationSmoother:
    """Exponential moving average over successive deviations.

    Per-frame landmark jitter is large enough to cross the thresholds on its
    own, which flickers the status and restarts the violation timer at random.
    Averaging over roughly the last second turns that into a stable reading
    while still tracking a genuine posture change quickly.
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

        def blend(new: float | None, old: float | None) -> float | None:
            # A metric that just became measurable (or stopped being) starts
            # fresh rather than blending against a value that is not there.
            if new is None:
                return None
            if old is None:
                return new
            return a * new + (1 - a) * old

        self._current = Deviation(
            head_tilt_deg=a * deviation.head_tilt_deg + (1 - a) * prev.head_tilt_deg,
            head_pitch_deg=a * deviation.head_pitch_deg + (1 - a) * prev.head_pitch_deg,
            shoulder_tilt_deg=blend(deviation.shoulder_tilt_deg, prev.shoulder_tilt_deg),
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
