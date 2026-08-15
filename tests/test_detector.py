import math

from posture_tracker.config import Settings
from posture_tracker.detector import (
    Deviation,
    DeviationSmoother,
    HeadPose,
    HysteresisTimer,
    Status,
    compute_baseline,
    compute_deviation,
    head_pose_from_matrix,
)


def rotation_matrix(roll_deg=0.0, pitch_deg=0.0, yaw_deg=0.0):
    """A 4x4 head transform built from the given Tait-Bryan angles."""
    x, y, z = map(math.radians, (pitch_deg, yaw_deg, roll_deg))
    cx, sx, cy, sy, cz, sz = (math.cos(x), math.sin(x), math.cos(y),
                              math.sin(y), math.cos(z), math.sin(z))
    return [
        [cy * cz, cz * sx * sy - cx * sz, cx * cz * sy + sx * sz, 0.0],
        [cy * sz, cx * cz + sx * sy * sz, -cz * sx + cx * sy * sz, 0.0],
        [-sy, cy * sx, cx * cy, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]


def pose(roll=0.0, pitch=0.0, yaw=0.0) -> HeadPose:
    return HeadPose(roll_deg=roll, pitch_deg=pitch, yaw_deg=yaw)


def test_head_pose_recovers_the_angles_it_was_built_from():
    recovered = head_pose_from_matrix(rotation_matrix(roll_deg=12.0, pitch_deg=-7.0, yaw_deg=20.0))
    assert abs(recovered.roll_deg - 12.0) < 0.01
    assert abs(recovered.pitch_deg - (-7.0)) < 0.01
    assert abs(recovered.yaw_deg - 20.0) < 0.01


def test_head_pose_of_an_identity_matrix_is_level():
    level = head_pose_from_matrix(rotation_matrix())
    assert abs(level.roll_deg) < 0.01
    assert abs(level.pitch_deg) < 0.01


def test_compute_baseline_takes_the_median():
    baseline = compute_baseline([pose(roll=1.0), pose(roll=2.0), pose(roll=90.0)])
    assert baseline.roll_deg == 2.0


def test_compute_baseline_ignores_an_outlier_sample():
    # One bad calibration frame must not drag the reference posture with it.
    samples = [pose(pitch=5.0) for _ in range(5)] + [pose(pitch=60.0)]
    assert compute_baseline(samples).pitch_deg == 5.0


def test_compute_baseline_requires_samples():
    import pytest

    with pytest.raises(ValueError):
        compute_baseline([])


def test_deviation_is_zero_against_its_own_baseline():
    p = pose(roll=-4.0, pitch=11.0)
    deviation = compute_deviation(p, compute_baseline([p]))
    assert deviation.roll_deg == 0.0
    assert deviation.pitch_deg == 0.0


def test_deviation_measures_the_change_from_the_baseline():
    baseline = compute_baseline([pose(roll=-4.0, pitch=11.0)])
    deviation = compute_deviation(pose(roll=6.0, pitch=25.0), baseline)
    assert abs(deviation.roll_deg - 10.0) < 1e-9
    assert abs(deviation.pitch_deg - 14.0) < 1e-9


def test_deviation_folds_the_wraparound_seam():
    # Readings either side of +/-180 describe nearly the same orientation and
    # must not read as a full turn apart.
    baseline = compute_baseline([pose(roll=179.0)])
    deviation = compute_deviation(pose(roll=-179.0), baseline)
    assert abs(deviation.roll_deg) < 5.0


def test_within_accepts_small_deviations_and_rejects_large_ones():
    settings = Settings()
    assert Deviation(roll_deg=1.0, pitch_deg=1.0).within(settings) is True
    assert Deviation(roll_deg=45.0, pitch_deg=0.0).within(settings) is False
    assert Deviation(roll_deg=0.0, pitch_deg=45.0).within(settings) is False


def dev(roll=0.0, pitch=0.0) -> Deviation:
    return Deviation(roll_deg=roll, pitch_deg=pitch)


def test_smoother_passes_the_first_sample_through_unchanged():
    assert DeviationSmoother(alpha=0.3).update(dev(roll=10.0)).roll_deg == 10.0


def test_smoother_damps_a_single_jitter_spike():
    smoother = DeviationSmoother(alpha=0.3)
    for _ in range(5):
        smoother.update(dev())
    # A lone bad frame must not drag the reading over an 8 deg threshold.
    assert smoother.update(dev(roll=30.0)).roll_deg < 10.0


def test_smoother_converges_on_a_sustained_change():
    smoother = DeviationSmoother(alpha=0.3)
    result = smoother.update(dev())
    for _ in range(20):
        result = smoother.update(dev(roll=30.0))
    assert result.roll_deg > 25.0


def test_smoother_reset_drops_history():
    smoother = DeviationSmoother(alpha=0.3)
    for _ in range(10):
        smoother.update(dev(roll=30.0))
    smoother.reset()
    assert smoother.update(dev()).roll_deg == 0.0


def test_smoother_rejects_invalid_alpha():
    import pytest

    with pytest.raises(ValueError):
        DeviationSmoother(alpha=0.0)
    with pytest.raises(ValueError):
        DeviationSmoother(alpha=1.5)


class FakeClock:
    def __init__(self, start: float = 0.0):
        self.t = start

    def __call__(self) -> float:
        return self.t

    def advance(self, seconds: float) -> None:
        self.t += seconds


def test_hysteresis_ok_when_posture_good():
    timer = HysteresisTimer(grace_period_seconds=7.0, clock=FakeClock())
    result = timer.update(posture_ok=True)
    assert result.status == Status.OK
    assert result.violation_seconds == 0.0


def test_hysteresis_warn_before_grace_period_elapses():
    clock = FakeClock()
    timer = HysteresisTimer(grace_period_seconds=7.0, clock=clock)
    timer.update(posture_ok=False)
    clock.advance(3.0)
    result = timer.update(posture_ok=False)
    assert result.status == Status.WARN
    assert result.violation_seconds == 3.0


def test_hysteresis_alert_after_grace_period_elapses():
    clock = FakeClock()
    timer = HysteresisTimer(grace_period_seconds=7.0, clock=clock)
    timer.update(posture_ok=False)
    clock.advance(7.5)
    result = timer.update(posture_ok=False)
    assert result.status == Status.ALERT
    assert result.violation_seconds >= 7.0


def test_hysteresis_resets_when_posture_returns_to_ok_before_grace_period():
    clock = FakeClock()
    timer = HysteresisTimer(grace_period_seconds=7.0, clock=clock)
    timer.update(posture_ok=False)
    clock.advance(5.0)
    assert timer.update(posture_ok=False).status == Status.WARN
    assert timer.update(posture_ok=True).status == Status.OK

    clock.advance(1.0)
    result = timer.update(posture_ok=False)
    assert result.status == Status.WARN
    assert result.violation_seconds == 0.0


def test_hysteresis_paused_when_face_lost_resets_timer():
    clock = FakeClock()
    timer = HysteresisTimer(grace_period_seconds=7.0, clock=clock)
    timer.update(posture_ok=False)
    clock.advance(6.9)
    assert timer.update(posture_ok=False).status == Status.WARN

    paused = timer.update(posture_ok=None)
    assert paused.status == Status.PAUSED
    assert paused.violation_seconds == 0.0

    clock.advance(0.5)
    result = timer.update(posture_ok=False)
    assert result.status == Status.WARN
    assert result.violation_seconds == 0.0


def test_hysteresis_notifies_once_after_notify_threshold():
    clock = FakeClock()
    timer = HysteresisTimer(grace_period_seconds=10.0, notify_after_seconds=5.0, clock=clock)
    timer.update(posture_ok=False)

    clock.advance(3.0)
    assert timer.update(posture_ok=False).should_notify is False

    clock.advance(2.5)  # total 5.5s, past the 5s notify threshold
    result = timer.update(posture_ok=False)
    assert result.status == Status.WARN  # still under the 10s overlay threshold
    assert result.should_notify is True

    clock.advance(1.0)
    # only fires once per continuous violation
    assert timer.update(posture_ok=False).should_notify is False


def test_hysteresis_notify_flag_resets_after_posture_recovers():
    clock = FakeClock()
    timer = HysteresisTimer(grace_period_seconds=10.0, notify_after_seconds=5.0, clock=clock)
    timer.update(posture_ok=False)
    clock.advance(6.0)
    assert timer.update(posture_ok=False).should_notify is True

    timer.update(posture_ok=True)
    timer.update(posture_ok=False)
    clock.advance(6.0)
    assert timer.update(posture_ok=False).should_notify is True


def test_hysteresis_without_notify_threshold_never_notifies():
    clock = FakeClock()
    timer = HysteresisTimer(grace_period_seconds=10.0, clock=clock)
    timer.update(posture_ok=False)
    clock.advance(20.0)
    result = timer.update(posture_ok=False)
    assert result.status == Status.ALERT
    assert result.should_notify is False
