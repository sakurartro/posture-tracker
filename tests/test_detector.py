from posture_tracker.detector import (
    Baseline,
    HysteresisTimer,
    Landmark,
    PoseLandmarks,
    Status,
    compute_baseline,
    compute_deviation,
)


def make_pose(
    nose=(0.5, 0.3),
    left_ear=(0.45, 0.28),
    right_ear=(0.55, 0.28),
    left_shoulder=(0.4, 0.5),
    right_shoulder=(0.6, 0.5),
    visibility=1.0,
) -> PoseLandmarks:
    return PoseLandmarks(
        nose=Landmark(*nose, visibility=visibility),
        left_ear=Landmark(*left_ear, visibility=visibility),
        right_ear=Landmark(*right_ear, visibility=visibility),
        left_shoulder=Landmark(*left_shoulder, visibility=visibility),
        right_shoulder=Landmark(*right_shoulder, visibility=visibility),
    )


def test_compute_baseline_averages_samples():
    samples = [make_pose(), make_pose()]
    baseline = compute_baseline(samples)
    assert baseline.head_tilt_deg == 0.0
    assert baseline.shoulder_tilt_deg == 0.0
    assert baseline.nose_to_shoulder_line > 0


def test_compute_deviation_zero_when_matching_baseline():
    pose = make_pose()
    baseline = compute_baseline([pose])
    deviation = compute_deviation(pose, baseline)
    assert deviation.head_tilt_deg == 0.0
    assert deviation.shoulder_tilt_deg == 0.0
    assert deviation.slouch_pct == 0.0


def test_compute_deviation_detects_head_tilt():
    baseline = compute_baseline([make_pose()])
    tilted = make_pose(left_ear=(0.45, 0.20), right_ear=(0.55, 0.28))
    deviation = compute_deviation(tilted, baseline)
    assert deviation.head_tilt_deg != 0.0


def test_compute_deviation_detects_slouch():
    baseline = compute_baseline([make_pose()])
    # Nose moves further from the shoulder line -> distance increases.
    leaning = make_pose(nose=(0.5, 0.1))
    deviation = compute_deviation(leaning, baseline)
    assert deviation.slouch_pct > 0


def test_compute_deviation_normalizes_angle_wraparound():
    # Baseline sits just below the +180 seam; the current angle sits just
    # below the -180 seam (nearly the same physical tilt). A naive
    # subtraction gives a ~358 degree jump; normalized it should be small.
    baseline = Baseline(head_tilt_deg=179.0, shoulder_tilt_deg=0.0, nose_to_shoulder_line=0.2)
    pose = make_pose(left_ear=(0.55, 0.281), right_ear=(0.45, 0.28))
    deviation = compute_deviation(pose, baseline)
    assert abs(deviation.head_tilt_deg) < 20.0


def test_baseline_requires_samples():
    import pytest

    with pytest.raises(ValueError):
        compute_baseline([])


class FakeClock:
    def __init__(self, start: float = 0.0):
        self.t = start

    def __call__(self) -> float:
        return self.t

    def advance(self, seconds: float) -> None:
        self.t += seconds


def test_hysteresis_ok_when_posture_good():
    clock = FakeClock()
    timer = HysteresisTimer(grace_period_seconds=7.0, clock=clock)
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

    ok_result = timer.update(posture_ok=True)
    assert ok_result.status == Status.OK

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
