from posture_tracker.config import Settings
from posture_tracker.detector import (
    Deviation,
    DeviationSmoother,
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


def make_camera_pose(head_dy=0.0, shoulder_dy=0.0, nose_y=0.30,
                     shoulder_y=0.50, shoulder_visibility=1.0) -> PoseLandmarks:
    """Mimics what a real webcam actually produces.

    MediaPipe labels landmarks anatomically and a webcam feed is not
    mirrored, so the subject's LEFT ear/shoulder lands at a LARGER x than the
    right one. `head_dy`/`shoulder_dy` lower the left-side landmark to tilt
    the corresponding line.
    """
    return PoseLandmarks(
        nose=Landmark(0.50, nose_y),
        left_ear=Landmark(0.55, 0.28 + head_dy),
        right_ear=Landmark(0.45, 0.28),
        left_shoulder=Landmark(0.60, shoulder_y + shoulder_dy, shoulder_visibility),
        right_shoulder=Landmark(0.40, shoulder_y, shoulder_visibility),
    )


def scaled(pose: PoseLandmarks, factor: float, shift: float = 0.0) -> PoseLandmarks:
    """The same posture seen from further away (everything shrinks towards a
    point) and shifted in frame."""

    def s(lm: Landmark) -> Landmark:
        return Landmark(x=lm.x * factor + shift, y=lm.y * factor + shift,
                        visibility=lm.visibility)

    return PoseLandmarks(
        nose=s(pose.nose),
        left_ear=s(pose.left_ear),
        right_ear=s(pose.right_ear),
        left_shoulder=s(pose.left_shoulder),
        right_shoulder=s(pose.right_shoulder),
    )


def test_compute_baseline_medians_samples():
    baseline = compute_baseline([make_pose(), make_pose()])
    assert baseline.head_tilt_deg == 0.0
    # No shoulder confidence bar was supplied, so shoulders go unjudged.
    assert baseline.shoulder_tilt_deg is None


def test_compute_deviation_zero_when_matching_baseline():
    pose = make_pose()
    baseline = compute_baseline([pose])
    deviation = compute_deviation(pose, baseline)
    assert deviation.head_tilt_deg == 0.0
    assert deviation.head_pitch_deg == 0.0


def test_compute_deviation_detects_head_tilt():
    baseline = compute_baseline([make_pose()])
    tilted = make_pose(left_ear=(0.45, 0.20), right_ear=(0.55, 0.28))
    deviation = compute_deviation(tilted, baseline)
    assert deviation.head_tilt_deg != 0.0


def test_head_pitch_ignores_distance_from_camera():
    # Rolling the chair back shrinks every on-screen length at once. That is
    # not a posture change and must not register as one -- the raw
    # nose-to-shoulder distance this replaced moved ~20% on chair movement
    # alone, well past the violation threshold.
    upright = make_camera_pose()
    baseline = compute_baseline([upright])

    for factor in (0.6, 0.8, 1.25):
        deviation = compute_deviation(scaled(upright, factor, shift=0.05), baseline)
        assert abs(deviation.head_pitch_deg) < 0.5, f"drifted at scale {factor}"


def test_head_pitch_detects_the_chin_dropping():
    baseline = compute_baseline([make_camera_pose()])
    # Same framing, but the nose sinks below the ear line.
    deviation = compute_deviation(make_camera_pose(nose_y=0.42), baseline)
    assert deviation.head_pitch_deg > 20.0


def test_baseline_is_level_for_non_mirrored_camera_layout():
    # Regression: with the left landmark at a larger x, a naive
    # atan2(dy, dx) lands near +/-180 deg -- right on the wraparound seam,
    # where sub-pixel jitter flips samples between +179 and -179. Averaging
    # those produced a baseline near 0 deg that matched no real posture, so
    # every later frame read as tens of degrees off and the app reported bad
    # posture permanently.
    samples = [make_camera_pose(head_dy=dy) for dy in (0.001, -0.001, 0.0005, -0.0005)]
    baseline = compute_baseline(samples, shoulder_min_visibility=0.7)

    assert abs(baseline.head_tilt_deg) < 5.0
    assert baseline.shoulder_tilt_deg is not None
    assert abs(baseline.shoulder_tilt_deg) < 5.0

    # ...and a level pose must then read as no deviation at all.
    deviation = compute_deviation(make_camera_pose(), baseline, shoulder_min_visibility=0.7)
    assert abs(deviation.head_tilt_deg) < 5.0
    assert deviation.shoulder_tilt_deg is not None
    assert abs(deviation.shoulder_tilt_deg) < 5.0


def test_shoulders_below_the_frame_edge_are_not_trusted():
    # MediaPipe still reports a position for joints it cannot see, so a high
    # visibility score alone is not enough -- on a laptop webcam the shoulders
    # sit past the bottom edge (y > 1.0) while still scoring up to 0.86.
    out_of_frame = make_camera_pose(shoulder_y=1.05, shoulder_visibility=0.86)
    assert out_of_frame.shoulders_usable(0.7) is False
    assert out_of_frame.face_visible(0.5) is True


def test_shoulder_tilt_is_skipped_when_shoulders_are_out_of_frame():
    baseline = compute_baseline(
        [make_camera_pose(shoulder_y=1.05)], shoulder_min_visibility=0.7
    )
    assert baseline.shoulder_tilt_deg is None

    deviation = compute_deviation(
        make_camera_pose(shoulder_y=1.05), baseline, shoulder_min_visibility=0.7
    )
    assert deviation.shoulder_tilt_deg is None


def test_an_unmeasurable_shoulder_does_not_count_as_bad_posture():
    settings = Settings()
    deviation = Deviation(head_tilt_deg=0.0, head_pitch_deg=0.0, shoulder_tilt_deg=None)
    assert deviation.within(settings) is True


def test_a_measured_crooked_shoulder_does_count():
    settings = Settings()
    deviation = Deviation(head_tilt_deg=0.0, head_pitch_deg=0.0, shoulder_tilt_deg=40.0)
    assert deviation.within(settings) is False


def test_real_tilt_still_detected_with_camera_layout():
    baseline = compute_baseline([make_camera_pose()])
    deviation = compute_deviation(make_camera_pose(head_dy=0.05), baseline)
    assert abs(deviation.head_tilt_deg) > 10.0


def test_baseline_median_ignores_an_outlier_sample():
    # One bad calibration frame must not drag the reference posture with it.
    samples = [make_camera_pose() for _ in range(5)] + [make_camera_pose(head_dy=0.4)]
    baseline = compute_baseline(samples)
    assert abs(baseline.head_tilt_deg) < 2.0


def test_compute_deviation_folds_tilt_seam_at_90_degrees():
    # Tilts live modulo 180, so a near-vertical line read as +89.9 in
    # calibration and -89.9 now is the same line, not a 180 degree change.
    upright = PoseLandmarks(
        nose=Landmark(0.5, 0.3),
        left_ear=Landmark(0.5000, 0.20),
        right_ear=Landmark(0.5001, 0.30),
        left_shoulder=Landmark(0.60, 0.50),
        right_shoulder=Landmark(0.40, 0.50),
    )
    flipped = PoseLandmarks(
        nose=Landmark(0.5, 0.3),
        left_ear=Landmark(0.5001, 0.20),
        right_ear=Landmark(0.5000, 0.30),
        left_shoulder=Landmark(0.60, 0.50),
        right_shoulder=Landmark(0.40, 0.50),
    )
    baseline = compute_baseline([upright])
    deviation = compute_deviation(flipped, baseline)
    assert abs(deviation.head_tilt_deg) < 5.0


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


def test_hysteresis_notifies_once_after_notify_threshold():
    clock = FakeClock()
    timer = HysteresisTimer(grace_period_seconds=10.0, notify_after_seconds=5.0, clock=clock)
    timer.update(posture_ok=False)

    clock.advance(3.0)
    result = timer.update(posture_ok=False)
    assert result.should_notify is False

    clock.advance(2.5)  # total 5.5s, past the 5s notify threshold
    result = timer.update(posture_ok=False)
    assert result.status == Status.WARN  # still under the 10s overlay threshold
    assert result.should_notify is True

    clock.advance(1.0)
    result = timer.update(posture_ok=False)
    assert result.should_notify is False  # only fires once per continuous violation


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


def dev(head=0.0, pitch=0.0, shoulder=None) -> Deviation:
    return Deviation(head_tilt_deg=head, head_pitch_deg=pitch, shoulder_tilt_deg=shoulder)


def test_smoother_passes_the_first_sample_through_unchanged():
    smoother = DeviationSmoother(alpha=0.3)
    result = smoother.update(dev(head=10.0))
    assert result.head_tilt_deg == 10.0


def test_smoother_damps_a_single_jitter_spike():
    smoother = DeviationSmoother(alpha=0.3)
    for _ in range(5):
        smoother.update(dev(head=0.0))

    spiked = smoother.update(dev(head=30.0))
    # A lone bad frame must not drag the reading over a 8 deg threshold.
    assert spiked.head_tilt_deg < 10.0


def test_smoother_converges_on_a_sustained_change():
    smoother = DeviationSmoother(alpha=0.3)
    smoother.update(dev(head=0.0))
    result = smoother.update(dev(head=30.0))
    for _ in range(19):
        result = smoother.update(dev(head=30.0))
    assert result.head_tilt_deg > 25.0


def test_smoother_reset_drops_history():
    smoother = DeviationSmoother(alpha=0.3)
    for _ in range(10):
        smoother.update(dev(head=30.0))

    smoother.reset()
    assert smoother.update(dev(head=0.0)).head_tilt_deg == 0.0


def test_smoother_rejects_invalid_alpha():
    import pytest

    with pytest.raises(ValueError):
        DeviationSmoother(alpha=0.0)
    with pytest.raises(ValueError):
        DeviationSmoother(alpha=1.5)
