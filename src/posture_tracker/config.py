"""Tunable constants and CLI argument parsing for posture-tracker."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

# Seconds of "get into position" countdown before calibration starts sampling.
# Without it the baseline is captured while you are still reaching for the
# keyboard, and a skewed baseline means permanent false "bad posture".
CALIBRATION_COUNTDOWN_SECONDS = 3.0
CALIBRATION_SECONDS = 3.0
# Seconds of continuous bad posture before a background desktop notification fires.
NOTIFY_AFTER_SECONDS = 5.0
# Seconds of continuous bad posture before the fullscreen overlay fires.
GRACE_PERIOD_SECONDS = 10.0
ANALYSIS_FPS = 5

# Sideways head tilt (roll) and forward chin drop (pitch), in degrees away
# from the calibrated posture. Tuned against a measured session: sitting
# upright the readings stayed within 0.8 deg of roll and 2.9 deg of pitch,
# while a genuine slouch sat around 8 deg of pitch -- so 6 deg splits the two
# with room on both sides.
HEAD_TILT_THRESHOLD_DEG = 8.0
HEAD_PITCH_THRESHOLD_DEG = 6.0

# Weight of the newest sample in the deviation moving average. Landmark jitter
# alone can cross a threshold; averaging over roughly the last second keeps the
# status stable without noticeably delaying a real posture change.
SMOOTHING_ALPHA = 0.35

DEFAULT_CAMERA_DEVICE = "/dev/video0"

# Ask for a widescreen capture rather than accepting the driver's default.
# A MacBook's FaceTime HD sensor offers 1280x720 and nothing else, so when
# OpenCV defaulted to 640x480 the driver produced that 4:3 frame by cropping
# the sides off the 16:9 sensor -- throwing away about a quarter of the
# horizontal field of view. That is enough to push someone sitting squarely in
# front of the laptop out past the edge of frame.
CAPTURE_WIDTH = 1280
CAPTURE_HEIGHT = 720

# Frames are shrunk by this factor before inference. The full-width capture is
# what preserves the field of view; the detector does not need the extra
# pixels, and downscaling keeps the per-frame cost where it was at 640x480.
INFERENCE_DOWNSCALE = 2

# Posture is read from the head's 3D orientation via MediaPipe's Face
# Landmarker. It replaced the Pose (whole-body) model, which on a laptop webcam
# had no body to look at: it extrapolated unseen joints and, on a motionless
# subject, produced 12 deg of roll noise against the ~8 deg being measured.
# Face Landmarker is also about seven times cheaper (8ms vs 56ms per frame) and,
# crucially, reports nothing at all when it cannot really see a face -- so
# "is the user in view" stops being a guess.
#
# The Tasks API needs a .task model file, downloaded once on first run and
# cached locally; no network calls happen after that.
FACE_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/latest/face_landmarker.task"
)
FACE_MODEL_PATH = Path.home() / ".local" / "share" / "posture-tracker" / "models" / "face_landmarker.task"


@dataclass(frozen=True)
class Settings:
    camera: str = DEFAULT_CAMERA_DEVICE
    check_camera: bool = False
    calibration_countdown_seconds: float = CALIBRATION_COUNTDOWN_SECONDS
    calibration_seconds: float = CALIBRATION_SECONDS
    smoothing_alpha: float = SMOOTHING_ALPHA
    notify_after_seconds: float = NOTIFY_AFTER_SECONDS
    grace_period_seconds: float = GRACE_PERIOD_SECONDS
    analysis_fps: int = ANALYSIS_FPS
    head_tilt_threshold_deg: float = HEAD_TILT_THRESHOLD_DEG
    head_pitch_threshold_deg: float = HEAD_PITCH_THRESHOLD_DEG


def parse_args(argv: list[str] | None = None) -> Settings:
    parser = argparse.ArgumentParser(
        prog="posture-tracker",
        description="Real-time posture tracker using a webcam.",
    )
    parser.add_argument("--camera", default=DEFAULT_CAMERA_DEVICE,
                         help=f"Camera device path or index (default: {DEFAULT_CAMERA_DEVICE})")
    parser.add_argument("--check-camera", action="store_true", dest="check_camera",
                         help="Open a mirrored preview to aim the camera, then exit. "
                              "Run this first if the tracker cannot see you.")
    parser.add_argument("--calibration-countdown", type=float, default=CALIBRATION_COUNTDOWN_SECONDS,
                         dest="calibration_countdown_seconds",
                         help="Seconds of countdown to get into position before calibration samples")
    parser.add_argument("--calibration-seconds", type=float, default=CALIBRATION_SECONDS,
                         help="Seconds to hold still during calibration")
    parser.add_argument("--smoothing", type=float, default=SMOOTHING_ALPHA, dest="smoothing_alpha",
                         help="Moving-average weight for the newest sample, 0-1 "
                              "(lower = steadier but slower to react)")
    parser.add_argument("--notify-after", type=float, default=NOTIFY_AFTER_SECONDS,
                         dest="notify_after_seconds",
                         help="Seconds of continuous bad posture before a desktop notification fires")
    parser.add_argument("--grace-period", type=float, default=GRACE_PERIOD_SECONDS,
                         dest="grace_period_seconds",
                         help="Seconds of continuous bad posture before the fullscreen overlay fires")
    parser.add_argument("--fps", type=int, default=ANALYSIS_FPS, dest="analysis_fps",
                         help="Analysis loop target frames per second")
    parser.add_argument("--head-tilt-threshold", type=float, default=HEAD_TILT_THRESHOLD_DEG,
                         dest="head_tilt_threshold_deg",
                         help="Allowed sideways head tilt deviation, degrees")
    parser.add_argument("--head-pitch-threshold", type=float, default=HEAD_PITCH_THRESHOLD_DEG,
                         dest="head_pitch_threshold_deg",
                         help="Allowed forward head/chin drop deviation, degrees (the slouch check)")

    ns = parser.parse_args(argv)
    return Settings(**vars(ns))
