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

HEAD_TILT_THRESHOLD_DEG = 8.0
SHOULDER_TILT_THRESHOLD_DEG = 10.0
SLOUCH_THRESHOLD_PCT = 12.0

# Weight of the newest sample in the deviation moving average. Landmark jitter
# alone can cross a threshold; averaging over roughly the last second keeps the
# status stable without noticeably delaying a real posture change.
SMOOTHING_ALPHA = 0.35

DEFAULT_CAMERA_DEVICE = "/dev/video0"

# Below this MediaPipe landmark visibility score, a keypoint is treated as absent.
MIN_LANDMARK_VISIBILITY = 0.5

# MediaPipe's legacy `mp.solutions.pose` API is no longer shipped in current
# pip wheels (verified absent in mediapipe 0.10.35 / 1.0.1 for Python 3.12) —
# only the Tasks API remains, which requires a downloadable .task model file.
# Downloaded once on first run and cached locally; no network calls after that.
POSE_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"
)
POSE_MODEL_PATH = Path.home() / ".local" / "share" / "posture-tracker" / "models" / "pose_landmarker_lite.task"


@dataclass(frozen=True)
class Settings:
    camera: str = DEFAULT_CAMERA_DEVICE
    calibration_countdown_seconds: float = CALIBRATION_COUNTDOWN_SECONDS
    calibration_seconds: float = CALIBRATION_SECONDS
    smoothing_alpha: float = SMOOTHING_ALPHA
    notify_after_seconds: float = NOTIFY_AFTER_SECONDS
    grace_period_seconds: float = GRACE_PERIOD_SECONDS
    analysis_fps: int = ANALYSIS_FPS
    head_tilt_threshold_deg: float = HEAD_TILT_THRESHOLD_DEG
    shoulder_tilt_threshold_deg: float = SHOULDER_TILT_THRESHOLD_DEG
    slouch_threshold_pct: float = SLOUCH_THRESHOLD_PCT


def parse_args(argv: list[str] | None = None) -> Settings:
    parser = argparse.ArgumentParser(
        prog="posture-tracker",
        description="Real-time posture tracker using a webcam.",
    )
    parser.add_argument("--camera", default=DEFAULT_CAMERA_DEVICE,
                         help=f"Camera device path or index (default: {DEFAULT_CAMERA_DEVICE})")
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
                         dest="head_tilt_threshold_deg", help="Allowed head tilt deviation, degrees")
    parser.add_argument("--shoulder-tilt-threshold", type=float, default=SHOULDER_TILT_THRESHOLD_DEG,
                         dest="shoulder_tilt_threshold_deg", help="Allowed shoulder tilt deviation, degrees")
    parser.add_argument("--slouch-threshold", type=float, default=SLOUCH_THRESHOLD_PCT,
                         dest="slouch_threshold_pct",
                         help="Allowed nose-to-shoulder-line distance change, percent")

    ns = parser.parse_args(argv)
    return Settings(**vars(ns))
