"""Tunable constants and CLI argument parsing for posture-tracker."""

from __future__ import annotations

import argparse
from dataclasses import dataclass

from posture_tracker import paths

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

def _default_camera_device() -> str:
    """Linux names cameras by device file; macOS and Windows use an index."""
    return "/dev/video0" if paths.is_linux() else "0"


DEFAULT_CAMERA_DEVICE = _default_camera_device()

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
FACE_MODEL_PATH = paths.data_dir() / "models" / "face_landmarker.task"


@dataclass(frozen=True)
class Settings:
    """Behaviour knobs.

    Deliberately not exposed on the command line: the app has a three-command
    surface (set up, check stats, stop) and every one of these has a value that
    was chosen from measurements rather than guessed. Edit them here if you
    need to.
    """

    camera: str = DEFAULT_CAMERA_DEVICE
    calibration_countdown_seconds: float = CALIBRATION_COUNTDOWN_SECONDS
    calibration_seconds: float = CALIBRATION_SECONDS
    smoothing_alpha: float = SMOOTHING_ALPHA
    notify_after_seconds: float = NOTIFY_AFTER_SECONDS
    grace_period_seconds: float = GRACE_PERIOD_SECONDS
    analysis_fps: int = ANALYSIS_FPS
    head_tilt_threshold_deg: float = HEAD_TILT_THRESHOLD_DEG
    head_pitch_threshold_deg: float = HEAD_PITCH_THRESHOLD_DEG


@dataclass(frozen=True)
class Command:
    """What the user asked for."""

    calibrate: bool = False  # redo calibration even if already set up
    stats: bool = False
    stop: bool = False
    foreground: bool = False  # run the tracker here; also what autostart uses


def parse_args(argv: list[str] | None = None) -> Command:
    parser = argparse.ArgumentParser(
        prog="posture-tracker",
        description=(
            "Reminds you to sit up straight, using your webcam. "
            "Run with no arguments to set it up: it checks the camera can see you, "
            "calibrates against your good posture, then keeps watching in the "
            "background and starts itself again at every login."
        ),
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--stats", action="store_true",
                       help="Show posture statistics for today, the last week and the last month")
    group.add_argument("--stop", action="store_true",
                       help="Stop the background tracker and remove it from autostart")
    group.add_argument("--calibrate", action="store_true",
                       help="Calibrate again, e.g. after moving the laptop or changing chair")
    group.add_argument("--foreground", action="store_true",
                       help="Run the tracker in this terminal, with the live dashboard, "
                            "instead of in the background (this is what autostart runs)")

    ns = parser.parse_args(argv)
    return Command(calibrate=ns.calibrate, stats=ns.stats, stop=ns.stop,
                   foreground=ns.foreground)
