"""Entry point: wires camera capture, pose detection, Rich dashboard and the
tkinter overlay together.

Threading model:
- Main thread: owns the persistent tkinter root and its mainloop/after()
  polling. tkinter must live on the main thread.
- Background daemon thread: opens the camera, runs MediaPipe Pose, computes
  posture deviations/hysteresis, drives the Rich Live dashboard, and writes
  into a small thread-safe SharedState that the main thread polls to decide
  whether to show the overlay.

Pose detection uses the MediaPipe Tasks API (PoseLandmarker) rather than the
legacy `mp.solutions.pose` API: the legacy solutions module is no longer
shipped in current mediapipe pip wheels for Python 3.12 (verified absent in
both 0.10.35 and 1.0.1). PoseLandmarker needs a small .task model file,
downloaded once on first run and cached under POSE_MODEL_PATH — no network
calls happen after that.
"""

from __future__ import annotations

import math
import signal
import subprocess
import sys
import threading
import time
import urllib.request
from datetime import datetime, timezone

import cv2
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions, vision
from rich.console import Console

from posture_tracker import overlay
from posture_tracker.config import (
    MIN_LANDMARK_VISIBILITY,
    POSE_MODEL_PATH,
    POSE_MODEL_URL,
    SHOULDER_MIN_VISIBILITY,
    Settings,
    parse_args,
)
from posture_tracker.detector import (
    Baseline,
    DeviationSmoother,
    HysteresisTimer,
    Landmark,
    PoseLandmarks,
    Status,
    compute_baseline,
    compute_deviation,
)
from posture_tracker.storage import SessionSummary, save_session
from posture_tracker.ui import Dashboard, DashboardState


class CameraError(RuntimeError):
    pass


_notify_available = True


def send_notification(title: str, message: str) -> None:
    """Fires a desktop notification via notify-send (libnotify). Works the
    same whether the app is in a foreground terminal or backgrounded with
    nohup/disown, since it talks to the desktop's notification daemon over
    D-Bus rather than the terminal. Silently no-ops if notify-send isn't
    installed, so a missing dependency never crashes tracking.

    Fire-and-forget on purpose: this runs on the capture thread, and waiting
    on the notification daemon would stall posture tracking for as long as it
    takes to answer.
    """
    global _notify_available
    if not _notify_available:
        return
    try:
        subprocess.Popen(
            ["notify-send", "-a", "Posture Tracker", "-u", "normal", title, message],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except FileNotFoundError:
        _notify_available = False


class SharedState:
    """Thread-safe flag read by the main (tkinter) thread, written by the
    background capture thread."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._show_overlay = False

    def set_show_overlay(self, value: bool) -> None:
        with self._lock:
            self._show_overlay = value

    def get_show_overlay(self) -> bool:
        with self._lock:
            return self._show_overlay


def _parse_camera_device(device: str) -> int | str:
    stripped = device.strip()
    if stripped.lstrip("-").isdigit():
        return int(stripped)
    return device


def open_camera(device: str) -> cv2.VideoCapture:
    """Opens the webcam through the native V4L2 backend.

    Left to itself OpenCV probes backends and settles on FFmpeg's
    video4linux2 demuxer, which costs ~1.1s of startup and prints
    "ioctl(VIDIOC_QBUF): Bad file descriptor" to stderr on the way (harmless,
    but it lands in the middle of the dashboard). Naming the backend skips the
    probe entirely: measured 2ms to open, and no stray output.
    """
    identifier = _parse_camera_device(device)
    cap = cv2.VideoCapture(identifier, cv2.CAP_V4L2)
    if not cap.isOpened():
        cap.release()
        raise CameraError(
            f"Could not open camera '{device}'. "
            "Check that the device is connected, accessible, and not in use by another application."
        )
    return cap


def ensure_pose_model(console: Console) -> None:
    """Downloads the PoseLandmarker .task model on first run and caches it
    locally. No-op if already present."""
    if POSE_MODEL_PATH.exists():
        return

    console.print(
        f"[cyan]Downloading pose model (one-time, ~6MB) to {POSE_MODEL_PATH}...[/cyan]"
    )
    POSE_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = POSE_MODEL_PATH.with_suffix(".tmp")
    try:
        urllib.request.urlretrieve(POSE_MODEL_URL, tmp_path)
        tmp_path.rename(POSE_MODEL_PATH)
    except OSError as exc:
        tmp_path.unlink(missing_ok=True)
        raise RuntimeError(
            f"Could not download the pose model from {POSE_MODEL_URL}: {exc}. "
            "Check your internet connection and try again."
        ) from exc


class PoseSource:
    """Wraps MediaPipe Tasks PoseLandmarker (VIDEO mode) for a single camera
    stream, tracking a monotonically increasing timestamp as detect_for_video
    requires."""

    def __init__(self, model_path):
        options = vision.PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(model_path)),
            running_mode=vision.RunningMode.VIDEO,
            num_poses=1,
            min_pose_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self._landmarker = vision.PoseLandmarker.create_from_options(options)
        self._start = time.monotonic()

    def detect(self, frame_bgr):
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        timestamp_ms = int((time.monotonic() - self._start) * 1000)
        return self._landmarker.detect_for_video(mp_image, timestamp_ms)

    def close(self) -> None:
        self._landmarker.close()

    def __enter__(self) -> "PoseSource":
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()


_IDX_NOSE = vision.PoseLandmark.NOSE.value
_IDX_LEFT_EAR = vision.PoseLandmark.LEFT_EAR.value
_IDX_RIGHT_EAR = vision.PoseLandmark.RIGHT_EAR.value
_IDX_LEFT_SHOULDER = vision.PoseLandmark.LEFT_SHOULDER.value
_IDX_RIGHT_SHOULDER = vision.PoseLandmark.RIGHT_SHOULDER.value

# A grab() that returns faster than this was served from the driver's queue,
# i.e. it handed back an already-captured (stale) frame. One that takes longer
# waited on the sensor, so it is current.
_QUEUED_GRAB_SECONDS = 0.008
_MAX_DRAIN_FRAMES = 8


def read_fresh_frame(cap: cv2.VideoCapture):
    """Returns the newest frame, dropping any the driver has queued up.

    The camera keeps producing at its native rate (30 fps here) while the
    analysis loop runs far slower, so frames pile up in the V4L2 queue and a
    plain read() hands back the *oldest* one -- measured at ~3 frames, i.e.
    ~100ms of lag, which shows up as the app being slow to notice that you
    straightened up. Neither CAP_PROP_BUFFERSIZE nor CAP_PROP_FPS is honoured
    by this backend, so drain the queue instead and keep only the last frame.
    """
    grabbed = False
    for _ in range(_MAX_DRAIN_FRAMES):
        started = time.monotonic()
        if not cap.grab():
            break
        grabbed = True
        if time.monotonic() - started > _QUEUED_GRAB_SECONDS:
            break  # this grab waited on the sensor, so it is a fresh frame

    if not grabbed:
        return None
    ok, frame = cap.retrieve()
    return frame if ok else None


def _read_landmarks(cap: cv2.VideoCapture, pose_source: PoseSource, min_visibility: float) -> PoseLandmarks | None:
    frame = read_fresh_frame(cap)
    if frame is None:
        return None

    result = pose_source.detect(frame)
    if not result.pose_landmarks:
        return None

    landmark_list = result.pose_landmarks[0]

    # MediaPipe normalizes x by frame width and y by height, so on a 4:3 frame
    # a vertical offset is stretched 1.33x against a horizontal one. Rescaling
    # x by the aspect ratio puts both axes in the same units, which is what
    # makes the tilt angles in detector.py real degrees.
    height, width = frame.shape[:2]
    aspect_ratio = width / height if height else 1.0

    def get(index: int) -> Landmark:
        lm = landmark_list[index]
        return Landmark(x=lm.x * aspect_ratio, y=lm.y, visibility=lm.visibility)

    pts = PoseLandmarks(
        nose=get(_IDX_NOSE),
        left_ear=get(_IDX_LEFT_EAR),
        right_ear=get(_IDX_RIGHT_EAR),
        left_shoulder=get(_IDX_LEFT_SHOULDER),
        right_shoulder=get(_IDX_RIGHT_SHOULDER),
    )
    if not pts.face_visible(min_visibility):
        return None
    return pts


def _setup_state(**overrides) -> DashboardState:
    """A blank dashboard state for the pre-tracking phases."""
    base = dict(
        status=Status.PAUSED,
        violation_seconds=0.0,
        session_seconds=0.0,
        good_posture_pct=100.0,
        violation_count=0,
        deviation=None,
    )
    base.update(overrides)
    return DashboardState(**base)


def _run_countdown(
    dashboard: Dashboard,
    cap: cv2.VideoCapture,
    settings: Settings,
    stop_event: threading.Event,
) -> None:
    """Gives the user a few seconds to actually sit the way they want the
    baseline captured, instead of calibrating against whatever posture they
    happened to be in while launching the app.

    Frames are pulled and discarded throughout so the camera's auto-exposure
    has settled before the samples that matter are taken (measured ramping
    from ~134 to ~163 mean brightness over the first second). Pose inference
    is deliberately skipped here -- nothing uses the result, and it is by far
    the most expensive part of a frame.
    """
    deadline = time.monotonic() + settings.calibration_countdown_seconds
    while not stop_event.is_set():
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        read_fresh_frame(cap)
        dashboard.update(_setup_state(countdown_seconds=math.ceil(remaining)))
        time.sleep(0.05)


def _calibrate(
    dashboard: Dashboard,
    cap: cv2.VideoCapture,
    pose_source: PoseSource,
    settings: Settings,
    stop_event: threading.Event,
) -> Baseline | None:
    samples: list[PoseLandmarks] = []
    deadline = time.monotonic() + settings.calibration_seconds
    frame_interval = 1.0 / settings.analysis_fps

    while time.monotonic() < deadline and not stop_event.is_set():
        loop_start = time.monotonic()
        landmarks = _read_landmarks(cap, pose_source, MIN_LANDMARK_VISIBILITY)
        if landmarks is not None:
            samples.append(landmarks)

        progress = min(
            100.0,
            (settings.calibration_seconds - (deadline - time.monotonic()))
            / settings.calibration_seconds
            * 100.0,
        )
        dashboard.update(_setup_state(calibrating=True, calibration_progress=progress))

        elapsed = time.monotonic() - loop_start
        sleep_time = frame_interval - elapsed
        if sleep_time > 0:
            time.sleep(sleep_time)

    if stop_event.is_set() or not samples:
        return None
    return compute_baseline(samples, shoulder_min_visibility=SHOULDER_MIN_VISIBILITY)


def run_capture_loop(
    cap: cv2.VideoCapture,
    settings: Settings,
    shared: SharedState,
    stop_event: threading.Event,
) -> None:
    console = Console()

    session_start = datetime.now(timezone.utc)
    total_elapsed = 0.0
    tracked_elapsed = 0.0
    good_elapsed = 0.0
    violation_count = 0
    good_pct = 100.0
    calibrated = False

    try:
        with PoseSource(POSE_MODEL_PATH) as pose_source, Dashboard() as dashboard:
            _run_countdown(dashboard, cap, settings, stop_event)
            if stop_event.is_set():
                return

            baseline = _calibrate(dashboard, cap, pose_source, settings, stop_event)
            if stop_event.is_set():
                return
            if baseline is None:
                console.print(
                    "[bold red]Calibration failed:[/bold red] no face detected in frame. "
                    "Check the lighting and camera position, then restart the application."
                )
                stop_event.set()
                return

            calibrated = True
            timer = HysteresisTimer(
                grace_period_seconds=settings.grace_period_seconds,
                notify_after_seconds=settings.notify_after_seconds,
            )
            smoother = DeviationSmoother(settings.smoothing_alpha)
            session_start = datetime.now(timezone.utc)
            prev_status = Status.OK
            frame_interval = 1.0 / settings.analysis_fps
            last_time = time.monotonic()

            while not stop_event.is_set():
                loop_start = time.monotonic()
                dt = loop_start - last_time
                last_time = loop_start

                landmarks = _read_landmarks(cap, pose_source, MIN_LANDMARK_VISIBILITY)
                if landmarks is None:
                    smoother.reset()
                    result = timer.update(None)
                    deviation = None
                else:
                    deviation = smoother.update(
                        compute_deviation(
                            landmarks, baseline, shoulder_min_visibility=SHOULDER_MIN_VISIBILITY
                        )
                    )
                    result = timer.update(deviation.within(settings))

                total_elapsed += dt
                # Time spent away from the desk is neither good nor bad posture,
                # so it must not drag the percentage down.
                if result.status != Status.PAUSED:
                    tracked_elapsed += dt
                    if result.status == Status.OK:
                        good_elapsed += dt
                if result.status == Status.ALERT and prev_status != Status.ALERT:
                    violation_count += 1
                prev_status = result.status

                if result.should_notify:
                    send_notification(
                        "Posture Tracker",
                        f"You've been slouching for {result.violation_seconds:.0f}s — straighten up.",
                    )

                good_pct = (good_elapsed / tracked_elapsed * 100.0) if tracked_elapsed > 0 else 100.0
                shared.set_show_overlay(result.status == Status.ALERT)

                dashboard.update(
                    DashboardState(
                        status=result.status,
                        violation_seconds=result.violation_seconds,
                        session_seconds=total_elapsed,
                        good_posture_pct=good_pct,
                        violation_count=violation_count,
                        deviation=deviation,
                    )
                )

                elapsed = time.monotonic() - loop_start
                sleep_time = frame_interval - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)
    finally:
        # In the finally block so a crash mid-loop still records the session
        # rather than silently losing it.
        if calibrated and total_elapsed > 0:
            try:
                save_session(
                    SessionSummary(
                        started_at=session_start,
                        duration_seconds=total_elapsed,
                        good_posture_pct=good_pct,
                        violation_count=violation_count,
                    )
                )
            except Exception as exc:  # storage must never block a clean shutdown
                console.print(f"[yellow]Could not save session stats: {exc}[/yellow]")
        shared.set_show_overlay(False)
        cap.release()
        stop_event.set()


def main() -> None:
    settings = parse_args(sys.argv[1:])
    console = Console()

    # Download before claiming the camera, so a slow first-run download does
    # not hold the device open (and locked away from other apps) meanwhile.
    try:
        ensure_pose_model(console)
    except RuntimeError as exc:
        console.print(f"[bold red]{exc}[/bold red]")
        sys.exit(1)

    try:
        cap = open_camera(settings.camera)
    except CameraError as exc:
        console.print(f"[bold red]{exc}[/bold red]")
        sys.exit(1)

    shared = SharedState()
    stop_event = threading.Event()

    # SIGINT (Ctrl+C) is turned into KeyboardInterrupt by Python's default
    # handler, but SIGTERM has no such default — and SIGTERM (not SIGINT) is
    # how backgrounded/daemonized processes are normally stopped (`kill`,
    # `systemctl stop`, session logout). Route it through the same stop_event
    # the rest of shutdown already uses so the camera and overlay are always
    # cleaned up, not just on Ctrl+C.
    def _handle_sigterm(signum, frame) -> None:
        stop_event.set()

    signal.signal(signal.SIGTERM, _handle_sigterm)

    thread = threading.Thread(
        target=run_capture_loop,
        args=(cap, settings, shared, stop_event),
        daemon=True,
    )
    thread.start()

    root = overlay.make_root()
    ov = overlay.Overlay(root)

    try:
        overlay.run_poll_loop(
            root,
            ov,
            should_show_overlay=shared.get_show_overlay,
            should_stop=stop_event.is_set,
        )
    except KeyboardInterrupt:
        pass
    finally:
        stop_event.set()
        thread.join(timeout=5)
        try:
            root.destroy()
        except Exception:
            pass


if __name__ == "__main__":
    main()
