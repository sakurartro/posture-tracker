"""Entry point: wires camera capture, head-pose detection, Rich dashboard and
the tkinter overlay together.

Threading model:
- Main thread: owns the persistent tkinter root and its mainloop/after()
  polling. tkinter must live on the main thread.
- Background daemon thread: opens the camera, runs the face detector, computes
  posture deviations/hysteresis, drives the Rich Live dashboard, and writes
  into a small thread-safe SharedState that the main thread polls to decide
  whether to show the overlay.

Detection uses MediaPipe's Face Landmarker, which reports the head's 3D
orientation directly. It needs a small .task model file, downloaded once on
first run and cached under FACE_MODEL_PATH — no network calls happen after
that. See detector.py for why this replaced the Pose (whole-body) model.
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

from posture_tracker import camera_check, overlay
from posture_tracker.config import (
    CAPTURE_HEIGHT,
    CAPTURE_WIDTH,
    FACE_MODEL_PATH,
    FACE_MODEL_URL,
    INFERENCE_DOWNSCALE,
    Settings,
    parse_args,
)
from posture_tracker.detector import (
    Baseline,
    DeviationSmoother,
    HeadPose,
    HysteresisTimer,
    Status,
    compute_baseline,
    compute_deviation,
    head_pose_from_matrix,
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
    # Widescreen, so the driver hands over the sensor's full field of view
    # instead of a cropped 4:3 window of it. See CAPTURE_WIDTH in config.
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAPTURE_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAPTURE_HEIGHT)
    return cap


def ensure_face_model(console: Console) -> None:
    """Downloads the Face Landmarker .task model on first run and caches it
    locally. No-op if already present."""
    if FACE_MODEL_PATH.exists():
        return

    console.print(
        f"[cyan]Downloading face model (one-time, ~4MB) to {FACE_MODEL_PATH}...[/cyan]"
    )
    FACE_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = FACE_MODEL_PATH.with_suffix(".tmp")
    try:
        urllib.request.urlretrieve(FACE_MODEL_URL, tmp_path)
        tmp_path.rename(FACE_MODEL_PATH)
    except OSError as exc:
        tmp_path.unlink(missing_ok=True)
        raise RuntimeError(
            f"Could not download the face model from {FACE_MODEL_URL}: {exc}. "
            "Check your internet connection and try again."
        ) from exc


class FaceSource:
    """Wraps MediaPipe Tasks FaceLandmarker (VIDEO mode) for a single camera
    stream, tracking a monotonically increasing timestamp as detect_for_video
    requires."""

    def __init__(self, model_path):
        options = vision.FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(model_path)),
            running_mode=vision.RunningMode.VIDEO,
            num_faces=1,
            output_facial_transformation_matrixes=True,
        )
        self._landmarker = vision.FaceLandmarker.create_from_options(options)
        self._start = time.monotonic()

    def _detect(self, frame_bgr):
        if INFERENCE_DOWNSCALE > 1:
            frame_bgr = cv2.resize(
                frame_bgr, None,
                fx=1 / INFERENCE_DOWNSCALE, fy=1 / INFERENCE_DOWNSCALE,
                interpolation=cv2.INTER_AREA,
            )
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        timestamp_ms = int((time.monotonic() - self._start) * 1000)
        return self._landmarker.detect_for_video(mp_image, timestamp_ms)

    def read_frame(self, cap):
        """Exposed so the camera preview shares this module's capture path."""
        return read_fresh_frame(cap)

    def head_pose(self, frame_bgr) -> HeadPose | None:
        """The head's orientation, or None when no face is really visible."""
        result = self._detect(frame_bgr)
        if not result.facial_transformation_matrixes:
            return None
        return head_pose_from_matrix(result.facial_transformation_matrixes[0])

    def face_bounds(self, frame_bgr):
        """Normalized (x0, y0, x1, y1) around the face, or None."""
        result = self._detect(frame_bgr)
        if not result.face_landmarks:
            return None
        points = result.face_landmarks[0]
        xs = [p.x for p in points]
        ys = [p.y for p in points]
        return min(xs), min(ys), max(xs), max(ys)

    def close(self) -> None:
        self._landmarker.close()

    def __enter__(self) -> "FaceSource":
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()


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


def _read_head_pose(cap: cv2.VideoCapture, face_source: FaceSource) -> HeadPose | None:
    frame = read_fresh_frame(cap)
    if frame is None:
        return None
    return face_source.head_pose(frame)


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
    from ~134 to ~163 mean brightness over the first second). Face inference
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
    face_source: FaceSource,
    settings: Settings,
    stop_event: threading.Event,
) -> Baseline | None:
    samples: list[HeadPose] = []
    deadline = time.monotonic() + settings.calibration_seconds
    frame_interval = 1.0 / settings.analysis_fps

    while time.monotonic() < deadline and not stop_event.is_set():
        loop_start = time.monotonic()
        pose = _read_head_pose(cap, face_source)
        if pose is not None:
            samples.append(pose)

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
    return compute_baseline(samples)


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
        with FaceSource(FACE_MODEL_PATH) as face_source, Dashboard() as dashboard:
            _run_countdown(dashboard, cap, settings, stop_event)
            if stop_event.is_set():
                return

            baseline = _calibrate(dashboard, cap, face_source, settings, stop_event)
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

                head_pose = _read_head_pose(cap, face_source)
                if head_pose is None:
                    smoother.reset()
                    result = timer.update(None)
                    deviation = None
                else:
                    deviation = smoother.update(compute_deviation(head_pose, baseline))
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
        ensure_face_model(console)
    except RuntimeError as exc:
        console.print(f"[bold red]{exc}[/bold red]")
        sys.exit(1)

    try:
        cap = open_camera(settings.camera)
    except CameraError as exc:
        console.print(f"[bold red]{exc}[/bold red]")
        sys.exit(1)

    if settings.check_camera:
        try:
            with FaceSource(FACE_MODEL_PATH) as face_source:
                framed = camera_check.run_preview(cap, face_source)
        finally:
            cap.release()
        if framed:
            console.print("[green]Camera looks good — run posture-tracker to start.[/green]")
        else:
            console.print(
                "[yellow]Your head was never fully in frame.[/yellow] Tracking needs to see "
                "your whole head, so aim the camera and run --check-camera again."
            )
        sys.exit(0 if framed else 1)

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
