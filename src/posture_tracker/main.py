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

from posture_tracker import camera_check, overlay, paths, service, storage, ui
from posture_tracker.quiet import native_stderr_silenced
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


_WINDOWS_TOAST_SCRIPT = """
[Windows.UI.Notifications.ToastNotificationManager, Windows.UI.Notifications, ContentType=WindowsRuntime] > $null
$xml = [Windows.UI.Notifications.ToastNotificationManager]::GetTemplateContent(
    [Windows.UI.Notifications.ToastTemplateType]::ToastText02)
$texts = $xml.GetElementsByTagName("text")
$texts.Item(0).AppendChild($xml.CreateTextNode("%TITLE%")) > $null
$texts.Item(1).AppendChild($xml.CreateTextNode("%BODY%")) > $null
$toast = [Windows.UI.Notifications.ToastNotification]::new($xml)
[Windows.UI.Notifications.ToastNotificationManager]::CreateToastNotifier("Posture Tracker").Show($toast)
"""


def _notification_command(title: str, message: str) -> list[str]:
    """The platform's way of raising a desktop notification."""
    if paths.is_macos():
        script = (f'display notification {_applescript_string(message)} '
                  f'with title {_applescript_string(title)}')
        return ["osascript", "-e", script]
    if paths.is_windows():
        script = (_WINDOWS_TOAST_SCRIPT
                  .replace("%TITLE%", title.replace('"', "'"))
                  .replace("%BODY%", message.replace('"', "'")))
        return ["powershell", "-NoProfile", "-NonInteractive", "-Command", script]
    return ["notify-send", "-a", "Posture Tracker", "-u", "normal", title, message]


def _applescript_string(value: str) -> str:
    escaped = value.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


def send_notification(title: str, message: str) -> None:
    """Raises a desktop notification, whatever the platform provides.

    Works the same whether the app is in a foreground terminal or running in
    the background, because it talks to the desktop's notification service
    rather than the terminal. Silently gives up if the platform's notifier is
    missing, so an absent dependency never crashes posture tracking.

    Fire-and-forget on purpose: this runs on the capture thread, and waiting
    on the notification service would stall tracking for as long as it takes
    to answer.
    """
    global _notify_available
    if not _notify_available:
        return
    try:
        subprocess.Popen(
            _notification_command(title, message),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except (FileNotFoundError, OSError):
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


def _capture_backend() -> int:
    """The platform's native capture API, named explicitly.

    Left to itself OpenCV probes backends. On Linux it settled on FFmpeg's
    video4linux2 demuxer, which cost ~1.1s of startup and printed
    "ioctl(VIDIOC_QBUF): Bad file descriptor" into the middle of the
    dashboard; naming V4L2 dropped that to 2ms with no stray output. The same
    reasoning picks AVFoundation on macOS and DirectShow on Windows, both of
    which also honour an explicit capture resolution more reliably than the
    probe result does.
    """
    if paths.is_macos():
        return cv2.CAP_AVFOUNDATION
    if paths.is_windows():
        return cv2.CAP_DSHOW
    return cv2.CAP_V4L2


def open_camera(device: str) -> cv2.VideoCapture:
    """Opens the webcam through the platform's native capture backend."""
    identifier = _parse_camera_device(device)
    cap = cv2.VideoCapture(identifier, _capture_backend())
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
        with native_stderr_silenced():
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


def _run_countdown(dashboard: Dashboard, cap, settings: Settings) -> None:
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
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return
        read_fresh_frame(cap)
        dashboard.update(_setup_state(countdown_seconds=math.ceil(remaining)))
        time.sleep(0.05)


def _collect_baseline(dashboard: Dashboard, cap, face_source: FaceSource,
                      settings: Settings) -> Baseline | None:
    samples: list[HeadPose] = []
    deadline = time.monotonic() + settings.calibration_seconds
    frame_interval = 1.0 / settings.analysis_fps

    while time.monotonic() < deadline:
        loop_start = time.monotonic()
        pose = _read_head_pose(cap, face_source)
        if pose is not None:
            samples.append(pose)

        progress = min(100.0, (settings.calibration_seconds
                               - (deadline - time.monotonic()))
                       / settings.calibration_seconds * 100.0)
        dashboard.update(_setup_state(calibrating=True, calibration_progress=progress))

        elapsed = time.monotonic() - loop_start
        if (sleep_time := frame_interval - elapsed) > 0:
            time.sleep(sleep_time)

    if not samples:
        return None
    return compute_baseline(samples)


def run_tracking(cap, settings: Settings, baseline: Baseline,
                 shared: SharedState, stop_event: threading.Event,
                 show_dashboard: bool) -> None:
    """The watching loop. Runs on a background thread while the main thread
    owns tkinter."""
    console = Console()
    session_start = datetime.now(timezone.utc)
    total_elapsed = tracked_elapsed = good_elapsed = 0.0
    violation_count = 0

    timer = HysteresisTimer(
        grace_period_seconds=settings.grace_period_seconds,
        notify_after_seconds=settings.notify_after_seconds,
    )
    smoother = DeviationSmoother(settings.smoothing_alpha)
    frame_interval = 1.0 / settings.analysis_fps
    prev_status = Status.OK
    last_time = time.monotonic()

    try:
        with FaceSource(FACE_MODEL_PATH) as face_source:
            # No dashboard when the output is a log file: Rich would fill it
            # with redraw escape codes, and nobody is watching anyway.
            dashboard = Dashboard() if show_dashboard else None
            if dashboard:
                dashboard.__enter__()
            try:
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
                    # Time spent away from the desk is neither good nor bad
                    # posture, so it must not skew the percentage.
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

                    shared.set_show_overlay(result.status == Status.ALERT)

                    if dashboard:
                        good_pct = (good_elapsed / tracked_elapsed * 100.0
                                    if tracked_elapsed > 0 else 100.0)
                        dashboard.update(DashboardState(
                            status=result.status,
                            violation_seconds=result.violation_seconds,
                            session_seconds=total_elapsed,
                            good_posture_pct=good_pct,
                            violation_count=violation_count,
                            deviation=deviation,
                        ))

                    elapsed = time.monotonic() - loop_start
                    if (sleep_time := frame_interval - elapsed) > 0:
                        time.sleep(sleep_time)
            finally:
                if dashboard:
                    dashboard.__exit__(None, None, None)
    finally:
        # In the finally block so a crash mid-session still records it.
        if total_elapsed > 0:
            try:
                save_session(SessionSummary(
                    started_at=session_start,
                    duration_seconds=total_elapsed,
                    tracked_seconds=tracked_elapsed,
                    good_seconds=good_elapsed,
                    violation_count=violation_count,
                ))
            except Exception as exc:  # storage must never block a clean shutdown
                console.print(f"[yellow]Could not save session stats: {exc}[/yellow]")
        shared.set_show_overlay(False)
        cap.release()
        stop_event.set()


def calibrate(cap, settings: Settings, console: Console) -> Baseline | None:
    """Camera check, countdown, then sampling. Returns None if it failed."""
    with FaceSource(FACE_MODEL_PATH) as face_source:
        console.print(
            "[bold]Step 1/2 — camera[/bold]  A preview will open. "
            "Sit how you normally work; it closes itself once it can see you."
        )
        if not camera_check.run_preview(cap, face_source, timeout_seconds=120):
            console.print(
                "[yellow]Could not get a clear view of your head.[/yellow] "
                "Aim the camera at yourself and try again."
            )
            return None

        console.print("[bold]Step 2/2 — calibration[/bold]  Sit the way you want to be reminded to sit.")
        with Dashboard() as dashboard:
            _run_countdown(dashboard, cap, settings)
            baseline = _collect_baseline(dashboard, cap, face_source, settings)

    if baseline is None:
        console.print("[bold red]Calibration failed:[/bold red] no face detected while sampling.")
    return baseline


def cmd_setup(settings: Settings, console: Console, force_calibrate: bool) -> int:
    """Default command: make sure the tracker is calibrated, installed and running."""
    try:
        ensure_face_model(console)
    except RuntimeError as exc:
        console.print(f"[bold red]{exc}[/bold red]")
        return 1

    saved = storage.load_baseline()
    if saved is not None and not force_calibrate:
        service.install_autostart()
        if service.running_pid() is not None:
            console.print("[green]Posture Tracker is already running.[/green] "
                          "Use --stats to see how you are doing, or --stop to turn it off.")
            return 0
        pid = service.start_background()
        console.print(f"[green]Posture Tracker is watching again[/green] (pid {pid}).")
        return 0

    # A tracker already holding the camera would block calibration.
    # stop_background waits for it to actually let the device go.
    if service.stop_background():
        console.print("Stopped the running tracker to recalibrate.")

    try:
        cap = open_camera(settings.camera)
    except CameraError as exc:
        console.print(f"[bold red]{exc}[/bold red]")
        return 1

    try:
        baseline = calibrate(cap, settings, console)
    finally:
        cap.release()

    if baseline is None:
        return 1

    storage.save_baseline(baseline.roll_deg, baseline.pitch_deg)
    service.install_autostart()
    pid = service.start_background()

    console.print()
    console.print("[bold green]All set.[/bold green]")
    console.print(f"  Watching in the background (pid {pid}), and starting itself at every login.")
    console.print("  [bold]posture-tracker --stats[/bold]  how your posture has been")
    console.print("  [bold]posture-tracker --stop[/bold]   stop it and remove it from autostart")
    return 0


def cmd_stats(console: Console) -> int:
    periods = storage.recent_stats()
    all_time = periods[-1]
    level = storage.level_progress(storage.points_earned(all_time.tracked_seconds))
    console.print(ui.render_stats(periods, level))
    if service.running_pid() is None:
        console.print("[dim]Tracker is not running. Run posture-tracker to start it.[/dim]")
    return 0


def cmd_stop(console: Console) -> int:
    was_running = service.stop_background()
    was_installed = service.remove_autostart()
    if was_running or was_installed:
        console.print("[green]Stopped.[/green] Removed from autostart; "
                      "run [bold]posture-tracker[/bold] to set it up again.")
    else:
        console.print("Posture Tracker was not running.")
    return 0


def cmd_foreground(settings: Settings, console: Console) -> int:
    """The tracker itself. Started by autostart, or by hand to watch it live."""
    if service.running_pid() is not None:
        console.print("[yellow]Posture Tracker is already running.[/yellow]")
        return 1

    saved = storage.load_baseline()
    if saved is None:
        console.print("[bold red]Not calibrated yet.[/bold red] "
                      "Run [bold]posture-tracker[/bold] first.")
        return 1
    baseline = Baseline(roll_deg=saved[0], pitch_deg=saved[1])

    try:
        ensure_face_model(console)
        cap = open_camera(settings.camera)
    except (RuntimeError, CameraError) as exc:
        console.print(f"[bold red]{exc}[/bold red]")
        return 1

    service.write_pid_file()
    shared = SharedState()
    stop_event = threading.Event()

    # SIGINT (Ctrl+C) becomes KeyboardInterrupt by default, but SIGTERM has no
    # such default -- and SIGTERM is how a background process is stopped
    # (`--stop`, `kill`, logout). Route it through the same shutdown path so
    # the camera and overlay are always cleaned up.
    signal.signal(signal.SIGTERM, lambda *_: stop_event.set())

    thread = threading.Thread(
        target=run_tracking,
        args=(cap, settings, baseline, shared, stop_event, sys.stdout.isatty()),
        daemon=True,
    )
    thread.start()

    root = overlay.make_root()
    ov = overlay.Overlay(root)
    try:
        overlay.run_poll_loop(
            root, ov,
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
        service.clear_pid_file()
    return 0


def main() -> None:
    command = parse_args(sys.argv[1:])
    settings = Settings()
    console = Console()

    if command.stats:
        sys.exit(cmd_stats(console))
    if command.stop:
        sys.exit(cmd_stop(console))
    if command.foreground:
        sys.exit(cmd_foreground(settings, console))
    sys.exit(cmd_setup(settings, console, force_calibrate=command.calibrate))


if __name__ == "__main__":
    main()
