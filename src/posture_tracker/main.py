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
    Settings,
    parse_args,
)
from posture_tracker.detector import (
    Baseline,
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
    installed, so a missing dependency never crashes tracking."""
    global _notify_available
    if not _notify_available:
        return
    try:
        subprocess.run(
            ["notify-send", "-a", "Posture Tracker", "-u", "normal", title, message],
            check=False,
            timeout=5,
        )
    except FileNotFoundError:
        _notify_available = False
    except subprocess.TimeoutExpired:
        pass


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
    identifier = _parse_camera_device(device)
    cap = cv2.VideoCapture(identifier)
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


def _read_landmarks(cap: cv2.VideoCapture, pose_source: PoseSource, min_visibility: float) -> PoseLandmarks | None:
    ok, frame = cap.read()
    if not ok:
        return None

    result = pose_source.detect(frame)
    if not result.pose_landmarks:
        return None

    landmark_list = result.pose_landmarks[0]
    p = vision.PoseLandmark

    def get(index: int) -> Landmark:
        lm = landmark_list[index]
        return Landmark(x=lm.x, y=lm.y, visibility=lm.visibility)

    pts = PoseLandmarks(
        nose=get(p.NOSE.value),
        left_ear=get(p.LEFT_EAR.value),
        right_ear=get(p.RIGHT_EAR.value),
        left_shoulder=get(p.LEFT_SHOULDER.value),
        right_shoulder=get(p.RIGHT_SHOULDER.value),
    )
    if not pts.visible(min_visibility):
        return None
    return pts


def _calibrate(cap, pose_source: PoseSource, settings: Settings, stop_event: threading.Event) -> Baseline | None:
    samples: list[PoseLandmarks] = []
    deadline = time.monotonic() + settings.calibration_seconds
    frame_interval = 1.0 / settings.analysis_fps

    with Dashboard() as dashboard:
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
            dashboard.update(
                DashboardState(
                    status=Status.PAUSED,
                    violation_seconds=0.0,
                    session_seconds=0.0,
                    good_posture_pct=100.0,
                    violation_count=0,
                    deviation=None,
                    calibrating=True,
                    calibration_progress=progress,
                )
            )

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

    try:
        with PoseSource(POSE_MODEL_PATH) as pose_source:
            baseline = _calibrate(cap, pose_source, settings, stop_event)
            if stop_event.is_set():
                return
            if baseline is None:
                console.print(
                    "[bold red]Calibration failed:[/bold red] no face detected in frame. "
                    "Check the lighting and camera position, then restart the application."
                )
                stop_event.set()
                return

            timer = HysteresisTimer(
                grace_period_seconds=settings.grace_period_seconds,
                notify_after_seconds=settings.notify_after_seconds,
            )
            session_start = datetime.now(timezone.utc)
            total_elapsed = 0.0
            good_elapsed = 0.0
            violation_count = 0
            prev_status = Status.OK
            good_pct = 100.0
            frame_interval = 1.0 / settings.analysis_fps
            last_time = time.monotonic()

            with Dashboard() as dashboard:
                while not stop_event.is_set():
                    loop_start = time.monotonic()
                    dt = loop_start - last_time
                    last_time = loop_start

                    landmarks = _read_landmarks(cap, pose_source, MIN_LANDMARK_VISIBILITY)
                    if landmarks is None:
                        result = timer.update(None)
                        deviation = None
                    else:
                        deviation = compute_deviation(landmarks, baseline)
                        result = timer.update(deviation.within(settings))

                    total_elapsed += dt
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

                    good_pct = (good_elapsed / total_elapsed * 100.0) if total_elapsed > 0 else 100.0
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

            save_session(
                SessionSummary(
                    started_at=session_start,
                    duration_seconds=total_elapsed,
                    good_posture_pct=good_pct,
                    violation_count=violation_count,
                )
            )
    finally:
        shared.set_show_overlay(False)
        cap.release()
        stop_event.set()


def main() -> None:
    settings = parse_args(sys.argv[1:])
    console = Console()

    try:
        cap = open_camera(settings.camera)
    except CameraError as exc:
        console.print(f"[bold red]{exc}[/bold red]")
        sys.exit(1)

    try:
        ensure_pose_model(console)
    except RuntimeError as exc:
        cap.release()
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
