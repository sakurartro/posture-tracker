# Posture Tracker

A real-time posture-tracking CLI app, using a webcam.
Runs fully locally: MediaPipe Pose + OpenCV for detection, Rich for the
terminal dashboard, a desktop notification (`notify-send`) after a short
continuous violation, and a fullscreen tkinter overlay if bad posture
persists even longer. Designed to run in the background — a normal desktop
notification doesn't need the terminal in focus, and the app shuts down
cleanly on both Ctrl+C and `kill`/`systemctl stop` (SIGTERM).

## Install

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

## Run

```bash
posture-tracker
```

or without installing the package:

```bash
python -m posture_tracker.main
```

On first run, the app downloads a small (~6MB) MediaPipe pose model to
`~/.local/share/posture-tracker/models/` — this needs internet once; every
run after that is fully offline.

Startup then runs a short countdown ("get into position") followed by a few
seconds of calibration — **sit the way you want to be reminded to sit**, since
everything afterwards is measured relative to that reference posture. If the
baseline is captured while you are still reaching for the keyboard, the app
will nag you for the rest of the session; restart it to recalibrate.

### Flags

| Flag | Default | Description |
|---|---|---|
| `--camera` | `/dev/video0` | Camera device path or index |
| `--calibration-countdown` | `3.0` | Countdown to get into position before calibration samples |
| `--calibration-seconds` | `3.0` | Calibration duration, seconds |
| `--smoothing` | `0.35` | Moving-average weight for the newest sample (lower = steadier, slower to react) |
| `--notify-after` | `5.0` | Seconds of continuous bad posture before a desktop notification fires |
| `--grace-period` | `10.0` | Seconds of continuous bad posture before the fullscreen overlay fires |
| `--fps` | `12` | Analysis loop frame rate |
| `--head-tilt-threshold` | `8.0` | Allowed head tilt deviation, degrees |
| `--shoulder-tilt-threshold` | `10.0` | Allowed shoulder tilt deviation, degrees |
| `--slouch-threshold` | `12.0` | Allowed nose-to-shoulder-line distance change, percent |

## Dashboard statuses

- **[OK]** — posture is good.
- **[WARN N sec]** — a violation is in progress but hasn't exceeded the overlay grace period yet.
  Once it passes `--notify-after` (default 5s) a one-off desktop notification fires.
- **[ALERT]** — `--grace-period` exceeded (default 10s), the fullscreen overlay is active.
- **[PAUSED]** — no face detected in frame (user stepped away), timer reset.

There are two independent thresholds for the same continuous violation:
a desktop notification at `--notify-after` seconds (quick heads-up, works
even if you're not looking at the terminal), and the fullscreen overlay at
`--grace-period` seconds (default is longer, since it's more disruptive).
The overlay closes automatically once posture is corrected, or manually via
`Esc`.

## Running in the background

The dashboard is nice to watch, but you don't need to: the desktop
notification is what tells you about a violation, so the app is meant to
be started once and left running in the background.

```bash
nohup posture-tracker > ~/.local/share/posture-tracker/run.log 2>&1 &
disown
```

Stop it later with `pkill -INT -f posture-tracker` (or `kill <pid>`/
`systemctl stop`, which sends SIGTERM) — either way the camera is released
and the session is saved to SQLite before exit.

## Data

Each session's summary (duration, % good posture, violation count) is saved
to SQLite: `~/.local/share/posture-tracker/posture.db`.

## Tests

```bash
pip install -e ".[dev]"
pytest
```

## Known limitations

- Designed for a single monitor; overlay behavior on multi-monitor setups
  hasn't been tested.
- Uses the MediaPipe Tasks API (`PoseLandmarker`), not the legacy
  `mp.solutions.pose` API: the legacy solutions module is no longer
  shipped in current mediapipe pip wheels for Python 3.12 (verified absent
  in 0.10.35 and 1.0.1). This means a one-time model download on first run
  (see above) instead of a fully bundled model.
- `requirements.txt`/`pyproject.toml` intentionally do not list
  `opencv-python` directly — mediapipe pulls in `opencv-contrib-python`
  transitively, and installing both creates a conflicting duplicate `cv2`
  package.
