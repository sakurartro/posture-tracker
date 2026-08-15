# Posture Tracker

A real-time posture-tracking CLI app, using a webcam.
Runs fully locally: MediaPipe Pose + OpenCV for detection, Rich for the
terminal dashboard, and tkinter for a fullscreen warning overlay when bad
posture persists too long.

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
run after that is fully offline. Then sit still for 3 seconds for
calibration. After that, the app tracks head tilt, shoulder tilt, and
slouching relative to your calibrated reference posture.

### Flags

| Flag | Default | Description |
|---|---|---|
| `--camera` | `/dev/video0` | Camera device path or index |
| `--calibration-seconds` | `3.0` | Calibration duration, seconds |
| `--grace-period` | `7.0` | Seconds of continuous bad posture before the overlay fires |
| `--fps` | `12` | Analysis loop frame rate |
| `--head-tilt-threshold` | `8.0` | Allowed head tilt deviation, degrees |
| `--shoulder-tilt-threshold` | `10.0` | Allowed shoulder tilt deviation, degrees |
| `--slouch-threshold` | `12.0` | Allowed nose-to-shoulder-line distance change, percent |

## Dashboard statuses

- **[OK]** — posture is good.
- **[WARN N sec]** — a violation is in progress but hasn't exceeded the grace period yet.
- **[ALERT]** — grace period exceeded, the fullscreen overlay is active.
- **[PAUSED]** — no face detected in frame (user stepped away), timer reset.

The overlay closes automatically once posture is corrected, or manually via `Esc`.

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
