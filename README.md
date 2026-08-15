# Posture Tracker

A real-time posture-tracking CLI app, using a webcam.

Runs fully locally: MediaPipe Face Landmarker reads the head's 3D orientation,
Rich draws a terminal dashboard, a desktop notification fires after a short
continuous slouch, and a fullscreen tkinter overlay appears if it goes on
longer. Designed to run in the background — a desktop notification doesn't need
the terminal in focus, and the app shuts down cleanly on both Ctrl+C and
`kill`/`systemctl stop` (SIGTERM).

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

On first run the app downloads a small (~4MB) MediaPipe face model to
`~/.local/share/posture-tracker/models/` — this needs internet once; every run
after that is fully offline.

Startup then runs a short countdown ("get into position") followed by a few
seconds of calibration — **sit the way you want to be reminded to sit**, since
everything afterwards is measured relative to that reference posture. If the
baseline is captured while you are still reaching for the keyboard, the app
will nag you for the rest of the session; restart it to recalibrate.

### If it cannot see you

```bash
posture-tracker --check-camera
```

Opens a mirrored preview showing exactly what the detector sees, with a box
around your head that turns green once you are properly in frame. The head has
to be fully inside the picture; sitting off to one side is fine.

### Flags

| Flag | Default | Description |
|---|---|---|
| `--camera` | `/dev/video0` | Camera device path or index |
| `--check-camera` | off | Preview the framing and exit |
| `--calibration-countdown` | `3.0` | Countdown to get into position before calibration |
| `--calibration-seconds` | `3.0` | Calibration duration, seconds |
| `--smoothing` | `0.35` | Moving-average weight for the newest sample (lower = steadier, slower to react) |
| `--notify-after` | `5.0` | Seconds of continuous bad posture before a desktop notification |
| `--grace-period` | `10.0` | Seconds of continuous bad posture before the fullscreen overlay |
| `--fps` | `5` | Analysis loop frame rate |
| `--head-tilt-threshold` | `8.0` | Allowed sideways head tilt, degrees |
| `--head-pitch-threshold` | `6.0` | Allowed forward chin drop, degrees (the slouch check) |

## Dashboard statuses

- **[OK]** — posture is good.
- **[WARN N sec]** — a violation is in progress but hasn't exceeded the overlay grace period yet.
  Once it passes `--notify-after` (default 5s) a one-off desktop notification fires.
- **[ALERT]** — `--grace-period` exceeded (default 10s), the fullscreen overlay is active.
- **[PAUSED]** — no face detected (user stepped away), timer reset.

There are two independent thresholds for the same continuous violation: a
desktop notification at `--notify-after` seconds (quick heads-up, works even if
you are not looking at the terminal), and the fullscreen overlay at
`--grace-period` seconds (longer by default, since it is more disruptive). The
overlay closes automatically once posture is corrected, or manually via `Esc`.

## Running in the background

The dashboard is nice to watch, but you don't need to: the desktop notification
is what tells you about a violation, so the app is meant to be started once and
left running.

```bash
nohup posture-tracker > ~/.local/share/posture-tracker/run.log 2>&1 &
disown
```

Sit properly *before* launching — calibration happens in the first few seconds
and you won't see the countdown.

Stop it with `pkill -INT -f posture-tracker` (or `kill`/`systemctl stop`, which
send SIGTERM) — either way the camera is released and the session is saved.

## Data

Each session's summary (duration, % good posture, violation count) is saved to
SQLite: `~/.local/share/posture-tracker/posture.db`.

## Tests

```bash
pip install -e ".[dev]"
pytest
```

## Design notes

**Why the face model and not MediaPipe Pose.** Pose is a whole-body model, and
on a laptop webcam there is no body in view — it reported shoulders it could
not see by extrapolating them, below the bottom edge of the frame in every one
of 130 measured samples. On a motionless subject its head roll drifted with
12° of noise against the ~8° being measured. Face Landmarker gives the head's
3D orientation directly, measures ~0.2° of roll noise in the same conditions,
costs about a seventh as much per frame, and reports nothing at all when it
cannot really see a face — which makes "is the user in view" an honest
question rather than a guess.

**Capture resolution matters more than it looks.** The app asks for 1280x720
rather than accepting the driver's default. A MacBook's FaceTime HD sensor
offers only that mode, so a 640x480 default was produced by cropping the sides
off the 16:9 sensor — throwing away about a quarter of the horizontal field of
view, enough to push someone sitting squarely in front of the laptop out past
the edge of frame. Frames are downscaled again before inference, so the wider
capture costs nothing.

**Fresh frames.** The camera produces at 30fps while analysis runs at 5fps, so
frames pile up in the V4L2 queue and a plain `read()` returns the *oldest* one
— measured ~3 frames of lag. Neither `CAP_PROP_BUFFERSIZE` nor `CAP_PROP_FPS`
is honoured by this backend, so the queue is drained explicitly and only the
newest frame kept.

## Known limitations

- Designed for a single monitor; overlay behaviour on multi-monitor setups
  hasn't been tested.
- Posture is judged from the head alone. A hunched back held with a level head
  will not be caught.
