# Posture Tracker

Reminds you to sit up straight, using your webcam. Runs entirely on your
machine — no video ever leaves it, and nothing is sent anywhere.

Set it up once. After that it starts itself at every login, watches quietly in
the background, and taps you on the shoulder when you have been slouching.

## Install

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

## Use

```bash
posture-tracker
```

That is the whole setup. It will:

1. **Check the camera can see you.** A preview opens showing what the detector
   sees, with a box around your head that turns green once you are properly in
   frame. Sit how you normally work; it closes by itself.
2. **Calibrate.** A short countdown, then a few seconds of sampling — sit the
   way you want to be reminded to sit, because everything afterwards is
   measured against that.
3. **Start watching**, in the background, and add itself to autostart so it
   comes back after a reboot.

Then there are two more commands:

```bash
posture-tracker --stats   # how your posture has been: today, this week, this month
posture-tracker --stop    # stop it and remove it from autostart
```

And one for when things move:

```bash
posture-tracker --calibrate   # after moving the laptop, or changing chair
```

On first run it downloads a small (~4MB) face model to
`~/.local/share/posture-tracker/models/`. That needs the internet once; every
run after that is fully offline.

## What it does when you slouch

| After | What happens |
|---|---|
| 5 seconds | A desktop notification — works whether or not a terminal is open |
| 10 seconds | A translucent fullscreen overlay: *"Straighten your back!"* |

Both timers reset the moment you sit up, so a brief lean to reach for something
never triggers anything. The overlay closes by itself once your posture is
back, or with `Esc`. If you leave your desk, tracking pauses — time away counts
as neither good nor bad posture.

## Watching it live

```bash
posture-tracker --foreground
```

Runs the tracker in your terminal with a live dashboard (status, session
stats, current head angles) instead of in the background. This is also what
the autostart entry runs.

## Where things live

| Path | What |
|---|---|
| `~/.local/share/posture-tracker/posture.db` | Session history, for `--stats` |
| `~/.local/share/posture-tracker/baseline.json` | Your calibrated posture |
| `~/.local/share/posture-tracker/tracker.log` | Background tracker output |
| `~/.config/autostart/posture-tracker.desktop` | Autostart entry |

## Tuning

There are no tuning flags on purpose — the surface is deliberately three
commands. Every threshold lives in `src/posture_tracker/config.py` with a note
on the measurement behind it, if you want to adjust one.

## Tests

```bash
pip install -e ".[dev]"
pytest
```

## Design notes

Three things caused nearly every problem worth knowing about:

**The face model, not MediaPipe Pose.** Pose is a whole-body model, and on a
laptop webcam there is no body in view — it reported shoulders it could not see
by extrapolating them, below the bottom edge of frame in every one of 130
measured samples. On a motionless subject its head roll carried 12° of noise
against the ~8° being measured. Face Landmarker gives the head's 3D orientation
directly, measures ~0.2° of roll noise in the same conditions, costs about a
seventh as much per frame, and reports nothing when it cannot really see a
face — which makes "is the user there" an honest question rather than a guess.

**Capture resolution decides the field of view.** The app asks for 1280x720
rather than accepting the driver's default. A MacBook's FaceTime HD sensor
offers only that mode, so a 640x480 default was produced by cropping the sides
off the 16:9 sensor — discarding about a quarter of the horizontal view.
Measured back to back without the subject moving: at 640x480 the face was not
detected at all; at 1280x720 it sat dead centre. Frames are downscaled again
before inference, so the wider capture costs nothing.

**Fresh frames.** The camera produces at 30fps while analysis runs at 5fps, so
frames pile up in the V4L2 queue and a plain `read()` returns the *oldest* one
— measured ~3 frames of lag, which reads as the app being slow to notice you
straightened up. Neither `CAP_PROP_BUFFERSIZE` nor `CAP_PROP_FPS` is honoured
by this backend, so the queue is drained explicitly and only the newest frame
kept.

## Known limitations

- Posture is judged from the head alone. A hunched back held with a level head
  will not be caught.
- The overlay is written for a single monitor; behaviour on multi-monitor
  setups has not been tested.
- Autostart uses the XDG spec, so it works on XFCE, GNOME and KDE, but not on
  desktops that ignore `~/.config/autostart`.
