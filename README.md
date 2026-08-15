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

## Tests

```bash
pip install -e ".[dev]"
pytest
```
