# Posture Tracker

Reminds you to sit up straight, using your webcam. Runs entirely on your
machine — no video ever leaves it, and nothing is sent anywhere.

Set it up once. After that it starts itself at every login, watches quietly in
the background, and taps you on the shoulder when you have been slouching.

## Install

**Linux**

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

**macOS** — *beta, may not work*

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

macOS will ask for camera permission the first time; allow it for your
terminal, or the tracker sees nothing. Autostart is installed as a LaunchAgent
and takes effect at your next login.

**Windows** — *beta, may not work*

```powershell
py -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
```

Autostart is installed as a launcher in the Start Menu's Startup folder.

### Platform support

Linux is where this was built and measured; everything below was verified
on Linux Mint XFCE. The macOS and Windows paths follow each platform's documented
conventions, but have not been run on real machines — treat them as beta and
expect the rough edges to be in setup rather than in the tracking itself.

| | Linux | macOS | Windows |
|---|---|---|---|
| Detection, overlay, dashboard, stats | verified | should work | should work |
| Desktop notifications | verified | `osascript`, untested | PowerShell toast, untested |
| Autostart | verified | LaunchAgent, untested | Startup folder, untested |
| Background start/stop | verified | should work | `taskkill`, untested |

If autostart or the background mode misbehaves on macOS or Windows, everything
still works run directly:

```bash
posture-tracker --foreground
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

On first run it downloads a small (~4MB) face model. That needs the internet
once; every run after that is fully offline. It, the database and your
calibration live in the platform's usual place:

| | Path |
|---|---|
| Linux | `~/.local/share/posture-tracker/` |
| macOS | `~/Library/Application Support/posture-tracker/` |
| Windows | `%LOCALAPPDATA%\posture-tracker\` |

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

## Tests

```bash
pip install -e ".[dev]"
pytest
```
