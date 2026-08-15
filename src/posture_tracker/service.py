"""Running in the background, and surviving a reboot.

Each desktop has its own idea of "start this at login":

- Linux: an XDG .desktop file in ~/.config/autostart, honoured by XFCE,
  GNOME and KDE alike, so no systemd unit is needed.
- macOS: a LaunchAgent plist in ~/Library/LaunchAgents.
- Windows: a launcher in the Start Menu's Startup folder.

Linux is the platform this was built and measured on; the macOS and Windows
paths follow the documented conventions but have not been run on real
machines.

The running tracker is tracked by a pid file rather than by scanning process
names, so stopping it cannot accidentally kill an unrelated process that
happens to share a name.
"""

from __future__ import annotations

import os
import plistlib
import signal
import subprocess
import sys
import time
from pathlib import Path

from posture_tracker import paths
from posture_tracker.storage import DATA_DIR

PID_FILE = DATA_DIR / "tracker.pid"
LOG_FILE = DATA_DIR / "tracker.log"

LAUNCH_AGENT_LABEL = "io.posturetracker.tracker"

_DESKTOP_ENTRY = """[Desktop Entry]
Type=Application
Name=Posture Tracker
Comment=Reminds you to sit up straight
Exec={exec_line}
Terminal=false
X-GNOME-Autostart-enabled=true
"""


def autostart_file() -> Path:
    if paths.is_macos():
        return Path.home() / "Library" / "LaunchAgents" / f"{LAUNCH_AGENT_LABEL}.plist"
    if paths.is_windows():
        return (Path.home() / "AppData" / "Roaming" / "Microsoft" / "Windows"
                / "Start Menu" / "Programs" / "Startup" / "posture-tracker.bat")
    return Path.home() / ".config" / "autostart" / "posture-tracker.desktop"


def executable_argv() -> list[str]:
    """The command that runs the tracker itself, as an argv list.

    Resolved from the interpreter currently running, so it points at this
    virtualenv rather than whatever `posture-tracker` might mean on PATH at
    login time -- when the venv is almost certainly not activated.

    A list rather than a string because the install path may well contain
    spaces (a Russian-locale desktop puts projects under "Рабочий стол", for
    one), and splitting such a command back apart tears the path in half.
    """
    suffix = ".exe" if paths.is_windows() else ""
    console_script = Path(sys.executable).with_name(f"posture-tracker{suffix}")
    if console_script.exists():
        return [str(console_script), "--foreground"]
    return [sys.executable, "-m", "posture_tracker.main", "--foreground"]


def _quote_desktop_arg(argument: str) -> str:
    """Quotes one argument for a .desktop Exec line.

    The Desktop Entry spec requires reserved characters to be double-quoted,
    with backslash, quote, backtick and dollar escaped inside the quotes.
    Without this, a path containing a space silently becomes two arguments and
    autostart fails at login with nothing to show for it.
    """
    if not any(c in argument for c in ' \t\n"\'\\><~|&;$*?#()`'):
        return argument
    escaped = (argument.replace("\\", "\\\\")
                       .replace('"', '\\"')
                       .replace("`", "\\`")
                       .replace("$", "\\$"))
    return f'"{escaped}"'


def desktop_exec_line() -> str:
    return " ".join(_quote_desktop_arg(a) for a in executable_argv())


def _autostart_contents() -> bytes:
    argv = executable_argv()
    if paths.is_macos():
        return plistlib.dumps({
            "Label": LAUNCH_AGENT_LABEL,
            "ProgramArguments": argv,
            "RunAtLoad": True,
        })
    if paths.is_windows():
        # `start ""` returns immediately and the empty title argument keeps a
        # quoted path from being mistaken for the window title.
        quoted = " ".join(f'"{a}"' if " " in a else a for a in argv)
        return f'@echo off\r\nstart "" {quoted}\r\n'.encode()
    return _DESKTOP_ENTRY.format(exec_line=desktop_exec_line()).encode()


def install_autostart() -> Path:
    target = autostart_file()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(_autostart_contents())
    return target


def remove_autostart() -> bool:
    """True if an entry was actually there."""
    target = autostart_file()
    existed = target.exists()
    target.unlink(missing_ok=True)
    return existed


def autostart_installed() -> bool:
    return autostart_file().exists()


def _process_alive(pid: int) -> bool:
    if paths.is_windows():
        # Windows has no signal-0 probe; ask the task list instead.
        result = subprocess.run(
            ["tasklist", "/FI", f"PID eq {pid}", "/NH"],
            capture_output=True, text=True,
        )
        return str(pid) in result.stdout
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def running_pid() -> int | None:
    """The pid of the live tracker, or None. Clears the file if it is stale."""
    try:
        pid = int(PID_FILE.read_text().strip())
    except (OSError, ValueError):
        return None
    if _process_alive(pid):
        return pid
    PID_FILE.unlink(missing_ok=True)
    return None


def write_pid_file(pid: int | None = None) -> None:
    PID_FILE.parent.mkdir(parents=True, exist_ok=True)
    PID_FILE.write_text(str(pid if pid is not None else os.getpid()))


def clear_pid_file() -> None:
    PID_FILE.unlink(missing_ok=True)


def start_background() -> int:
    """Launches the tracker detached from this terminal and returns its pid."""
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG_FILE, "ab") as log:
        if paths.is_windows():
            # No console window, and detached so closing this one leaves the
            # tracker alone.
            flags = (getattr(subprocess, "DETACHED_PROCESS", 0)
                     | getattr(subprocess, "CREATE_NO_WINDOW", 0))
            process = subprocess.Popen(
                executable_argv(),
                stdout=log, stderr=log, stdin=subprocess.DEVNULL,
                creationflags=flags,
            )
        else:
            # Its own session, so a shell sending SIGHUP on exit does not take
            # the tracker with it.
            process = subprocess.Popen(
                executable_argv(),
                stdout=log, stderr=log, stdin=subprocess.DEVNULL,
                start_new_session=True,
            )
    return process.pid


def _terminate(pid: int) -> None:
    if paths.is_windows():
        # taskkill asks the process to close rather than shooting it, which
        # gives it the chance to release the camera and save the session.
        subprocess.run(["taskkill", "/PID", str(pid), "/T"],
                       capture_output=True)
        return
    os.kill(pid, signal.SIGTERM)


def stop_background(wait_timeout: float = 10.0) -> bool:
    """Asks a running tracker to shut down and waits for it. True if one was
    running.

    Waiting matters: the tracker releases the camera on its way out, and
    recalibration opens the camera immediately afterwards. Returning while the
    old process still held the device would fail with "camera busy". It also
    means --stop only returns once the camera is genuinely free.
    """
    pid = running_pid()
    if pid is None:
        return False

    try:
        _terminate(pid)
    except ProcessLookupError:
        clear_pid_file()
        return True

    deadline = time.monotonic() + wait_timeout
    while time.monotonic() < deadline and _process_alive(pid):
        time.sleep(0.1)

    clear_pid_file()
    return True
