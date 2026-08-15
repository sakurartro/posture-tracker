"""Running in the background, and surviving a reboot.

Autostart uses the XDG spec (~/.config/autostart/*.desktop), which XFCE, GNOME
and KDE all honour, so there is no need for a systemd unit or anything
desktop-specific.

The running tracker is tracked by a pid file rather than by scanning process
names, so stopping it cannot accidentally kill an unrelated process that
happens to share a name.
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
from pathlib import Path

from posture_tracker.storage import DATA_DIR

AUTOSTART_DIR = Path.home() / ".config" / "autostart"
AUTOSTART_FILE = AUTOSTART_DIR / "posture-tracker.desktop"
PID_FILE = DATA_DIR / "tracker.pid"
LOG_FILE = DATA_DIR / "tracker.log"

_DESKTOP_ENTRY = """[Desktop Entry]
Type=Application
Name=Posture Tracker
Comment=Reminds you to sit up straight
Exec={exec_line}
Terminal=false
X-GNOME-Autostart-enabled=true
"""


def executable_argv() -> list[str]:
    """The command that runs the tracker itself, as an argv list.

    Resolved from the interpreter currently running, so it points at this
    virtualenv rather than whatever `posture-tracker` might mean on PATH at
    login time -- when the venv is almost certainly not activated.

    A list rather than a string because the install path may well contain
    spaces (a Russian-locale desktop puts projects under "Рабочий стол", for
    one), and splitting such a command back apart tears the path in half.
    """
    console_script = Path(sys.executable).with_name("posture-tracker")
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


def install_autostart() -> Path:
    AUTOSTART_DIR.mkdir(parents=True, exist_ok=True)
    AUTOSTART_FILE.write_text(_DESKTOP_ENTRY.format(exec_line=desktop_exec_line()))
    return AUTOSTART_FILE


def remove_autostart() -> bool:
    """True if an entry was actually there."""
    existed = AUTOSTART_FILE.exists()
    AUTOSTART_FILE.unlink(missing_ok=True)
    return existed


def autostart_installed() -> bool:
    return AUTOSTART_FILE.exists()


def _process_alive(pid: int) -> bool:
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
    """Launches the tracker detached from this terminal and returns its pid.

    start_new_session puts it in its own session, so closing the terminal (or
    the shell sending SIGHUP on exit) does not take the tracker with it.
    """
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG_FILE, "ab") as log:
        process = subprocess.Popen(
            executable_argv(),
            stdout=log,
            stderr=log,
            stdin=subprocess.DEVNULL,
            start_new_session=True,
        )
    return process.pid


def stop_background() -> bool:
    """Asks a running tracker to shut down. True if one was running."""
    pid = running_pid()
    if pid is None:
        return False
    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    clear_pid_file()
    return True
