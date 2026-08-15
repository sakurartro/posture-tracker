"""Where this app keeps its files, per operating system.

Each platform has its own convention, and putting a dot-directory in a
Windows or macOS home folder would be wrong in a way users notice.

Linux and Windows support are what the app was built and measured on;
macOS and Windows paths follow the documented conventions but have not been
exercised on real machines.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

APP_NAME = "posture-tracker"


def is_windows() -> bool:
    return sys.platform == "win32"


def is_macos() -> bool:
    return sys.platform == "darwin"


def is_linux() -> bool:
    return sys.platform.startswith("linux")


def data_dir() -> Path:
    """Application data: the model, the database, the calibrated baseline."""
    if is_windows():
        base = os.environ.get("LOCALAPPDATA")
        root = Path(base) if base else Path.home() / "AppData" / "Local"
        return root / APP_NAME
    if is_macos():
        return Path.home() / "Library" / "Application Support" / APP_NAME
    # Linux and other unixes: XDG.
    base = os.environ.get("XDG_DATA_HOME")
    root = Path(base) if base else Path.home() / ".local" / "share"
    return root / APP_NAME
