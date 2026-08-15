"""tkinter fullscreen warning overlay.

Must run in the main thread. The app keeps a single persistent, hidden
Tk root alive for the whole program lifetime and shows/hides a fullscreen
toplevel on top of it, polled from a `root.after()` loop against shared
state produced by the background detection thread.
"""

from __future__ import annotations

import tkinter as tk
from typing import Callable

OVERLAY_TEXT = "Straighten your back! "
OVERLAY_BG = "black"
OVERLAY_FG = "white"
OVERLAY_ALPHA = 0.75
POLL_INTERVAL_MS = 100


class Overlay:
    def __init__(self, root: tk.Tk):
        self._root = root
        self._toplevel: tk.Toplevel | None = None

    @property
    def visible(self) -> bool:
        return self._toplevel is not None

    def show(self) -> None:
        if self._toplevel is not None:
            return

        top = tk.Toplevel(self._root)
        top.attributes("-fullscreen", True)
        top.attributes("-topmost", True)
        top.attributes("-alpha", OVERLAY_ALPHA)
        top.configure(bg=OVERLAY_BG)
        top.bind("<Escape>", lambda _event: self.hide())
        top.focus_force()

        label = tk.Label(
            top,
            text=OVERLAY_TEXT,
            font=("Sans", 48, "bold"),
            bg=OVERLAY_BG,
            fg=OVERLAY_FG,
        )
        label.pack(expand=True)

        self._toplevel = top

    def hide(self) -> None:
        if self._toplevel is None:
            return
        self._toplevel.destroy()
        self._toplevel = None


def make_root() -> tk.Tk:
    root = tk.Tk()
    root.withdraw()
    return root


def run_poll_loop(
    root: tk.Tk,
    overlay: Overlay,
    should_show_overlay: Callable[[], bool],
    should_stop: Callable[[], bool],
) -> None:
    """Schedules periodic polling of shared state to show/hide the overlay
    and to stop the Tk mainloop once the background thread signals exit."""

    def tick() -> None:
        if should_stop():
            root.quit()
            return

        if should_show_overlay():
            overlay.show()
        else:
            overlay.hide()

        root.after(POLL_INTERVAL_MS, tick)

    root.after(POLL_INTERVAL_MS, tick)
    root.mainloop()
