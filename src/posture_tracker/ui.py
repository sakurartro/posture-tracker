"""Rich Live dashboard rendering for the posture tracker."""

from __future__ import annotations

from dataclasses import dataclass

from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from posture_tracker.detector import Deviation, Status

_STATUS_STYLE = {
    Status.OK: ("green", "[OK] Sitting straight"),
    Status.WARN: ("yellow", "[WARN] Straighten up"),
    Status.ALERT: ("red", "[ALERT] Overlay active"),
    Status.PAUSED: ("grey58", "[PAUSED] No one in frame"),
}


@dataclass(frozen=True)
class DashboardState:
    status: Status
    violation_seconds: float
    session_seconds: float
    good_posture_pct: float
    violation_count: int
    deviation: Deviation | None
    calibrating: bool = False
    calibration_progress: float = 0.0
    countdown_seconds: float | None = None


def _status_panel(state: DashboardState) -> Panel:
    style, label = _STATUS_STYLE[state.status]
    body = Text(label, style=f"bold {style}")
    if state.status == Status.WARN:
        body.append(f"\nViolation timer: {state.violation_seconds:.1f} sec")
    return Panel(body, title="Status", border_style=style)


def _stats_table(state: DashboardState) -> Table:
    table = Table(show_header=False, box=None, padding=(0, 1))
    table.add_row("Session time", _format_duration(state.session_seconds))
    table.add_row("Good posture", f"{state.good_posture_pct:.0f}%")
    table.add_row("Violations", str(state.violation_count))
    return table


def _deviation_table(state: DashboardState) -> Table:
    table = Table(show_header=True, header_style="bold", box=None, padding=(0, 1))
    table.add_column("Metric")
    table.add_column("Deviation", justify="right")
    if state.deviation is None:
        table.add_row("—", "no data")
    else:
        d = state.deviation
        table.add_row("Head tilt", f"{d.head_tilt_deg:+.1f}°")
        table.add_row("Shoulder tilt", f"{d.shoulder_tilt_deg:+.1f}°")
        table.add_row("Slouch", f"{d.slouch_pct:+.1f}%")
    return table


def _format_duration(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def render_dashboard(state: DashboardState) -> Panel:
    if state.countdown_seconds is not None:
        body = Text(
            f"Sit the way you want to be reminded to sit.\n"
            f"Calibrating in {state.countdown_seconds:.0f}...",
            style="bold cyan",
        )
        return Panel(body, title="Posture Tracker — get ready")

    if state.calibrating:
        body = Text(
            f"Hold still... calibrating {state.calibration_progress:.0f}%",
            style="bold cyan",
        )
        return Panel(body, title="Posture Tracker — calibration")

    group = Group(
        _status_panel(state),
        Panel(_stats_table(state), title="Session stats"),
        Panel(_deviation_table(state), title="Current deviations"),
    )
    return Panel(group, title="Posture Tracker")


class Dashboard:
    """Thin wrapper around rich.Live for the main loop to push state into."""

    def __init__(self, console: Console | None = None, refresh_per_second: float = 4.0):
        self._console = console or Console()
        self._live = Live(
            console=self._console,
            refresh_per_second=refresh_per_second,
            screen=False,
        )

    def __enter__(self) -> "Dashboard":
        self._live.__enter__()
        return self

    def __exit__(self, *exc_info) -> None:
        self._live.__exit__(*exc_info)

    def update(self, state: DashboardState) -> None:
        self._live.update(render_dashboard(state))
