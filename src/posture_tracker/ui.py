"""Rich Live dashboard rendering for the posture tracker."""

from __future__ import annotations

from dataclasses import dataclass

from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from posture_tracker.detector import Deviation, Status
from posture_tracker.storage import LevelProgress, MAX_LEVEL

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
        table.add_row("Head tilt (sideways)", f"{d.roll_deg:+.1f}°")
        table.add_row("Head pitch (slouch)", f"{d.pitch_deg:+.1f}°")
    return table


def _format_duration(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def _posture_style(pct: float) -> str:
    if pct >= 85:
        return "green"
    if pct >= 60:
        return "yellow"
    return "red"


def _progress_bar(pct: float, width: int = 20) -> str:
    filled = round(width * min(max(pct, 0.0), 100.0) / 100.0)
    return "█" * filled + "░" * (width - filled)


def _level_panel(progress: LevelProgress) -> Panel:
    body = Text()
    body.append(f"Level {progress.level}/{MAX_LEVEL}", style="bold cyan")
    body.append(f"   {progress.points} pts earned (1 pt / minute tracked)", style="dim")
    if progress.points_for_next_level is None:
        body.append("\nMax level reached!", style="bold green")
    else:
        bar = _progress_bar(progress.progress_pct)
        body.append(
            f"\n[{bar}] {progress.points_into_level}/{progress.points_for_next_level} "
            "to next level"
        )
    return Panel(body, title="Level")


def render_stats(periods, level: LevelProgress) -> Panel:
    """A table of PeriodStats rows (today, week, month, all time) plus the
    level earned from all-time tracked minutes."""
    table = Table(show_header=True, header_style="bold", box=None, padding=(0, 2))
    table.add_column("Period")
    table.add_column("Tracked", justify="right")
    table.add_column("Good posture", justify="right")
    table.add_column("Violations", justify="right")
    table.add_column("Sessions", justify="right")

    for p in periods:
        if p.tracked_seconds <= 0:
            table.add_row(p.label, "—", "[dim]no data[/dim]", "—", "0")
            continue
        pct = p.good_posture_pct
        table.add_row(
            p.label,
            _format_duration(p.tracked_seconds),
            f"[{_posture_style(pct)}]{pct:.0f}%[/{_posture_style(pct)}]",
            str(p.violation_count),
            str(p.session_count),
        )

    group = Group(table, _level_panel(level))
    return Panel(group, title="Posture Tracker — statistics")


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
