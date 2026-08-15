"""Persistent state: the calibrated baseline, and per-session posture history.

Sessions store *seconds*, not a percentage. Aggregating percentages across
sessions of different lengths would weight a two-minute session the same as an
eight-hour one, so the weekly and monthly figures would be wrong; keeping the
raw seconds means the totals are exact and the percentage is derived at the
end.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

DATA_DIR = Path.home() / ".local" / "share" / "posture-tracker"
DEFAULT_DB_PATH = DATA_DIR / "posture.db"
BASELINE_PATH = DATA_DIR / "baseline.json"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    started_at TEXT NOT NULL,
    duration_seconds REAL NOT NULL,
    tracked_seconds REAL NOT NULL,
    good_seconds REAL NOT NULL,
    violation_count INTEGER NOT NULL
);
"""


@dataclass(frozen=True)
class SessionSummary:
    started_at: datetime
    duration_seconds: float
    # Time the user was actually in frame. Time spent away from the desk is
    # neither good nor bad posture and must not skew the percentage.
    tracked_seconds: float
    good_seconds: float
    violation_count: int

    @property
    def good_posture_pct(self) -> float:
        if self.tracked_seconds <= 0:
            return 100.0
        return self.good_seconds / self.tracked_seconds * 100.0


@dataclass(frozen=True)
class PeriodStats:
    """Sessions rolled up over a time window."""

    label: str
    session_count: int
    tracked_seconds: float
    good_seconds: float
    violation_count: int

    @property
    def good_posture_pct(self) -> float:
        if self.tracked_seconds <= 0:
            return 100.0
        return self.good_seconds / self.tracked_seconds * 100.0


def _migrate(conn: sqlite3.Connection) -> None:
    """Brings a database written by an older version up to the current schema.

    Earlier releases stored a good-posture *percentage* per session. Rebuild
    the table around seconds instead, deriving the old rows' seconds from
    their percentage so existing history is not thrown away. Done as an
    explicit rebuild rather than ALTER ... DROP COLUMN so it works regardless
    of the SQLite version.
    """
    columns = {row[1] for row in conn.execute("PRAGMA table_info(sessions)")}
    if not columns or "tracked_seconds" in columns:
        return

    conn.execute("ALTER TABLE sessions RENAME TO sessions_old")
    conn.execute(_SCHEMA)
    conn.execute(
        "INSERT INTO sessions "
        "(started_at, duration_seconds, tracked_seconds, good_seconds, violation_count) "
        "SELECT started_at, duration_seconds, duration_seconds, "
        "       duration_seconds * good_posture_pct / 100.0, violation_count "
        "FROM sessions_old"
    )
    conn.execute("DROP TABLE sessions_old")


def _connect(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    _migrate(conn)
    conn.execute(_SCHEMA)
    conn.commit()
    return conn


def save_session(summary: SessionSummary, db_path: Path = DEFAULT_DB_PATH) -> None:
    with _connect(db_path) as conn:
        conn.execute(
            "INSERT INTO sessions "
            "(started_at, duration_seconds, tracked_seconds, good_seconds, violation_count) "
            "VALUES (?, ?, ?, ?, ?)",
            (
                summary.started_at.astimezone(timezone.utc).isoformat(),
                summary.duration_seconds,
                summary.tracked_seconds,
                summary.good_seconds,
                summary.violation_count,
            ),
        )


def load_sessions(db_path: Path = DEFAULT_DB_PATH) -> list[SessionSummary]:
    with _connect(db_path) as conn:
        rows = conn.execute(
            "SELECT started_at, duration_seconds, tracked_seconds, good_seconds, violation_count "
            "FROM sessions ORDER BY started_at"
        ).fetchall()

    return [
        SessionSummary(
            started_at=datetime.fromisoformat(row[0]),
            duration_seconds=row[1],
            tracked_seconds=row[2],
            good_seconds=row[3],
            violation_count=row[4],
        )
        for row in rows
    ]


def period_stats(
    label: str,
    since: datetime | None,
    db_path: Path = DEFAULT_DB_PATH,
    now: datetime | None = None,
) -> PeriodStats:
    """Rolls up every session started at or after `since` (all time if None)."""
    query = (
        "SELECT COUNT(*), COALESCE(SUM(tracked_seconds), 0), "
        "COALESCE(SUM(good_seconds), 0), COALESCE(SUM(violation_count), 0) FROM sessions"
    )
    params: tuple = ()
    if since is not None:
        query += " WHERE started_at >= ?"
        params = (since.astimezone(timezone.utc).isoformat(),)

    with _connect(db_path) as conn:
        count, tracked, good, violations = conn.execute(query, params).fetchone()

    return PeriodStats(
        label=label,
        session_count=count,
        tracked_seconds=tracked,
        good_seconds=good,
        violation_count=violations,
    )


def recent_stats(db_path: Path = DEFAULT_DB_PATH, now: datetime | None = None) -> list[PeriodStats]:
    """Today / last 7 days / last 30 days / all time."""
    now = now or datetime.now(timezone.utc)
    return [
        period_stats("Today", now - timedelta(days=1), db_path),
        period_stats("Last 7 days", now - timedelta(days=7), db_path),
        period_stats("Last 30 days", now - timedelta(days=30), db_path),
        period_stats("All time", None, db_path),
    ]


def save_baseline(roll_deg: float, pitch_deg: float, path: Path = BASELINE_PATH) -> None:
    """Stores the calibrated posture so later runs -- especially the one the
    desktop autostarts -- do not have to ask the user to calibrate again."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "roll_deg": roll_deg,
        "pitch_deg": pitch_deg,
        "calibrated_at": datetime.now(timezone.utc).isoformat(),
    }))


def load_baseline(path: Path = BASELINE_PATH) -> tuple[float, float] | None:
    """Returns (roll, pitch), or None if the app has never been calibrated."""
    try:
        data = json.loads(path.read_text())
        return float(data["roll_deg"]), float(data["pitch_deg"])
    except (OSError, ValueError, KeyError):
        return None


def clear_baseline(path: Path = BASELINE_PATH) -> None:
    path.unlink(missing_ok=True)
