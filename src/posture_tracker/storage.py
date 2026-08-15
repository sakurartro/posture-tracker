"""SQLite storage for per-session posture summaries.

Only writes a summary row per completed session. Weekly/monthly aggregation
is out of scope for this project but this table is the foundation for it.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

DEFAULT_DB_PATH = Path.home() / ".local" / "share" / "posture-tracker" / "posture.db"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    started_at TEXT NOT NULL,
    duration_seconds REAL NOT NULL,
    good_posture_pct REAL NOT NULL,
    violation_count INTEGER NOT NULL
);
"""


@dataclass(frozen=True)
class SessionSummary:
    started_at: datetime
    duration_seconds: float
    good_posture_pct: float
    violation_count: int


def _connect(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.execute(_SCHEMA)
    return conn


def save_session(summary: SessionSummary, db_path: Path = DEFAULT_DB_PATH) -> None:
    with _connect(db_path) as conn:
        conn.execute(
            "INSERT INTO sessions (started_at, duration_seconds, good_posture_pct, violation_count) "
            "VALUES (?, ?, ?, ?)",
            (
                summary.started_at.astimezone(timezone.utc).isoformat(),
                summary.duration_seconds,
                summary.good_posture_pct,
                summary.violation_count,
            ),
        )


def load_sessions(db_path: Path = DEFAULT_DB_PATH) -> list[SessionSummary]:
    with _connect(db_path) as conn:
        rows = conn.execute(
            "SELECT started_at, duration_seconds, good_posture_pct, violation_count "
            "FROM sessions ORDER BY started_at"
        ).fetchall()

    return [
        SessionSummary(
            started_at=datetime.fromisoformat(row[0]),
            duration_seconds=row[1],
            good_posture_pct=row[2],
            violation_count=row[3],
        )
        for row in rows
    ]
