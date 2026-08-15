from datetime import datetime, timezone
from pathlib import Path

from posture_tracker.storage import SessionSummary, load_sessions, save_session


def test_save_and_load_session(tmp_path: Path):
    db_path = tmp_path / "posture.db"
    summary = SessionSummary(
        started_at=datetime(2026, 8, 15, 10, 0, tzinfo=timezone.utc),
        duration_seconds=1234.5,
        good_posture_pct=87.3,
        violation_count=4,
    )

    save_session(summary, db_path=db_path)
    loaded = load_sessions(db_path=db_path)

    assert len(loaded) == 1
    assert loaded[0].duration_seconds == 1234.5
    assert loaded[0].good_posture_pct == 87.3
    assert loaded[0].violation_count == 4
    assert loaded[0].started_at == summary.started_at


def test_multiple_sessions_ordered_by_start_time(tmp_path: Path):
    db_path = tmp_path / "posture.db"
    first = SessionSummary(
        started_at=datetime(2026, 8, 14, 9, 0, tzinfo=timezone.utc),
        duration_seconds=600,
        good_posture_pct=90,
        violation_count=1,
    )
    second = SessionSummary(
        started_at=datetime(2026, 8, 15, 9, 0, tzinfo=timezone.utc),
        duration_seconds=700,
        good_posture_pct=80,
        violation_count=2,
    )

    save_session(second, db_path=db_path)
    save_session(first, db_path=db_path)

    loaded = load_sessions(db_path=db_path)
    assert [s.started_at for s in loaded] == [first.started_at, second.started_at]


def test_db_directory_is_created_if_missing(tmp_path: Path):
    db_path = tmp_path / "nested" / "dir" / "posture.db"
    summary = SessionSummary(
        started_at=datetime(2026, 8, 15, 10, 0, tzinfo=timezone.utc),
        duration_seconds=10,
        good_posture_pct=100,
        violation_count=0,
    )
    save_session(summary, db_path=db_path)
    assert db_path.exists()
