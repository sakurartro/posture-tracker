from datetime import datetime, timedelta, timezone
from pathlib import Path

from posture_tracker.storage import (
    MAX_LEVEL,
    SessionSummary,
    level_progress,
    load_baseline,
    load_sessions,
    period_stats,
    points_earned,
    save_baseline,
    save_session,
)


def summary(started_at, duration=600.0, tracked=600.0, good=500.0, violations=2):
    return SessionSummary(
        started_at=started_at,
        duration_seconds=duration,
        tracked_seconds=tracked,
        good_seconds=good,
        violation_count=violations,
    )


def test_save_and_load_session(tmp_path: Path):
    db = tmp_path / "posture.db"
    started = datetime(2026, 8, 15, 10, 0, tzinfo=timezone.utc)
    save_session(summary(started), db_path=db)

    loaded = load_sessions(db_path=db)
    assert len(loaded) == 1
    assert loaded[0].started_at == started
    assert loaded[0].tracked_seconds == 600.0
    assert loaded[0].good_seconds == 500.0
    assert loaded[0].violation_count == 2


def test_good_posture_pct_is_derived_from_seconds():
    assert summary(datetime.now(timezone.utc), tracked=100, good=75).good_posture_pct == 75.0


def test_good_posture_pct_when_never_tracked():
    # Nothing observed is not a failing grade.
    assert summary(datetime.now(timezone.utc), tracked=0, good=0).good_posture_pct == 100.0


def test_db_directory_is_created_if_missing(tmp_path: Path):
    db = tmp_path / "nested" / "dir" / "posture.db"
    save_session(summary(datetime.now(timezone.utc)), db_path=db)
    assert db.exists()


def test_period_stats_only_counts_sessions_inside_the_window(tmp_path: Path):
    db = tmp_path / "posture.db"
    now = datetime(2026, 8, 15, 12, 0, tzinfo=timezone.utc)
    save_session(summary(now - timedelta(days=2), tracked=100, good=100, violations=0), db_path=db)
    save_session(summary(now - timedelta(days=40), tracked=900, good=0, violations=9), db_path=db)

    week = period_stats("week", now - timedelta(days=7), db_path=db)
    assert week.session_count == 1
    assert week.tracked_seconds == 100
    assert week.violation_count == 0

    everything = period_stats("all", None, db_path=db)
    assert everything.session_count == 2
    assert everything.tracked_seconds == 1000


def test_period_stats_weights_by_time_not_by_session(tmp_path: Path):
    # A short perfect session must not cancel out a long bad one; averaging
    # per-session percentages would say 50%, the truth is 10%.
    db = tmp_path / "posture.db"
    now = datetime(2026, 8, 15, 12, 0, tzinfo=timezone.utc)
    save_session(summary(now, tracked=100, good=100, violations=0), db_path=db)
    save_session(summary(now, tracked=900, good=0, violations=5), db_path=db)

    stats = period_stats("all", None, db_path=db)
    assert stats.good_posture_pct == 10.0
    assert stats.violation_count == 5


def test_period_stats_with_no_sessions_is_empty_not_an_error(tmp_path: Path):
    stats = period_stats("all", None, db_path=tmp_path / "empty.db")
    assert stats.session_count == 0
    assert stats.tracked_seconds == 0


def test_migrates_a_database_written_by_an_older_version(tmp_path: Path):
    import sqlite3

    db = tmp_path / "old.db"
    conn = sqlite3.connect(db)
    conn.execute(
        "CREATE TABLE sessions ("
        "id INTEGER PRIMARY KEY AUTOINCREMENT, started_at TEXT NOT NULL, "
        "duration_seconds REAL NOT NULL, good_posture_pct REAL NOT NULL, "
        "violation_count INTEGER NOT NULL)"
    )
    conn.execute(
        "INSERT INTO sessions (started_at, duration_seconds, good_posture_pct, violation_count) "
        "VALUES (?, ?, ?, ?)",
        (datetime(2026, 8, 1, tzinfo=timezone.utc).isoformat(), 1000.0, 80.0, 3),
    )
    conn.commit()
    conn.close()

    # History survives, with seconds derived from the old percentage.
    loaded = load_sessions(db_path=db)
    assert len(loaded) == 1
    assert loaded[0].tracked_seconds == 1000.0
    assert loaded[0].good_seconds == 800.0
    assert loaded[0].violation_count == 3

    # And the migrated database still accepts new rows.
    save_session(summary(datetime(2026, 8, 2, tzinfo=timezone.utc)), db_path=db)
    assert len(load_sessions(db_path=db)) == 2


def test_baseline_round_trip(tmp_path: Path):
    path = tmp_path / "baseline.json"
    save_baseline(-1.25, 6.5, path=path)
    assert load_baseline(path=path) == (-1.25, 6.5)


def test_load_baseline_when_never_calibrated(tmp_path: Path):
    assert load_baseline(path=tmp_path / "missing.json") is None


def test_load_baseline_survives_a_corrupt_file(tmp_path: Path):
    path = tmp_path / "baseline.json"
    path.write_text("{ not json")
    assert load_baseline(path=path) is None


def test_points_earned_is_one_per_full_minute():
    assert points_earned(0) == 0
    assert points_earned(59) == 0  # partial minutes don't count
    assert points_earned(60) == 1
    assert points_earned(179) == 2


def test_level_progress_starts_at_level_one_with_zero_points():
    progress = level_progress(0)
    assert progress.level == 1
    assert progress.points_into_level == 0
    assert progress.points_for_next_level == 50


def test_level_progress_advances_at_each_threshold():
    assert level_progress(49).level == 1
    assert level_progress(50).level == 2
    assert level_progress(199).level == 2
    assert level_progress(200).level == 3


def test_level_progress_caps_at_max_level():
    progress = level_progress(1_000_000)
    assert progress.level == MAX_LEVEL
    assert progress.points_for_next_level is None
    assert progress.progress_pct == 100.0


def test_level_progress_pct_is_relative_to_current_level():
    # Level 2 starts at 50 points and level 3 at 200, so 125 is halfway through.
    progress = level_progress(125)
    assert progress.level == 2
    assert progress.progress_pct == 50.0
