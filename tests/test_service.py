import os

import pytest

from posture_tracker import service


@pytest.fixture
def paths(tmp_path, monkeypatch):
    """Redirect autostart and pid file into a temp dir."""
    autostart = tmp_path / "autostart" / "posture-tracker.desktop"
    pid_file = tmp_path / "tracker.pid"
    monkeypatch.setattr(service, "AUTOSTART_DIR", autostart.parent)
    monkeypatch.setattr(service, "AUTOSTART_FILE", autostart)
    monkeypatch.setattr(service, "PID_FILE", pid_file)
    return autostart, pid_file


def test_install_autostart_writes_a_desktop_entry(paths):
    autostart, _ = paths
    service.install_autostart()

    content = autostart.read_text()
    assert "[Desktop Entry]" in content
    assert "Type=Application" in content
    # Must launch the tracker itself, not the setup flow, or every login would
    # reopen the camera preview.
    assert "--foreground" in content


def test_autostart_is_reported_and_removed(paths):
    assert service.autostart_installed() is False
    service.install_autostart()
    assert service.autostart_installed() is True

    assert service.remove_autostart() is True
    assert service.autostart_installed() is False
    # Removing twice is not an error.
    assert service.remove_autostart() is False


def test_autostart_command_points_at_this_interpreter(paths):
    # Autostart runs at login with no virtualenv activated, so a bare
    # "posture-tracker" on PATH would not resolve.
    argv = service.executable_argv()
    assert argv[0].startswith("/")
    assert argv[-1] == "--foreground"


def test_desktop_exec_line_quotes_a_path_containing_spaces(monkeypatch):
    # A Russian-locale desktop puts projects under "Рабочий стол"; an unquoted
    # Exec line would split that path in half and autostart would silently do
    # nothing at login.
    monkeypatch.setattr(
        service, "executable_argv",
        lambda: ["/home/me/Рабочий стол/pose/.venv/bin/posture-tracker", "--foreground"],
    )
    line = service.desktop_exec_line()
    assert line == '"/home/me/Рабочий стол/pose/.venv/bin/posture-tracker" --foreground'


def test_desktop_exec_line_leaves_a_plain_path_alone(monkeypatch):
    monkeypatch.setattr(
        service, "executable_argv",
        lambda: ["/opt/pose/bin/posture-tracker", "--foreground"],
    )
    assert service.desktop_exec_line() == "/opt/pose/bin/posture-tracker --foreground"


def test_running_pid_reports_a_live_process(paths):
    _, pid_file = paths
    service.write_pid_file(os.getpid())
    assert service.running_pid() == os.getpid()


def test_running_pid_clears_a_stale_file(paths):
    _, pid_file = paths
    # PID 2^22 is above the default pid_max, so it cannot be live.
    pid_file.parent.mkdir(parents=True, exist_ok=True)
    pid_file.write_text("4194304")

    assert service.running_pid() is None
    assert not pid_file.exists()


def test_running_pid_ignores_a_garbage_file(paths):
    _, pid_file = paths
    pid_file.parent.mkdir(parents=True, exist_ok=True)
    pid_file.write_text("not a pid")
    assert service.running_pid() is None


def test_running_pid_when_nothing_was_ever_started(paths):
    assert service.running_pid() is None


def test_stop_background_reports_when_nothing_runs(paths):
    assert service.stop_background() is False


def test_stop_background_waits_for_the_process_to_actually_exit(paths, monkeypatch):
    # It has to outlive the SIGTERM by a moment, because the tracker releases
    # the camera on its way out and recalibration opens it straight after.
    service.write_pid_file(4242)

    signalled: list[tuple[int, int]] = []
    alive_polls: list[int] = []

    def fake_kill(pid, sig):
        signalled.append((pid, sig))

    def fake_alive(pid):
        # Alive for the first couple of polls, then gone.
        if len(alive_polls) < 2:
            alive_polls.append(pid)
            return True
        return False

    monkeypatch.setattr(service.os, "kill", fake_kill)
    monkeypatch.setattr(service, "_process_alive", fake_alive)

    assert service.stop_background(wait_timeout=5.0) is True
    assert signalled and signalled[0][0] == 4242
    assert len(alive_polls) == 2  # it polled rather than returning immediately
    assert not service.PID_FILE.exists()


def test_stop_background_gives_up_after_the_timeout(paths, monkeypatch):
    # A wedged process must not hang the command forever.
    service.write_pid_file(4242)
    monkeypatch.setattr(service.os, "kill", lambda pid, sig: None)
    monkeypatch.setattr(service, "_process_alive", lambda pid: True)

    assert service.stop_background(wait_timeout=0.3) is True
    assert not service.PID_FILE.exists()
