import os
import plistlib

import pytest

from posture_tracker import paths, service


@pytest.fixture
def sandbox(tmp_path, monkeypatch):
    """Point the autostart entry and pid file at a temp dir."""
    autostart = tmp_path / "autostart" / "posture-tracker.desktop"
    monkeypatch.setattr(service, "autostart_file", lambda: autostart)
    monkeypatch.setattr(service, "PID_FILE", tmp_path / "tracker.pid")
    return autostart


def on_platform(monkeypatch, name: str) -> None:
    """Pretend we are running on another OS."""
    monkeypatch.setattr(paths, "is_windows", lambda: name == "win32")
    monkeypatch.setattr(paths, "is_macos", lambda: name == "darwin")
    monkeypatch.setattr(paths, "is_linux", lambda: name == "linux")


def test_install_autostart_writes_a_desktop_entry(sandbox, monkeypatch):
    on_platform(monkeypatch, "linux")
    service.install_autostart()

    content = sandbox.read_text()
    assert "[Desktop Entry]" in content
    # Must launch the tracker itself, not the setup flow, or every login would
    # reopen the camera preview.
    assert "--foreground" in content


def test_autostart_is_reported_and_removed(sandbox, monkeypatch):
    on_platform(monkeypatch, "linux")
    assert service.autostart_installed() is False

    service.install_autostart()
    assert service.autostart_installed() is True

    assert service.remove_autostart() is True
    assert service.autostart_installed() is False
    # Removing twice is not an error.
    assert service.remove_autostart() is False


def test_macos_autostart_is_a_launch_agent(sandbox, monkeypatch):
    on_platform(monkeypatch, "darwin")
    service.install_autostart()

    plist = plistlib.loads(sandbox.read_bytes())
    assert plist["RunAtLoad"] is True
    assert plist["ProgramArguments"][-1] == "--foreground"
    # An argv list, so a path containing spaces stays one argument.
    assert isinstance(plist["ProgramArguments"], list)


def test_windows_autostart_is_a_startup_launcher(sandbox, monkeypatch):
    on_platform(monkeypatch, "win32")
    service.install_autostart()

    content = sandbox.read_bytes().decode()
    assert content.startswith("@echo off")
    assert "--foreground" in content
    # `start ""` supplies the empty window title, so a quoted path is not
    # mistaken for one.
    assert 'start ""' in content


def test_autostart_paths_differ_per_platform(monkeypatch):
    on_platform(monkeypatch, "linux")
    assert service.autostart_file().suffix == ".desktop"
    on_platform(monkeypatch, "darwin")
    assert service.autostart_file().suffix == ".plist"
    on_platform(monkeypatch, "win32")
    assert service.autostart_file().suffix == ".bat"


def test_autostart_command_points_at_this_interpreter(sandbox):
    # Autostart runs at login with no virtualenv activated, so a bare
    # "posture-tracker" on PATH would not resolve.
    argv = service.executable_argv()
    assert argv[0].startswith("/") or ":" in argv[0]  # absolute on posix or windows
    assert argv[-1] == "--foreground"


def test_desktop_exec_line_quotes_a_path_containing_spaces(monkeypatch):
    # A Russian-locale desktop puts projects under "Рабочий стол"; an unquoted
    # Exec line would split that path in half and autostart would silently do
    # nothing at login.
    monkeypatch.setattr(
        service, "executable_argv",
        lambda: ["/home/me/Рабочий стол/pose/.venv/bin/posture-tracker", "--foreground"],
    )
    assert service.desktop_exec_line() == (
        '"/home/me/Рабочий стол/pose/.venv/bin/posture-tracker" --foreground'
    )


def test_desktop_exec_line_leaves_a_plain_path_alone(monkeypatch):
    monkeypatch.setattr(
        service, "executable_argv",
        lambda: ["/opt/pose/bin/posture-tracker", "--foreground"],
    )
    assert service.desktop_exec_line() == "/opt/pose/bin/posture-tracker --foreground"


def test_running_pid_reports_a_live_process(sandbox):
    service.write_pid_file(os.getpid())
    assert service.running_pid() == os.getpid()


def test_running_pid_clears_a_stale_file(sandbox):
    # PID 2^22 is above the default pid_max, so it cannot be live.
    service.PID_FILE.parent.mkdir(parents=True, exist_ok=True)
    service.PID_FILE.write_text("4194304")

    assert service.running_pid() is None
    assert not service.PID_FILE.exists()


def test_running_pid_ignores_a_garbage_file(sandbox):
    service.PID_FILE.parent.mkdir(parents=True, exist_ok=True)
    service.PID_FILE.write_text("not a pid")
    assert service.running_pid() is None


def test_running_pid_when_nothing_was_ever_started(sandbox):
    assert service.running_pid() is None


def test_stop_background_reports_when_nothing_runs(sandbox):
    assert service.stop_background() is False


def test_stop_background_waits_for_the_process_to_actually_exit(sandbox, monkeypatch):
    # It has to outlive the terminate request by a moment, because the tracker
    # releases the camera on its way out and recalibration opens it straight
    # after.
    service.write_pid_file(4242)

    terminated: list[int] = []
    alive_polls: list[int] = []

    def fake_alive(pid):
        if len(alive_polls) < 2:
            alive_polls.append(pid)
            return True
        return False

    monkeypatch.setattr(service, "_terminate", lambda pid: terminated.append(pid))
    monkeypatch.setattr(service, "_process_alive", fake_alive)

    assert service.stop_background(wait_timeout=5.0) is True
    assert terminated == [4242]
    assert len(alive_polls) == 2  # polled rather than returning immediately
    assert not service.PID_FILE.exists()


def test_stop_background_gives_up_after_the_timeout(sandbox, monkeypatch):
    # A wedged process must not hang the command forever.
    service.write_pid_file(4242)
    monkeypatch.setattr(service, "_terminate", lambda pid: None)
    monkeypatch.setattr(service, "_process_alive", lambda pid: True)

    assert service.stop_background(wait_timeout=0.3) is True
    assert not service.PID_FILE.exists()
