from pathlib import Path

import pytest

from posture_tracker import paths


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    monkeypatch.delenv("XDG_DATA_HOME", raising=False)
    monkeypatch.delenv("LOCALAPPDATA", raising=False)


def on_platform(monkeypatch, name: str) -> None:
    monkeypatch.setattr(paths.sys, "platform", name)


def test_linux_uses_xdg_data_home(monkeypatch, tmp_path):
    on_platform(monkeypatch, "linux")
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path))
    assert paths.data_dir() == tmp_path / "posture-tracker"


def test_linux_falls_back_to_local_share(monkeypatch):
    on_platform(monkeypatch, "linux")
    assert paths.data_dir() == Path.home() / ".local" / "share" / "posture-tracker"


def test_macos_uses_application_support(monkeypatch):
    on_platform(monkeypatch, "darwin")
    expected = Path.home() / "Library" / "Application Support" / "posture-tracker"
    assert paths.data_dir() == expected


def test_windows_uses_localappdata(monkeypatch, tmp_path):
    on_platform(monkeypatch, "win32")
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))
    assert paths.data_dir() == tmp_path / "posture-tracker"


def test_windows_falls_back_when_localappdata_is_unset(monkeypatch):
    on_platform(monkeypatch, "win32")
    assert paths.data_dir() == Path.home() / "AppData" / "Local" / "posture-tracker"


def test_platform_predicates_are_mutually_exclusive(monkeypatch):
    for name, expected in (("linux", "linux"), ("darwin", "macos"), ("win32", "windows")):
        on_platform(monkeypatch, name)
        flags = {
            "linux": paths.is_linux(),
            "macos": paths.is_macos(),
            "windows": paths.is_windows(),
        }
        assert flags[expected] is True
        assert sum(flags.values()) == 1
