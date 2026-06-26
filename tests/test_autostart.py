"""Tests for the macOS LaunchAgent generation (pure, no launchctl calls)."""

import plistlib
from pathlib import Path

from dendr import autostart


def test_plist_path_location():
    p = autostart.plist_path()
    assert p.name == "com.dendr.daemon.plist"
    assert p.parent.name == "LaunchAgents"


def test_program_args_uses_module_invocation():
    args = autostart.program_args(python="/venv/bin/python")
    assert args == ["/venv/bin/python", "-m", "dendr", "daemon"]


def test_program_args_includes_data_dir():
    args = autostart.program_args(Path("/data/dendr"), python="/venv/bin/python")
    assert args[-2:] == ["--data-dir", "/data/dendr"]


def test_render_plist_roundtrips_and_sets_launch_keys():
    args = autostart.program_args(Path("/data/dendr"), python="/venv/bin/python")
    raw = autostart.render_plist(
        args,
        stdout_path="/logs/daemon.out.log",
        stderr_path="/logs/daemon.err.log",
        working_dir="/vault",
    )
    d = plistlib.loads(raw)
    assert d["Label"] == "com.dendr.daemon"
    assert d["ProgramArguments"] == args
    assert d["RunAtLoad"] is True
    assert d["KeepAlive"] is True
    assert d["StandardOutPath"] == "/logs/daemon.out.log"
    assert d["StandardErrorPath"] == "/logs/daemon.err.log"
    assert d["WorkingDirectory"] == "/vault"


def test_render_plist_omits_optional_keys_when_absent():
    raw = autostart.render_plist(autostart.program_args(python="/p"))
    d = plistlib.loads(raw)
    assert "StandardOutPath" not in d
    assert "WorkingDirectory" not in d
