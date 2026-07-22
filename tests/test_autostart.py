"""Tests for the macOS LaunchAgent generation (pure, no launchctl calls)."""

import plistlib
from pathlib import Path

from dendr import autostart


def test_plist_path_location():
    p = autostart.plist_path()
    assert p.name == "com.dendr.ingest.plist"
    assert p.parent.name == "LaunchAgents"


def test_legacy_plist_path_matches_old_daemon_label():
    p = autostart.plist_path(autostart.LEGACY_LAUNCH_AGENT_LABEL)
    assert p.name == "com.dendr.daemon.plist"


def test_program_args_uses_module_invocation():
    args = autostart.program_args(python="/venv/bin/python")
    assert args == ["/venv/bin/python", "-m", "dendr", "ingest"]


def test_program_args_includes_data_dir():
    args = autostart.program_args(Path("/data/dendr"), python="/venv/bin/python")
    assert args[-2:] == ["--data-dir", "/data/dendr"]


def test_render_plist_roundtrips_and_sets_launch_keys():
    args = autostart.program_args(Path("/data/dendr"), python="/venv/bin/python")
    raw = autostart.render_plist(
        args,
        interval_seconds=900,
        stdout_path="/logs/ingest.out.log",
        stderr_path="/logs/ingest.err.log",
        working_dir="/vault",
    )
    d = plistlib.loads(raw)
    assert d["Label"] == "com.dendr.ingest"
    assert d["ProgramArguments"] == args
    assert d["RunAtLoad"] is True
    assert d["StartInterval"] == 900
    assert d["StandardOutPath"] == "/logs/ingest.out.log"
    assert d["StandardErrorPath"] == "/logs/ingest.err.log"
    assert d["WorkingDirectory"] == "/vault"


def test_render_plist_omits_optional_keys_when_absent():
    raw = autostart.render_plist(autostart.program_args(python="/p"))
    d = plistlib.loads(raw)
    assert "StandardOutPath" not in d
    assert "WorkingDirectory" not in d
