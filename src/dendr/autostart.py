"""macOS launchd LaunchAgent generation for running ingest on a schedule.

Pure helpers (plist rendering, paths, launchctl invocation) live here so the CLI
layer stays thin and the rendering is unit-testable without touching launchctl.
The agent runs ``<python> -m dendr ingest`` with ``RunAtLoad`` (once at login)
and ``StartInterval`` (every N seconds thereafter) — each run is a single ingest
cycle that exits, not a long-lived process.
"""

from __future__ import annotations

import os
import plistlib
import subprocess
import sys
from pathlib import Path

LAUNCH_AGENT_LABEL = "com.dendr.ingest"

# Pre-v8 label: the watcher daemon this agent replaced. `autostart install`
# cleans up an agent still running under this label so it doesn't keep
# respawning (KeepAlive) alongside the new scheduled one.
LEGACY_LAUNCH_AGENT_LABEL = "com.dendr.daemon"


def plist_path(label: str = LAUNCH_AGENT_LABEL) -> Path:
    """Path of the per-user LaunchAgent plist for the given label."""
    return Path.home() / "Library" / "LaunchAgents" / f"{label}.plist"


def program_args(data_dir: Path | None = None, python: str | None = None) -> list[str]:
    """Argv launchd should exec.

    Uses ``-m dendr`` against the *current* interpreter (``sys.executable``) so
    the agent runs in the exact environment that installed it — no reliance on a
    console script being on ``PATH`` (launchd runs with a bare environment).
    """
    args = [python or sys.executable, "-m", "dendr", "ingest"]
    if data_dir is not None:
        args += ["--data-dir", str(data_dir)]
    return args


def build_plist_dict(
    args: list[str],
    *,
    label: str = LAUNCH_AGENT_LABEL,
    interval_seconds: int = 900,
    stdout_path: str | None = None,
    stderr_path: str | None = None,
    working_dir: str | None = None,
) -> dict:
    """Assemble the launchd property-list dict for the scheduled ingest agent."""
    d: dict = {
        "Label": label,
        "ProgramArguments": args,
        "RunAtLoad": True,
        "StartInterval": interval_seconds,
        # Run at lowered priority — this is a background batch job, not an
        # interactive one.
        "ProcessType": "Background",
    }
    if stdout_path:
        d["StandardOutPath"] = stdout_path
    if stderr_path:
        d["StandardErrorPath"] = stderr_path
    if working_dir:
        d["WorkingDirectory"] = working_dir
    return d


def render_plist(args: list[str], **kwargs) -> bytes:
    """Render the LaunchAgent plist as XML bytes (plistlib handles escaping)."""
    return plistlib.dumps(build_plist_dict(args, **kwargs))


def _run(cmd: list[str]) -> tuple[int, str]:
    """Run a command, returning (returncode, combined output). Never raises."""
    try:
        proc = subprocess.run(  # noqa: S603 — fixed argv, no shell
            cmd, capture_output=True, text=True, check=False
        )
    except OSError as e:  # launchctl missing (non-macOS)
        return 1, str(e)
    return proc.returncode, (proc.stdout + proc.stderr).strip()


def load_agent(path: Path, label: str = LAUNCH_AGENT_LABEL) -> tuple[int, str]:
    """Bootstrap (load) the agent into the user's GUI domain.

    Tries the modern ``bootstrap`` verb first, falling back to the legacy
    ``load -w`` for older macOS where ``bootstrap`` behaves differently.
    """
    domain = f"gui/{os.getuid()}"
    rc, out = _run(["launchctl", "bootstrap", domain, str(path)])
    if rc == 0:
        return rc, out
    legacy_rc, legacy_out = _run(["launchctl", "load", "-w", str(path)])
    if legacy_rc == 0:
        return legacy_rc, legacy_out
    # Surface the modern error — it's usually the more informative one.
    return rc, out or legacy_out


def unload_agent(path: Path, label: str = LAUNCH_AGENT_LABEL) -> tuple[int, str]:
    """Bootout (unload) the agent. Idempotent — absent agent is treated as OK."""
    domain = f"gui/{os.getuid()}"
    rc, out = _run(["launchctl", "bootout", f"{domain}/{label}"])
    if rc == 0:
        return rc, out
    return _run(["launchctl", "unload", "-w", str(path)])


def is_loaded(label: str = LAUNCH_AGENT_LABEL) -> bool:
    """Whether launchd currently has the agent loaded."""
    rc, _ = _run(["launchctl", "list", label])
    return rc == 0


def remove_legacy_agent(label: str = LEGACY_LAUNCH_AGENT_LABEL) -> Path | None:
    """Unload + delete a pre-v8 watcher-daemon agent still installed under `label`.

    Returns its plist path if one was found and removed, else None.
    """
    path = plist_path(label)
    if not path.exists():
        return None
    unload_agent(path, label=label)
    path.unlink()
    return path
