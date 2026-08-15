"""Filesystem helpers.

`atomic_write_text` is used for the writes that overwrite irreplaceable
user-authored files in the vault: the daily notes (block-ref injection and
task-closure write-back) and the digest (`digest.md` / `_digest_prompt.md`). A
plain `Path.write_text` truncates the file and then writes, so a crash, power
loss, or the launchd agent being killed mid-write leaves a truncated — and via
iCloud, replicated — user journal. Writing to a temp file in the same directory
and `os.replace`-ing it into place makes the swap atomic: a reader only ever
sees either the old complete file or the new one. (Append-only logs and
regenerable machine files — the activity log, `.dendr` marker, `config.json`,
queue items — don't route through here; they carry no irreplaceable content.)
"""

from __future__ import annotations

import os
import re
import tempfile
from datetime import datetime
from pathlib import Path

# Shared with pipeline.py's source-date extraction, so both stay in sync.
DATE_PATTERN = r"\d{4}-\d{2}-\d{2}"

# A sync-conflict copy ("<note> 2.md", "<note> (1).md", ".sync-conflict-…") is a
# byte copy of a real note. Shared by pipeline.py (Daily/, where the real
# invariant — a block ref claimed by only one file per scan — does the actual
# deduping, and this only sorts conflict-shaped names last so the canonical
# note claims refs first) and garden.py (Pages/, which has no block-ref
# mechanism to fall back on, so conflict-shaped names are skipped outright).
_CONFLICT_RE = re.compile(
    rf"(?:"
    r"conflicted copy"  # Dropbox / generic sync services
    r"|\s\(\d+\)$"  # Obsidian Sync: "2026-07-01 (1)"
    rf"|{DATE_PATTERN}\s+\d+$"  # iCloud: "2026-07-01 2"
    r")",
    re.IGNORECASE,
)


def is_conflicted_copy(path: Path) -> bool:
    """True if `path` looks like a sync-conflict duplicate of another note."""
    return _CONFLICT_RE.search(path.stem) is not None


def scan_order(path: Path) -> tuple[bool, str]:
    """Sort key: canonical notes before conflict-shaped names, then by name."""
    return (is_conflicted_copy(path), path.name)


def iso_week_label(dt: datetime) -> str:
    """Return an ISO week label like '2026-W15' (zero-padded, sortable)."""
    iso_year, iso_week, _ = dt.isocalendar()
    return f"{iso_year:04d}-W{iso_week:02d}"


def atomic_write_text(path: Path, text: str, encoding: str = "utf-8") -> None:
    """Write `text` to `path` atomically (temp file in the same dir + replace).

    The temp file is dot-prefixed so Obsidian ignores the momentary extra file,
    and fsync'd before the rename so the bytes are durable if the machine loses
    power immediately after. `os.replace` is atomic on the same filesystem.
    """
    path = Path(path)
    directory = path.parent
    directory.mkdir(parents=True, exist_ok=True)

    fd, tmp = tempfile.mkstemp(dir=directory, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding=encoding) as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        # Never leave the temp file behind on failure; re-raise the original.
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
