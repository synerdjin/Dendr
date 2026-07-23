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
import tempfile
from pathlib import Path


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
