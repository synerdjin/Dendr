"""Cross-process advisory lock so ingest and reprocess never run at once.

`dendr ingest` runs on a schedule (the launchd agent, every 15 min) while
`dendr reprocess` is a manual command that `shutil.rmtree`s the processing/
queue and blanks every block hash. If the two overlap, reprocess can delete a
queue item an ingest has mid-flight. This is an `flock` on a file in `data_dir`:
non-blocking, so the loser fails fast rather than stalling, and released by the
kernel when the process exits — even on crash — so there is no stale lock to
clean up.
"""

from __future__ import annotations

import contextlib
import fcntl
import logging
from collections.abc import Iterator

from dendr.config import Config

logger = logging.getLogger(__name__)


class LockHeld(RuntimeError):
    """Raised when the ingest lock is already held by another process."""


@contextlib.contextmanager
def ingest_lock(config: Config) -> Iterator[None]:
    """Hold the exclusive ingest lock for the duration, or raise LockHeld.

    The lock is advisory (only cooperating callers — `ingest` and `reprocess` —
    take it), non-blocking, and released when the file object closes, including
    on process exit.
    """
    config.data_dir.mkdir(parents=True, exist_ok=True)
    lock_path = config.lock_path
    fd = lock_path.open("w")
    try:
        fcntl.flock(fd.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError as e:
        fd.close()
        # Logged (not just echoed) so a wedged run that skips every scheduled
        # tick is visible to DENDR_LOG_JSON aggregation, not only on stderr.
        logger.warning("Ingest lock %s is held by another process", lock_path)
        raise LockHeld(
            f"another ingest or reprocess is already running (lock: {lock_path})"
        ) from e
    try:
        yield
    finally:
        # Closing the fd releases the flock — no explicit LOCK_UN needed.
        fd.close()
