"""Two-phase commit queue for the ingestion pipeline.

Blocks move through: pending/ → processing/ → done/
On crash, anything in processing/ is replayed on restart.
"""

from __future__ import annotations

import json
import logging
import shutil
from datetime import datetime
from pathlib import Path

from dendr.config import Config
from dendr.models import QueueItem

logger = logging.getLogger(__name__)


def enqueue(config: Config, item: QueueItem) -> Path:
    """Add a block to the pending queue."""
    config.pending_dir.mkdir(parents=True, exist_ok=True)
    path = config.pending_dir / f"{item.block_id}.json"
    data = {
        "block_id": item.block_id,
        "source_file": item.source_file,
        "block_hash": item.block_hash,
        "block_text": item.block_text,
        "private": item.private,
        "attachment_path": item.attachment_path,
        "attachment_type": item.attachment_type,
        "created_at": item.created_at.isoformat(),
    }
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return path


def claim_for_processing(config: Config, block_id: str) -> bool:
    """Move item from pending to processing (atomic on same filesystem)."""
    src = config.pending_dir / f"{block_id}.json"
    dst = config.processing_dir / f"{block_id}.json"
    if not src.exists():
        return False
    config.processing_dir.mkdir(parents=True, exist_ok=True)
    shutil.move(str(src), str(dst))
    return True


def mark_done(config: Config, block_id: str) -> None:
    """Move item from processing to done."""
    src = config.processing_dir / f"{block_id}.json"
    dst = config.done_dir / f"{block_id}.json"
    if src.exists():
        config.done_dir.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src), str(dst))


def get_pending(config: Config) -> list[QueueItem]:
    """List all pending items."""
    config.pending_dir.mkdir(parents=True, exist_ok=True)
    items: list[QueueItem] = []
    for path in sorted(config.pending_dir.glob("*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            items.append(
                QueueItem(
                    block_id=data["block_id"],
                    source_file=data["source_file"],
                    block_hash=data["block_hash"],
                    block_text=data["block_text"],
                    private=data.get("private", False),
                    attachment_path=data.get("attachment_path"),
                    attachment_type=data.get("attachment_type"),
                    created_at=datetime.fromisoformat(data["created_at"]),
                )
            )
        except (json.JSONDecodeError, KeyError) as e:
            logger.warning("Corrupt queue item %s: %s", path, e)
    return items


def get_stale_processing(config: Config) -> list[QueueItem]:
    """Find items stuck in processing/ (from a crash). These need replay."""
    config.processing_dir.mkdir(parents=True, exist_ok=True)
    items: list[QueueItem] = []
    for path in config.processing_dir.glob("*.json"):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            items.append(
                QueueItem(
                    block_id=data["block_id"],
                    source_file=data["source_file"],
                    block_hash=data["block_hash"],
                    block_text=data["block_text"],
                    private=data.get("private", False),
                    attachment_path=data.get("attachment_path"),
                    attachment_type=data.get("attachment_type"),
                    created_at=datetime.fromisoformat(data["created_at"]),
                )
            )
        except (json.JSONDecodeError, KeyError) as e:
            logger.warning("Corrupt processing item %s: %s", path, e)
    return items


def recover_stale(config: Config) -> int:
    """Move stale processing items back to pending for replay.

    Returns the count of recovered items.
    """
    stale = get_stale_processing(config)
    for item in stale:
        src = config.processing_dir / f"{item.block_id}.json"
        dst = config.pending_dir / f"{item.block_id}.json"
        if src.exists():
            shutil.move(str(src), str(dst))
    if stale:
        logger.info("Recovered %d stale processing items", len(stale))
    return len(stale)


def pending_count(config: Config) -> int:
    """Quick count of pending items."""
    config.pending_dir.mkdir(parents=True, exist_ok=True)
    return len(list(config.pending_dir.glob("*.json")))


def cleanup_done(config: Config, keep_days: int = 30) -> int:
    """Remove done items older than keep_days."""
    config.done_dir.mkdir(parents=True, exist_ok=True)
    cutoff = datetime.now().timestamp() - (keep_days * 86400)
    removed = 0
    for path in config.done_dir.glob("*.json"):
        if path.stat().st_mtime < cutoff:
            path.unlink()
            removed += 1
    return removed
