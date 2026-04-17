"""Core ingestion pipeline — parse, privacy, embed, commit.

Raw-text architecture:
  1. Reconcile closures from digest.md
  2. Parse daily notes, find dirty blocks (hash-based dedup)
  3. Embed block text (Nomic)
  4. Commit blocks + embeddings + checkbox-driven task events
"""

from __future__ import annotations

import logging
import re
import sqlite3
import time
from datetime import datetime
from pathlib import Path

from dendr import db, queue
from dendr.config import Config
from dendr.llm import LLMClient
from dendr.metrics import BLOCKS_PROCESSED, INGEST_CYCLE_SECONDS
from dendr.models import (
    CHECKBOX_CLOSED,
    CHECKBOX_NONE,
    CHECKBOX_OPEN,
    Block,
    QueueItem,
)
from dendr.parser import inject_block_ids, parse_closures, parse_daily_note
from dendr.privacy import filter_blocks

logger = logging.getLogger(__name__)

_DATE_RE = re.compile(r"(\d{4}-\d{2}-\d{2})")


def _extract_source_date(source_file: str) -> str:
    """Extract YYYY-MM-DD from a daily note filename."""
    m = _DATE_RE.search(Path(source_file).stem)
    return m.group(1) if m else datetime.now().strftime("%Y-%m-%d")


def scan_daily_notes(config: Config, conn: sqlite3.Connection) -> list[Block]:
    """Scan Daily/ for new or changed blocks."""
    dirty_blocks: list[Block] = []
    daily_dir = config.daily_dir

    if not daily_dir.exists():
        return dirty_blocks

    for note_path in sorted(daily_dir.glob("*.md")):
        blocks = parse_daily_note(note_path, config.attachments_dir)
        inject_block_ids(note_path, blocks)

        for block in blocks:
            stored_hash = db.get_block_hash(conn, block.block_id)
            if stored_hash != block.block_hash:
                dirty_blocks.append(block)

    return dirty_blocks


def queue_dirty_blocks(config: Config, dirty_blocks: list[Block]) -> int:
    """Add dirty blocks to the pending queue after privacy filtering."""
    filter_blocks(dirty_blocks)
    count = 0
    for block in dirty_blocks:
        item = QueueItem(
            block_id=block.block_id,
            source_file=block.source_file,
            block_hash=block.block_hash,
            block_text=block.text,
            checkbox_state=block.checkbox_state,
            private=block.private,
            attachment_path=block.attachment_path,
            attachment_type=block.attachment_type,
        )
        queue.enqueue(config, item)
        count += 1
    return count


def _track_checkbox_transition(conn: sqlite3.Connection, item: QueueItem) -> None:
    """Log task_events for checkbox-driven transitions.

    Creates a `created` event when a task first appears as open, and a
    `closed` event (reason='done') when `- [ ]` becomes `- [x]`.
    """
    if item.checkbox_state == CHECKBOX_NONE:
        return

    existing = db.get_block(conn, item.block_id)
    source_date = _extract_source_date(item.source_file)

    if existing is None:
        if item.checkbox_state == CHECKBOX_OPEN:
            db.insert_task_event(
                conn, item.block_id, "created", source_date, source="auto"
            )
        elif item.checkbox_state == CHECKBOX_CLOSED:
            # Task created and closed in the same write — log both so
            # completion stats still reflect it.
            db.insert_task_event(
                conn, item.block_id, "created", source_date, source="auto"
            )
            db.insert_task_event(
                conn,
                item.block_id,
                "closed",
                source_date,
                reason="done",
                source="auto",
            )
        return

    old_state = existing["checkbox_state"]
    if old_state == item.checkbox_state:
        return

    if old_state == CHECKBOX_OPEN and item.checkbox_state == CHECKBOX_CLOSED:
        db.insert_task_event(
            conn,
            item.block_id,
            "closed",
            source_date,
            reason="done",
            source="auto",
        )
    elif old_state == CHECKBOX_CLOSED and item.checkbox_state == CHECKBOX_OPEN:
        db.insert_task_event(
            conn,
            item.block_id,
            "created",
            source_date,
            source="auto",
        )


def process_queue(config: Config, conn: sqlite3.Connection, llm: LLMClient) -> int:
    """Process pending queue items: embed raw text, commit."""
    queue.recover_stale(config)
    pending = queue.get_pending(config)
    if not pending:
        return 0

    claimed: list[QueueItem] = []
    for item in pending:
        if queue.claim_for_processing(config, item.block_id):
            claimed.append(item)
    if not claimed:
        return 0

    # ── Embed all blocks (embedding model stays loaded) ──────────────
    total = len(claimed)
    logger.info("Embedding %d blocks", total)
    embeddings: dict[str, bytes | None] = {}
    for idx, item in enumerate(claimed, 1):
        try:
            logger.info("Embedding block %d/%d: %s", idx, total, item.block_id)
            embeddings[item.block_id] = llm.embed(item.block_text)
        except Exception as e:
            logger.warning("Failed to embed block %s: %s", item.block_id, e)
            embeddings[item.block_id] = None

    # ── Commit blocks + embeddings + task events ─────────────────────
    logger.info("Committing %d blocks", total)
    processed = 0

    for item in claimed:
        block = Block(
            block_id=item.block_id,
            source_file=item.source_file,
            line_start=0,
            line_end=0,
            text=item.block_text,
            block_hash=item.block_hash,
            checkbox_state=item.checkbox_state,
            private=item.private,
            attachment_path=item.attachment_path,
            attachment_type=item.attachment_type,
        )
        source_date = _extract_source_date(item.source_file)

        try:
            conn.execute("BEGIN")
            try:
                _track_checkbox_transition(conn, item)
                db.upsert_block(conn, block, source_date)

                ann_embedding = embeddings.get(item.block_id)
                if ann_embedding is not None:
                    try:
                        db.insert_block_embedding(conn, item.block_id, ann_embedding)
                    except Exception as e:
                        logger.warning(
                            "Failed to store block embedding %s: %s",
                            item.block_id,
                            e,
                        )

                conn.execute("COMMIT")
            except Exception:
                conn.execute("ROLLBACK")
                raise

            queue.mark_done(config, item.block_id)
            processed += 1
            BLOCKS_PROCESSED.inc()

        except Exception as e:
            logger.error("Failed to process block %s: %s", item.block_id, e)
            continue

    if processed > 0:
        config.append_activity_log(
            f"INGEST processed {processed} blocks",
        )

    return processed


# Closure event details for each user-driven status. Maps the marker
# status in the digest to (completion_status, event_reason) pairs.
_CLOSURE_DETAILS = {
    "done": ("done", "done"),
    "abandoned": ("abandoned", "abandoned"),
    "snoozed": ("snoozed", "snoozed"),
    "still-live": ("open", "reopened"),
}


def reconcile_closures(config: Config, conn: sqlite3.Connection) -> int:
    """Apply closure markers from Wiki/digest.md to block rows.

    Runs before the scan/ingest phase so user closures are in place
    before any re-parse can log spurious checkbox transitions.
    """
    digest_path = config.wiki_dir / "digest.md"
    if not digest_path.exists():
        return 0

    try:
        text = digest_path.read_text(encoding="utf-8")
    except OSError as e:
        logger.warning("Could not read digest for closures: %s", e)
        return 0

    closures = parse_closures(text)
    if not closures:
        return 0

    applied = 0
    today = datetime.now().strftime("%Y-%m-%d")

    for closure in closures:
        existing = db.get_block(conn, closure.block_id)
        if existing is None:
            continue

        details = _CLOSURE_DETAILS.get(closure.status)
        if details is None:
            continue
        new_completion, event_reason = details

        old_completion = existing["completion_status"]
        if old_completion == new_completion:
            continue

        db.update_completion_status(conn, closure.block_id, new_completion)

        event_type = "created" if event_reason == "reopened" else "closed"
        db.insert_task_event(
            conn,
            closure.block_id,
            event_type,
            today,
            reason=event_reason,
            source="user",
        )
        applied += 1

    if applied > 0:
        logger.info("Applied %d closure(s) from digest", applied)

    return applied


def run_ingest(config: Config, conn: sqlite3.Connection, llm: LLMClient) -> dict:
    """Full ingest cycle: reconcile closures -> scan -> queue -> process."""
    logger.info("Starting ingest cycle...")
    t0 = time.monotonic()

    closures_applied = reconcile_closures(config, conn)
    if closures_applied:
        logger.info("Reconciled %d closures from digest", closures_applied)

    dirty = scan_daily_notes(config, conn)
    queued = queue_dirty_blocks(config, dirty)
    logger.info("Found %d dirty blocks, queued %d", len(dirty), queued)

    processed = process_queue(config, conn, llm)
    logger.info("Processed %d blocks", processed)

    INGEST_CYCLE_SECONDS.observe(time.monotonic() - t0)

    return {
        "closures_applied": closures_applied,
        "dirty_blocks": len(dirty),
        "queued": queued,
        "processed": processed,
    }
