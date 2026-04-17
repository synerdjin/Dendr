"""Core ingestion pipeline — parse, privacy, embed, commit."""

from __future__ import annotations

import logging
import re
import sqlite3
import time
from datetime import datetime
from pathlib import Path

import numpy as np

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
    """Scan Daily/ for new or changed blocks.

    Hashes are compared in-memory per-file using a single bulk fetch to
    avoid one SELECT per block.
    """
    dirty_blocks: list[Block] = []
    daily_dir = config.daily_dir

    if not daily_dir.exists():
        return dirty_blocks

    for note_path in sorted(daily_dir.glob("*.md")):
        blocks = parse_daily_note(note_path, config.attachments_dir)
        inject_block_ids(note_path, blocks)

        stored = db.get_block_hashes(conn, [b.block_id for b in blocks])
        for block in blocks:
            if stored.get(block.block_id) != block.block_hash:
                dirty_blocks.append(block)

    return dirty_blocks


def queue_dirty_blocks(config: Config, dirty_blocks: list[Block]) -> int:
    """Tag privacy and enqueue dirty blocks."""
    filter_blocks(dirty_blocks)
    for block in dirty_blocks:
        queue.enqueue(
            config,
            QueueItem(
                block_id=block.block_id,
                source_file=block.source_file,
                block_hash=block.block_hash,
                block_text=block.text,
                checkbox_state=block.checkbox_state,
                private=block.private,
                attachment_path=block.attachment_path,
                attachment_type=block.attachment_type,
            ),
        )
    return len(dirty_blocks)


def _track_checkbox_transition(
    conn: sqlite3.Connection,
    item: QueueItem,
    source_date: str,
    existing: sqlite3.Row | None,
) -> None:
    """Log task_events for checkbox-driven transitions."""
    if item.checkbox_state == CHECKBOX_NONE:
        return

    if existing is None:
        db.insert_task_event(conn, item.block_id, "created", source_date)
        if item.checkbox_state == CHECKBOX_CLOSED:
            db.insert_task_event(
                conn, item.block_id, "closed", source_date, reason="done"
            )
        return

    old_state = existing["checkbox_state"]
    if old_state == item.checkbox_state:
        return

    if item.checkbox_state == CHECKBOX_CLOSED and old_state == CHECKBOX_OPEN:
        db.insert_task_event(conn, item.block_id, "closed", source_date, reason="done")
    elif item.checkbox_state == CHECKBOX_OPEN and old_state == CHECKBOX_CLOSED:
        db.insert_task_event(conn, item.block_id, "created", source_date)


def _embed_all(
    llm: LLMClient, claimed: list[QueueItem]
) -> dict[str, np.ndarray | None]:
    """Embed every block. Falls back to per-item on batch failure."""
    logger.info("Embedding %d blocks", len(claimed))
    try:
        vecs = llm.embed_batch([i.block_text for i in claimed])
        return {item.block_id: vec for item, vec in zip(claimed, vecs)}
    except Exception as e:
        logger.warning("Batch embed failed, falling back per-item: %s", e)

    out: dict[str, np.ndarray | None] = {}
    for item in claimed:
        try:
            out[item.block_id] = llm.embed(item.block_text)
        except Exception as e:
            logger.warning("Failed to embed %s: %s", item.block_id, e)
            out[item.block_id] = None
    return out


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

    embeddings = _embed_all(llm, claimed)

    existing_rows = db.get_blocks(conn, [i.block_id for i in claimed])
    logger.info("Committing %d blocks", len(claimed))
    processed = 0

    for item in claimed:
        source_date = _extract_source_date(item.source_file)
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

        try:
            conn.execute("BEGIN")
            try:
                _track_checkbox_transition(
                    conn, item, source_date, existing_rows.get(item.block_id)
                )
                db.upsert_block(conn, block, source_date)

                embedding = embeddings.get(item.block_id)
                if embedding is not None:
                    db.upsert_block_embedding(conn, item.block_id, embedding)

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
        config.append_activity_log(f"INGEST processed {processed} blocks")

    return processed


# Maps a digest closure status to (new completion_status, task_event reason).
_CLOSURE_DETAILS = {
    "done": ("done", "done"),
    "abandoned": ("abandoned", "abandoned"),
    "snoozed": ("snoozed", "snoozed"),
    "still-live": ("open", "reopened"),
}


def reconcile_closures(config: Config, conn: sqlite3.Connection) -> int:
    """Apply closure markers from Wiki/digest.md to block rows.

    Runs before the scan phase so user closures are in place before any
    re-parse can log spurious checkbox transitions.
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

    existing = db.get_blocks(conn, [c.block_id for c in closures])
    applied = 0
    today = datetime.now().strftime("%Y-%m-%d")

    for closure in closures:
        row = existing.get(closure.block_id)
        if row is None:
            continue

        details = _CLOSURE_DETAILS.get(closure.status)
        if details is None:
            continue
        new_completion, event_reason = details

        if row["completion_status"] == new_completion:
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
