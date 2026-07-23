"""Core ingestion pipeline — parse, embed, commit."""

from __future__ import annotations

import logging
import re
import sqlite3
import time
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

from dendr import db, queue
from dendr.config import Config
from dendr.llm import LLMClient
from dendr.metrics import (
    BLOCKS_ERROR,
    BLOCKS_PROCESSED,
    EMBED_THROUGHPUT,
    INGEST_CYCLE_SECONDS,
    TASKS_CLOSED,
)
from dendr.models import (
    CHECKBOX_CLOSED,
    CHECKBOX_NONE,
    CHECKBOX_OPEN,
    CLOSURE_ABANDONED,
    CLOSURE_DONE,
    CLOSURE_SNOOZED,
    CLOSURE_STILL_LIVE,
    COMPLETION_ABANDONED,
    COMPLETION_DONE,
    COMPLETION_OPEN,
    COMPLETION_SNOOZED,
    COMPLETION_TERMINAL,
    EVENT_CLOSED,
    EVENT_CREATED,
    REASON_ABANDONED,
    REASON_DONE,
    REASON_REOPENED,
    REASON_SNOOZED,
    REASON_WOKE,
    SNOOZE_DEFAULT_DAYS,
    SOURCE_USER,
    Block,
    QueueItem,
)
from dendr.parser import (
    close_task_in_source,
    inject_block_ids,
    parse_closures,
    parse_daily_note,
)

logger = logging.getLogger(__name__)

_DATE_RE = re.compile(r"(\d{4}-\d{2}-\d{2})")

# A sync-conflict copy ("<note> 2.md", "<note> (1).md", ".sync-conflict-…") is a
# byte copy of a real note and carries the SAME `^dendr-<ulid>` refs, so
# ingesting it alongside the original makes each block's stored text flip-flop
# between the two files every cycle. We don't trust any single vendor's naming
# to catch it — scan_daily_notes enforces the real invariant (a block_id comes
# from only one file per scan). This pattern only *sorts* conflict-shaped names
# last so the canonical note claims its refs first; a note whose refs are
# actually unique is still ingested even if its name happens to match here.
_CONFLICT_RE = re.compile(
    r"(?:"
    r"conflicted copy"  # Dropbox / generic sync services
    r"|\s\(\d+\)$"  # Obsidian Sync: "2026-07-01 (1)"
    rf"|{_DATE_RE.pattern}\s+\d+$"  # iCloud: "2026-07-01 2"
    r")",
    re.IGNORECASE,
)


def _is_conflicted_copy(path: Path) -> bool:
    """True if `path` looks like a sync-conflict duplicate of a daily note."""
    return _CONFLICT_RE.search(path.stem) is not None


def _scan_order(path: Path) -> tuple[bool, str]:
    """Sort key: canonical notes before conflict-shaped names, then by name.

    On a cold start (neither file recorded yet) this decides which of two files
    sharing a block ref is treated as canonical — the non-conflict-shaped name.
    """
    return (_is_conflicted_copy(path), path.name)


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

    claimed: dict[str, str] = {}  # block_id -> file that first claimed it
    for note_path in sorted(daily_dir.glob("*.md"), key=_scan_order):
        blocks = parse_daily_note(note_path, config.attachments_dir)

        # If a ref here was already claimed by another file this scan, this is a
        # sync-conflict copy of that file — skip it (don't inject into it or
        # ingest it) so its stale text can't overwrite the canonical block.
        clash = next(
            (
                b.block_id
                for b in blocks
                if claimed.get(b.block_id, str(note_path)) != str(note_path)
            ),
            None,
        )
        if clash is not None:
            logger.warning(
                "Skipping %s: block ref %s already claimed by %s (sync-conflict copy?)",
                note_path.name,
                clash,
                claimed[clash],
            )
            continue

        inject_block_ids(note_path, blocks)
        for block in blocks:
            claimed[block.block_id] = str(note_path)

        stored = db.get_block_hashes(conn, [b.block_id for b in blocks])
        for block in blocks:
            if stored.get(block.block_id) != block.block_hash:
                dirty_blocks.append(block)

    return dirty_blocks


def queue_dirty_blocks(config: Config, dirty_blocks: list[Block]) -> int:
    """Enqueue dirty blocks, skipping ones already dead-lettered unchanged.

    A block that poisoned the queue is moved to dead/ but its hash is never
    committed to `blocks`, so the scan keeps flagging it dirty. Without this
    guard it would re-embed and re-dead on every cycle. A dead block whose
    content changed (different hash) gets a fresh attempt.
    """
    dead = queue.get_dead_hashes(config)
    # Skip blocks whose exact content already dead-lettered; a changed hash
    # (edited since it died) falls through to a fresh attempt.
    to_queue = [b for b in dirty_blocks if dead.get(b.block_id) != b.block_hash]
    for block in to_queue:
        if block.block_id in dead:
            queue.clear_dead(config, block.block_id)  # edited since it died; retry
        queue.enqueue(
            config,
            QueueItem(
                block_id=block.block_id,
                source_file=block.source_file,
                block_hash=block.block_hash,
                block_text=block.text,
                checkbox_state=block.checkbox_state,
                attachment_path=block.attachment_path,
                attachment_type=block.attachment_type,
            ),
        )
    return len(to_queue)


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
        db.insert_task_event(conn, item.block_id, EVENT_CREATED, source_date)
        if item.checkbox_state == CHECKBOX_CLOSED:
            db.insert_task_event(
                conn, item.block_id, EVENT_CLOSED, source_date, reason=REASON_DONE
            )
        return

    old_state = existing["checkbox_state"]
    if old_state == item.checkbox_state:
        return

    if item.checkbox_state == CHECKBOX_CLOSED:
        # A digest closure already logged a user-sourced close and ticked the
        # source checkbox; don't double-count that echo as an auto close.
        if existing["completion_status"] in COMPLETION_TERMINAL:
            return
        # A plain block turned straight into a done task (none→closed) is both a
        # creation and a close; open→closed is just the close.
        if old_state == CHECKBOX_NONE:
            db.insert_task_event(conn, item.block_id, EVENT_CREATED, source_date)
        db.insert_task_event(
            conn, item.block_id, EVENT_CLOSED, source_date, reason=REASON_DONE
        )
        TASKS_CLOSED.labels(source="auto").inc()
    elif item.checkbox_state == CHECKBOX_OPEN:
        # Reopened in the source note. A terminal user closure (done/abandoned)
        # is stale once the user unchecks the source line — clear it, or the
        # task stays invisible to get_open_tasks and the digest review forever.
        # Covers closed→open (reopen after `done`) and none→open (reopen after
        # `abandoned`, whose `- [-]` mark parses as checkbox_state=none).
        if existing["completion_status"] in COMPLETION_TERMINAL:
            db.update_completion_status(conn, item.block_id, None)
            db.insert_task_event(
                conn,
                item.block_id,
                EVENT_CREATED,
                source_date,
                reason=REASON_REOPENED,
            )
        else:
            # A newly-added open checkbox (none→open) or an un-done task
            # (closed→open) is a fresh task appearance either way.
            db.insert_task_event(conn, item.block_id, EVENT_CREATED, source_date)


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

    embed_t0 = time.monotonic()
    embeddings = _embed_all(llm, claimed)
    embed_elapsed = time.monotonic() - embed_t0
    rate = len(claimed) / embed_elapsed if embed_elapsed > 0 else 0
    EMBED_THROUGHPUT.set(rate)
    logger.info(
        "Embedded %d blocks in %.1fs (%.1f blocks/sec)",
        len(claimed),
        embed_elapsed,
        rate,
    )

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
            queue.mark_dead(config, item.block_id)
            BLOCKS_ERROR.inc()
            continue

    if processed > 0:
        config.append_activity_log(f"INGEST processed {processed} blocks")

    return processed


# Maps a digest closure status to (new completion_status, task_event reason).
# `snoozed` is handled separately (_apply_snooze) because it also carries a wake
# date and must not write completion_status the same blunt way.
_CLOSURE_DETAILS = {
    CLOSURE_DONE: (COMPLETION_DONE, REASON_DONE),
    CLOSURE_ABANDONED: (COMPLETION_ABANDONED, REASON_ABANDONED),
    CLOSURE_STILL_LIVE: (COMPLETION_OPEN, REASON_REOPENED),
}

# Closures that also flip the checkbox in the source daily note, mapped to the
# Markdown mark written there (Tasks plugin: `x` = done, `-` = cancelled).
# Snoozed/still-live stay open in source, so they're absent here.
#
# Note: `-` is deliberately NOT in parser._CHECKBOX_RE, so a re-parsed `- [-]`
# line reads as checkbox_state=none (no transition logged) while completion_status
# stays 'abandoned' (authoritative). Done's `x` IS a checkbox mark, so its echo is
# instead suppressed by the completion_status guard in _track_checkbox_transition.
_SOURCE_WRITEBACK_MARK = {
    CLOSURE_DONE: "x",
    CLOSURE_ABANDONED: "-",
}


def _default_snooze_until() -> str:
    """Wake date for a snooze with no explicit `until:` — a week out."""
    return (datetime.now() + timedelta(days=SNOOZE_DEFAULT_DAYS)).strftime("%Y-%m-%d")


def _apply_snooze(
    conn: sqlite3.Connection, closure, row: sqlite3.Row, today: str
) -> bool:
    """Snooze a task until its wake date. Returns True if anything changed.

    The wake date is the marker's `until:` value or a week out by default. A
    past/expired date is ignored here — wake_snoozed_tasks handles it, and
    re-snoozing to a past date would just fight the wake and flip-flop.
    """
    wake_until = closure.wake_date or _default_snooze_until()
    if wake_until <= today:
        return False
    if (
        row["completion_status"] == COMPLETION_SNOOZED
        and row["snooze_until"] == wake_until
    ):
        return False  # already snoozed to the same date
    db.set_snooze(conn, closure.block_id, wake_until)
    db.insert_task_event(
        conn,
        closure.block_id,
        EVENT_CLOSED,
        today,
        reason=REASON_SNOOZED,
        source=SOURCE_USER,
    )
    TASKS_CLOSED.labels(source="user").inc()
    return True


def wake_snoozed_tasks(conn: sqlite3.Connection) -> int:
    """Resurface snoozed tasks whose wake date has arrived.

    Clears completion_status (so get_open_tasks and the digest Task Review see
    them again) and logs a `woke` event. Returns the number woken.
    """
    today = datetime.now().strftime("%Y-%m-%d")
    due = db.get_due_snoozed_blocks(conn, today)
    for block_id in due:
        db.update_completion_status(conn, block_id, None)
        db.insert_task_event(conn, block_id, EVENT_CREATED, today, reason=REASON_WOKE)
    if due:
        logger.info("Woke %d snoozed task(s)", len(due))
    return len(due)


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
        digest_mtime = digest_path.stat().st_mtime
    except OSError as e:
        logger.warning("Could not read digest for closures: %s", e)
        return 0

    closures = parse_closures(text)
    if not closures:
        return 0

    existing = db.get_blocks(conn, [c.block_id for c in closures])
    applied = 0
    today = datetime.now().strftime("%Y-%m-%d")
    # Markers carry no timestamp, so the digest file's mtime stands in for
    # "when the user last asserted these closures". A source-checkbox reopen
    # logged after that is newer information and wins over a stale marker.
    # Compared as strings against task_events.created_at — both sides must
    # stay naive-local ISO format for the ordering to hold.
    digest_edited_at = datetime.fromtimestamp(digest_mtime).isoformat()

    for closure in closures:
        row = existing.get(closure.block_id)
        if row is None:
            continue

        if closure.status == CLOSURE_SNOOZED:
            if _apply_snooze(conn, closure, row, today):
                applied += 1
            continue

        details = _CLOSURE_DETAILS.get(closure.status)
        if details is None:
            continue
        new_completion, event_reason = details

        if row["completion_status"] == new_completion:
            continue

        if new_completion in COMPLETION_TERMINAL:
            reopened_at = db.get_latest_reopen_event_time(conn, closure.block_id)
            if reopened_at is not None and reopened_at > digest_edited_at:
                continue

        db.update_completion_status(conn, closure.block_id, new_completion)

        event_type = EVENT_CREATED if event_reason == REASON_REOPENED else EVENT_CLOSED
        db.insert_task_event(
            conn,
            closure.block_id,
            event_type,
            today,
            reason=event_reason,
            source=SOURCE_USER,
        )
        if event_type == EVENT_CLOSED:
            TASKS_CLOSED.labels(source="user").inc()
        # Mirror the close back into the source daily note so the checkbox is
        # actually ticked there (no more hunting for the task to close it).
        mark = _SOURCE_WRITEBACK_MARK.get(closure.status)
        source_file = row["source_file"]
        if (
            mark
            and source_file
            and close_task_in_source(Path(source_file), closure.block_id, mark, today)
        ):
            logger.info("Closed task %s in source %s", closure.block_id, source_file)

        applied += 1

    if applied > 0:
        logger.info("Applied %d closure(s) from digest", applied)

    return applied


def run_ingest(config: Config, conn: sqlite3.Connection, llm: LLMClient) -> dict:
    """Full ingest cycle: reconcile closures -> scan -> queue -> process."""
    logger.info("Starting ingest cycle...")
    t0 = time.monotonic()

    # Elapsed snooze timers first, then apply the user's newest digest edits.
    woke = wake_snoozed_tasks(conn)

    closures_applied = reconcile_closures(config, conn)
    if closures_applied:
        logger.info("Reconciled %d closures from digest", closures_applied)

    dirty = scan_daily_notes(config, conn)
    queued = queue_dirty_blocks(config, dirty)
    logger.info("Found %d dirty blocks, queued %d", len(dirty), queued)

    processed = process_queue(config, conn, llm)

    elapsed = time.monotonic() - t0
    rate = processed / elapsed if elapsed > 0 else 0
    logger.info(
        "Processed %d blocks in %.1fs (%.1f blocks/sec)",
        processed,
        elapsed,
        rate,
    )

    INGEST_CYCLE_SECONDS.observe(elapsed)

    return {
        "woke_snoozed": woke,
        "closures_applied": closures_applied,
        "dirty_blocks": len(dirty),
        "queued": queued,
        "processed": processed,
        "elapsed_sec": round(elapsed, 1),
        "blocks_per_sec": round(rate, 1),
    }
