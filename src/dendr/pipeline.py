"""Core ingestion pipeline — orchestrates block annotation and storage.

Annotation-first architecture:
  1. Reconcile closures from digest.md
  2. Parse daily notes, find dirty blocks
  3. Annotate blocks (tagger model) + embed annotation text
  4. Commit annotations, annotation embeddings, task events
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
    Block,
    BlockAnnotation,
    BlockType,
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
            existing = db.get_block_state(conn, block.block_id)
            if existing is None or existing["block_hash"] != block.block_hash:
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
            private=block.private,
            attachment_path=block.attachment_path,
            attachment_type=block.attachment_type,
        )
        queue.enqueue(config, item)
        count += 1
    return count


def _build_annotation(
    item: QueueItem, raw: dict, model_version: str
) -> BlockAnnotation:
    """Build a BlockAnnotation from raw LLM output."""
    raw_type = raw.get("block_type", "observation")
    try:
        block_type = BlockType(raw_type)
    except ValueError:
        block_type = BlockType.OBSERVATION

    return BlockAnnotation(
        block_id=item.block_id,
        source_file=item.source_file,
        source_date=_extract_source_date(item.source_file),
        original_text=item.block_text,
        gist=raw.get("gist", ""),
        block_type=block_type,
        life_areas=raw.get("life_areas", []),
        emotional_valence=float(raw.get("emotional_valence", 0.0)),
        intensity=float(raw.get("intensity", 0.5)),
        urgency=raw.get("urgency"),
        importance=raw.get("importance"),
        completion_status=raw.get("completion_status"),
        causal_links=raw.get("causal_links", []),
        concepts=raw.get("concepts", []),
        private=item.private,
        model_version=model_version,
        prompt_version=LLMClient.ANNOTATION_PROMPT_VERSION,
    )


# Statuses the user can set via the digest closure flow. If any of these
# are already on record for a block, a re-annotation that returns
# open/None must NOT reopen them — the tagger only reads the raw text
# ("- [ ] task"), so it would clobber user-driven closures on every
# re-ingest.
_STICKY_CLOSED_STATUSES = {"done", "abandoned", "snoozed", "still-live"}


def _track_task_lifecycle(
    conn: sqlite3.Connection, annotation: BlockAnnotation
) -> None:
    """Detect task status transitions and log lifecycle events.

    Compares the new annotation against any existing annotation for the same
    block_id. If completion_status changed (e.g. open -> done), log it.
    If this is a new task/plan block, log a 'created' event.

    Mutates `annotation.completion_status` to preserve sticky user-driven
    closures when the tagger tries to reopen them.
    """
    if annotation.block_type.value not in ("task", "plan"):
        return

    existing = db.get_block_annotation(conn, annotation.block_id)
    new_status = annotation.completion_status
    source_date = annotation.source_date

    if existing is None:
        # New task — log creation
        if new_status in (None, "open"):
            db.insert_task_event(conn, annotation.block_id, "created", source_date)
        return

    old_status = existing["completion_status"]

    # Sticky closures: once the user has closed a task via the digest
    # flow, a subsequent re-annotation must not silently reopen it.
    if old_status in _STICKY_CLOSED_STATUSES and new_status in (None, "open"):
        annotation.completion_status = old_status
        return

    if old_status == new_status:
        return

    if new_status == "done":
        db.insert_task_event(conn, annotation.block_id, "completed", source_date)
    elif new_status == "abandoned":
        db.insert_task_event(conn, annotation.block_id, "abandoned", source_date)
    elif new_status == "blocked":
        db.insert_task_event(conn, annotation.block_id, "blocked", source_date)


def process_queue(config: Config, conn: sqlite3.Connection, llm: LLMClient) -> int:
    """Process pending queue items through annotation + embedding.

    Phase 1 (tagger + embedding model): annotate all blocks + embed annotation text
    Phase 2 (DB commit):                persist annotations, embeddings, task events
    """
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

    # ── Phase 1: Annotation + embedding ──────────────────────────────
    total = len(claimed)
    logger.info("Phase 1/2: annotating & embedding %d blocks", total)
    phase1: dict[str, tuple[QueueItem, BlockAnnotation, bytes | None]] = {}

    for idx, item in enumerate(claimed, 1):
        try:
            logger.info(
                "Phase 1/2: annotating block %d/%d: %s", idx, total, item.block_id
            )
            raw_ann = llm.annotate_block(item.block_text)
            annotation = _build_annotation(
                item, raw_ann, llm.config.models.tagger_model
            )

            embed_text = annotation.gist or item.block_text
            try:
                ann_embedding = llm.embed(embed_text)
            except Exception as e:
                logger.warning("Failed to embed annotation %s: %s", item.block_id, e)
                ann_embedding = None

            phase1[item.block_id] = (item, annotation, ann_embedding)
        except Exception as e:
            logger.error("Failed to annotate block %s: %s", item.block_id, e)

    # ── Phase 2: Commit annotations ──────────────────────────────────
    total2 = len(phase1)
    logger.info("Phase 2/2: committing %d blocks", total2)
    processed = 0

    for block_id, (item, annotation, ann_embedding) in phase1.items():
        try:
            conn.execute("BEGIN")
            try:
                _track_task_lifecycle(conn, annotation)

                db.upsert_block_annotation(conn, annotation)

                if ann_embedding is not None:
                    try:
                        db.insert_annotation_embedding(
                            conn, item.block_id, ann_embedding
                        )
                    except Exception as e:
                        logger.warning(
                            "Failed to store annotation embedding %s: %s",
                            block_id,
                            e,
                        )

                db.upsert_block_state(
                    conn,
                    item.block_id,
                    item.source_file,
                    item.block_hash,
                    annotation.model_version,
                    annotation.prompt_version,
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


# Closure event_type for each user-driven status.
_CLOSURE_EVENT_TYPES = {
    "done": "completed",
    "abandoned": "abandoned",
    "snoozed": "snoozed",
    "still-live": "reopened",
}


def reconcile_closures(config: Config, conn: sqlite3.Connection) -> int:
    """Apply closure markers from Wiki/digest.md to block annotations.

    Runs before the scan/ingest phase so user closures are in place
    before any re-annotation can clobber them.
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
        existing = db.get_block_annotation(conn, closure.block_id)
        if existing is None:
            continue

        old_status = existing["completion_status"]
        new_status = closure.status

        # "still-live" reopens a closed task — set status back to open.
        stored_status = "open" if new_status == "still-live" else new_status

        if old_status == stored_status:
            continue

        db.update_completion_status(conn, closure.block_id, stored_status)
        event_type = _CLOSURE_EVENT_TYPES.get(new_status)
        if event_type:
            db.insert_task_event(
                conn,
                closure.block_id,
                event_type,
                today,
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
