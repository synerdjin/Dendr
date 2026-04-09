"""Core ingestion pipeline — orchestrates the full block-to-wiki flow.

Ties together: parser → privacy → queue → enrichment → canonicalization
→ claim store → wiki update.
"""

from __future__ import annotations

import logging
import sqlite3
import time
from datetime import datetime
from pathlib import Path

from dendr import db, queue
from dendr.metrics import (
    BACKPRESSURE_ACTIVE,
    BLOCKS_PROCESSED,
    CLAIMS_EXTRACTED,
    CLAIMS_REINFORCED,
    CONTRADICTIONS_DETECTED,
    INGEST_CYCLE_SECONDS,
)
from dendr.canonicalize import canonicalize_concepts
from dendr.config import Config
from dendr.enrichment import enrich_block
from dendr.llm import LLMClient
from dendr.models import Block, Claim, ClaimStatus, Concept, PageType, QueueItem
from dendr.parser import get_file_hash, inject_block_ids, parse_daily_note
from dendr.privacy import filter_blocks
from dendr.wiki import (
    append_activity_log,
    append_evidence,
    ensure_page,
    update_index,
)

logger = logging.getLogger(__name__)


def scan_daily_notes(config: Config, conn: sqlite3.Connection) -> list[Block]:
    """Scan Daily/ for new or changed blocks.

    Returns only dirty blocks (new or hash-changed since last processing).
    """
    dirty_blocks: list[Block] = []
    daily_dir = config.daily_dir

    if not daily_dir.exists():
        return dirty_blocks

    for note_path in sorted(daily_dir.glob("*.md")):
        blocks = parse_daily_note(note_path, config.attachments_dir)

        # Inject block IDs into the file for any blocks that lack them
        inject_block_ids(note_path, blocks)

        # Check which blocks are dirty
        for block in blocks:
            existing = db.get_block_state(conn, block.block_id)
            if existing is None or existing["block_hash"] != block.block_hash:
                dirty_blocks.append(block)

    return dirty_blocks


def queue_dirty_blocks(config: Config, dirty_blocks: list[Block]) -> int:
    """Add dirty blocks to the pending queue after privacy filtering.

    Returns count of queued items.
    """
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


def process_queue(config: Config, conn: sqlite3.Connection, llm: LLMClient) -> int:
    """Process all pending queue items through the enrichment pipeline.

    Returns count of processed items.
    """
    # Recover any items stuck from a prior crash
    queue.recover_stale(config)

    pending = queue.get_pending(config)
    if not pending:
        return 0

    # Check backpressure
    total_pending = len(pending)
    # Rough estimate: each daily note produces ~20 blocks
    estimated_days = total_pending / 20
    shallow = estimated_days > config.backpressure_days

    BACKPRESSURE_ACTIVE.set(1 if shallow else 0)

    if shallow:
        logger.warning(
            "Backpressure: ~%.0f days queued (threshold: %d). Using shallow enrichment.",
            estimated_days,
            config.backpressure_days,
        )

    # Get existing concept slugs for the enrichment prompts
    existing_concepts = [
        r["slug"] for r in db.get_all_concepts(conn)
    ]

    processed = 0
    for item in pending:
        # Phase 1: claim for processing
        if not queue.claim_for_processing(config, item.block_id):
            continue

        try:
            # Build a Block from the queue item
            block = Block(
                block_id=item.block_id,
                source_file=item.source_file,
                line_start=0,
                line_end=0,
                text=item.block_text,
                block_hash=item.block_hash,
                private=item.private,
                attachment_path=item.attachment_path,
                attachment_type=item.attachment_type,
            )

            # Enrich
            result = enrich_block(block, llm, existing_concepts, shallow=shallow)

            # Begin transaction
            conn.execute("BEGIN")

            try:
                # Canonicalize concepts
                slug_map = canonicalize_concepts(
                    result.concepts, llm, conn, config
                )

                # Process each claim
                source_ref = Path(item.source_file).stem
                for claim_data in result.claims:
                    # Determine canonical concept slug for this claim
                    concept_slug = ""
                    for c in claim_data.concepts:
                        if c in slug_map:
                            concept_slug = slug_map[c]
                            break
                    if not concept_slug and slug_map:
                        concept_slug = next(iter(slug_map.values()))

                    sp_key = f"{claim_data.subject}|{claim_data.predicate}"

                    # Check for existing identical claim (reinforce)
                    existing = db.find_similar_claim(
                        conn, sp_key, claim_data.object
                    )
                    if existing:
                        db.reinforce_claim(conn, existing["id"])
                        CLAIMS_REINFORCED.inc()
                        continue

                    # Check for contradictions
                    contradictions = db.find_contradictions(
                        conn, sp_key, claim_data.object
                    )

                    # Insert the new claim
                    claim = Claim(
                        id=None,
                        text=claim_data.text,
                        subject=claim_data.subject,
                        predicate=claim_data.predicate,
                        object=claim_data.object,
                        subject_predicate=sp_key,
                        concept_slug=concept_slug,
                        source_block_ref=item.block_id,
                        source_file_hash=item.block_hash,
                        created_at=datetime.now(),
                        updated_at=datetime.now(),
                        confidence=claim_data.confidence,
                        status=ClaimStatus.CREATED,
                        private=block.private,
                        model_version=result.model_version,
                        prompt_version=result.prompt_version,
                    )
                    new_id = db.insert_claim(conn, claim)
                    CLAIMS_EXTRACTED.inc()

                    # Embed the claim
                    try:
                        emb = llm.embed(claim_data.text)
                        db.insert_claim_embedding(conn, new_id, emb)
                    except Exception as e:
                        logger.warning("Failed to embed claim %d: %s", new_id, e)

                    # Handle contradictions
                    for contra in contradictions:
                        db.challenge_claim(conn, contra["id"])
                        CONTRADICTIONS_DETECTED.inc()
                        db.append_log(conn, "contradiction", {
                            "new_claim_id": new_id,
                            "challenged_id": contra["id"],
                            "subject_predicate": sp_key,
                        })

                # Ensure concept pages exist and append evidence
                for candidate, slug in slug_map.items():
                    title = candidate.replace("-", " ").title()
                    evidence = [
                        c.text for c in result.claims
                        if any(cc in slug_map and slug_map[cc] == slug for cc in c.concepts)
                    ]
                    if not evidence:
                        evidence = [f"Referenced in {source_ref}"]

                    ensure_page(config, conn, slug, title, PageType.CONCEPT)
                    append_evidence(
                        config, conn, llm, slug, title,
                        evidence, source_ref, PageType.CONCEPT,
                    )

                # Record block as processed
                db.upsert_block_state(
                    conn,
                    item.block_id,
                    item.source_file,
                    item.block_hash,
                    result.model_version,
                    result.prompt_version,
                )

                conn.execute("COMMIT")
            except Exception:
                conn.execute("ROLLBACK")
                raise

            # Phase 2: mark done (only after successful commit)
            queue.mark_done(config, item.block_id)
            processed += 1
            BLOCKS_PROCESSED.inc()

            # Update existing concepts list for next iteration
            existing_concepts = [r["slug"] for r in db.get_all_concepts(conn)]

        except Exception as e:
            logger.error("Failed to process block %s: %s", item.block_id, e)
            # Leave in processing/ — will be recovered on next run
            continue

    # Update index after batch
    if processed > 0:
        update_index(config, conn)
        append_activity_log(
            config,
            f"INGEST processed {processed} blocks, {len(existing_concepts)} concepts",
        )

    return processed


def run_ingest(config: Config, conn: sqlite3.Connection, llm: LLMClient) -> dict:
    """Full ingest cycle: scan → queue → process.

    Returns stats dict.
    """
    logger.info("Starting ingest cycle...")
    t0 = time.monotonic()

    dirty = scan_daily_notes(config, conn)
    queued = queue_dirty_blocks(config, dirty)
    logger.info("Found %d dirty blocks, queued %d", len(dirty), queued)

    processed = process_queue(config, conn, llm)
    logger.info("Processed %d blocks", processed)

    INGEST_CYCLE_SECONDS.observe(time.monotonic() - t0)

    return {
        "dirty_blocks": len(dirty),
        "queued": queued,
        "processed": processed,
    }
