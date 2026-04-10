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
from typing import Any

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
from dendr.models import Block, Claim, ClaimKind, ClaimStatus, PageType, QueueItem
from dendr.parser import inject_block_ids, parse_daily_note
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

    Batches work into three phases to minimize GPU model swaps:
      Phase 1 (enrichment model): extract claims from all blocks
      Phase 2 (embedding model):  canonicalize concepts + embed claims
      Phase 3 (enrichment model): generate wiki sections + commit

    Returns count of processed items.
    """
    # Recover any items stuck from a prior crash
    queue.recover_stale(config)

    pending = queue.get_pending(config)
    if not pending:
        return 0

    # Check backpressure
    total_pending = len(pending)
    estimated_days = total_pending / 20
    shallow = estimated_days > config.backpressure_days

    BACKPRESSURE_ACTIVE.set(1 if shallow else 0)

    if shallow:
        logger.warning(
            "Backpressure: ~%.0f days queued (threshold: %d). Using shallow enrichment.",
            estimated_days,
            config.backpressure_days,
        )

    existing_concepts = [r["slug"] for r in db.get_all_concepts(conn)]

    # Claim all items for processing upfront
    claimed: list[QueueItem] = []
    for item in pending:
        if queue.claim_for_processing(config, item.block_id):
            claimed.append(item)

    if not claimed:
        return 0

    # ── Phase 1: Enrichment (enrichment model stays loaded) ──────────
    total = len(claimed)
    logger.info("Phase 1/3: enriching %d blocks", total)
    enriched: dict[str, tuple[QueueItem, Block, Any]] = {}
    for idx, item in enumerate(claimed, 1):
        try:
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
            logger.info(
                "Phase 1/3: enriching block %d/%d: %s", idx, total, item.block_id
            )
            result = enrich_block(block, llm, existing_concepts, shallow=shallow)
            enriched[item.block_id] = (item, block, result)
        except Exception as e:
            logger.error("Failed to enrich block %s: %s", item.block_id, e)

    # ── Phase 2: Canonicalization & embedding (embedding model stays loaded) ─
    total2 = len(enriched)
    logger.info("Phase 2/3: canonicalizing & embedding %d blocks", total2)
    # Per-block: slug_map and pre-computed claim embeddings
    phase2: dict[str, tuple[QueueItem, Block, Any, dict[str, str], dict[int, Any]]] = {}
    for idx2, (block_id, (item, block, result)) in enumerate(enriched.items(), 1):
        try:
            logger.info("Phase 2/3: embedding block %d/%d: %s", idx2, total2, block_id)
            slug_map = canonicalize_concepts(result.concepts, llm, conn, config)

            claim_embeddings: dict[int, Any] = {}
            for i, claim_data in enumerate(result.claims):
                try:
                    claim_embeddings[i] = llm.embed(claim_data.text)
                except Exception as e:
                    logger.warning("Failed to embed claim text: %s", e)

            phase2[block_id] = (item, block, result, slug_map, claim_embeddings)
        except Exception as e:
            logger.error("Failed to canonicalize block %s: %s", block_id, e)

    # ── Phase 3: Wiki generation & DB commits (enrichment model stays loaded) ─
    total3 = len(phase2)
    logger.info("Phase 3/3: writing wiki pages & committing %d blocks", total3)
    processed = 0
    for block_id, (item, block, result, slug_map, claim_embeddings) in phase2.items():
        try:
            logger.info(
                "Phase 3/3: committing block %d/%d: %s", processed + 1, total3, block_id
            )
            conn.execute("BEGIN")
            try:
                source_ref = Path(item.source_file).stem
                for i, claim_data in enumerate(result.claims):
                    concept_slug = ""
                    for c in claim_data.concepts:
                        if c in slug_map:
                            concept_slug = slug_map[c]
                            break
                    if not concept_slug and slug_map:
                        concept_slug = next(iter(slug_map.values()))

                    sp_key = f"{claim_data.subject}|{claim_data.predicate}"

                    existing = db.find_similar_claim(conn, sp_key, claim_data.object)
                    if existing:
                        db.reinforce_claim(conn, existing["id"])
                        CLAIMS_REINFORCED.inc()
                        continue

                    contradictions = db.find_contradictions(
                        conn, sp_key, claim_data.object
                    )

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
                        kind=claim_data.kind,
                        private=block.private,
                        model_version=result.model_version,
                        prompt_version=result.prompt_version,
                    )
                    new_id = db.insert_claim(conn, claim)
                    CLAIMS_EXTRACTED.inc()

                    if i in claim_embeddings:
                        db.insert_claim_embedding(conn, new_id, claim_embeddings[i])

                    for contra in contradictions:
                        db.challenge_claim(conn, contra["id"])
                        CONTRADICTIONS_DETECTED.inc()
                        db.append_log(
                            conn,
                            "contradiction",
                            {
                                "new_claim_id": new_id,
                                "challenged_id": contra["id"],
                                "subject_predicate": sp_key,
                            },
                        )

                for candidate, slug in slug_map.items():
                    title = candidate.replace("-", " ").title()
                    evidence = [
                        c.text
                        for c in result.claims
                        if any(
                            cc in slug_map and slug_map[cc] == slug for cc in c.concepts
                        )
                    ]
                    if not evidence:
                        evidence = [f"Referenced in {source_ref}"]

                    ensure_page(config, conn, slug, title, PageType.CONCEPT)
                    append_evidence(
                        config,
                        conn,
                        llm,
                        slug,
                        title,
                        evidence,
                        source_ref,
                        PageType.CONCEPT,
                    )

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

            queue.mark_done(config, item.block_id)
            processed += 1
            BLOCKS_PROCESSED.inc()

        except Exception as e:
            logger.error("Failed to process block %s: %s", item.block_id, e)
            continue

    # Update existing concepts list once at end
    existing_concepts = [r["slug"] for r in db.get_all_concepts(conn)]

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
