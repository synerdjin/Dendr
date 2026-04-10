"""Core ingestion pipeline — orchestrates the full block-to-wiki flow.

Annotation-first architecture:
  1. Parse daily notes, find dirty blocks
  2. Annotate blocks (tagger model — fast, always runs)
  3. Embed annotations for semantic search
  4. Extract claims (enrichment model — optional, skip in backpressure)
  5. Semantic claim dedup (replaces exact-match SPO)
  6. Wiki page update
"""

from __future__ import annotations

import logging
import re
import sqlite3
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from dendr import db, queue
from dendr.canonicalize import canonicalize_concepts
from dendr.config import Config
from dendr.enrichment import enrich_block
from dendr.llm import LLMClient
from dendr.metrics import (
    BACKPRESSURE_ACTIVE,
    BLOCKS_PROCESSED,
    CLAIMS_EXTRACTED,
    CLAIMS_REINFORCED,
    INGEST_CYCLE_SECONDS,
)
from dendr.models import (
    Block,
    BlockAnnotation,
    BlockType,
    Claim,
    ClaimStatus,
    PageType,
    QueueItem,
)
from dendr.parser import inject_block_ids, parse_daily_note
from dendr.privacy import filter_blocks
from dendr.wiki import (
    append_activity_log,
    append_evidence,
    ensure_page,
    update_index,
)

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
        emotional_labels=raw.get("emotional_labels", []),
        intensity=float(raw.get("intensity", 0.5)),
        urgency=raw.get("urgency"),
        importance=raw.get("importance"),
        completion_status=raw.get("completion_status"),
        epistemic_status=raw.get("epistemic_status", "certain"),
        causal_links=raw.get("causal_links", []),
        concepts=raw.get("concepts", []),
        entities=raw.get("entities", []),
        private=item.private,
        model_version=model_version,
        prompt_version=LLMClient.ANNOTATION_PROMPT_VERSION,
    )


def process_queue(config: Config, conn: sqlite3.Connection, llm: LLMClient) -> int:
    """Process all pending queue items through the annotation + enrichment pipeline.

    Three phases to minimize GPU model swaps:
      Phase 1 (tagger model):     annotate all blocks
      Phase 2 (embedding model):  embed + canonicalize concepts
      Phase 3 (enrichment model): extract claims + generate wiki sections
    """
    queue.recover_stale(config)
    pending = queue.get_pending(config)
    if not pending:
        return 0

    total_pending = len(pending)
    estimated_days = total_pending / 20
    shallow = estimated_days > config.backpressure_days
    BACKPRESSURE_ACTIVE.set(1 if shallow else 0)

    if shallow:
        logger.warning(
            "Backpressure: ~%.0f days queued (threshold: %d). Skipping claim extraction.",
            estimated_days,
            config.backpressure_days,
        )

    existing_concepts = [r["slug"] for r in db.get_all_concepts(conn)]

    # Claim all items for processing
    claimed: list[QueueItem] = []
    for item in pending:
        if queue.claim_for_processing(config, item.block_id):
            claimed.append(item)
    if not claimed:
        return 0

    # ── Phase 1: Annotation (tagger model) ────────────────────────────
    total = len(claimed)
    logger.info("Phase 1/3: annotating %d blocks", total)
    annotated: dict[str, tuple[QueueItem, BlockAnnotation]] = {}

    for idx, item in enumerate(claimed, 1):
        try:
            logger.info(
                "Phase 1/3: annotating block %d/%d: %s", idx, total, item.block_id
            )
            raw_ann = llm.annotate_block(item.block_text)
            annotation = _build_annotation(
                item, raw_ann, llm.config.models.tagger_model
            )
            annotated[item.block_id] = (item, annotation)
        except Exception as e:
            logger.error("Failed to annotate block %s: %s", item.block_id, e)

    # ── Phase 2: Embedding + canonicalization ─────────────────────────
    total2 = len(annotated)
    logger.info("Phase 2/3: embedding & canonicalizing %d blocks", total2)

    phase2: dict[str, tuple[QueueItem, BlockAnnotation, dict[str, str]]] = {}
    enriched: dict[str, Any] = {}

    for idx2, (block_id, (item, annotation)) in enumerate(annotated.items(), 1):
        try:
            logger.info("Phase 2/3: embedding block %d/%d: %s", idx2, total2, block_id)
            slug_map = canonicalize_concepts(annotation.concepts, llm, conn, config)
            phase2[block_id] = (item, annotation, slug_map)

            # Also run enrichment if not in backpressure mode
            if not shallow:
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
                result = enrich_block(block, llm, existing_concepts)
                # Pre-embed claim texts for semantic dedup
                claim_embeddings = {}
                for i, claim_data in enumerate(result.claims):
                    try:
                        claim_embeddings[i] = llm.embed(claim_data.text)
                    except Exception as e:
                        logger.warning("Failed to embed claim: %s", e)
                enriched[block_id] = (result, slug_map, claim_embeddings)
        except Exception as e:
            logger.error("Failed in phase 2 for block %s: %s", block_id, e)

    # ── Phase 3: Commit annotations + claims + wiki ───────────────────
    total3 = len(phase2)
    logger.info("Phase 3/3: committing %d blocks", total3)
    processed = 0

    for block_id, (item, annotation, slug_map) in phase2.items():
        try:
            conn.execute("BEGIN")
            try:
                # Store annotation
                db.upsert_block_annotation(conn, annotation)

                source_ref = Path(item.source_file).stem

                # Process claims (if enrichment ran)
                if block_id in enriched:
                    result, slug_map, claim_embeddings = enriched[block_id]

                    for i, claim_data in enumerate(result.claims):
                        concept_slug = ""
                        for c in claim_data.concepts:
                            if c in slug_map:
                                concept_slug = slug_map[c]
                                break
                        if not concept_slug and slug_map:
                            concept_slug = next(iter(slug_map.values()))

                        # Semantic dedup: check if similar claim exists
                        if i in claim_embeddings:
                            existing = db.find_similar_claim_semantic(
                                conn, claim_embeddings[i], similarity_threshold=0.92
                            )
                            if existing:
                                db.reinforce_claim(conn, existing["id"])
                                CLAIMS_REINFORCED.inc()
                                continue

                        claim = Claim(
                            id=None,
                            text=claim_data.text,
                            concept_slug=concept_slug,
                            source_block_ref=item.block_id,
                            source_file_hash=item.block_hash,
                            created_at=datetime.now(),
                            updated_at=datetime.now(),
                            confidence=claim_data.confidence,
                            status=ClaimStatus.CREATED,
                            kind=claim_data.kind,
                            private=item.private,
                            model_version=result.model_version,
                            prompt_version=result.prompt_version,
                        )
                        new_id = db.insert_claim(conn, claim)
                        CLAIMS_EXTRACTED.inc()

                        if i in claim_embeddings:
                            db.insert_claim_embedding(conn, new_id, claim_embeddings[i])

                # Update wiki pages for concepts
                for candidate, slug in slug_map.items():
                    title = candidate.replace("-", " ").title()
                    evidence = []
                    if block_id in enriched:
                        result_data = enriched[block_id][0]
                        evidence = [
                            c.text
                            for c in result_data.claims
                            if any(
                                cc in slug_map and slug_map[cc] == slug
                                for cc in c.concepts
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

    existing_concepts = [r["slug"] for r in db.get_all_concepts(conn)]

    if processed > 0:
        update_index(config, conn)
        append_activity_log(
            config,
            f"INGEST processed {processed} blocks, {len(existing_concepts)} concepts",
        )

    return processed


def run_ingest(config: Config, conn: sqlite3.Connection, llm: LLMClient) -> dict:
    """Full ingest cycle: scan -> queue -> process."""
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
