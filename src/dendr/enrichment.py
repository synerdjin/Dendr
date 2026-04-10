"""Enrichment pipeline — transforms raw blocks into structured claims and concepts."""

from __future__ import annotations

import logging

from dendr.llm import LLMClient
from dendr.models import Block, ClaimKind, EnrichmentResult, ExtractedClaim

logger = logging.getLogger(__name__)


def enrich_block(
    block: Block,
    llm: LLMClient,
    existing_concepts: list[str],
    shallow: bool = False,
) -> EnrichmentResult:
    """Run the enrichment pipeline on a single block.

    If shallow=True (backpressure mode), only tag — skip full claim extraction.
    """
    result = EnrichmentResult(
        block_id=block.block_id,
        block_hash=block.block_hash,
        source_file=block.source_file,
        model_version=llm.config.models.enrichment_model,
        prompt_version=LLMClient.ENRICHMENT_PROMPT_VERSION,
    )

    text = block.text

    # Handle attachments: extract text first
    if block.is_attachment_ref and block.attachment_path:
        if block.attachment_type == "pdf":
            extracted = llm.extract_text_from_pdf(block.attachment_path)
            text = f"{block.text}\n\n[Extracted from PDF]\n{extracted}"
        elif block.attachment_type == "image":
            extracted = llm.extract_text_from_image(block.attachment_path)
            text = f"{block.text}\n\n[Image description]\n{extracted}"
        # audio is v2

    if not text.strip():
        return result

    if shallow:
        # Backpressure: quick tag only
        tags = llm.tag_block(text)
        result.concepts = tags.get("concepts", [])
        result.entities = tags.get("entities", [])
        result.model_version = llm.config.models.tagger_model
        return result

    # Full enrichment
    raw = llm.enrich_block(text, existing_concepts)

    for claim_data in raw.get("claims", []):
        try:
            raw_kind = claim_data.get("kind", "statement")
            try:
                kind = ClaimKind(raw_kind)
            except ValueError:
                kind = ClaimKind.STATEMENT
            claim = ExtractedClaim(
                text=claim_data["text"],
                subject=claim_data.get("subject", ""),
                predicate=claim_data.get("predicate", ""),
                object=claim_data.get("object", ""),
                confidence=float(claim_data.get("confidence", 0.5)),
                concepts=claim_data.get("concepts", []),
                entities=claim_data.get("entities", []),
                kind=kind,
            )
            result.claims.append(claim)
        except (KeyError, TypeError, ValueError) as e:
            logger.warning("Skipping malformed claim: %s", e)

    result.concepts = raw.get("concepts", [])
    result.entities = raw.get("entities", [])
    result.related_slugs = raw.get("related_slugs", [])

    return result


def enrich_blocks(
    blocks: list[Block],
    llm: LLMClient,
    existing_concepts: list[str],
    backpressure_threshold: int = 0,
) -> list[EnrichmentResult]:
    """Enrich a batch of blocks.

    If len(blocks) > backpressure_threshold > 0, switch to shallow mode.
    """
    shallow = backpressure_threshold > 0 and len(blocks) > backpressure_threshold
    if shallow:
        logger.info(
            "Backpressure: %d blocks exceed threshold %d, using shallow enrichment",
            len(blocks),
            backpressure_threshold,
        )

    results: list[EnrichmentResult] = []
    for i, block in enumerate(blocks):
        if block.private:
            logger.debug("Skipping private block %s for enrichment", block.block_id)
            # Still tag locally but mark result as private-sourced
        logger.info("Enriching block %d/%d: %s", i + 1, len(blocks), block.block_id)
        result = enrich_block(block, llm, existing_concepts, shallow=shallow)
        results.append(result)

    return results
