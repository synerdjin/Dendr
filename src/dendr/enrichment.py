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

    if not text.strip():
        return result

    if shallow:
        tags = llm.tag_block(text)
        result.concepts = tags.get("concepts", [])
        result.entities = tags.get("entities", [])
        result.model_version = llm.config.models.tagger_model
        return result

    # Full enrichment (simplified — no SPO)
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
                confidence=float(claim_data.get("confidence", 0.5)),
                kind=kind,
                concepts=claim_data.get("concepts", []),
            )
            result.claims.append(claim)
        except (KeyError, TypeError, ValueError) as e:
            logger.warning("Skipping malformed claim: %s", e)

    result.concepts = raw.get("concepts", [])
    result.entities = raw.get("entities", [])
    result.related_slugs = raw.get("related_slugs", [])

    return result
