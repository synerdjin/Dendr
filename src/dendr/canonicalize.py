"""Concept canonicalization — prevent slug drift via embedding-based dedup.

Before creating any new concept slug, we embed the candidate and
ANN-search existing slugs. If cosine similarity >= threshold, reuse
the existing slug.
"""

from __future__ import annotations

import logging
import re
import sqlite3

from dendr.config import Config
from dendr.db import find_nearest_concept, insert_concept_embedding
from dendr.llm import LLMClient
from dendr.metrics import CANONICALIZATION_NEW, CANONICALIZATION_REUSE

logger = logging.getLogger(__name__)


def _slugify(text: str) -> str:
    """Convert text to a valid slug."""
    slug = text.lower().strip()
    slug = re.sub(r"[^a-z0-9\s-]", "", slug)
    slug = re.sub(r"[\s_]+", "-", slug)
    slug = re.sub(r"-+", "-", slug)
    return slug.strip("-")


def _cosine_distance_to_similarity(distance: float) -> float:
    """sqlite-vec returns cosine distance; convert to similarity."""
    return 1.0 - distance


def canonicalize_concept(
    candidate: str,
    llm: LLMClient,
    conn: sqlite3.Connection,
    config: Config,
) -> str:
    """Resolve a candidate concept name to a canonical slug.

    Returns an existing slug if a near-match exists (cosine >= threshold),
    otherwise creates and returns a new slug.
    """
    slug = _slugify(candidate)
    if not slug:
        return ""

    # Check exact match first (cheap)
    existing = conn.execute(
        "SELECT slug FROM concepts WHERE slug = ?", (slug,)
    ).fetchone()
    if existing:
        return existing["slug"]

    # Embed the candidate
    embedding = llm.embed(candidate)

    # ANN search existing concepts
    nearest = find_nearest_concept(conn, embedding, top_k=3)

    for existing_slug, distance in nearest:
        similarity = _cosine_distance_to_similarity(distance)
        if similarity >= config.canonicalization_threshold:
            logger.info(
                "Canonicalized '%s' → '%s' (similarity=%.3f)",
                candidate,
                existing_slug,
                similarity,
            )
            CANONICALIZATION_REUSE.inc()
            return existing_slug

    # No match — this is a genuinely new concept. Store its embedding.
    insert_concept_embedding(conn, slug, embedding)
    logger.info("New concept slug: '%s'", slug)
    CANONICALIZATION_NEW.inc()
    return slug


def canonicalize_concepts(
    candidates: list[str],
    llm: LLMClient,
    conn: sqlite3.Connection,
    config: Config,
) -> dict[str, str]:
    """Canonicalize a batch of concept candidates.

    Returns a mapping of {original_candidate: canonical_slug}.
    """
    result: dict[str, str] = {}
    for candidate in candidates:
        if not candidate.strip():
            continue
        slug = canonicalize_concept(candidate, llm, conn, config)
        if slug:
            result[candidate] = slug
    return result
