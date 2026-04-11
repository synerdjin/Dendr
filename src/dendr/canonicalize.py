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
    page_type: str = "concept",
) -> str:
    """Resolve a candidate name to a canonical slug for the given page_type.

    Returns an existing slug if a near-match exists (cosine >= threshold),
    otherwise creates and returns a new slug. Concepts and entities are
    canonicalized in separate namespaces — a concept and entity that
    happen to slugify identically would collide on the slug PK, so on
    cross-type collision we log and skip rather than corrupt the row.
    """
    slug = _slugify(candidate)
    if not slug:
        return ""

    # Exact match in the same page_type namespace
    existing = conn.execute(
        "SELECT slug, page_type FROM concepts WHERE slug = ?", (slug,)
    ).fetchone()
    if existing:
        if existing["page_type"] == page_type:
            return existing["slug"]
        # Cross-type collision: the slug is already in use by a different
        # page_type. Skip rather than overwrite the existing row.
        logger.warning(
            "Slug '%s' already exists as %s; skipping %s candidate '%s'",
            slug,
            existing["page_type"],
            page_type,
            candidate,
        )
        return ""

    # Embed the candidate
    embedding = llm.embed(candidate)

    # ANN search within the same page_type
    nearest = find_nearest_concept(conn, embedding, top_k=3, page_type=page_type)

    for existing_slug, distance in nearest:
        similarity = _cosine_distance_to_similarity(distance)
        if similarity >= config.canonicalization_threshold:
            logger.info(
                "Canonicalized '%s' → '%s' (similarity=%.3f, %s)",
                candidate,
                existing_slug,
                similarity,
                page_type,
            )
            CANONICALIZATION_REUSE.inc()
            return existing_slug

    # No match — this is a genuinely new slug. Store its embedding.
    insert_concept_embedding(conn, slug, embedding)
    logger.info("New %s slug: '%s'", page_type, slug)
    CANONICALIZATION_NEW.inc()
    return slug


def canonicalize_concepts(
    candidates: list[str],
    llm: LLMClient,
    conn: sqlite3.Connection,
    config: Config,
    page_type: str = "concept",
) -> dict[str, str]:
    """Canonicalize a batch of candidates.

    Returns a mapping of {original_candidate: canonical_slug}.
    """
    result: dict[str, str] = {}
    for candidate in candidates:
        if not candidate.strip():
            continue
        slug = canonicalize_concept(candidate, llm, conn, config, page_type=page_type)
        if slug:
            result[candidate] = slug
    return result
