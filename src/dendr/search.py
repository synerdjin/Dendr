"""Search server — localhost:7777 exposing FTS5 + semantic search.

Claude Code and Obsidian plugins hit this for retrieval.
"""

from __future__ import annotations

import logging
import sqlite3
import time

from fastapi import FastAPI, Query, Response
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
from pydantic import BaseModel

from dendr import db
from dendr.config import Config
from dendr.llm import LLMClient
from dendr.metrics import SEARCH_REQUEST_SECONDS

logger = logging.getLogger(__name__)

app = FastAPI(title="Dendr Search", version="0.1.0")

# Module-level state (set by run_server)
_config: Config | None = None
_conn: sqlite3.Connection | None = None
_llm: LLMClient | None = None


class SearchResult(BaseModel):
    claim_id: int
    text: str
    concept_slug: str
    confidence: float
    status: str
    source_block_ref: str
    score_type: str  # "fts" or "semantic"


class SearchResponse(BaseModel):
    query: str
    results: list[SearchResult]
    total: int


class ConceptResult(BaseModel):
    slug: str
    title: str
    page_type: str
    page_path: str


@app.get("/metrics")
def metrics() -> Response:
    """Prometheus metrics endpoint."""
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/search", response_model=SearchResponse)
def search(
    q: str = Query(..., min_length=1, description="Search query"),
    mode: str = Query("hybrid", description="fts, semantic, or hybrid"),
    limit: int = Query(20, ge=1, le=100),
    include_private: bool = Query(False),
) -> SearchResponse:
    """Search claims via full-text, semantic, or hybrid."""
    assert _conn is not None and _llm is not None
    t0 = time.monotonic()

    results: list[SearchResult] = []
    seen_ids: set[int] = set()

    if mode in ("fts", "hybrid"):
        fts_rows = db.search_claims_fts(
            _conn, q, limit=limit, include_private=include_private
        )
        for row in fts_rows:
            if row["id"] not in seen_ids:
                seen_ids.add(row["id"])
                results.append(
                    SearchResult(
                        claim_id=row["id"],
                        text=row["text"],
                        concept_slug=row["concept_slug"],
                        confidence=row["confidence"],
                        status=row["status"],
                        source_block_ref=row["source_block_ref"],
                        score_type="fts",
                    )
                )

    if mode in ("semantic", "hybrid"):
        try:
            query_emb = _llm.embed(q)
            sem_rows = db.search_claims_semantic(
                _conn, query_emb, limit=limit, include_private=include_private
            )
            for row in sem_rows:
                if row["id"] not in seen_ids:
                    seen_ids.add(row["id"])
                    results.append(
                        SearchResult(
                            claim_id=row["id"],
                            text=row["text"],
                            concept_slug=row["concept_slug"],
                            confidence=row["confidence"],
                            status=row["status"],
                            source_block_ref=row["source_block_ref"],
                            score_type="semantic",
                        )
                    )
        except Exception as e:
            logger.warning("Semantic search failed: %s", e)

    SEARCH_REQUEST_SECONDS.labels(mode=mode).observe(time.monotonic() - t0)
    return SearchResponse(query=q, results=results[:limit], total=len(results))


@app.get("/concepts", response_model=list[ConceptResult])
def list_concepts() -> list[ConceptResult]:
    """List all concepts."""
    assert _conn is not None
    rows = db.get_all_concepts(_conn)
    return [
        ConceptResult(
            slug=r["slug"],
            title=r["title"],
            page_type=r["page_type"],
            page_path=r["page_path"],
        )
        for r in rows
    ]


@app.get("/concept/{slug}/claims")
def concept_claims(slug: str, include_private: bool = False) -> list[SearchResult]:
    """Get all claims for a concept."""
    assert _conn is not None
    rows = db.get_claims_for_concept(_conn, slug, include_private=include_private)
    return [
        SearchResult(
            claim_id=r["id"],
            text=r["text"],
            concept_slug=r["concept_slug"],
            confidence=r["confidence"],
            status=r["status"],
            source_block_ref=r["source_block_ref"],
            score_type="direct",
        )
        for r in rows
    ]


@app.get("/stats")
def stats() -> dict:
    """Get knowledge base statistics."""
    assert _conn is not None
    return db.get_stats(_conn)


def run_server(config: Config) -> None:
    """Start the search server."""
    import uvicorn

    global _config, _conn, _llm
    _config = config
    _conn = db.connect(config.db_path)
    db.init_schema(_conn)
    _llm = LLMClient(config)

    logger.info("Starting search server on port %d", config.search_port)
    uvicorn.run(app, host="127.0.0.1", port=config.search_port, log_level="info")
