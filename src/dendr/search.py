"""Search server — localhost:7777 exposing FTS5 + semantic search over annotations."""

from __future__ import annotations

import json
import logging
import sqlite3
import time

from fastapi import FastAPI, Query, Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from pydantic import BaseModel

from dendr import db
from dendr.config import Config
from dendr.llm import LLMClient
from dendr.metrics import SEARCH_REQUEST_SECONDS

logger = logging.getLogger(__name__)

app = FastAPI(title="Dendr Search", version="0.1.0")

_config: Config | None = None
_conn: sqlite3.Connection | None = None
_llm: LLMClient | None = None


class AnnotationResult(BaseModel):
    block_id: str
    source_file: str
    source_date: str
    gist: str
    block_type: str
    life_areas: list[str]
    concepts: list[str]
    entities: list[str]
    urgency: str | None = None
    importance: str | None = None
    completion_status: str | None = None
    score_type: str  # "fts" or "semantic"


class SearchResponse(BaseModel):
    query: str
    results: list[AnnotationResult]
    total: int


def _row_to_result(row: sqlite3.Row, score_type: str) -> AnnotationResult:
    return AnnotationResult(
        block_id=row["block_id"],
        source_file=row["source_file"],
        source_date=row["source_date"],
        gist=row["gist"],
        block_type=row["block_type"],
        life_areas=json.loads(row["life_areas"] or "[]"),
        concepts=json.loads(row["concepts"] or "[]"),
        entities=json.loads(row["entities"] or "[]"),
        urgency=row["urgency"],
        importance=row["importance"],
        completion_status=row["completion_status"],
        score_type=score_type,
    )


@app.get("/metrics")
def metrics() -> Response:
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/search", response_model=SearchResponse)
def search(
    q: str = Query(..., min_length=1, description="Search query"),
    mode: str = Query("hybrid", description="fts, semantic, or hybrid"),
    limit: int = Query(20, ge=1, le=100),
    include_private: bool = Query(False),
) -> SearchResponse:
    """Search block annotations via full-text, semantic, or hybrid."""
    assert _conn is not None and _llm is not None
    t0 = time.monotonic()

    results: list[AnnotationResult] = []
    seen_ids: set[str] = set()

    if mode in ("fts", "hybrid"):
        fts_rows = db.search_annotations_fts(
            _conn, q, limit=limit, include_private=include_private
        )
        for row in fts_rows:
            if row["block_id"] in seen_ids:
                continue
            seen_ids.add(row["block_id"])
            results.append(_row_to_result(row, "fts"))

    if mode in ("semantic", "hybrid"):
        try:
            query_emb = _llm.embed(q)
            sem_rows = db.search_annotations_semantic(
                _conn, query_emb, limit=limit, include_private=include_private
            )
            for row in sem_rows:
                if row["block_id"] in seen_ids:
                    continue
                seen_ids.add(row["block_id"])
                results.append(_row_to_result(row, "semantic"))
        except Exception as e:
            logger.warning("Semantic search failed: %s", e)

    SEARCH_REQUEST_SECONDS.labels(mode=mode).observe(time.monotonic() - t0)
    return SearchResponse(query=q, results=results[:limit], total=len(results))


@app.get("/stats")
def stats() -> dict:
    assert _conn is not None
    return db.get_stats(_conn)


def run_server(config: Config) -> None:
    import uvicorn

    global _config, _conn, _llm
    _config = config
    _conn = db.connect(config.db_path)
    db.init_schema(_conn)
    _llm = LLMClient(config)

    logger.info("Starting search server on port %d", config.search_port)
    uvicorn.run(app, host="127.0.0.1", port=config.search_port, log_level="info")
