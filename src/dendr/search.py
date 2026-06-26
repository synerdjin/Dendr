"""Search server — localhost:7777 exposing FTS5 + semantic search over blocks."""

from __future__ import annotations

import logging
import sqlite3
import threading
import time
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, Query, Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from pydantic import BaseModel

from dendr import db
from dendr.config import Config
from dendr.llm import LLMClient
from dendr.metrics import SEARCH_REQUEST_SECONDS

logger = logging.getLogger(__name__)

app = FastAPI(title="Dendr Search", version="0.1.0")

_db_path: Path | None = None
_llm: LLMClient | None = None
_llm_lock = threading.Lock()

ScoreType = Literal["fts", "semantic", "hybrid"]


class BlockResult(BaseModel):
    block_id: str
    source_file: str
    source_date: str
    text: str
    checkbox_state: str
    completion_status: str | None = None
    score_type: ScoreType
    score: float | None = None


class SearchResponse(BaseModel):
    query: str
    results: list[BlockResult]
    total: int


def _row_to_result(
    row: sqlite3.Row, score_type: ScoreType, *, similarity: float | None = None
) -> BlockResult:
    score = round(similarity, 4) if similarity is not None else None
    return BlockResult(**db.block_row_to_dict(row), score_type=score_type, score=score)


def _get_conn() -> sqlite3.Connection:
    """Create a per-request connection (thread-safe with WAL mode)."""
    if _db_path is None:
        raise RuntimeError("Search server not initialized")
    return db.connect(_db_path)


@app.get("/metrics")
def metrics() -> Response:
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/search", response_model=SearchResponse)
def search(
    q: str = Query(..., min_length=1, description="Search query"),
    mode: str = Query("hybrid", description="fts, semantic, or hybrid"),
    limit: int = Query(20, ge=1, le=100),
    include_private: bool = Query(False),
    min_score: float = Query(
        0.25,
        ge=0.0,
        le=1.0,
        description="Minimum similarity score for semantic results (0-1)",
    ),
) -> SearchResponse:
    """Search blocks via full-text, semantic, or hybrid."""
    if _llm is None:
        raise RuntimeError("Search server not initialized")
    conn = _get_conn()
    try:
        t0 = time.monotonic()
        # Pull a deeper candidate pool than `limit` for hybrid so fusion has
        # material to rerank; pure modes fetch exactly what they return.
        pool = limit * 2 if mode == "hybrid" else limit

        fts_rows: list[sqlite3.Row] = []
        if mode in ("fts", "hybrid"):
            fts_rows = db.search_blocks_fts(
                conn, q, limit=pool, include_private=include_private
            )

        sem_pairs: list[tuple[sqlite3.Row, float]] = []
        if mode in ("semantic", "hybrid"):
            try:
                with _llm_lock:
                    query_emb = _llm.embed(q, kind="query")
                sem_pairs = db.search_blocks_semantic(
                    conn,
                    query_emb,
                    limit=pool,
                    include_private=include_private,
                    min_similarity=min_score,
                )
            except Exception as e:
                logger.warning("Semantic search failed: %s", e)

        results: list[BlockResult] = []
        if mode == "hybrid":
            for row, rrf_score, _sim in db.rrf_fuse(fts_rows, sem_pairs, limit):
                results.append(
                    BlockResult(
                        **db.block_row_to_dict(row),
                        score_type="hybrid",
                        score=round(rrf_score, 6),
                    )
                )
        elif mode == "semantic":
            for row, similarity in sem_pairs[:limit]:
                results.append(_row_to_result(row, "semantic", similarity=similarity))
        else:  # fts
            for row in fts_rows[:limit]:
                results.append(_row_to_result(row, "fts"))

        SEARCH_REQUEST_SECONDS.labels(mode=mode).observe(time.monotonic() - t0)
        return SearchResponse(query=q, results=results, total=len(results))
    finally:
        conn.close()


@app.get("/stats")
def stats() -> dict:
    conn = _get_conn()
    try:
        return db.get_stats(conn)
    finally:
        conn.close()


def run_server(config: Config, *, host: str = "127.0.0.1") -> None:
    import uvicorn

    global _db_path, _llm
    _db_path = config.db_path

    init_conn = db.connect(config.db_path)
    db.init_schema(init_conn)
    init_conn.close()

    _llm = LLMClient(config)

    logger.info("Starting search server on port %d", config.search_port)
    uvicorn.run(app, host=host, port=config.search_port, log_level="info")
