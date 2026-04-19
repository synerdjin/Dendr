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

ScoreType = Literal["fts", "semantic"]


class BlockResult(BaseModel):
    block_id: str
    source_file: str
    source_date: str
    text: str
    checkbox_state: str
    completion_status: str | None = None
    score_type: ScoreType
    score: float | None = None  # 0-1 similarity (semantic only; 1 = identical)


class SearchResponse(BaseModel):
    query: str
    results: list[BlockResult]
    total: int


def _cosine_similarity(distance: float) -> float:
    """Convert cosine distance [0,2] to similarity [0,1] where 1 = identical."""
    return 1.0 - distance / 2.0


def _row_to_result(
    row: sqlite3.Row, score_type: ScoreType, *, distance: float | None = None
) -> BlockResult:
    score = round(_cosine_similarity(distance), 4) if distance is not None else None
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

        results: list[BlockResult] = []
        seen_ids: set[str] = set()

        if mode in ("fts", "hybrid"):
            fts_rows = db.search_blocks_fts(
                conn, q, limit=limit, include_private=include_private
            )
            for row in fts_rows:
                if row["block_id"] in seen_ids:
                    continue
                seen_ids.add(row["block_id"])
                results.append(_row_to_result(row, "fts"))

        if mode in ("semantic", "hybrid"):
            try:
                with _llm_lock:
                    query_emb = _llm.embed(q)
                # similarity = 1 - dist/2, so dist = 2*(1 - similarity)
                max_distance = 2.0 * (1.0 - min_score)
                sem_pairs = db.search_blocks_semantic(
                    conn,
                    query_emb,
                    limit=limit,
                    include_private=include_private,
                    max_distance=max_distance,
                )
                for row, distance in sem_pairs:
                    if row["block_id"] in seen_ids:
                        continue
                    seen_ids.add(row["block_id"])
                    results.append(_row_to_result(row, "semantic", distance=distance))
            except Exception as e:
                logger.warning("Semantic search failed: %s", e)

        SEARCH_REQUEST_SECONDS.labels(mode=mode).observe(time.monotonic() - t0)
        return SearchResponse(query=q, results=results[:limit], total=len(results))
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
