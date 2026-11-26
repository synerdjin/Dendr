"""MCP server — expose Dendr search over stdio for remote/desktop Claude clients.

The iCloud-synced vault makes `digest.md` readable on any device, but the
SQLite index is local-only and the FastAPI search server binds localhost. This
module wraps the same search code paths as `search.py` as Model Context
Protocol tools so a Claude Desktop / Claude Code client can query the knowledge
base directly.

The query logic lives in `_search_blocks` / `_stats` (plain functions, no `mcp`
dependency) so it is unit-testable without the optional `mcp` package. The
stdio wiring in `run_mcp_server` imports `mcp` lazily.
"""

from __future__ import annotations

import logging
import sqlite3
import threading

from dendr import db
from dendr.config import Config

logger = logging.getLogger(__name__)


def _search_blocks(
    conn: sqlite3.Connection,
    llm,
    q: str,
    *,
    mode: str = "hybrid",
    limit: int = 20,
    min_score: float = 0.25,
    lock: threading.Lock | None = None,
) -> list[dict]:
    """Run FTS / semantic / hybrid search and return plain dicts.

    Mirrors the `/search` endpoint in search.py: hybrid fuses the two rankings
    with Reciprocal Rank Fusion. `llm` may be None when only `mode="fts"` is
    needed (semantic results are then skipped gracefully).
    """
    pool = limit * 2 if mode == "hybrid" else limit

    fts_rows: list[sqlite3.Row] = []
    if mode in ("fts", "hybrid"):
        fts_rows = db.search_blocks_fts(conn, q, limit=pool)

    sem_pairs: list[tuple[sqlite3.Row, float]] = []
    if mode in ("semantic", "hybrid") and llm is not None:
        try:
            if lock is not None:
                with lock:
                    query_emb = llm.embed(q, kind="query")
            else:
                query_emb = llm.embed(q, kind="query")
            sem_pairs = db.search_blocks_semantic(
                conn, query_emb, limit=pool, min_similarity=min_score
            )
        except Exception as e:  # noqa: BLE001 — semantic is best-effort
            logger.warning("Semantic search failed: %s", e)

    results: list[dict] = []
    if mode == "hybrid":
        for row, rrf_score, _sim in db.rrf_fuse(fts_rows, sem_pairs, limit):
            results.append(
                {
                    **db.block_row_to_dict(row),
                    "score_type": "hybrid",
                    "score": round(rrf_score, 6),
                }
            )
    elif mode == "semantic":
        for row, similarity in sem_pairs[:limit]:
            results.append(
                {
                    **db.block_row_to_dict(row),
                    "score_type": "semantic",
                    "score": round(similarity, 4),
                }
            )
    else:  # fts
        for row in fts_rows[:limit]:
            results.append(
                {**db.block_row_to_dict(row), "score_type": "fts", "score": None}
            )

    return results


def _stats(conn: sqlite3.Connection) -> dict:
    """Knowledge-base stats — wraps db.get_stats for the `stats` tool."""
    return db.get_stats(conn)


def run_mcp_server(config: Config) -> None:
    """Boot the stdio MCP server exposing `search` and `stats` tools.

    Requires the optional `mcp` package (`pip install dendr[mcp]`).
    """
    try:
        from mcp.server.fastmcp import FastMCP
    except ImportError as e:  # pragma: no cover - import guard
        raise RuntimeError(
            "The MCP server needs the 'mcp' package. Install with: "
            "pip install 'dendr[mcp]'"
        ) from e

    from dendr.llm import LLMClient

    init_conn = db.connect(config.db_path)
    db.init_schema(init_conn)
    init_conn.close()

    llm = LLMClient(config)
    llm_lock = threading.Lock()
    db_path = config.db_path

    mcp = FastMCP("dendr")

    @mcp.tool()
    def search(
        q: str, mode: str = "hybrid", limit: int = 20, min_score: float = 0.25
    ) -> list[dict]:
        """Search the user's Dendr knowledge base of daily-note blocks.

        mode: 'fts' (keyword), 'semantic' (embeddings), or 'hybrid' (both,
        fused with Reciprocal Rank Fusion). Returns raw blocks with
        source_date, text, checkbox_state, completion_status, and a score
        (RRF for hybrid, cosine similarity for semantic).
        """
        conn = db.connect(db_path)
        try:
            return _search_blocks(
                conn,
                llm,
                q,
                mode=mode,
                limit=limit,
                min_score=min_score,
                lock=llm_lock,
            )
        finally:
            conn.close()

    @mcp.tool()
    def stats() -> dict:
        """Return knowledge-base statistics (total blocks, open tasks)."""
        conn = db.connect(db_path)
        try:
            return _stats(conn)
        finally:
            conn.close()

    logger.info("Starting Dendr MCP server (stdio)")
    mcp.run()
