"""Tests for the MCP server search logic (no `mcp` dependency required)."""

import tempfile
from pathlib import Path

from dendr.db import connect, init_schema, upsert_block
from dendr.mcp_server import _search_blocks, _stats
from dendr.models import CHECKBOX_NONE, CHECKBOX_OPEN, Block


def _temp_db():
    f = tempfile.NamedTemporaryFile(suffix=".sqlite", delete=False)
    f.close()
    conn = connect(Path(f.name))
    init_schema(conn)
    return conn


def _block(block_id: str, text: str, *, checkbox=CHECKBOX_NONE):
    return Block(
        block_id=block_id,
        source_file="Daily/x.md",
        line_start=0,
        line_end=0,
        text=text,
        block_hash=block_id,
        checkbox_state=checkbox,
    )


def test_search_blocks_fts_returns_matches():
    conn = _temp_db()
    upsert_block(conn, _block("b1", "thinking about coffee and sleep"), "2026-01-01")
    upsert_block(conn, _block("b2", "unrelated note about taxes"), "2026-01-01")

    results = _search_blocks(conn, None, "coffee", mode="fts")
    ids = [r["block_id"] for r in results]
    assert ids == ["b1"]
    assert results[0]["score_type"] == "fts"
    assert results[0]["score"] is None
    assert results[0]["text"] == "thinking about coffee and sleep"


def test_search_blocks_semantic_skipped_without_llm():
    """mode=hybrid with llm=None still returns FTS hits and no crash."""
    conn = _temp_db()
    upsert_block(conn, _block("b1", "coffee musings"), "2026-01-01")
    results = _search_blocks(conn, None, "coffee", mode="hybrid")
    assert [r["block_id"] for r in results] == ["b1"]
    assert results[0]["score_type"] == "hybrid"


def test_stats_wraps_get_stats():
    conn = _temp_db()
    upsert_block(
        conn, _block("t1", "[ ] open task", checkbox=CHECKBOX_OPEN), "2026-01-01"
    )
    s = _stats(conn)
    assert s["blocks"] == 1
    assert s["open_tasks"] == 1
