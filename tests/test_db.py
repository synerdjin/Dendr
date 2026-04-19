"""Tests for the database layer."""

import tempfile
from pathlib import Path

import numpy as np

from dendr.db import (
    connect,
    get_block,
    get_block_hashes,
    get_open_tasks,
    get_section_effectiveness,
    get_stats,
    init_schema,
    insert_task_event,
    search_blocks_fts,
    search_blocks_semantic,
    update_completion_status,
    upsert_block,
    upsert_block_embedding,
    upsert_feedback_score,
)
from dendr.models import CHECKBOX_CLOSED, CHECKBOX_NONE, CHECKBOX_OPEN, Block


def _temp_db():
    f = tempfile.NamedTemporaryFile(suffix=".sqlite", delete=False)
    f.close()
    conn = connect(Path(f.name))
    init_schema(conn)
    return conn


def _make_block(**kwargs) -> Block:
    defaults = dict(
        block_id="dendr-test-1",
        source_file="Daily/2026-04-08.md",
        line_start=0,
        line_end=0,
        text="Test block content",
        block_hash="abc123",
        checkbox_state=CHECKBOX_NONE,
    )
    defaults.update(kwargs)
    return Block(**defaults)


def test_stats_empty():
    conn = _temp_db()
    s = get_stats(conn)
    assert s["blocks"] == 0
    assert s["open_tasks"] == 0


def test_upsert_block_roundtrip():
    conn = _temp_db()
    block = _make_block()
    upsert_block(conn, block, "2026-04-08")

    retrieved = get_block(conn, "dendr-test-1")
    assert retrieved is not None
    assert retrieved["text"] == "Test block content"
    assert retrieved["source_date"] == "2026-04-08"
    assert retrieved["checkbox_state"] == CHECKBOX_NONE


def test_upsert_block_update_changes_text_and_hash():
    conn = _temp_db()
    upsert_block(conn, _make_block(text="first"), "2026-04-08")
    upsert_block(
        conn,
        _make_block(text="second", block_hash="def456"),
        "2026-04-08",
    )

    retrieved = get_block(conn, "dendr-test-1")
    assert retrieved["text"] == "second"
    assert retrieved["block_hash"] == "def456"


def test_upsert_preserves_completion_status():
    """User-set completion_status must survive a source-file re-ingest."""
    conn = _temp_db()
    upsert_block(
        conn,
        _make_block(checkbox_state=CHECKBOX_OPEN),
        "2026-04-08",
    )
    update_completion_status(conn, "dendr-test-1", "done")

    # Re-ingest: same block_id, same checkbox, different hash (edited text)
    upsert_block(
        conn,
        _make_block(
            checkbox_state=CHECKBOX_OPEN,
            text="Test block edited",
            block_hash="xyz789",
        ),
        "2026-04-08",
    )

    retrieved = get_block(conn, "dendr-test-1")
    assert retrieved["completion_status"] == "done"
    assert retrieved["text"] == "Test block edited"


def test_get_block_hashes():
    conn = _temp_db()
    assert get_block_hashes(conn, []) == {}
    assert get_block_hashes(conn, ["missing"]) == {}
    upsert_block(conn, _make_block(block_id="a", block_hash="h1"), "2026-04-08")
    upsert_block(conn, _make_block(block_id="b", block_hash="h2"), "2026-04-08")
    assert get_block_hashes(conn, ["a", "b", "missing"]) == {"a": "h1", "b": "h2"}


def test_stats_counts_open_tasks_only():
    conn = _temp_db()
    upsert_block(
        conn,
        _make_block(block_id="t1", checkbox_state=CHECKBOX_OPEN),
        "2026-04-08",
    )
    upsert_block(
        conn,
        _make_block(block_id="t2", checkbox_state=CHECKBOX_CLOSED),
        "2026-04-08",
    )
    upsert_block(
        conn,
        _make_block(block_id="t3", checkbox_state=CHECKBOX_NONE),
        "2026-04-08",
    )
    s = get_stats(conn)
    assert s["blocks"] == 3
    assert s["open_tasks"] == 1


def test_get_open_tasks_excludes_user_closed():
    conn = _temp_db()
    upsert_block(
        conn, _make_block(block_id="t1", checkbox_state=CHECKBOX_OPEN), "2026-04-08"
    )
    upsert_block(
        conn, _make_block(block_id="t2", checkbox_state=CHECKBOX_OPEN), "2026-04-08"
    )
    update_completion_status(conn, "t2", "abandoned")

    rows = get_open_tasks(conn)
    ids = [r["block_id"] for r in rows]
    assert ids == ["t1"]


def test_fts_finds_raw_text():
    conn = _temp_db()
    upsert_block(conn, _make_block(text="The quick brown fox jumps over"), "2026-04-08")
    rows = search_blocks_fts(conn, "brown fox")
    assert len(rows) == 1
    assert rows[0]["block_id"] == "dendr-test-1"


def test_fts_excludes_private_when_requested():
    conn = _temp_db()
    upsert_block(
        conn,
        _make_block(
            block_id="priv",
            text="sensitive content",
            private=True,
        ),
        "2026-04-08",
    )
    upsert_block(
        conn,
        _make_block(block_id="pub", text="sensitive content"),
        "2026-04-08",
    )
    rows = search_blocks_fts(conn, "sensitive", include_private=False)
    ids = [r["block_id"] for r in rows]
    assert "priv" not in ids
    assert "pub" in ids


def test_fts_no_stale_tokens_on_update():
    """Regression: external-content FTS5 must not retain old tokens after update."""
    conn = _temp_db()
    upsert_block(conn, _make_block(text="zebra elephant mongoose"), "2026-04-08")
    assert len(search_blocks_fts(conn, "zebra")) == 1

    upsert_block(
        conn,
        _make_block(text="turtle panda", block_hash="h2"),
        "2026-04-08",
    )
    assert search_blocks_fts(conn, "zebra") == []
    assert len(search_blocks_fts(conn, "panda")) == 1


# ── Feedback tests ────────────────────────────────────────────────────


def test_feedback_scores():
    conn = _temp_db()
    upsert_feedback_score(conn, "2026-04-03", "narrative", True, "good stuff")
    upsert_feedback_score(conn, "2026-04-03", "open-loops", False, "")

    scores = get_section_effectiveness(conn)
    assert scores["narrative"] == 1.0
    assert scores["open-loops"] == 0.0


# ── Task events ───────────────────────────────────────────────────────


def test_task_event_with_reason():
    conn = _temp_db()
    insert_task_event(conn, "t1", "created", "2026-04-01")
    insert_task_event(conn, "t1", "closed", "2026-04-05", reason="done", source="user")

    rows = conn.execute(
        "SELECT event_type, reason, source FROM task_events WHERE block_id = ? "
        "ORDER BY id",
        ("t1",),
    ).fetchall()
    assert rows[0]["event_type"] == "created"
    assert rows[0]["reason"] is None
    assert rows[0]["source"] == "auto"
    assert rows[1]["event_type"] == "closed"
    assert rows[1]["reason"] == "done"
    assert rows[1]["source"] == "user"


# ── Semantic search tests ────────────────────────────────────────────

DIM = 768


def _normalize(v: np.ndarray) -> np.ndarray:
    return v / np.linalg.norm(v)


def _insert_block_with_embedding(conn, block_id, text, embedding, source_date):
    block = _make_block(block_id=block_id, text=text, block_hash=block_id)
    upsert_block(conn, block, source_date)
    upsert_block_embedding(conn, block_id, embedding)
    conn.commit()


def test_semantic_search_returns_similarity():
    """Semantic results include similarity values in [0, 1]."""
    conn = _temp_db()
    emb = _normalize(np.random.default_rng(42).standard_normal(DIM).astype(np.float32))
    _insert_block_with_embedding(conn, "blk-1", "close match", emb, "2026-04-08")

    results = search_blocks_semantic(conn, emb, limit=10)
    assert len(results) == 1
    row, similarity = results[0]
    assert row["block_id"] == "blk-1"
    assert similarity > 0.99  # near-identical vector should have similarity ~1.0


def test_semantic_search_similarity_ordering():
    """Results are ordered by descending similarity."""
    conn = _temp_db()
    rng = np.random.default_rng(42)
    query = _normalize(rng.standard_normal(DIM).astype(np.float32))

    close_emb = _normalize(query + rng.standard_normal(DIM).astype(np.float32) * 0.01)
    _insert_block_with_embedding(conn, "blk-close", "close", close_emb, "2026-04-08")

    mid_emb = _normalize(query + rng.standard_normal(DIM).astype(np.float32) * 0.5)
    _insert_block_with_embedding(conn, "blk-mid", "mid", mid_emb, "2026-04-08")

    results = search_blocks_semantic(conn, query, limit=10)
    assert len(results) >= 2
    assert results[0][0]["block_id"] == "blk-close"
    assert results[0][1] > results[1][1]


def test_semantic_search_min_similarity_filters():
    """Results below min_similarity are excluded."""
    conn = _temp_db()
    rng = np.random.default_rng(42)
    query = _normalize(rng.standard_normal(DIM).astype(np.float32))

    close_emb = query.copy()
    _insert_block_with_embedding(conn, "blk-close", "close", close_emb, "2026-04-08")

    far_emb = -query
    _insert_block_with_embedding(conn, "blk-far", "far", far_emb, "2026-04-08")

    results = search_blocks_semantic(conn, query, limit=10, min_similarity=0.95)
    ids = [r[0]["block_id"] for r in results]
    assert "blk-close" in ids
    assert "blk-far" not in ids


def test_semantic_search_empty():
    """Semantic search on empty vec table returns empty list."""
    conn = _temp_db()
    query = np.zeros(DIM, dtype=np.float32)
    results = search_blocks_semantic(conn, query, limit=10)
    assert results == []
