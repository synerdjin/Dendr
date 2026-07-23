"""Tests for the database layer."""

import sqlite3
import tempfile
from datetime import datetime, timedelta
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
    rrf_fuse,
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


def test_get_open_tasks_truncation_keeps_oldest():
    """Regression (F14): when open tasks exceed the limit, the OLDEST must be
    retained (Task Review targets them), and the result stays newest-first."""
    conn = _temp_db()
    # Five open tasks on distinct dates, inserted newest-first to prove the
    # kept set is chosen by date, not insertion order.
    for day in (5, 4, 3, 2, 1):
        upsert_block(
            conn,
            _make_block(block_id=f"t{day}", checkbox_state=CHECKBOX_OPEN),
            f"2026-04-0{day}",
        )

    rows = get_open_tasks(conn, limit=3)
    ids = [r["block_id"] for r in rows]
    # The three OLDEST (04-01..04-03) survive; returned newest-first.
    assert ids == ["t3", "t2", "t1"]


def test_fts_finds_raw_text():
    conn = _temp_db()
    upsert_block(conn, _make_block(text="The quick brown fox jumps over"), "2026-04-08")
    rows = search_blocks_fts(conn, "brown fox")
    assert len(rows) == 1
    assert rows[0]["block_id"] == "dendr-test-1"


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


def test_fts_special_characters_do_not_raise():
    """Regression (F10): FTS5 syntax chars in a user query must not crash."""
    conn = _temp_db()
    upsert_block(
        conn, _make_block(text='meeting about the "roadmap" (Q3)'), "2026-04-08"
    )
    # Each of these would raise sqlite3.OperationalError against a raw MATCH.
    for q in ['"unbalanced', "roadmap*", "-foo", "(Q3", "NEAR:", "a AND", ""]:
        rows = search_blocks_fts(conn, q)
        assert isinstance(rows, list)  # no exception, well-formed result


def test_fts_quoted_query_still_matches():
    """Sanitizing to phrase tokens must not break ordinary term search."""
    conn = _temp_db()
    upsert_block(conn, _make_block(text="quarterly roadmap review"), "2026-04-08")
    assert len(search_blocks_fts(conn, "roadmap review")) == 1
    assert len(search_blocks_fts(conn, '"roadmap"')) == 1


# ── Schema migration ─────────────────────────────────────────────────


def test_init_schema_drops_legacy_private_column():
    """Pre-v5 DBs carried a `private` column; init_schema must drop it
    without losing existing rows, and be safe to run more than once."""
    f = tempfile.NamedTemporaryFile(suffix=".sqlite", delete=False)
    f.close()
    path = Path(f.name)

    raw = sqlite3.connect(str(path))
    raw.execute(
        """
        CREATE TABLE blocks (
            id                INTEGER PRIMARY KEY AUTOINCREMENT,
            block_id          TEXT NOT NULL UNIQUE,
            source_file       TEXT NOT NULL,
            source_date       TEXT NOT NULL,
            text              TEXT NOT NULL,
            block_hash        TEXT NOT NULL,
            checkbox_state    TEXT NOT NULL DEFAULT 'none',
            completion_status TEXT,
            private           INTEGER NOT NULL DEFAULT 0,
            attachment_path   TEXT,
            attachment_type   TEXT,
            created_at        TEXT NOT NULL,
            updated_at        TEXT NOT NULL
        )
        """
    )
    raw.execute(
        "INSERT INTO blocks (block_id, source_file, source_date, text, "
        "block_hash, private, created_at, updated_at) "
        "VALUES ('b1', 'f', '2026-01-01', 'hi', 'h1', 1, 'now', 'now')"
    )
    raw.commit()
    raw.close()

    conn = connect(path)
    init_schema(conn)

    cols = [row[1] for row in conn.execute("PRAGMA table_info(blocks)")]
    assert "private" not in cols

    row = get_block(conn, "b1")
    assert row["text"] == "hi"

    init_schema(conn)  # must be idempotent
    cols_again = [row[1] for row in conn.execute("PRAGMA table_info(blocks)")]
    assert "private" not in cols_again


def _vec_available(conn: sqlite3.Connection) -> bool:
    try:
        conn.execute("CREATE VIRTUAL TABLE _vec_probe USING vec0(embedding float[4])")
        conn.execute("DROP TABLE _vec_probe")
        return True
    except sqlite3.OperationalError:
        return False


def test_blocks_vec_migration_reembeds(tmp_path):
    """Regression (F13): migrating blocks_vec from L2 to cosine drops every
    embedding, so init_schema must blank block_hash to force a re-embed —
    otherwise semantic search stays silently empty until a manual reprocess."""
    import pytest

    path = tmp_path / "state.sqlite"
    conn = connect(path)
    if not _vec_available(conn):
        pytest.skip("sqlite-vec not available")

    init_schema(conn)
    upsert_block(conn, _make_block(block_hash="orig"), "2026-04-08")

    # Recreate blocks_vec the pre-v4 way: default (L2) metric, no distance_metric.
    conn.execute("DROP TABLE blocks_vec")
    conn.execute(
        "CREATE VIRTUAL TABLE blocks_vec "
        "USING vec0(embedding float[768], block_id text)"
    )

    init_schema(conn)  # detects L2 → migrates to cosine + blanks hashes

    vec_sql = conn.execute(
        "SELECT sql FROM sqlite_master WHERE name='blocks_vec'"
    ).fetchone()[0]
    assert "distance_metric" in vec_sql.lower()
    assert get_block(conn, "dendr-test-1")["block_hash"] == ""


def test_blocks_vec_migration_is_atomic_on_failure(tmp_path, monkeypatch):
    """If the hash-blank fails mid-migration, the whole thing rolls back so the
    L2 table remains and the migration re-fires next time — rather than landing
    cosine with stale hashes, which would leave semantic search silently empty."""
    import pytest

    from dendr import db as dbmod

    path = tmp_path / "state.sqlite"
    conn = connect(path)
    if not _vec_available(conn):
        pytest.skip("sqlite-vec not available")

    init_schema(conn)
    upsert_block(conn, _make_block(block_hash="orig"), "2026-04-08")
    conn.execute("DROP TABLE blocks_vec")
    conn.execute(
        "CREATE VIRTUAL TABLE blocks_vec "
        "USING vec0(embedding float[768], block_id text)"
    )

    def boom(_conn):
        raise sqlite3.OperationalError("simulated crash mid-migration")

    monkeypatch.setattr(dbmod, "mark_all_blocks_dirty", boom)
    init_schema(conn)  # migration fails, rolls back; error swallowed at debug

    # The DROP+CREATE were rolled back: blocks_vec is still L2, hash untouched.
    vec_sql = conn.execute(
        "SELECT sql FROM sqlite_master WHERE name='blocks_vec'"
    ).fetchone()[0]
    assert "distance_metric" not in vec_sql.lower()
    assert get_block(conn, "dendr-test-1")["block_hash"] == "orig"


# ── Feedback tests ────────────────────────────────────────────────────


def test_feedback_scores():
    conn = _temp_db()
    # Use a date comfortably inside get_section_effectiveness's 12-week
    # lookback window; a hardcoded date sitting on the boundary is flaky.
    recent = (datetime.now() - timedelta(weeks=1)).isoformat()[:10]
    upsert_feedback_score(conn, recent, "narrative", True, "good stuff")
    upsert_feedback_score(conn, recent, "open-loops", False, "")

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


def _rows_by_id(conn, ids):
    for bid in ids:
        block = _make_block(block_id=bid, text=f"text {bid}", block_hash=bid)
        upsert_block(conn, block, "2026-04-08")
    conn.commit()
    return {r["block_id"]: r for r in conn.execute("SELECT * FROM blocks")}


def test_rrf_fuse_rewards_agreement():
    """A doc ranked highly by both FTS and semantic wins the fusion."""
    conn = _temp_db()
    rows = _rows_by_id(conn, ["a", "b", "c"])
    fts_rows = [rows["b"], rows["a"], rows["c"]]  # b #1
    sem_pairs = [(rows["b"], 0.9), (rows["c"], 0.8), (rows["a"], 0.7)]  # b #1
    fused = rrf_fuse(fts_rows, sem_pairs, limit=10)
    ids = [r["block_id"] for r, _score, _sim in fused]
    assert ids[0] == "b"
    # Semantic similarity is surfaced for docs that appeared in the semantic list.
    sim_by_id = {r["block_id"]: sim for r, _score, sim in fused}
    assert sim_by_id["b"] == 0.9


def test_rrf_fuse_includes_single_list_doc():
    """A strong semantic-only hit still surfaces (no starvation by FTS)."""
    conn = _temp_db()
    rows = _rows_by_id(conn, ["a", "b", "z"])
    fts_rows = [rows["a"], rows["b"]]  # z absent from FTS
    sem_pairs = [(rows["z"], 0.95)]  # z is the top semantic hit
    fused = rrf_fuse(fts_rows, sem_pairs, limit=10)
    ids = [r["block_id"] for r, _score, _sim in fused]
    assert "z" in ids
    # similarity is None for FTS-only docs
    sim_by_id = {r["block_id"]: sim for r, _score, sim in fused}
    assert sim_by_id["a"] is None


def test_rrf_fuse_respects_limit():
    conn = _temp_db()
    rows = _rows_by_id(conn, ["a", "b", "c"])
    fts_rows = [rows["a"], rows["b"], rows["c"]]
    fused = rrf_fuse(fts_rows, [], limit=2)
    assert len(fused) == 2
