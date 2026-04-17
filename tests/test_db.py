"""Tests for the database layer."""

import tempfile
from pathlib import Path

from dendr.db import (
    connect,
    get_block,
    get_block_hash,
    get_open_tasks,
    get_section_effectiveness,
    get_stats,
    init_schema,
    insert_task_event,
    search_blocks_fts,
    update_completion_status,
    upsert_block,
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


def test_get_block_hash():
    conn = _temp_db()
    assert get_block_hash(conn, "missing") is None
    upsert_block(conn, _make_block(block_hash="h1"), "2026-04-08")
    assert get_block_hash(conn, "dendr-test-1") == "h1"


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
