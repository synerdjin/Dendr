"""Tests for the database layer."""

import tempfile
from datetime import datetime
from pathlib import Path

from dendr.db import (
    connect,
    init_schema,
    insert_claim,
    find_contradictions,
    find_similar_claim,
    reinforce_claim,
    challenge_claim,
    supersede_claim,
    upsert_concept,
    append_log,
    search_claims_fts,
    get_stats,
    upsert_block_state,
    get_block_state,
)
from dendr.models import Claim, ClaimStatus, Concept, PageType


def _temp_db():
    f = tempfile.NamedTemporaryFile(suffix=".sqlite", delete=False)
    f.close()
    conn = connect(Path(f.name))
    init_schema(conn)
    return conn


def _make_claim(**kwargs) -> Claim:
    defaults = dict(
        id=None,
        text="Test claim",
        subject="X",
        predicate="uses",
        object="Y",
        subject_predicate="X|uses",
        concept_slug="test-concept",
        source_block_ref="block-1",
        source_file_hash="abc123",
        created_at=datetime.now(),
        updated_at=datetime.now(),
        confidence=0.8,
        status=ClaimStatus.CREATED,
    )
    defaults.update(kwargs)
    return Claim(**defaults)


def test_insert_and_find_claim():
    conn = _temp_db()
    claim = _make_claim()
    claim_id = insert_claim(conn, claim)
    assert claim_id > 0

    found = find_similar_claim(conn, "X|uses", "Y")
    assert found is not None
    assert found["text"] == "Test claim"


def test_contradiction_detection():
    conn = _temp_db()
    c1 = _make_claim(text="X uses Postgres", object="Postgres")
    id1 = insert_claim(conn, c1)

    contradictions = find_contradictions(conn, "X|uses", "SQLite")
    assert len(contradictions) == 1
    assert contradictions[0]["id"] == id1


def test_reinforce_claim():
    conn = _temp_db()
    c = _make_claim(confidence=0.7)
    cid = insert_claim(conn, c)
    reinforce_claim(conn, cid)

    row = conn.execute("SELECT * FROM claims WHERE id = ?", (cid,)).fetchone()
    assert row["status"] == "reinforced"
    assert row["confidence"] > 0.7


def test_supersede_claim():
    conn = _temp_db()
    c1 = _make_claim(text="old claim")
    id1 = insert_claim(conn, c1)
    c2 = _make_claim(text="new claim")
    id2 = insert_claim(conn, c2)

    supersede_claim(conn, id1, id2)
    row = conn.execute("SELECT * FROM claims WHERE id = ?", (id1,)).fetchone()
    assert row["status"] == "superseded"
    assert row["superseded_by"] == id2


def test_fts_search():
    conn = _temp_db()
    c = _make_claim(text="machine learning is powerful", concept_slug="ml")
    insert_claim(conn, c)

    results = search_claims_fts(conn, "machine learning")
    assert len(results) >= 1
    assert results[0]["concept_slug"] == "ml"


def test_stats():
    conn = _temp_db()
    insert_claim(conn, _make_claim())
    s = get_stats(conn)
    assert s["active_claims"] == 1
    assert s["concepts"] == 0


def test_block_state():
    conn = _temp_db()
    upsert_block_state(conn, "b1", "daily.md", "hash1", "model1", "v1")
    state = get_block_state(conn, "b1")
    assert state is not None
    assert state["block_hash"] == "hash1"

    # Update
    upsert_block_state(conn, "b1", "daily.md", "hash2", "model1", "v1")
    state = get_block_state(conn, "b1")
    assert state["block_hash"] == "hash2"


def test_concept_upsert():
    conn = _temp_db()
    c = Concept(
        slug="ml",
        title="Machine Learning",
        page_type=PageType.CONCEPT,
        created_at=datetime.now(),
        updated_at=datetime.now(),
        page_path="Wiki/concepts/ml.md",
    )
    upsert_concept(conn, c)
    row = conn.execute("SELECT * FROM concepts WHERE slug = 'ml'").fetchone()
    assert row["title"] == "Machine Learning"


def test_log():
    conn = _temp_db()
    append_log(conn, "test_event", {"key": "value"})
    rows = conn.execute("SELECT * FROM log").fetchall()
    assert len(rows) == 1
    assert rows[0]["kind"] == "test_event"
