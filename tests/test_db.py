"""Tests for the database layer."""

import tempfile
from datetime import datetime
from pathlib import Path

from dendr.db import (
    connect,
    init_schema,
    insert_claim,
    reinforce_claim,
    supersede_claim,
    upsert_concept,
    upsert_block_annotation,
    get_block_annotation,
    append_log,
    search_claims_fts,
    get_stats,
    upsert_block_state,
    get_block_state,
    get_significant_blocks,
    get_open_tasks_annotated,
    get_life_area_distribution,
    get_emotional_trajectory,
    upsert_feedback_score,
    get_section_effectiveness,
)
from dendr.models import (
    BlockAnnotation,
    BlockType,
    Claim,
    ClaimStatus,
    Concept,
    PageType,
)


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


def _make_annotation(**kwargs) -> BlockAnnotation:
    defaults = dict(
        block_id="dendr-test-1",
        source_file="Daily/2026-04-08.md",
        source_date="2026-04-08",
        original_text="Test block content",
        gist="Test gist",
        block_type=BlockType.OBSERVATION,
        life_areas=["work"],
        emotional_valence=0.0,
        emotional_labels=[],
        intensity=0.5,
        concepts=["test-concept"],
        entities=[],
    )
    defaults.update(kwargs)
    return BlockAnnotation(**defaults)


def test_insert_and_find_claim():
    conn = _temp_db()
    claim = _make_claim()
    claim_id = insert_claim(conn, claim)
    assert claim_id > 0


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
    assert s["annotations"] == 0


def test_block_state():
    conn = _temp_db()
    upsert_block_state(conn, "b1", "daily.md", "hash1", "model1", "v1")
    state = get_block_state(conn, "b1")
    assert state is not None
    assert state["block_hash"] == "hash1"

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


# ── Block annotation tests ───────────────────────────────────────────


def test_upsert_block_annotation():
    conn = _temp_db()
    ann = _make_annotation()
    ann_id = upsert_block_annotation(conn, ann)
    assert ann_id > 0

    retrieved = get_block_annotation(conn, "dendr-test-1")
    assert retrieved is not None
    assert retrieved["gist"] == "Test gist"
    assert retrieved["block_type"] == "observation"
    assert retrieved["source_date"] == "2026-04-08"


def test_upsert_block_annotation_update():
    conn = _temp_db()
    ann = _make_annotation(gist="Original gist")
    upsert_block_annotation(conn, ann)

    ann.gist = "Updated gist"
    upsert_block_annotation(conn, ann)

    retrieved = get_block_annotation(conn, "dendr-test-1")
    assert retrieved["gist"] == "Updated gist"


def test_annotation_json_fields():
    conn = _temp_db()
    ann = _make_annotation(
        life_areas=["work", "health"],
        emotional_labels=["frustrated", "anxious"],
        causal_links=["overwork -> burnout"],
        concepts=["burnout", "project-x"],
        entities=["Alice"],
    )
    upsert_block_annotation(conn, ann)

    import json

    retrieved = get_block_annotation(conn, "dendr-test-1")
    assert json.loads(retrieved["life_areas"]) == ["work", "health"]
    assert json.loads(retrieved["emotional_labels"]) == ["frustrated", "anxious"]
    assert json.loads(retrieved["causal_links"]) == ["overwork -> burnout"]
    assert json.loads(retrieved["concepts"]) == ["burnout", "project-x"]


def test_get_significant_blocks():
    conn = _temp_db()
    # High intensity block
    upsert_block_annotation(
        conn,
        _make_annotation(
            block_id="high",
            intensity=0.9,
            gist="Very important",
        ),
    )
    # Low intensity block
    upsert_block_annotation(
        conn,
        _make_annotation(
            block_id="low",
            intensity=0.1,
            gist="Not important",
        ),
    )

    results = get_significant_blocks(conn, "2026-04-01")
    assert len(results) == 2
    assert results[0]["block_id"] == "high"  # highest intensity first


def test_get_open_tasks_annotated():
    conn = _temp_db()
    upsert_block_annotation(
        conn,
        _make_annotation(
            block_id="task1",
            block_type=BlockType.TASK,
            completion_status="open",
            importance="high",
            gist="Fix CI",
        ),
    )
    upsert_block_annotation(
        conn,
        _make_annotation(
            block_id="done1",
            block_type=BlockType.TASK,
            completion_status="done",
            gist="Done task",
        ),
    )
    upsert_block_annotation(
        conn,
        _make_annotation(
            block_id="obs1",
            block_type=BlockType.OBSERVATION,
            gist="Just an observation",
        ),
    )

    tasks = get_open_tasks_annotated(conn)
    assert len(tasks) == 1
    assert tasks[0]["gist"] == "Fix CI"


def test_get_life_area_distribution():
    conn = _temp_db()
    upsert_block_annotation(
        conn,
        _make_annotation(block_id="b1", life_areas=["work", "health"]),
    )
    upsert_block_annotation(
        conn,
        _make_annotation(block_id="b2", life_areas=["work"]),
    )

    dist = get_life_area_distribution(conn, "2026-04-01")
    assert "work" in dist
    assert dist["work"] > dist.get("health", 0)


def test_get_emotional_trajectory():
    conn = _temp_db()
    trajectory = get_emotional_trajectory(conn, weeks=2)
    assert len(trajectory) == 2
    for w in trajectory:
        assert "avg_valence" in w
        assert "block_count" in w


# ── Feedback tests ────────────────────────────────────────────────────


def test_feedback_scores():
    conn = _temp_db()
    upsert_feedback_score(conn, "2026-04-03", "narrative", True, "good stuff")
    upsert_feedback_score(conn, "2026-04-03", "open-loops", False, "")

    scores = get_section_effectiveness(conn)
    assert scores["narrative"] == 1.0
    assert scores["open-loops"] == 0.0


def test_stats_includes_annotations():
    conn = _temp_db()
    upsert_block_annotation(conn, _make_annotation())
    s = get_stats(conn)
    assert s["annotations"] == 1
