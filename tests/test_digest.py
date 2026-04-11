"""Tests for the weekly digest feature."""

import tempfile
from datetime import datetime, timedelta
from pathlib import Path

from dendr.db import (
    connect,
    get_dropped_threads,
    init_schema,
    insert_claim,
)
from dendr.digest import (
    SectionFeedback,
    _age_days,
    _age_suffix,
    _annotation_to_dict,
    _render_task_review,
    build_synthesis_prompt,
    ingest_feedback,
    parse_feedback,
    render_local_digest,
)
from dendr.models import BlockAnnotation, BlockType, Claim, ClaimKind, ClaimStatus


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
        kind=ClaimKind.STATEMENT,
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


def test_claim_kind_persisted():
    conn = _temp_db()
    claim = _make_claim(kind=ClaimKind.TASK, text="Fix the CI pipeline")
    cid = insert_claim(conn, claim)
    row = conn.execute("SELECT kind FROM claims WHERE id = ?", (cid,)).fetchone()
    assert row["kind"] == "task"


def test_claim_kind_default():
    conn = _temp_db()
    claim = _make_claim()
    cid = insert_claim(conn, claim)
    row = conn.execute("SELECT kind FROM claims WHERE id = ?", (cid,)).fetchone()
    assert row["kind"] == "statement"


def test_get_dropped_threads():
    conn = _temp_db()
    old = datetime.now() - timedelta(weeks=3)

    insert_claim(
        conn,
        _make_claim(
            concept_slug="forgotten-topic",
            text="Some note about X",
            created_at=old,
        ),
    )
    insert_claim(
        conn,
        _make_claim(
            concept_slug="active-topic",
            text="First mention",
            created_at=old,
        ),
    )
    insert_claim(
        conn,
        _make_claim(
            concept_slug="active-topic",
            text="Second mention",
        ),
    )

    before = (datetime.now() - timedelta(weeks=2)).isoformat()
    dropped = get_dropped_threads(conn, before)
    slugs = [r["concept_slug"] for r in dropped]
    assert "forgotten-topic" in slugs
    assert "active-topic" not in slugs


def test_render_local_digest_with_annotations():
    """render_local_digest produces valid markdown from annotation-based data."""
    today = datetime.now().strftime("%Y-%m-%d")
    two_days_ago = (datetime.now() - timedelta(days=2)).strftime("%Y-%m-%d")
    data = {
        "generated_at": datetime.now().isoformat(),
        "period_start": (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d"),
        "period_end": today,
        "stats": {
            "active_claims": 42,
            "concepts": 10,
            "challenged_claims": 2,
            "annotations": 15,
        },
        "narrative_blocks": [
            {
                "block_id": "b1",
                "source_date": two_days_ago,
                "original_text": "I'm feeling burned out",
                "gist": "Feeling burned out from project work",
                "block_type": "reflection",
                "life_areas": ["work", "health"],
                "emotional_valence": -0.6,
                "emotional_labels": ["burned_out"],
                "intensity": 0.9,
                "urgency": None,
                "importance": None,
                "completion_status": None,
                "epistemic_status": "certain",
                "causal_links": ["3 months pushing -> burnout"],
                "concepts": ["burnout"],
                "entities": [],
            },
        ],
        "patterns": {
            "recurring_topics": [
                {
                    "concept": "burnout",
                    "mentions": 4,
                    "avg_valence": -0.5,
                    "trend": "worsening",
                },
                {
                    "concept": "rust",
                    "mentions": 1,
                    "avg_valence": 0.3,
                    "trend": "stable",
                },
            ],
            "life_area_distribution": {"work": 65, "health": 20, "learning": 15},
            "emotional_trajectory": [
                {"week_start": "2026-03-25", "avg_valence": -0.2, "block_count": 5},
                {"week_start": "2026-04-01", "avg_valence": -0.5, "block_count": 7},
            ],
            "open_tasks": [
                {
                    "block_id": "t1",
                    "source_date": two_days_ago,
                    "original_text": "Fix CI pipeline",
                    "gist": "Fix CI pipeline",
                    "block_type": "task",
                    "life_areas": ["work"],
                    "emotional_valence": -0.2,
                    "emotional_labels": [],
                    "intensity": 0.7,
                    "urgency": "this_week",
                    "importance": "high",
                    "completion_status": "open",
                    "epistemic_status": "certain",
                    "causal_links": [],
                    "concepts": ["ci"],
                    "entities": [],
                },
            ],
            "completed_recently": [],
            "stale_tasks": [],
        },
        "contradictions": [
            {
                "id": 1,
                "text": "X uses Postgres",
                "concept_slug": "database",
                "confidence": 0.8,
                "created_at": datetime.now().isoformat(),
            },
        ],
        "dropped_threads": [
            {
                "concept_slug": "forgotten",
                "text": "Something about X",
                "created_at": (datetime.now() - timedelta(weeks=3)).isoformat(),
            },
        ],
        "section_effectiveness": {},
    }

    result = render_local_digest(data)

    assert "# Weekly Digest" in result
    assert "What's On Your Mind" in result
    assert "burnout" in result.lower()
    assert "Open Loops" in result
    assert "Fix CI" in result
    assert "Patterns" in result
    assert "[[burnout]]" in result
    assert "work: 65%" in result
    assert "Contradictions" in result
    assert "Dropped Threads" in result
    assert "[[forgotten]]" in result
    assert "feedback:" in result


def test_render_local_digest_empty():
    data = {
        "generated_at": datetime.now().isoformat(),
        "period_start": (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d"),
        "period_end": datetime.now().strftime("%Y-%m-%d"),
        "stats": {
            "active_claims": 0,
            "concepts": 0,
            "challenged_claims": 0,
            "annotations": 0,
        },
        "narrative_blocks": [],
        "patterns": {
            "recurring_topics": [],
            "life_area_distribution": {},
            "emotional_trajectory": [],
            "open_tasks": [],
            "completed_recently": [],
            "stale_tasks": [],
        },
        "contradictions": [],
        "dropped_threads": [],
        "section_effectiveness": {},
    }

    result = render_local_digest(data)
    assert "# Weekly Digest" in result
    assert "No notable insights this week" in result


def test_build_synthesis_prompt():
    data = {
        "generated_at": datetime.now().isoformat(),
        "period_start": (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d"),
        "period_end": datetime.now().strftime("%Y-%m-%d"),
        "stats": {
            "active_claims": 5,
            "concepts": 2,
            "challenged_claims": 0,
            "annotations": 3,
        },
        "narrative_blocks": [
            {
                "block_id": "b1",
                "source_date": "2026-04-08",
                "original_text": "Test note",
                "gist": "Test gist",
                "block_type": "observation",
                "life_areas": ["work"],
                "emotional_valence": 0.0,
                "emotional_labels": [],
                "intensity": 0.5,
                "urgency": None,
                "importance": None,
                "completion_status": None,
                "epistemic_status": "certain",
                "causal_links": [],
                "concepts": ["test"],
                "entities": [],
            },
        ],
        "patterns": {
            "recurring_topics": [],
            "life_area_distribution": {},
            "emotional_trajectory": [],
            "open_tasks": [],
            "completed_recently": [],
            "stale_tasks": [],
        },
        "contradictions": [],
        "dropped_threads": [],
        "section_effectiveness": {"narrative": 0.8},
    }

    prompt = build_synthesis_prompt(data)
    assert "weekly advisor" in prompt
    assert "narrative_blocks" in prompt
    assert "Reframes" in prompt
    assert "section_effectiveness" in prompt
    assert "Test note" in prompt


# ── Feedback tests ────────────────────────────────────────────────────


def test_parse_feedback_filled():
    text = """## Open Loops
- stuff

<!-- feedback:open-loops
useful: yes
note: CI is done, remove it
-->

## Dropped Threads

<!-- feedback:dropped-threads
useful: no
note:
-->
"""
    feedback = parse_feedback(text)
    assert len(feedback) == 2

    ol = next(f for f in feedback if f.section == "open-loops")
    assert ol.useful is True
    assert ol.note == "CI is done, remove it"

    dt = next(f for f in feedback if f.section == "dropped-threads")
    assert dt.useful is False


def test_parse_feedback_empty():
    text = """<!-- feedback:open-loops
useful:
note:
-->
"""
    feedback = parse_feedback(text)
    assert len(feedback) == 0


def test_parse_feedback_note_only():
    text = """<!-- feedback:contradictions
useful:
note: both are true actually, different contexts
-->"""
    feedback = parse_feedback(text)
    assert len(feedback) == 1
    assert feedback[0].useful is None
    assert "both are true" in feedback[0].note


# ── Age helpers + Task Review ─────────────────────────────────────────


def test_age_days_today():
    today = datetime.now().strftime("%Y-%m-%d")
    assert _age_days(today) == 0


def test_age_days_past():
    three_weeks = (datetime.now() - timedelta(days=21)).strftime("%Y-%m-%d")
    assert _age_days(three_weeks) == 21


def test_age_days_malformed():
    assert _age_days("not-a-date") == 0
    assert _age_days("") == 0


def test_age_suffix_words():
    today = datetime.now().strftime("%Y-%m-%d")
    yesterday = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
    three_weeks = (datetime.now() - timedelta(days=21)).strftime("%Y-%m-%d")
    two_months = (datetime.now() - timedelta(days=62)).strftime("%Y-%m-%d")
    assert _age_suffix(today) == "written today"
    assert _age_suffix(yesterday) == "written 1d ago"
    assert _age_suffix(three_weeks) == "written 3w ago"
    assert "mo ago" in _age_suffix(two_months)


def test_render_task_review_empty_with_fresh_only():
    # All fresh tasks (<7d) should route to Open Loops, not Task Review.
    today = datetime.now().strftime("%Y-%m-%d")
    data = {
        "generated_at": datetime.now().isoformat(),
        "period_start": today,
        "period_end": today,
        "stats": {
            "active_claims": 0,
            "concepts": 0,
            "challenged_claims": 0,
            "annotations": 1,
        },
        "narrative_blocks": [],
        "patterns": {
            "recurring_topics": [],
            "life_area_distribution": {},
            "emotional_trajectory": [],
            "open_tasks": [
                {
                    "block_id": "fresh-1",
                    "source_date": today,
                    "gist": "Fresh task",
                    "life_areas": ["work"],
                    "urgency": "today",
                    "importance": "high",
                    "completion_status": "open",
                }
            ],
            "completed_recently": [],
            "stale_tasks": [],
        },
        "contradictions": [],
        "dropped_threads": [],
        "section_effectiveness": {},
    }
    result = render_local_digest(data)
    assert "Task Review" not in result
    assert "Open Loops (1 fresh)" in result
    assert "[today when written]" in result


def test_render_task_review_buckets():
    twelve_days = (datetime.now() - timedelta(days=12)).strftime("%Y-%m-%d")
    three_weeks = (datetime.now() - timedelta(days=21)).strftime("%Y-%m-%d")
    two_months = (datetime.now() - timedelta(days=62)).strftime("%Y-%m-%d")

    tasks = [
        {
            "block_id": "dendr-early",
            "source_date": twelve_days,
            "gist": "Fresh-ish task",
            "life_areas": ["work"],
        },
        {
            "block_id": "dendr-mid",
            "source_date": three_weeks,
            "gist": "Middle task",
            "life_areas": ["health"],
        },
        {
            "block_id": "dendr-old",
            "source_date": two_months,
            "gist": "Ancient task",
            "life_areas": [],
        },
    ]

    rendered = _render_task_review(tasks)
    assert "Task Review (3 open" in rendered
    # All three age buckets present, oldest first
    assert "### 1m+ old" in rendered
    assert "### 2-4w old" in rendered
    assert "### 1-2w old" in rendered
    # Oldest renders before newest
    assert rendered.index("1m+ old") < rendered.index("2-4w old")
    assert rendered.index("2-4w old") < rendered.index("1-2w old")
    # Every task has a closure marker with its block_id
    assert "<!-- closure:dendr-early status:open -->" in rendered
    assert "<!-- closure:dendr-mid status:open -->" in rendered
    assert "<!-- closure:dendr-old status:open -->" in rendered
    # Checkbox is unchecked
    assert "- [ ] **Fresh-ish task**" in rendered


def test_render_local_digest_includes_task_review_section():
    three_weeks = (datetime.now() - timedelta(days=21)).strftime("%Y-%m-%d")
    today = datetime.now().strftime("%Y-%m-%d")
    data = {
        "generated_at": datetime.now().isoformat(),
        "period_start": (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d"),
        "period_end": today,
        "stats": {
            "active_claims": 0,
            "concepts": 0,
            "challenged_claims": 0,
            "annotations": 1,
        },
        "narrative_blocks": [],
        "patterns": {
            "recurring_topics": [],
            "life_area_distribution": {},
            "emotional_trajectory": [],
            "open_tasks": [
                {
                    "block_id": "dendr-stale-1",
                    "source_date": three_weeks,
                    "gist": "Old unresolved thing",
                    "life_areas": ["work"],
                    "urgency": "today",
                    "importance": "high",
                    "completion_status": "open",
                }
            ],
            "completed_recently": [],
            "stale_tasks": [],
        },
        "contradictions": [],
        "dropped_threads": [],
        "section_effectiveness": {},
    }
    result = render_local_digest(data)
    assert "## Task Review" in result
    assert "Old unresolved thing" in result
    assert "<!-- closure:dendr-stale-1 status:open -->" in result
    # No Open Loops header since only stale tasks
    assert "## Open Loops" not in result
    # feedback marker for task-review section
    assert "feedback:task-review" in result


def test_annotation_to_dict_age_days():
    # Build a fake sqlite3.Row-ish dict via _annotation_to_dict.
    # Easier path: round-trip through a real DB.
    from dendr.db import connect, init_schema, upsert_block_annotation
    from dendr.models import BlockAnnotation, BlockType

    f = tempfile.NamedTemporaryFile(suffix=".sqlite", delete=False)
    f.close()
    conn = connect(Path(f.name))
    init_schema(conn)

    eight_days_ago = (datetime.now() - timedelta(days=8)).strftime("%Y-%m-%d")
    ann = BlockAnnotation(
        block_id="test-age",
        source_file="Daily/x.md",
        source_date=eight_days_ago,
        original_text="stuff",
        gist="a gist",
        block_type=BlockType.TASK,
        life_areas=["work"],
        emotional_valence=0.0,
        emotional_labels=[],
        intensity=0.5,
        urgency="today",
        importance="high",
        completion_status="open",
        concepts=[],
        entities=[],
    )
    upsert_block_annotation(conn, ann)

    row = conn.execute(
        "SELECT * FROM block_annotations WHERE block_id = ?", ("test-age",)
    ).fetchone()

    d = _annotation_to_dict(row)
    assert d["age_days"] == 8
    assert d["urgency"] == "today"
    assert d["source_date"] == eight_days_ago


def test_ingest_feedback_creates_claims():
    conn = _temp_db()
    feedback = [
        SectionFeedback(section="open-loops", useful=True, note="CI task is done"),
        SectionFeedback(section="contradictions", useful=False, note=""),
    ]

    stats = ingest_feedback(conn, feedback, "2026-04-10")
    assert stats["logged_ratings"] == 2
    assert stats["ingested_claims"] == 1

    rows = conn.execute(
        "SELECT * FROM claims WHERE source_block_ref LIKE 'digest-feedback%'"
    ).fetchall()
    assert len(rows) == 1
    assert rows[0]["text"] == "CI task is done"

    fb_rows = conn.execute("SELECT * FROM feedback_scores").fetchall()
    assert len(fb_rows) == 2
