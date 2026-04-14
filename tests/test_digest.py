"""Tests for the weekly digest feature."""

import tempfile
from datetime import datetime, timedelta
from pathlib import Path

from dendr.db import (
    connect,
    init_schema,
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
from dendr.models import BlockAnnotation, BlockType


def _temp_db():
    f = tempfile.NamedTemporaryFile(suffix=".sqlite", delete=False)
    f.close()
    conn = connect(Path(f.name))
    init_schema(conn)
    return conn


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
        intensity=0.5,
        concepts=["test-concept"],
        entities=[],
    )
    defaults.update(kwargs)
    return BlockAnnotation(**defaults)


def test_render_local_digest_with_annotations():
    """render_local_digest produces valid markdown from annotation-based data."""
    today = datetime.now().strftime("%Y-%m-%d")
    two_days_ago = (datetime.now() - timedelta(days=2)).strftime("%Y-%m-%d")
    data = {
        "generated_at": datetime.now().isoformat(),
        "period_start": (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d"),
        "period_end": today,
        "stats": {
            "concepts": 10,
            "annotations": 15,
            "open_tasks": 1,
        },
        "this_period": {
            "narrative_blocks": [
                {
                    "block_id": "b1",
                    "source_date": two_days_ago,
                    "original_text": "I'm feeling burned out",
                    "gist": "Feeling burned out from project work",
                    "block_type": "reflection",
                    "life_areas": ["work", "health"],
                    "emotional_valence": -0.6,
                    "intensity": 0.9,
                    "urgency": None,
                    "importance": None,
                    "completion_status": None,
                    "causal_links": ["3 months pushing -> burnout"],
                    "concepts": ["burnout"],
                    "entities": [],
                },
            ],
            "new_open_tasks": [
                {
                    "block_id": "t1",
                    "source_date": two_days_ago,
                    "original_text": "Fix CI pipeline",
                    "gist": "Fix CI pipeline",
                    "block_type": "task",
                    "life_areas": ["work"],
                    "emotional_valence": -0.2,
                    "intensity": 0.7,
                    "urgency": "this_week",
                    "importance": "high",
                    "completion_status": "open",
                    "causal_links": [],
                    "concepts": ["ci"],
                    "entities": [],
                },
            ],
        },
        "carried_forward": {
            "open_tasks": [],
            "stale_tasks": [],
        },
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
            "completed_recently": [],
            "task_lifecycle": {},
        },
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
    assert "feedback:" in result


def test_render_local_digest_empty():
    data = {
        "generated_at": datetime.now().isoformat(),
        "period_start": (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d"),
        "period_end": datetime.now().strftime("%Y-%m-%d"),
        "stats": {
            "concepts": 0,
            "annotations": 0,
            "open_tasks": 0,
        },
        "this_period": {"narrative_blocks": [], "new_open_tasks": []},
        "carried_forward": {"open_tasks": [], "stale_tasks": []},
        "patterns": {
            "recurring_topics": [],
            "life_area_distribution": {},
            "emotional_trajectory": [],
            "completed_recently": [],
            "task_lifecycle": {},
        },
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
            "concepts": 2,
            "annotations": 3,
            "open_tasks": 0,
        },
        "user_context": "",
        "this_period": {
            "narrative_blocks": [
                {
                    "block_id": "b1",
                    "source_date": "2026-04-08",
                    "original_text": "Test note",
                    "gist": "Test gist",
                    "block_type": "observation",
                    "life_areas": ["work"],
                    "emotional_valence": 0.0,
                    "intensity": 0.5,
                    "urgency": None,
                    "importance": None,
                    "completion_status": None,
                    "causal_links": [],
                    "concepts": ["test"],
                    "entities": [],
                },
            ],
            "new_open_tasks": [],
        },
        "carried_forward": {"open_tasks": [], "stale_tasks": []},
        "patterns": {
            "recurring_topics": [],
            "life_area_distribution": {},
            "emotional_trajectory": [],
            "completed_recently": [],
            "task_lifecycle": {},
        },
        "section_effectiveness": {"narrative": 0.8},
    }

    prompt = build_synthesis_prompt(data)
    assert "reviewing a week" in prompt
    assert "this_period.narrative_blocks" in prompt
    assert "carried_forward.open_tasks" in prompt
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
            "concepts": 0,
            "annotations": 1,
            "open_tasks": 1,
        },
        "this_period": {
            "narrative_blocks": [],
            "new_open_tasks": [
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
        },
        "carried_forward": {"open_tasks": [], "stale_tasks": []},
        "patterns": {
            "recurring_topics": [],
            "life_area_distribution": {},
            "emotional_trajectory": [],
            "completed_recently": [],
            "task_lifecycle": {},
        },
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
            "concepts": 0,
            "annotations": 1,
            "open_tasks": 1,
        },
        "this_period": {"narrative_blocks": [], "new_open_tasks": []},
        "carried_forward": {
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
            "stale_tasks": [],
        },
        "patterns": {
            "recurring_topics": [],
            "life_area_distribution": {},
            "emotional_trajectory": [],
            "completed_recently": [],
            "task_lifecycle": {},
        },
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


def test_normalize_annotation_raw_coerces_nullish_strings():
    from dendr.llm import _normalize_annotation_raw

    raw = {
        "gist": "x",
        "urgency": "null",
        "importance": "N/A",
        "completion_status": "",
    }
    out = _normalize_annotation_raw(raw)
    assert out["urgency"] is None
    assert out["importance"] is None
    assert out["completion_status"] is None


def test_normalize_annotation_raw_preserves_real_values():
    from dendr.llm import _normalize_annotation_raw

    raw = {
        "urgency": "today",
        "importance": "high",
        "completion_status": "open",
    }
    out = _normalize_annotation_raw(raw)
    assert out["urgency"] == "today"
    assert out["importance"] == "high"
    assert out["completion_status"] == "open"


def test_load_user_context_missing(tmp_path):
    from dendr.config import Config
    from dendr.digest import _load_user_context

    config = Config(vault_path=tmp_path, data_dir=tmp_path / "data")
    config.wiki_dir.mkdir(parents=True, exist_ok=True)
    assert _load_user_context(config) == ""


def test_load_user_context_present(tmp_path):
    from dendr.config import Config
    from dendr.digest import _load_user_context

    config = Config(vault_path=tmp_path, data_dir=tmp_path / "data")
    config.wiki_dir.mkdir(parents=True, exist_ok=True)
    ctx = "I'm a senior engineer.\nActive goal: ship Dendr.\n"
    (config.wiki_dir / "_user_context.md").write_text(ctx, encoding="utf-8")
    assert _load_user_context(config) == ctx.strip()


def test_build_synthesis_prompt_injects_user_context():
    data = {
        "generated_at": datetime.now().isoformat(),
        "period_start": "2026-04-03",
        "period_end": "2026-04-10",
        "stats": {"concepts": 0, "annotations": 0, "open_tasks": 0},
        "user_context": "Senior engineer working on Dendr. No kids, live alone.",
        "this_period": {"narrative_blocks": [], "new_open_tasks": []},
        "carried_forward": {"open_tasks": [], "stale_tasks": []},
        "patterns": {
            "recurring_topics": [],
            "life_area_distribution": {},
            "emotional_trajectory": [],
            "completed_recently": [],
            "task_lifecycle": {},
        },
        "section_effectiveness": {},
    }
    prompt = build_synthesis_prompt(data)
    assert "Who the user is" in prompt
    assert "Senior engineer working on Dendr" in prompt


def test_build_synthesis_prompt_missing_user_context():
    data = {
        "generated_at": datetime.now().isoformat(),
        "period_start": "2026-04-03",
        "period_end": "2026-04-10",
        "stats": {"concepts": 0, "annotations": 0, "open_tasks": 0},
        "this_period": {"narrative_blocks": [], "new_open_tasks": []},
        "carried_forward": {"open_tasks": [], "stale_tasks": []},
        "patterns": {
            "recurring_topics": [],
            "life_area_distribution": {},
            "emotional_trajectory": [],
            "completed_recently": [],
            "task_lifecycle": {},
        },
        "section_effectiveness": {},
    }
    prompt = build_synthesis_prompt(data)
    assert "_user_context.md" in prompt


def test_gather_digest_data_splits_new_vs_carried_forward(tmp_path):
    """_gather_digest_data sorts open tasks into this_period vs carried_forward."""
    from dendr.config import Config
    from dendr.db import upsert_block_annotation
    from dendr.digest import _gather_digest_data

    config = Config(vault_path=tmp_path, data_dir=tmp_path / "data")
    config.wiki_dir.mkdir(parents=True, exist_ok=True)

    conn = _temp_db()

    # Task written 2 days ago → this_period.new_open_tasks
    two_days_ago = (datetime.now() - timedelta(days=2)).strftime("%Y-%m-%d")
    upsert_block_annotation(
        conn,
        BlockAnnotation(
            block_id="new-task",
            source_file="Daily/x.md",
            source_date=two_days_ago,
            original_text="new task",
            gist="new task",
            block_type=BlockType.TASK,
            life_areas=["work"],
            emotional_valence=0.0,
            intensity=0.5,
            completion_status="open",
            concepts=[],
            entities=[],
        ),
    )

    # Task written 3 weeks ago → carried_forward.open_tasks
    three_weeks_ago = (datetime.now() - timedelta(days=21)).strftime("%Y-%m-%d")
    upsert_block_annotation(
        conn,
        BlockAnnotation(
            block_id="old-task",
            source_file="Daily/x.md",
            source_date=three_weeks_ago,
            original_text="old task",
            gist="old task",
            block_type=BlockType.TASK,
            life_areas=[],
            emotional_valence=0.0,
            intensity=0.5,
            completion_status="open",
            concepts=[],
            entities=[],
        ),
    )

    data = _gather_digest_data(config, conn, weeks=1)

    new_ids = [t["block_id"] for t in data["this_period"]["new_open_tasks"]]
    carried_ids = [t["block_id"] for t in data["carried_forward"]["open_tasks"]]
    assert "new-task" in new_ids
    assert "old-task" not in new_ids
    assert "old-task" in carried_ids
    assert "new-task" not in carried_ids


def test_ingest_feedback_logs_ratings():
    conn = _temp_db()
    feedback = [
        SectionFeedback(section="open-loops", useful=True, note="CI task is done"),
        SectionFeedback(section="patterns", useful=False, note=""),
    ]

    stats = ingest_feedback(conn, feedback, "2026-04-10")
    assert stats["logged_ratings"] == 2

    fb_rows = conn.execute("SELECT * FROM feedback_scores").fetchall()
    assert len(fb_rows) == 2
