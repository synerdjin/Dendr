"""Tests for the weekly digest feature."""

import tempfile
from datetime import datetime, timedelta
from pathlib import Path

from dendr.db import connect, init_schema, upsert_block
from dendr.digest import (
    SectionFeedback,
    _age_days,
    _age_suffix,
    _block_to_dict,
    _render_task_review,
    build_synthesis_prompt,
    ingest_feedback,
    parse_feedback,
    render_local_digest,
)
from dendr.models import CHECKBOX_NONE, CHECKBOX_OPEN, Block


def _temp_db():
    f = tempfile.NamedTemporaryFile(suffix=".sqlite", delete=False)
    f.close()
    conn = connect(Path(f.name))
    init_schema(conn)
    return conn


def test_render_local_digest_basic():
    today = datetime.now().strftime("%Y-%m-%d")
    two_days_ago = (datetime.now() - timedelta(days=2)).strftime("%Y-%m-%d")
    data = {
        "generated_at": datetime.now().isoformat(),
        "period_start": (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d"),
        "period_end": today,
        "stats": {"blocks": 15, "open_tasks": 1},
        "this_period": {
            "blocks": [
                {
                    "block_id": "b1",
                    "source_date": two_days_ago,
                    "age_days": 2,
                    "text": "I'm feeling burned out from project work",
                    "checkbox_state": CHECKBOX_NONE,
                    "completion_status": None,
                }
            ],
            "new_open_tasks": [
                {
                    "block_id": "t1",
                    "source_date": two_days_ago,
                    "age_days": 2,
                    "text": "[ ] Fix CI pipeline",
                    "checkbox_state": CHECKBOX_OPEN,
                    "completion_status": None,
                }
            ],
        },
        "carried_forward": {"open_tasks": []},
        "section_effectiveness": {},
    }

    result = render_local_digest(data)
    assert "# Weekly Digest" in result
    assert "This Week" in result
    assert "burned out" in result
    assert "Open Loops" in result
    assert "Fix CI" in result
    assert "feedback:" in result


def test_render_local_digest_empty():
    today = datetime.now().strftime("%Y-%m-%d")
    data = {
        "generated_at": datetime.now().isoformat(),
        "period_start": (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d"),
        "period_end": today,
        "stats": {"blocks": 0, "open_tasks": 0},
        "this_period": {"blocks": [], "new_open_tasks": []},
        "carried_forward": {"open_tasks": []},
        "section_effectiveness": {},
    }

    result = render_local_digest(data)
    assert "# Weekly Digest" in result
    assert "No notable activity" in result


def test_build_synthesis_prompt_contains_raw_text():
    data = {
        "generated_at": datetime.now().isoformat(),
        "period_start": (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d"),
        "period_end": datetime.now().strftime("%Y-%m-%d"),
        "stats": {"blocks": 3, "open_tasks": 0},
        "user_context": "",
        "this_period": {
            "blocks": [
                {
                    "block_id": "b1",
                    "source_date": "2026-04-08",
                    "age_days": 9,
                    "text": "I finished the prototype today",
                    "checkbox_state": CHECKBOX_NONE,
                    "completion_status": None,
                }
            ],
            "new_open_tasks": [],
        },
        "carried_forward": {"open_tasks": []},
        "section_effectiveness": {"narrative": 0.8},
    }

    prompt = build_synthesis_prompt(data)
    assert "retrospective coach" in prompt
    assert "this_period.blocks" in prompt
    assert "carried_forward.open_tasks" in prompt
    assert "Reframes" in prompt
    assert "section_effectiveness" in prompt
    assert "finished the prototype" in prompt
    # No pre-tagged affect fields bleeding into the prompt anymore
    assert "emotional_valence" not in prompt
    assert "life_areas" not in prompt
    # Coaching scaffolding regression guards
    assert "Anti-sycophancy" in prompt
    assert "prior_digests" in prompt
    assert "One thing" in prompt
    assert "Safety" in prompt


# ── Feedback parsing ──────────────────────────────────────────────────


def test_parse_feedback_filled():
    text = """## Open Loops
- stuff

<!-- feedback:open-loops
useful: yes
note: CI is done, remove it
-->
"""
    feedback = parse_feedback(text)
    assert len(feedback) == 1
    assert feedback[0].section == "open-loops"
    assert feedback[0].useful is True
    assert feedback[0].note == "CI is done, remove it"


def test_parse_feedback_empty():
    text = """<!-- feedback:open-loops
useful:
note:
-->
"""
    assert parse_feedback(text) == []


def test_parse_feedback_note_only():
    text = """<!-- feedback:contradictions
useful:
note: both are true actually, different contexts
-->"""
    feedback = parse_feedback(text)
    assert len(feedback) == 1
    assert feedback[0].useful is None
    assert "both are true" in feedback[0].note


# ── Age helpers ───────────────────────────────────────────────────────


def test_age_days():
    today = datetime.now().strftime("%Y-%m-%d")
    three_weeks = (datetime.now() - timedelta(days=21)).strftime("%Y-%m-%d")
    assert _age_days(today) == 0
    assert _age_days(three_weeks) == 21
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


# ── Task review rendering ─────────────────────────────────────────────


def test_render_task_review_buckets():
    twelve_days = (datetime.now() - timedelta(days=12)).strftime("%Y-%m-%d")
    three_weeks = (datetime.now() - timedelta(days=21)).strftime("%Y-%m-%d")
    two_months = (datetime.now() - timedelta(days=62)).strftime("%Y-%m-%d")

    tasks = [
        {
            "block_id": "dendr-early",
            "source_date": twelve_days,
            "text": "[ ] Fresh-ish task",
        },
        {
            "block_id": "dendr-mid",
            "source_date": three_weeks,
            "text": "[ ] Middle task",
        },
        {
            "block_id": "dendr-old",
            "source_date": two_months,
            "text": "[ ] Ancient task",
        },
    ]

    rendered = _render_task_review(tasks)
    assert "Task Review (3 open" in rendered
    # All three age buckets present, oldest first
    assert "### 1m+ old" in rendered
    assert "### 2-4w old" in rendered
    assert "### 1-2w old" in rendered
    assert rendered.index("1m+ old") < rendered.index("2-4w old")
    assert rendered.index("2-4w old") < rendered.index("1-2w old")
    # Every task has a closure marker with its block_id
    assert "<!-- closure:dendr-early status:open -->" in rendered
    assert "<!-- closure:dendr-mid status:open -->" in rendered
    assert "<!-- closure:dendr-old status:open -->" in rendered
    assert "- [ ] **Fresh-ish task**" in rendered


def test_local_digest_renders_task_review_for_carried_forward():
    three_weeks = (datetime.now() - timedelta(days=21)).strftime("%Y-%m-%d")
    today = datetime.now().strftime("%Y-%m-%d")
    data = {
        "generated_at": datetime.now().isoformat(),
        "period_start": (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d"),
        "period_end": today,
        "stats": {"blocks": 1, "open_tasks": 1},
        "this_period": {"blocks": [], "new_open_tasks": []},
        "carried_forward": {
            "open_tasks": [
                {
                    "block_id": "dendr-stale-1",
                    "source_date": three_weeks,
                    "age_days": 21,
                    "text": "[ ] Old unresolved thing",
                    "checkbox_state": CHECKBOX_OPEN,
                    "completion_status": None,
                }
            ],
        },
        "section_effectiveness": {},
    }
    result = render_local_digest(data)
    assert "## Task Review" in result
    assert "Old unresolved thing" in result
    assert "<!-- closure:dendr-stale-1 status:open -->" in result
    # No Open Loops header since only stale tasks
    assert "## Open Loops" not in result
    assert "feedback:task-review" in result


# ── _block_to_dict / DB roundtrip ─────────────────────────────────────


def test_block_to_dict_computes_age_days():
    f = tempfile.NamedTemporaryFile(suffix=".sqlite", delete=False)
    f.close()
    conn = connect(Path(f.name))
    init_schema(conn)

    eight_days_ago = (datetime.now() - timedelta(days=8)).strftime("%Y-%m-%d")
    upsert_block(
        conn,
        Block(
            block_id="test-age",
            source_file="Daily/x.md",
            line_start=0,
            line_end=0,
            text="[ ] something",
            block_hash="h",
            checkbox_state=CHECKBOX_OPEN,
        ),
        eight_days_ago,
    )

    row = conn.execute(
        "SELECT * FROM blocks WHERE block_id = ?", ("test-age",)
    ).fetchone()

    d = _block_to_dict(row)
    assert d["age_days"] == 8
    assert d["source_date"] == eight_days_ago
    assert d["checkbox_state"] == CHECKBOX_OPEN
    assert d["completion_status"] is None


# ── User context ──────────────────────────────────────────────────────


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
        "stats": {"blocks": 0, "open_tasks": 0},
        "user_context": "Senior engineer working on Dendr. No kids, live alone.",
        "this_period": {"blocks": [], "new_open_tasks": []},
        "carried_forward": {"open_tasks": []},
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
        "stats": {"blocks": 0, "open_tasks": 0},
        "this_period": {"blocks": [], "new_open_tasks": []},
        "carried_forward": {"open_tasks": []},
        "section_effectiveness": {},
    }
    prompt = build_synthesis_prompt(data)
    assert "_user_context.md" in prompt


def test_gather_digest_data_splits_new_vs_carried_forward(tmp_path):
    """_gather_digest_data sorts open tasks into this_period vs carried_forward."""
    from dendr.config import Config
    from dendr.digest import _gather_digest_data

    config = Config(vault_path=tmp_path, data_dir=tmp_path / "data")
    config.wiki_dir.mkdir(parents=True, exist_ok=True)

    conn = _temp_db()

    two_days_ago = (datetime.now() - timedelta(days=2)).strftime("%Y-%m-%d")
    upsert_block(
        conn,
        Block(
            block_id="new-task",
            source_file="Daily/x.md",
            line_start=0,
            line_end=0,
            text="[ ] new task",
            block_hash="h1",
            checkbox_state=CHECKBOX_OPEN,
        ),
        two_days_ago,
    )

    three_weeks_ago = (datetime.now() - timedelta(days=21)).strftime("%Y-%m-%d")
    upsert_block(
        conn,
        Block(
            block_id="old-task",
            source_file="Daily/x.md",
            line_start=0,
            line_end=0,
            text="[ ] old task",
            block_hash="h2",
            checkbox_state=CHECKBOX_OPEN,
        ),
        three_weeks_ago,
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
        SectionFeedback(section="narrative", useful=False, note=""),
    ]

    stats = ingest_feedback(conn, feedback, "2026-04-10")
    assert stats["logged_ratings"] == 2

    fb_rows = conn.execute("SELECT * FROM feedback_scores").fetchall()
    assert len(fb_rows) == 2


# ── Prior-digest archive tests ─────────────────────────────────────────


def test_archive_and_load_prior_digests_round_trip(tmp_path):
    """_archive_digest copies to Wiki/digests/ and load_prior_digests reads it back."""
    from dendr.config import Config
    from dendr.digest import _archive_digest, load_prior_digests

    config = Config(vault_path=tmp_path, data_dir=tmp_path / "data")
    config.wiki_dir.mkdir(parents=True, exist_ok=True)

    digest_path = config.wiki_dir / "digest.md"
    digest_path.write_text(
        "---\ntype: digest\ngenerated: 2026-04-10T09:00:00\n---\n\n"
        "# Weekly Digest\n\nSome content.\n",
        encoding="utf-8",
    )

    _archive_digest(config, digest_path)

    # 2026-04-10 falls in ISO week 15 of 2026
    archived = config.digests_archive_dir / "2026-W15.md"
    assert archived.exists()
    assert "Some content." in archived.read_text(encoding="utf-8")

    prior = load_prior_digests(config)
    assert len(prior) == 1
    assert prior[0]["iso_week"] == "2026-W15"
    assert "Some content." in prior[0]["content"]


def test_archive_digest_noop_when_missing(tmp_path):
    """_archive_digest silently does nothing if digest.md doesn't exist yet."""
    from dendr.config import Config
    from dendr.digest import _archive_digest

    config = Config(vault_path=tmp_path, data_dir=tmp_path / "data")
    config.wiki_dir.mkdir(parents=True, exist_ok=True)

    digest_path = config.wiki_dir / "digest.md"  # doesn't exist
    _archive_digest(config, digest_path)  # must not raise

    assert not config.digests_archive_dir.exists() or not any(
        config.digests_archive_dir.iterdir()
    )


def test_load_prior_digests_returns_newest_first(tmp_path):
    """load_prior_digests sorts archive files descending and caps at n."""
    from dendr.config import Config
    from dendr.digest import load_prior_digests

    config = Config(vault_path=tmp_path, data_dir=tmp_path / "data")
    config.digests_archive_dir.mkdir(parents=True, exist_ok=True)

    for week in ["2026-W12", "2026-W13", "2026-W14", "2026-W15", "2026-W16"]:
        (config.digests_archive_dir / f"{week}.md").write_text(
            f"digest for {week}", encoding="utf-8"
        )

    prior = load_prior_digests(config, n=3)
    assert [p["iso_week"] for p in prior] == ["2026-W16", "2026-W15", "2026-W14"]
