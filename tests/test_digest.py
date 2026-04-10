"""Tests for the weekly digest feature."""

import tempfile
from datetime import datetime, timedelta
from pathlib import Path

from dendr.db import (
    connect,
    init_schema,
    insert_claim,
    get_recent_claims,
    get_open_tasks,
    get_concept_frequencies,
    get_all_contradictions,
    get_dropped_threads,
)
from dendr.db import append_log
from dendr.digest import (
    build_synthesis_prompt,
    get_feedback_history,
    ingest_feedback,
    parse_feedback,
    render_local_digest,
    SectionFeedback,
)
from dendr.models import Claim, ClaimKind, ClaimStatus


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
        kind=ClaimKind.STATEMENT,
    )
    defaults.update(kwargs)
    return Claim(**defaults)


def test_claim_kind_persisted():
    """claim_kind is stored and retrieved correctly."""
    conn = _temp_db()
    claim = _make_claim(kind=ClaimKind.TASK, text="Fix the CI pipeline")
    cid = insert_claim(conn, claim)
    row = conn.execute("SELECT kind FROM claims WHERE id = ?", (cid,)).fetchone()
    assert row["kind"] == "task"


def test_claim_kind_default():
    """Default claim_kind is 'statement'."""
    conn = _temp_db()
    claim = _make_claim()
    cid = insert_claim(conn, claim)
    row = conn.execute("SELECT kind FROM claims WHERE id = ?", (cid,)).fetchone()
    assert row["kind"] == "statement"


def test_get_open_tasks():
    """get_open_tasks returns only task/intention claims."""
    conn = _temp_db()
    insert_claim(conn, _make_claim(kind=ClaimKind.TASK, text="Buy groceries"))
    insert_claim(conn, _make_claim(kind=ClaimKind.INTENTION, text="Learn Rust"))
    insert_claim(conn, _make_claim(kind=ClaimKind.STATEMENT, text="Sky is blue"))
    insert_claim(conn, _make_claim(kind=ClaimKind.QUESTION, text="What is X?"))

    tasks = get_open_tasks(conn)
    assert len(tasks) == 2
    texts = {r["text"] for r in tasks}
    assert "Buy groceries" in texts
    assert "Learn Rust" in texts
    assert "Sky is blue" not in texts


def test_get_recent_claims():
    """get_recent_claims filters by date and excludes private/superseded."""
    conn = _temp_db()
    now = datetime.now()
    old = now - timedelta(days=30)

    insert_claim(conn, _make_claim(text="Recent", created_at=now))
    insert_claim(conn, _make_claim(text="Old", created_at=old))
    insert_claim(conn, _make_claim(text="Private", created_at=now, private=True))

    since = (now - timedelta(days=7)).isoformat()
    recent = get_recent_claims(conn, since)
    texts = [r["text"] for r in recent]
    assert "Recent" in texts
    assert "Old" not in texts
    assert "Private" not in texts


def test_get_concept_frequencies():
    """get_concept_frequencies counts correctly."""
    conn = _temp_db()
    now = datetime.now()
    since = (now - timedelta(days=7)).isoformat()

    for _ in range(3):
        insert_claim(conn, _make_claim(
            concept_slug="rust",
            created_at=now,
            subject=f"s{_}", subject_predicate=f"s{_}|uses",
        ))
    insert_claim(conn, _make_claim(
        concept_slug="python", created_at=now,
        subject="s9", subject_predicate="s9|uses",
    ))

    freq = get_concept_frequencies(conn, since)
    assert freq[0] == ("rust", 3)
    assert freq[1] == ("python", 1)


def test_get_all_contradictions():
    """get_all_contradictions finds conflicting claims."""
    conn = _temp_db()
    insert_claim(conn, _make_claim(
        text="X uses Postgres", object="Postgres",
    ))
    insert_claim(conn, _make_claim(
        text="X uses SQLite", object="SQLite",
    ))

    contras = get_all_contradictions(conn)
    assert len(contras) == 1
    assert contras[0]["claim_a"]["object"] == "Postgres"
    assert contras[0]["claim_b"]["object"] == "SQLite"


def test_get_dropped_threads():
    """get_dropped_threads finds single-mention concepts."""
    conn = _temp_db()
    old = datetime.now() - timedelta(weeks=3)

    insert_claim(conn, _make_claim(
        concept_slug="forgotten-topic", text="Some note about X",
        created_at=old,
    ))
    # This concept has 2 mentions — should not appear
    insert_claim(conn, _make_claim(
        concept_slug="active-topic", text="First mention",
        created_at=old, subject="a1", subject_predicate="a1|is",
    ))
    insert_claim(conn, _make_claim(
        concept_slug="active-topic", text="Second mention",
        subject="a2", subject_predicate="a2|is",
    ))

    before = (datetime.now() - timedelta(weeks=2)).isoformat()
    dropped = get_dropped_threads(conn, before)
    slugs = [r["concept_slug"] for r in dropped]
    assert "forgotten-topic" in slugs
    assert "active-topic" not in slugs


def test_render_local_digest_with_data():
    """render_local_digest produces valid markdown with all sections."""
    data = {
        "generated_at": datetime.now().isoformat(),
        "period_start": (datetime.now() - timedelta(days=7)).isoformat(),
        "period_end": datetime.now().isoformat(),
        "stats": {"active_claims": 42, "concepts": 10, "challenged_claims": 2},
        "recent_claims": [
            {
                "id": 1, "text": "Test claim", "subject": "X",
                "predicate": "uses", "object": "Y", "kind": "statement",
                "concept_slug": "test", "confidence": 0.8,
                "created_at": datetime.now().isoformat(),
                "source_block_ref": "block-1",
            },
        ],
        "open_tasks": [
            {
                "id": 2, "text": "Fix CI", "kind": "task",
                "concept_slug": "ci", "confidence": 0.7,
                "created_at": datetime.now().isoformat(),
                "status": "created",
            },
        ],
        "contradictions": [
            {
                "subject_predicate": "X|uses",
                "claim_a": {"id": 1, "text": "X uses Postgres", "object": "Postgres", "confidence": 0.8, "created_at": datetime.now().isoformat()},
                "claim_b": {"id": 2, "text": "X uses SQLite", "object": "SQLite", "confidence": 0.7, "created_at": datetime.now().isoformat()},
            },
        ],
        "emerging_themes": [
            {"concept": "rust", "mentions": 5},
            {"concept": "python", "mentions": 1},
        ],
        "dropped_threads": [
            {"concept_slug": "forgotten", "text": "Something about X", "created_at": (datetime.now() - timedelta(weeks=3)).isoformat()},
        ],
    }

    result = render_local_digest(data)

    assert "# Weekly Digest" in result
    assert "## Contradictions" in result
    assert "## Open Loops" in result
    assert "## Emerging Themes" in result
    assert "[[rust]]" in result
    # python has only 1 mention, below threshold of 2
    assert "[[python]]" not in result
    assert "## Dropped Threads" in result
    assert "[[forgotten]]" in result
    assert "type: digest" in result


def test_render_local_digest_empty():
    """render_local_digest handles empty data gracefully."""
    data = {
        "generated_at": datetime.now().isoformat(),
        "period_start": (datetime.now() - timedelta(days=7)).isoformat(),
        "period_end": datetime.now().isoformat(),
        "stats": {"active_claims": 0, "concepts": 0, "challenged_claims": 0},
        "recent_claims": [],
        "open_tasks": [],
        "contradictions": [],
        "emerging_themes": [],
        "dropped_threads": [],
    }

    result = render_local_digest(data)
    assert "# Weekly Digest" in result
    assert "No notable insights this week" in result


def test_build_synthesis_prompt():
    """build_synthesis_prompt produces a valid prompt with data."""
    data = {
        "generated_at": datetime.now().isoformat(),
        "period_start": (datetime.now() - timedelta(days=7)).isoformat(),
        "period_end": datetime.now().isoformat(),
        "stats": {"active_claims": 5, "concepts": 2, "challenged_claims": 0},
        "recent_claims": [
            {
                "id": 1, "text": "Test", "subject": "X",
                "predicate": "is", "object": "Y", "kind": "statement",
                "concept_slug": "test", "confidence": 0.9,
                "created_at": datetime.now().isoformat(),
                "source_block_ref": "b1",
            },
        ],
        "open_tasks": [],
        "contradictions": [],
        "emerging_themes": [],
        "dropped_threads": [],
    }

    prompt = build_synthesis_prompt(data)
    assert "weekly digest synthesizer" in prompt
    assert '"text": "Test"' in prompt
    assert "Contradictions" in prompt
    assert "Open Loops" in prompt
    assert "Reframes" in prompt


# --- Feedback tests ---


def test_render_includes_feedback_markers():
    """render_local_digest includes feedback comment blocks in each section."""
    data = {
        "generated_at": datetime.now().isoformat(),
        "period_start": (datetime.now() - timedelta(days=7)).isoformat(),
        "period_end": datetime.now().isoformat(),
        "stats": {"active_claims": 5, "concepts": 2, "challenged_claims": 1},
        "recent_claims": [
            {
                "id": 1, "text": "Claim", "subject": "X",
                "predicate": "is", "object": "Y", "kind": "statement",
                "concept_slug": "test", "confidence": 0.9,
                "created_at": datetime.now().isoformat(),
                "source_block_ref": "b1",
            },
        ],
        "open_tasks": [
            {
                "id": 2, "text": "Do thing", "kind": "task",
                "concept_slug": "stuff", "confidence": 0.7,
                "created_at": datetime.now().isoformat(),
                "status": "created",
            },
        ],
        "contradictions": [
            {
                "subject_predicate": "X|uses",
                "claim_a": {"id": 1, "text": "A", "object": "A", "confidence": 0.8, "created_at": datetime.now().isoformat()},
                "claim_b": {"id": 2, "text": "B", "object": "B", "confidence": 0.7, "created_at": datetime.now().isoformat()},
            },
        ],
        "emerging_themes": [{"concept": "rust", "mentions": 3}],
        "dropped_threads": [
            {"concept_slug": "old", "text": "Old thing", "created_at": (datetime.now() - timedelta(weeks=3)).isoformat()},
        ],
    }

    result = render_local_digest(data)
    assert "<!-- feedback:contradictions" in result
    assert "<!-- feedback:open-loops" in result
    assert "<!-- feedback:emerging-themes" in result
    assert "<!-- feedback:dropped-threads" in result
    assert "<!-- feedback:activity" in result


def test_parse_feedback_filled():
    """parse_feedback extracts user responses from comment blocks."""
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
    assert dt.note == ""


def test_parse_feedback_empty():
    """parse_feedback ignores untouched feedback blocks."""
    text = """## Open Loops
- stuff

<!-- feedback:open-loops
useful:
note:
-->
"""
    feedback = parse_feedback(text)
    assert len(feedback) == 0


def test_parse_feedback_note_only():
    """parse_feedback accepts blocks with just a note (no useful rating)."""
    text = """<!-- feedback:contradictions
useful:
note: both are true actually, different contexts
-->"""
    feedback = parse_feedback(text)
    assert len(feedback) == 1
    assert feedback[0].section == "contradictions"
    assert feedback[0].useful is None
    assert "both are true" in feedback[0].note


def test_ingest_feedback_creates_claims():
    """ingest_feedback stores notes as claims and logs ratings."""
    conn = _temp_db()
    feedback = [
        SectionFeedback(section="open-loops", useful=True, note="CI task is done"),
        SectionFeedback(section="contradictions", useful=False, note=""),
    ]

    stats = ingest_feedback(conn, feedback, "2026-04-10")
    assert stats["logged_ratings"] == 2
    assert stats["ingested_claims"] == 1  # only "CI task is done" has a note

    # Verify the claim was created
    rows = conn.execute(
        "SELECT * FROM claims WHERE source_block_ref LIKE 'digest-feedback%'"
    ).fetchall()
    assert len(rows) == 1
    assert rows[0]["text"] == "CI task is done"
    assert rows[0]["confidence"] == 0.9

    # Verify log entries
    log_rows = conn.execute(
        "SELECT * FROM log WHERE kind = 'digest_feedback'"
    ).fetchall()
    assert len(log_rows) == 2


def test_get_feedback_history():
    """get_feedback_history returns recent feedback log entries."""
    conn = _temp_db()
    append_log(conn, "digest_feedback", {
        "section": "open-loops", "useful": True,
        "note": "good stuff", "digest_date": "2026-04-03",
    })
    append_log(conn, "digest_feedback", {
        "section": "contradictions", "useful": False,
        "note": "", "digest_date": "2026-04-03",
    })
    append_log(conn, "other_event", {"key": "value"})

    history = get_feedback_history(conn)
    assert len(history) == 2
    sections = {h["section"] for h in history}
    assert "open-loops" in sections
    assert "contradictions" in sections


def test_synthesis_prompt_includes_feedback_context():
    """build_synthesis_prompt mentions feedback history when present."""
    data = {
        "generated_at": datetime.now().isoformat(),
        "period_start": (datetime.now() - timedelta(days=7)).isoformat(),
        "period_end": datetime.now().isoformat(),
        "stats": {"active_claims": 1, "concepts": 1, "challenged_claims": 0},
        "recent_claims": [],
        "open_tasks": [],
        "contradictions": [],
        "emerging_themes": [],
        "dropped_threads": [],
        "feedback_history": [
            {"section": "open-loops", "useful": True, "note": "keep these coming"},
        ],
    }

    prompt = build_synthesis_prompt(data)
    assert "feedback_history" in prompt
    assert "keep these coming" in prompt
    assert "deprioritize or skip" in prompt
