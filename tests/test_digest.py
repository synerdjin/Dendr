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
from dendr.digest import (
    _gather_digest_data,
    build_synthesis_prompt,
    render_local_digest,
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
