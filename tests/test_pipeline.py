"""Tests for pipeline helpers: closure reconciliation + sticky-status rule."""

from __future__ import annotations

import tempfile
from pathlib import Path

from dendr.config import Config
from dendr.db import connect, init_schema, upsert_block_annotation
from dendr.models import BlockAnnotation, BlockType
from dendr.pipeline import _track_task_lifecycle, reconcile_closures


def _temp_db():
    f = tempfile.NamedTemporaryFile(suffix=".sqlite", delete=False)
    f.close()
    conn = connect(Path(f.name))
    init_schema(conn)
    return conn


def _temp_vault(digest_text: str | None = None) -> Config:
    tmp = Path(tempfile.mkdtemp(prefix="dendr-test-"))
    wiki_dir = tmp / "Wiki"
    wiki_dir.mkdir(parents=True)
    (tmp / "Daily").mkdir()
    (tmp / "Attachments").mkdir()
    if digest_text is not None:
        (wiki_dir / "digest.md").write_text(digest_text, encoding="utf-8")

    data_dir = tmp / ".data"
    data_dir.mkdir()
    return Config(vault_path=tmp, data_dir=data_dir)


def _make_task(block_id: str, status: str = "open") -> BlockAnnotation:
    return BlockAnnotation(
        block_id=block_id,
        source_file="Daily/2026-04-01.md",
        source_date="2026-04-01",
        original_text=f"- [ ] task for {block_id}",
        gist=f"task {block_id}",
        block_type=BlockType.TASK,
        life_areas=["work"],
        completion_status=status,
    )


# ── preserve-closed-status rule ───────────────────────────────────────


def test_track_task_lifecycle_preserves_done_on_reopen():
    """A user-closed task must not be reopened by a re-annotation."""
    conn = _temp_db()
    # Existing annotation: done
    existing = _make_task("dendr-a", status="done")
    upsert_block_annotation(conn, existing)

    # Incoming annotation from the tagger says it's open again
    incoming = _make_task("dendr-a", status="open")
    _track_task_lifecycle(conn, incoming)

    # The sticky rule must have mutated the incoming annotation
    assert incoming.completion_status == "done"


def test_track_task_lifecycle_preserves_abandoned():
    conn = _temp_db()
    upsert_block_annotation(conn, _make_task("dendr-b", status="abandoned"))
    incoming = _make_task("dendr-b", status=None)
    _track_task_lifecycle(conn, incoming)
    assert incoming.completion_status == "abandoned"


def test_track_task_lifecycle_allows_auto_done_transition():
    """If the user genuinely wrote - [x] in their daily note, close it."""
    conn = _temp_db()
    upsert_block_annotation(conn, _make_task("dendr-c", status="open"))
    incoming = _make_task("dendr-c", status="done")
    _track_task_lifecycle(conn, incoming)
    assert incoming.completion_status == "done"

    # Event logged
    events = conn.execute(
        "SELECT event_type FROM task_events WHERE block_id = ?", ("dendr-c",)
    ).fetchall()
    assert any(e["event_type"] == "completed" for e in events)


def test_track_task_lifecycle_logs_created_for_new_tasks():
    conn = _temp_db()
    incoming = _make_task("dendr-d", status="open")
    _track_task_lifecycle(conn, incoming)
    events = conn.execute(
        "SELECT event_type FROM task_events WHERE block_id = ?", ("dendr-d",)
    ).fetchall()
    assert len(events) == 1
    assert events[0]["event_type"] == "created"


# ── reconcile_closures ────────────────────────────────────────────────


def test_reconcile_closures_applies_done():
    digest = """# Weekly Digest

## Task Review

- [x] **Do the thing** — *written 3w ago* <!-- closure:dendr-apply-done status:open -->
"""
    config = _temp_vault(digest_text=digest)
    conn = _temp_db()
    upsert_block_annotation(conn, _make_task("dendr-apply-done", status="open"))

    applied = reconcile_closures(config, conn)
    assert applied == 1

    row = conn.execute(
        "SELECT completion_status FROM block_annotations WHERE block_id = ?",
        ("dendr-apply-done",),
    ).fetchone()
    assert row["completion_status"] == "done"

    ev = conn.execute(
        "SELECT event_type, source FROM task_events WHERE block_id = ?",
        ("dendr-apply-done",),
    ).fetchone()
    assert ev["event_type"] == "completed"
    assert ev["source"] == "user"


def test_reconcile_closures_abandoned_via_explicit_status():
    digest = (
        """- [ ] **Plan** — *5w ago* <!-- closure:dendr-abandon status:abandoned -->"""
    )
    config = _temp_vault(digest_text=digest)
    conn = _temp_db()
    upsert_block_annotation(conn, _make_task("dendr-abandon", status="open"))

    applied = reconcile_closures(config, conn)
    assert applied == 1

    row = conn.execute(
        "SELECT completion_status FROM block_annotations WHERE block_id = ?",
        ("dendr-abandon",),
    ).fetchone()
    assert row["completion_status"] == "abandoned"


def test_reconcile_closures_still_live_resets_to_open():
    digest = """- [ ] **Nope, keep** — *6w ago* <!-- closure:dendr-keep status:still-live -->"""
    config = _temp_vault(digest_text=digest)
    conn = _temp_db()
    # Task was previously closed as abandoned by mistake; user says no, keep it
    upsert_block_annotation(conn, _make_task("dendr-keep", status="abandoned"))

    applied = reconcile_closures(config, conn)
    assert applied == 1
    row = conn.execute(
        "SELECT completion_status FROM block_annotations WHERE block_id = ?",
        ("dendr-keep",),
    ).fetchone()
    assert row["completion_status"] == "open"


def test_reconcile_closures_noop_when_already_matches():
    digest = """- [ ] **A** — *3w ago* <!-- closure:dendr-e status:open -->"""
    config = _temp_vault(digest_text=digest)
    conn = _temp_db()
    upsert_block_annotation(conn, _make_task("dendr-e", status="open"))
    applied = reconcile_closures(config, conn)
    assert applied == 0


def test_reconcile_closures_no_digest_file():
    config = _temp_vault(digest_text=None)
    conn = _temp_db()
    assert reconcile_closures(config, conn) == 0


def test_reconcile_closures_missing_block():
    """Closure for a block_id that doesn't exist in the DB is ignored."""
    digest = """- [x] **Ghost** <!-- closure:dendr-missing status:open -->"""
    config = _temp_vault(digest_text=digest)
    conn = _temp_db()
    # No annotation inserted
    assert reconcile_closures(config, conn) == 0
