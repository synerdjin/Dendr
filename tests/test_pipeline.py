"""Tests for pipeline helpers: closure reconciliation + checkbox transitions."""

from __future__ import annotations

import tempfile
from pathlib import Path

from dendr.config import Config
from dendr.db import (
    connect,
    get_block,
    init_schema,
    update_completion_status,
    upsert_block,
)
from dendr.models import (
    CHECKBOX_CLOSED,
    CHECKBOX_NONE,
    CHECKBOX_OPEN,
    Block,
    QueueItem,
)
from dendr.pipeline import _track_checkbox_transition, reconcile_closures


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


def _make_task_block(block_id: str, checkbox: str = CHECKBOX_OPEN) -> Block:
    return Block(
        block_id=block_id,
        source_file="Daily/2026-04-01.md",
        line_start=0,
        line_end=0,
        text=f"[{'x' if checkbox == CHECKBOX_CLOSED else ' '}] task for {block_id}",
        block_hash=f"h-{block_id}",
        checkbox_state=checkbox,
    )


def _make_queue_item(block_id: str, checkbox: str = CHECKBOX_OPEN) -> QueueItem:
    return QueueItem(
        block_id=block_id,
        source_file="Daily/2026-04-01.md",
        block_hash=f"h-{block_id}",
        block_text="[ ] something",
        checkbox_state=checkbox,
    )


# ── Checkbox transitions log the right task_events ────────────────────


def test_new_open_task_logs_created():
    conn = _temp_db()
    _track_checkbox_transition(conn, _make_queue_item("t1", CHECKBOX_OPEN))

    rows = conn.execute(
        "SELECT event_type, reason FROM task_events WHERE block_id = ?", ("t1",)
    ).fetchall()
    assert len(rows) == 1
    assert rows[0]["event_type"] == "created"


def test_new_closed_task_logs_created_and_closed():
    conn = _temp_db()
    _track_checkbox_transition(conn, _make_queue_item("t1", CHECKBOX_CLOSED))

    rows = conn.execute(
        "SELECT event_type, reason FROM task_events WHERE block_id = ? ORDER BY id",
        ("t1",),
    ).fetchall()
    assert [r["event_type"] for r in rows] == ["created", "closed"]
    assert rows[1]["reason"] == "done"


def test_checkbox_open_to_closed_logs_closed():
    conn = _temp_db()
    upsert_block(conn, _make_task_block("t1", CHECKBOX_OPEN), "2026-04-01")
    _track_checkbox_transition(conn, _make_queue_item("t1", CHECKBOX_CLOSED))

    rows = conn.execute(
        "SELECT event_type, reason FROM task_events WHERE block_id = ? ORDER BY id",
        ("t1",),
    ).fetchall()
    assert len(rows) == 1
    assert rows[0]["event_type"] == "closed"
    assert rows[0]["reason"] == "done"


def test_checkbox_unchanged_logs_nothing():
    conn = _temp_db()
    upsert_block(conn, _make_task_block("t1", CHECKBOX_OPEN), "2026-04-01")
    _track_checkbox_transition(conn, _make_queue_item("t1", CHECKBOX_OPEN))
    rows = conn.execute(
        "SELECT event_type FROM task_events WHERE block_id = ?", ("t1",)
    ).fetchall()
    assert rows == []


def test_non_task_block_logs_nothing():
    conn = _temp_db()
    _track_checkbox_transition(conn, _make_queue_item("t1", CHECKBOX_NONE))
    rows = conn.execute("SELECT COUNT(*) as n FROM task_events").fetchone()
    assert rows["n"] == 0


# ── reconcile_closures ────────────────────────────────────────────────


def test_reconcile_closures_applies_done():
    digest = """# Weekly Digest

## Task Review

- [x] **Do the thing** — *written 3w ago* <!-- closure:dendr-apply-done status:open -->
"""
    config = _temp_vault(digest_text=digest)
    conn = _temp_db()
    upsert_block(conn, _make_task_block("dendr-apply-done"), "2026-04-01")

    applied = reconcile_closures(config, conn)
    assert applied == 1

    row = get_block(conn, "dendr-apply-done")
    assert row["completion_status"] == "done"

    ev = conn.execute(
        "SELECT event_type, reason, source FROM task_events WHERE block_id = ?",
        ("dendr-apply-done",),
    ).fetchone()
    assert ev["event_type"] == "closed"
    assert ev["reason"] == "done"
    assert ev["source"] == "user"


def test_reconcile_closures_abandoned():
    digest = (
        """- [ ] **Plan** — *5w ago* <!-- closure:dendr-abandon status:abandoned -->"""
    )
    config = _temp_vault(digest_text=digest)
    conn = _temp_db()
    upsert_block(conn, _make_task_block("dendr-abandon"), "2026-04-01")

    applied = reconcile_closures(config, conn)
    assert applied == 1

    row = get_block(conn, "dendr-abandon")
    assert row["completion_status"] == "abandoned"


def test_reconcile_closures_still_live_reopens():
    digest = """- [ ] **Nope, keep** — *6w ago* <!-- closure:dendr-keep status:still-live -->"""
    config = _temp_vault(digest_text=digest)
    conn = _temp_db()
    upsert_block(conn, _make_task_block("dendr-keep"), "2026-04-01")
    update_completion_status(conn, "dendr-keep", "abandoned")

    applied = reconcile_closures(config, conn)
    assert applied == 1
    row = get_block(conn, "dendr-keep")
    assert row["completion_status"] == "open"

    ev = conn.execute(
        "SELECT event_type, reason FROM task_events WHERE block_id = ?",
        ("dendr-keep",),
    ).fetchone()
    assert ev["event_type"] == "created"
    assert ev["reason"] == "reopened"


def test_reconcile_closures_noop_when_already_matches():
    digest = """- [ ] **A** — *3w ago* <!-- closure:dendr-e status:open -->"""
    config = _temp_vault(digest_text=digest)
    conn = _temp_db()
    upsert_block(conn, _make_task_block("dendr-e"), "2026-04-01")
    update_completion_status(conn, "dendr-e", "open")
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
    assert reconcile_closures(config, conn) == 0
