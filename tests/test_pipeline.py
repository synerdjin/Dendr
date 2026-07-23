"""Tests for pipeline helpers: closure reconciliation + checkbox transitions."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import numpy as np

from dendr.config import Config
from dendr.db import (
    connect,
    get_block,
    get_open_tasks,
    init_schema,
    set_snooze,
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
from dendr.pipeline import _track_checkbox_transition, reconcile_closures, run_ingest


SOURCE_DATE = "2026-04-01"


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
    _track_checkbox_transition(
        conn, _make_queue_item("t1", CHECKBOX_OPEN), SOURCE_DATE, None
    )

    rows = conn.execute(
        "SELECT event_type, reason FROM task_events WHERE block_id = ?", ("t1",)
    ).fetchall()
    assert len(rows) == 1
    assert rows[0]["event_type"] == "created"


def test_new_closed_task_logs_created_and_closed():
    conn = _temp_db()
    _track_checkbox_transition(
        conn, _make_queue_item("t1", CHECKBOX_CLOSED), SOURCE_DATE, None
    )

    rows = conn.execute(
        "SELECT event_type, reason FROM task_events WHERE block_id = ? ORDER BY id",
        ("t1",),
    ).fetchall()
    assert [r["event_type"] for r in rows] == ["created", "closed"]
    assert rows[1]["reason"] == "done"


def test_checkbox_open_to_closed_logs_closed():
    conn = _temp_db()
    upsert_block(conn, _make_task_block("t1", CHECKBOX_OPEN), SOURCE_DATE)
    existing = get_block(conn, "t1")
    _track_checkbox_transition(
        conn, _make_queue_item("t1", CHECKBOX_CLOSED), SOURCE_DATE, existing
    )

    rows = conn.execute(
        "SELECT event_type, reason FROM task_events WHERE block_id = ? ORDER BY id",
        ("t1",),
    ).fetchall()
    assert len(rows) == 1
    assert rows[0]["event_type"] == "closed"
    assert rows[0]["reason"] == "done"


def test_checkbox_unchanged_logs_nothing():
    conn = _temp_db()
    upsert_block(conn, _make_task_block("t1", CHECKBOX_OPEN), SOURCE_DATE)
    existing = get_block(conn, "t1")
    _track_checkbox_transition(
        conn, _make_queue_item("t1", CHECKBOX_OPEN), SOURCE_DATE, existing
    )
    rows = conn.execute(
        "SELECT event_type FROM task_events WHERE block_id = ?", ("t1",)
    ).fetchall()
    assert rows == []


def test_non_task_block_logs_nothing():
    conn = _temp_db()
    _track_checkbox_transition(
        conn, _make_queue_item("t1", CHECKBOX_NONE), SOURCE_DATE, None
    )
    rows = conn.execute("SELECT COUNT(*) as n FROM task_events").fetchone()
    assert rows["n"] == 0


def test_none_to_open_logs_created():
    """Regression (F11): adding a checkbox to a plain block (none->open) is a
    task creation and must be recorded."""
    conn = _temp_db()
    upsert_block(conn, _make_task_block("t1", CHECKBOX_NONE), SOURCE_DATE)
    existing = get_block(conn, "t1")
    _track_checkbox_transition(
        conn, _make_queue_item("t1", CHECKBOX_OPEN), SOURCE_DATE, existing
    )
    rows = conn.execute(
        "SELECT event_type, reason FROM task_events WHERE block_id = ? ORDER BY id",
        ("t1",),
    ).fetchall()
    assert [r["event_type"] for r in rows] == ["created"]
    assert rows[0]["reason"] is None  # a plain creation, not a reopen


def test_none_to_closed_logs_created_and_closed():
    """Regression (F11): a plain block turned straight into a done task
    (none->closed) is both a creation and a close."""
    conn = _temp_db()
    upsert_block(conn, _make_task_block("t1", CHECKBOX_NONE), SOURCE_DATE)
    existing = get_block(conn, "t1")
    _track_checkbox_transition(
        conn, _make_queue_item("t1", CHECKBOX_CLOSED), SOURCE_DATE, existing
    )
    rows = conn.execute(
        "SELECT event_type, reason FROM task_events WHERE block_id = ? ORDER BY id",
        ("t1",),
    ).fetchall()
    assert [r["event_type"] for r in rows] == ["created", "closed"]
    assert rows[1]["reason"] == "done"


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


def test_reconcile_closures_writes_back_to_source():
    """Closing a task in the digest ticks the checkbox in its daily note."""
    config = _temp_vault()
    note = config.daily_dir / "2026-04-01.md"
    note.write_text("- [ ] ship it ^dendr-wb\n", encoding="utf-8")
    (config.wiki_dir / "digest.md").write_text(
        "- [x] **Ship it** <!-- closure:dendr-wb status:open -->",
        encoding="utf-8",
    )

    conn = _temp_db()
    block = _make_task_block("dendr-wb")
    block.source_file = str(note)
    upsert_block(conn, block, "2026-04-01")

    assert reconcile_closures(config, conn) == 1
    assert get_block(conn, "dendr-wb")["completion_status"] == "done"

    text = note.read_text()
    assert text.startswith("- [x] ship it ✅ ")
    assert text.rstrip().endswith("^dendr-wb")


def test_reconcile_closures_abandoned_cancels_in_source():
    config = _temp_vault()
    note = config.daily_dir / "2026-04-01.md"
    note.write_text("- [ ] drop it ^dendr-ab\n", encoding="utf-8")
    (config.wiki_dir / "digest.md").write_text(
        "- [ ] **Drop it** <!-- closure:dendr-ab status:abandoned -->",
        encoding="utf-8",
    )

    conn = _temp_db()
    block = _make_task_block("dendr-ab")
    block.source_file = str(note)
    upsert_block(conn, block, "2026-04-01")

    assert reconcile_closures(config, conn) == 1
    assert note.read_text().startswith("- [-] drop it ❌ ")


def test_reconcile_closures_snoozed_leaves_source_open():
    config = _temp_vault()
    note = config.daily_dir / "2026-04-01.md"
    note.write_text("- [ ] later ^dendr-sn\n", encoding="utf-8")
    (config.wiki_dir / "digest.md").write_text(
        "- [ ] **Later** <!-- closure:dendr-sn status:snoozed -->",
        encoding="utf-8",
    )

    conn = _temp_db()
    block = _make_task_block("dendr-sn")
    block.source_file = str(note)
    upsert_block(conn, block, "2026-04-01")

    assert reconcile_closures(config, conn) == 1
    assert note.read_text() == "- [ ] later ^dendr-sn\n"  # untouched


# -- Snooze wake (F5) --


def test_snooze_sets_default_wake_a_week_out():
    from datetime import datetime, timedelta

    config = _temp_vault(
        digest_text="- [ ] **Later** <!-- closure:dendr-sn status:snoozed -->"
    )
    conn = _temp_db()
    upsert_block(conn, _make_task_block("dendr-sn"), "2026-04-01")

    assert reconcile_closures(config, conn) == 1
    row = get_block(conn, "dendr-sn")
    assert row["completion_status"] == "snoozed"
    expected = (datetime.now() + timedelta(days=7)).strftime("%Y-%m-%d")
    assert row["snooze_until"] == expected
    assert "dendr-sn" not in [r["block_id"] for r in get_open_tasks(conn)]


def test_snooze_honours_explicit_until_date():
    config = _temp_vault(
        digest_text=(
            "- [ ] **Later** <!-- closure:dendr-sn status:snoozed until:2099-01-15 -->"
        )
    )
    conn = _temp_db()
    upsert_block(conn, _make_task_block("dendr-sn"), "2026-04-01")

    assert reconcile_closures(config, conn) == 1
    assert get_block(conn, "dendr-sn")["snooze_until"] == "2099-01-15"


def test_snooze_reconcile_is_idempotent():
    config = _temp_vault(
        digest_text=(
            "- [ ] **Later** <!-- closure:dendr-sn status:snoozed until:2099-01-15 -->"
        )
    )
    conn = _temp_db()
    upsert_block(conn, _make_task_block("dendr-sn"), "2026-04-01")

    assert reconcile_closures(config, conn) == 1
    assert reconcile_closures(config, conn) == 0  # same date, nothing to do


def test_due_snoozed_task_wakes_and_reappears():
    from dendr.pipeline import wake_snoozed_tasks

    conn = _temp_db()
    upsert_block(conn, _make_task_block("dendr-sn"), "2026-04-01")
    set_snooze(conn, "dendr-sn", "2000-01-01")
    assert "dendr-sn" not in [r["block_id"] for r in get_open_tasks(conn)]

    assert wake_snoozed_tasks(conn) == 1
    row = get_block(conn, "dendr-sn")
    assert row["completion_status"] is None
    assert row["snooze_until"] is None
    assert "dendr-sn" in [r["block_id"] for r in get_open_tasks(conn)]

    ev = conn.execute(
        "SELECT event_type, reason FROM task_events WHERE block_id = ? ORDER BY id DESC",
        ("dendr-sn",),
    ).fetchone()
    assert ev["event_type"] == "created"
    assert ev["reason"] == "woke"


def test_future_snooze_does_not_wake():
    from dendr.pipeline import wake_snoozed_tasks

    conn = _temp_db()
    upsert_block(conn, _make_task_block("dendr-sn"), "2026-04-01")
    set_snooze(conn, "dendr-sn", "2099-01-01")
    assert wake_snoozed_tasks(conn) == 0
    assert get_block(conn, "dendr-sn")["completion_status"] == "snoozed"


def test_expired_snooze_marker_does_not_refire_after_wake():
    """A stale `snoozed until:<past>` marker left in the digest must not
    re-snooze a task the wake step already resurfaced (no flip-flop)."""
    from dendr.pipeline import wake_snoozed_tasks

    config = _temp_vault(
        digest_text=(
            "- [ ] **Later** <!-- closure:dendr-sn status:snoozed until:2000-01-01 -->"
        )
    )
    conn = _temp_db()
    upsert_block(conn, _make_task_block("dendr-sn"), "2026-04-01")
    set_snooze(conn, "dendr-sn", "2000-01-01")

    assert wake_snoozed_tasks(conn) == 1
    assert reconcile_closures(config, conn) == 0
    assert get_block(conn, "dendr-sn")["completion_status"] is None


def test_transition_suppressed_when_user_already_closed():
    """The source-flip echo from a digest close isn't double-logged as auto."""
    conn = _temp_db()
    upsert_block(conn, _make_task_block("t1", CHECKBOX_OPEN), SOURCE_DATE)
    update_completion_status(conn, "t1", "done")
    existing = get_block(conn, "t1")

    _track_checkbox_transition(
        conn, _make_queue_item("t1", CHECKBOX_CLOSED), SOURCE_DATE, existing
    )
    rows = conn.execute(
        "SELECT event_type FROM task_events WHERE block_id = ?", ("t1",)
    ).fetchall()
    assert rows == []


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


# ── Reopening a user-closed task in the source note ───────────────────


class _StubLLM:
    """Deterministic embeddings so run_ingest works without model weights."""

    def embed(self, text, kind="document"):
        return np.zeros(768, dtype=np.float32)

    def embed_batch(self, texts, kind="document"):
        return [self.embed(t) for t in texts]


def _close_via_digest(config: Config, conn, note: Path, block_id: str) -> None:
    """Drive the full digest-close flow: marker → reconcile → write-back echo."""
    (config.wiki_dir / "digest.md").write_text(
        f"- [x] **task** <!-- closure:{block_id} status:open -->\n",
        encoding="utf-8",
    )
    run_ingest(config, conn, _StubLLM())
    row = get_block(conn, block_id)
    assert row["completion_status"] == "done"
    assert note.read_text().startswith("- [x]")


def test_reopen_in_source_after_digest_close_restores_open_task():
    """Unchecking the source checkbox after a digest close must make the task
    queryable as open again — not leave completion_status stuck at 'done'."""
    config = _temp_vault()
    config.ensure_dirs()
    conn = _temp_db()

    note = config.daily_dir / "2026-04-01.md"
    note.write_text("- [ ] ship the report ^dendr-ro\n", encoding="utf-8")
    run_ingest(config, conn, _StubLLM())

    _close_via_digest(config, conn, note, "dendr-ro")
    (config.wiki_dir / "digest.md").unlink()

    # User reopens the task in the daily note.
    note.write_text("- [ ] ship the report ^dendr-ro\n", encoding="utf-8")
    run_ingest(config, conn, _StubLLM())

    row = get_block(conn, "dendr-ro")
    assert row["checkbox_state"] == "open"
    assert row["completion_status"] is None
    assert [r["block_id"] for r in get_open_tasks(conn)] == ["dendr-ro"]

    ev = conn.execute(
        "SELECT event_type, reason, source FROM task_events "
        "WHERE block_id = ? ORDER BY id DESC LIMIT 1",
        ("dendr-ro",),
    ).fetchone()
    assert ev["event_type"] == "created"
    assert ev["reason"] == "reopened"
    assert ev["source"] == "auto"


def test_stale_digest_marker_does_not_reclose_reopened_task():
    """A closure marker left in digest.md must not fight a later source reopen:
    subsequent ingests may not re-close the task or re-tick the checkbox."""
    config = _temp_vault()
    config.ensure_dirs()
    conn = _temp_db()

    note = config.daily_dir / "2026-04-01.md"
    note.write_text("- [ ] ship the report ^dendr-st\n", encoding="utf-8")
    run_ingest(config, conn, _StubLLM())

    _close_via_digest(config, conn, note, "dendr-st")

    # Digest (with its marker) predates the reopen; make mtime reflect that.
    digest_path = config.wiki_dir / "digest.md"
    old = digest_path.stat().st_mtime - 3600
    os.utime(digest_path, (old, old))

    # User reopens in the daily note; the stale marker is still in digest.md.
    note.write_text("- [ ] ship the report ^dendr-st\n", encoding="utf-8")
    run_ingest(config, conn, _StubLLM())
    run_ingest(config, conn, _StubLLM())

    row = get_block(conn, "dendr-st")
    assert row["checkbox_state"] == "open"
    assert row["completion_status"] is None
    assert note.read_text().startswith("- [ ] ")
    assert [r["block_id"] for r in get_open_tasks(conn)] == ["dendr-st"]


def test_fresh_digest_edit_after_reopen_can_reclose():
    """A digest edit newer than the reopen is a genuine user decision and
    must still apply."""
    config = _temp_vault()
    config.ensure_dirs()
    conn = _temp_db()

    note = config.daily_dir / "2026-04-01.md"
    note.write_text("- [ ] ship the report ^dendr-rc\n", encoding="utf-8")
    run_ingest(config, conn, _StubLLM())

    _close_via_digest(config, conn, note, "dendr-rc")

    note.write_text("- [ ] ship the report ^dendr-rc\n", encoding="utf-8")
    run_ingest(config, conn, _StubLLM())
    assert get_block(conn, "dendr-rc")["completion_status"] is None

    # User re-closes by editing the digest again (mtime now after the reopen).
    digest_path = config.wiki_dir / "digest.md"
    new = digest_path.stat().st_mtime + 3600
    os.utime(digest_path, (new, new))
    run_ingest(config, conn, _StubLLM())

    assert get_block(conn, "dendr-rc")["completion_status"] == "done"
    assert note.read_text().startswith("- [x]")


# ── Dead-letter queue ─────────────────────────────────────────────────


def test_mark_dead_moves_item_out_of_processing():
    """Poison items should end up in dead/, not stay in processing/ forever."""
    from dendr import queue

    config = _temp_vault()
    config.ensure_dirs()
    item = _make_queue_item("dendr-poison")
    queue.enqueue(config, item)
    assert queue.claim_for_processing(config, item.block_id)
    queue.mark_dead(config, item.block_id)

    assert not (config.processing_dir / f"{item.block_id}.json").exists()
    assert (config.dead_dir / f"{item.block_id}.json").exists()

    # recover_stale should not pull a dead item back to pending.
    assert queue.recover_stale(config) == 0


def test_dead_lettered_block_not_reenqueued_until_edited():
    """Regression (F9): a poison block's hash is never committed, so the scan
    keeps flagging it dirty. queue_dirty_blocks must not re-enqueue it every
    cycle — but an edit (new hash) clears the record and retries."""
    from dendr import queue
    from dendr.pipeline import queue_dirty_blocks

    config = _temp_vault()
    config.ensure_dirs()

    from dataclasses import replace

    block = Block(
        block_id="dendr-poison",
        source_file="Daily/2026-04-01.md",
        line_start=0,
        line_end=0,
        text="bad block",
        block_hash="h1",
    )
    # First pass enqueues it; simulate a commit failure that dead-letters it.
    assert queue_dirty_blocks(config, [block]) == 1
    assert queue.claim_for_processing(config, "dendr-poison")
    queue.mark_dead(config, "dendr-poison")

    # Next scan still sees it dirty (hash uncommitted) — must NOT re-enqueue.
    assert queue_dirty_blocks(config, [block]) == 0
    assert queue.pending_count(config) == 0

    # The user edits the block: different hash → fresh attempt, record cleared.
    edited = replace(block, text="fixed block", block_hash="h2")
    assert queue_dirty_blocks(config, [edited]) == 1
    assert queue.pending_count(config) == 1
    assert not (config.dead_dir / "dendr-poison.json").exists()


# ── iCloud conflicted-copy filtering (F2) ─────────────────────────────


def test_conflicted_copy_is_not_ingested():
    """An iCloud "<date> 2.md" conflict shares block refs with the canonical
    note; ingesting it would flip-flop the stored text. It must be skipped."""
    from dendr.db import get_block

    config = _temp_vault()
    config.ensure_dirs()
    conn = _temp_db()

    note = config.daily_dir / "2026-04-01.md"
    note.write_text("decided to STAY at my job ^dendr-cc\n", encoding="utf-8")
    # iCloud drops an older-content conflicted copy carrying the same ref.
    conflict = config.daily_dir / "2026-04-01 2.md"
    conflict.write_text("thinking about quitting ^dendr-cc\n", encoding="utf-8")

    run_ingest(config, conn, _StubLLM())
    run_ingest(config, conn, _StubLLM())  # a second cycle must not flip it

    assert get_block(conn, "dendr-cc")["text"] == "decided to STAY at my job"


def test_is_conflicted_copy_patterns():
    from dendr.pipeline import _is_conflicted_copy

    assert _is_conflicted_copy(Path("Daily/2026-04-01 2.md"))
    assert _is_conflicted_copy(Path("Daily/2026-04-01 (1).md"))
    assert _is_conflicted_copy(Path("Daily/2026-04-01 (conflicted copy 2026-04-02).md"))
    # Canonical daily notes are never treated as conflicts.
    assert not _is_conflicted_copy(Path("Daily/2026-04-01.md"))


def test_conflict_shaped_name_with_unique_refs_is_still_ingested():
    """The skip is keyed on a genuine block-ref clash, not the filename alone:
    a conflict-shaped name whose refs are unique must still be ingested."""
    from dendr.db import get_block

    config = _temp_vault()
    config.ensure_dirs()
    conn = _temp_db()

    canonical = config.daily_dir / "2026-04-01.md"
    canonical.write_text("real note ^dendr-real\n", encoding="utf-8")
    # A "(2)"-shaped name, but with its OWN unique ref — not a conflict copy.
    other = config.daily_dir / "2026-04-01 (2).md"
    other.write_text("a different note ^dendr-other\n", encoding="utf-8")

    run_ingest(config, conn, _StubLLM())

    assert get_block(conn, "dendr-real")["text"] == "real note"
    assert get_block(conn, "dendr-other")["text"] == "a different note"


def test_conflicted_copy_skipped_regardless_of_scan_order():
    """Even when the conflict copy sorts first alphabetically, block-ref
    ownership (not sort position) decides which file wins."""
    from dendr.db import get_block

    config = _temp_vault()
    config.ensure_dirs()
    conn = _temp_db()

    # "<date> 2.md" sorts before "<date>.md" (space < dot), so a naive
    # first-seen-wins would pick the conflict. _scan_order must prevent that.
    canonical = config.daily_dir / "2026-04-01.md"
    canonical.write_text("keep this ^dendr-x\n", encoding="utf-8")
    conflict = config.daily_dir / "2026-04-01 2.md"
    conflict.write_text("stale copy ^dendr-x\n", encoding="utf-8")

    run_ingest(config, conn, _StubLLM())
    assert get_block(conn, "dendr-x")["text"] == "keep this"
