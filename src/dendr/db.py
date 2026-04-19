"""Database schema and operations for Dendr's knowledge store.

Tables:
  - blocks: raw block text + structural metadata
  - blocks_fts: FTS5 over raw block text (kept in sync via triggers)
  - blocks_vec: sqlite-vec embeddings over raw block text
  - feedback_scores: per-section digest feedback for tuning
  - task_events: task lifecycle events (created, closed)
"""

from __future__ import annotations

import logging
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

from dendr.models import CHECKBOX_OPEN, COMPLETION_OPEN, Block

logger = logging.getLogger(__name__)


def _load_vec(conn: sqlite3.Connection) -> None:
    """Load sqlite-vec on this connection. Silent no-op if unavailable."""
    try:
        import sqlite_vec
    except ImportError:
        return
    try:
        conn.enable_load_extension(True)
        sqlite_vec.load(conn)
        conn.enable_load_extension(False)
    except sqlite3.OperationalError as e:
        logger.debug("sqlite-vec load failed: %s", e, exc_info=True)


def connect(db_path: Path, *, check_same_thread: bool = True) -> sqlite3.Connection:
    """Open (or create) the Dendr state database."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(
        str(db_path), isolation_level=None, check_same_thread=check_same_thread
    )
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.row_factory = sqlite3.Row
    _load_vec(conn)
    return conn


def init_schema(conn: sqlite3.Connection) -> None:
    """Create all tables if they don't exist."""
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS blocks (
            id                INTEGER PRIMARY KEY AUTOINCREMENT,
            block_id          TEXT NOT NULL UNIQUE,
            source_file       TEXT NOT NULL,
            source_date       TEXT NOT NULL,
            text              TEXT NOT NULL,
            block_hash        TEXT NOT NULL,
            checkbox_state    TEXT NOT NULL DEFAULT 'none',
            completion_status TEXT,
            private           INTEGER NOT NULL DEFAULT 0,
            attachment_path   TEXT,
            attachment_type   TEXT,
            created_at        TEXT NOT NULL,
            updated_at        TEXT NOT NULL
        );

        CREATE INDEX IF NOT EXISTS idx_blocks_date
            ON blocks(source_date);
        CREATE INDEX IF NOT EXISTS idx_blocks_checkbox
            ON blocks(checkbox_state);
        CREATE INDEX IF NOT EXISTS idx_blocks_completion
            ON blocks(completion_status);

        CREATE TABLE IF NOT EXISTS feedback_scores (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            digest_date TEXT NOT NULL,
            section     TEXT NOT NULL,
            useful      INTEGER,
            note        TEXT DEFAULT '',
            created_at  TEXT NOT NULL,
            UNIQUE(digest_date, section)
        );

        CREATE TABLE IF NOT EXISTS task_events (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            block_id        TEXT NOT NULL,
            event_type      TEXT NOT NULL,
            reason          TEXT,
            source_date     TEXT NOT NULL,
            source          TEXT NOT NULL DEFAULT 'auto',
            created_at      TEXT NOT NULL
        );

        CREATE INDEX IF NOT EXISTS idx_task_events_block
            ON task_events(block_id);
        CREATE INDEX IF NOT EXISTS idx_task_events_type
            ON task_events(event_type);
        """
    )

    conn.execute(
        """
        CREATE VIRTUAL TABLE IF NOT EXISTS blocks_fts
        USING fts5(text, content=blocks, content_rowid=id)
        """
    )

    # External-content FTS5 requires explicit sync: plain INSERT OR REPLACE
    # leaves stale tokens on updates. Triggers keep the index consistent.
    conn.executescript(
        """
        CREATE TRIGGER IF NOT EXISTS blocks_fts_ai AFTER INSERT ON blocks BEGIN
            INSERT INTO blocks_fts(rowid, text) VALUES (new.id, new.text);
        END;

        CREATE TRIGGER IF NOT EXISTS blocks_fts_ad AFTER DELETE ON blocks BEGIN
            INSERT INTO blocks_fts(blocks_fts, rowid, text)
                VALUES('delete', old.id, old.text);
        END;

        CREATE TRIGGER IF NOT EXISTS blocks_fts_au AFTER UPDATE ON blocks BEGIN
            INSERT INTO blocks_fts(blocks_fts, rowid, text)
                VALUES('delete', old.id, old.text);
            INSERT INTO blocks_fts(rowid, text) VALUES (new.id, new.text);
        END;
        """
    )

    try:
        conn.execute(
            "CREATE VIRTUAL TABLE IF NOT EXISTS blocks_vec "
            "USING vec0(embedding float[768] distance_metric=cosine, block_id text)"
        )
    except sqlite3.OperationalError as e:
        logger.debug("blocks_vec creation skipped: %s", e)

    # Migration: recreate blocks_vec with cosine distance metric if an older
    # DB created it with the sqlite-vec default (L2). Detected by inspecting
    # the CREATE statement for the `distance_metric` keyword.
    try:
        row = conn.execute(
            "SELECT sql FROM sqlite_master WHERE type='table' AND name='blocks_vec'"
        ).fetchone()
        if row and "distance_metric" not in (row[0] or "").lower():
            logger.info("Migrating blocks_vec to cosine distance metric")
            conn.execute("DROP TABLE blocks_vec")
            conn.execute(
                "CREATE VIRTUAL TABLE blocks_vec "
                "USING vec0(embedding float[768] distance_metric=cosine, block_id text)"
            )
    except sqlite3.OperationalError as e:
        logger.debug("blocks_vec migration skipped: %s", e)

    # Drop tables from pre-v3 schemas so re-running init on an old DB is idempotent.
    for table in (
        "block_annotations",
        "annotations_fts",
        "annotations_vec",
        "block_state",
        "concepts",
        "concepts_vec",
        "claims",
        "claims_fts",
        "claims_vec",
        "page_hashes",
        "log",
    ):
        conn.execute(f"DROP TABLE IF EXISTS {table}")


# ── Block operations ──────────────────────────────────────────────────


def upsert_block(conn: sqlite3.Connection, block: Block, source_date: str) -> int:
    """Insert or update a block. Returns the row id.

    `completion_status` is intentionally not touched on update so user-driven
    closures (via the digest review flow) survive re-ingest of the source file.
    FTS index is maintained by triggers on `blocks`.
    """
    now = datetime.now().isoformat()
    cur = conn.execute(
        """
        INSERT INTO blocks (
            block_id, source_file, source_date, text, block_hash,
            checkbox_state, completion_status, private,
            attachment_path, attachment_type, created_at, updated_at
        ) VALUES (?, ?, ?, ?, ?, ?, NULL, ?, ?, ?, ?, ?)
        ON CONFLICT(block_id) DO UPDATE SET
            source_file = excluded.source_file,
            source_date = excluded.source_date,
            text = excluded.text,
            block_hash = excluded.block_hash,
            checkbox_state = excluded.checkbox_state,
            private = excluded.private,
            attachment_path = excluded.attachment_path,
            attachment_type = excluded.attachment_type,
            updated_at = excluded.updated_at
        """,
        (
            block.block_id,
            block.source_file,
            source_date,
            block.text,
            block.block_hash,
            block.checkbox_state,
            int(block.private),
            block.attachment_path,
            block.attachment_type,
            now,
            now,
        ),
    )
    row_id = cur.lastrowid
    if row_id is None or row_id == 0:
        row = conn.execute(
            "SELECT id FROM blocks WHERE block_id = ?", (block.block_id,)
        ).fetchone()
        row_id = row["id"] if row else 0
    return row_id


def get_block(conn: sqlite3.Connection, block_id: str) -> sqlite3.Row | None:
    """Get a block by its block_id."""
    return conn.execute(
        "SELECT * FROM blocks WHERE block_id = ?", (block_id,)
    ).fetchone()


def block_row_to_dict(row: sqlite3.Row) -> dict:
    """Serialize a blocks row to a plain dict.

    Callers (digest, search) layer extra fields on top; keep this the single
    point that knows the blocks column shape.
    """
    return {
        "block_id": row["block_id"],
        "source_file": row["source_file"],
        "source_date": row["source_date"],
        "text": row["text"],
        "checkbox_state": row["checkbox_state"],
        "completion_status": row["completion_status"],
    }


def get_block_hashes(conn: sqlite3.Connection, block_ids: list[str]) -> dict[str, str]:
    """Bulk hash lookup. Returns {block_id: block_hash} for known blocks."""
    if not block_ids:
        return {}
    placeholders = ",".join("?" * len(block_ids))
    rows = conn.execute(
        f"SELECT block_id, block_hash FROM blocks WHERE block_id IN ({placeholders})",  # noqa: S608
        block_ids,
    ).fetchall()
    return {r["block_id"]: r["block_hash"] for r in rows}


def get_blocks(
    conn: sqlite3.Connection, block_ids: list[str]
) -> dict[str, sqlite3.Row]:
    """Bulk block lookup. Returns {block_id: row} for known blocks."""
    if not block_ids:
        return {}
    placeholders = ",".join("?" * len(block_ids))
    rows = conn.execute(
        f"SELECT * FROM blocks WHERE block_id IN ({placeholders})",  # noqa: S608
        block_ids,
    ).fetchall()
    return {r["block_id"]: r for r in rows}


def upsert_block_embedding(
    conn: sqlite3.Connection, block_id: str, embedding: np.ndarray
) -> None:
    """Insert or replace a block's embedding vector."""
    conn.execute("DELETE FROM blocks_vec WHERE block_id = ?", (block_id,))
    conn.execute(
        "INSERT INTO blocks_vec(embedding, block_id) VALUES (?, ?)",
        (embedding.astype(np.float32).tobytes(), block_id),
    )


# ── Feedback operations ───────────────────────────────────────────────


def upsert_feedback_score(
    conn: sqlite3.Connection,
    digest_date: str,
    section: str,
    useful: bool | None,
    note: str,
) -> None:
    """Store a feedback score for a digest section."""
    now = datetime.now().isoformat()
    conn.execute(
        """
        INSERT INTO feedback_scores (digest_date, section, useful, note, created_at)
        VALUES (?, ?, ?, ?, ?)
        ON CONFLICT(digest_date, section) DO UPDATE SET
            useful = excluded.useful,
            note = excluded.note
        """,
        (
            digest_date,
            section,
            1 if useful else (0 if useful is False else None),
            note,
            now,
        ),
    )


def get_section_effectiveness(
    conn: sqlite3.Connection, lookback_weeks: int = 12
) -> dict[str, float]:
    """Per-section usefulness ratio from feedback history."""
    cutoff = (datetime.now() - timedelta(weeks=lookback_weeks)).isoformat()[:10]
    rows = conn.execute(
        """
        SELECT section,
               SUM(CASE WHEN useful = 1 THEN 1 ELSE 0 END) as yes_count,
               SUM(CASE WHEN useful IS NOT NULL THEN 1 ELSE 0 END) as total
        FROM feedback_scores
        WHERE digest_date >= ?
        GROUP BY section
        """,
        (cutoff,),
    ).fetchall()
    return {
        r["section"]: r["yes_count"] / r["total"] if r["total"] > 0 else 0.5
        for r in rows
    }


# ── Task lifecycle operations ─────────────────────────────────────────


def insert_task_event(
    conn: sqlite3.Connection,
    block_id: str,
    event_type: str,
    source_date: str,
    reason: str | None = None,
    source: str = "auto",
) -> None:
    """Record a task lifecycle event."""
    conn.execute(
        """
        INSERT INTO task_events
            (block_id, event_type, reason, source_date, source, created_at)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (
            block_id,
            event_type,
            reason,
            source_date,
            source,
            datetime.now().isoformat(),
        ),
    )


def update_completion_status(
    conn: sqlite3.Connection, block_id: str, status: str | None
) -> bool:
    """Set completion_status on an existing block. Returns True if a row updated."""
    cur = conn.execute(
        """
        UPDATE blocks
           SET completion_status = ?, updated_at = ?
         WHERE block_id = ?
        """,
        (status, datetime.now().isoformat(), block_id),
    )
    return cur.rowcount > 0


# ── Search ────────────────────────────────────────────────────────────


def search_blocks_fts(
    conn: sqlite3.Connection,
    query: str,
    limit: int = 50,
    include_private: bool = True,
) -> list[sqlite3.Row]:
    """Full-text search across blocks."""
    q = """
        SELECT b.* FROM blocks b
        JOIN blocks_fts f ON b.id = f.rowid
        WHERE blocks_fts MATCH ?
    """
    if not include_private:
        q += " AND b.private = 0"
    q += " LIMIT ?"
    return conn.execute(q, (query, limit)).fetchall()


def search_blocks_semantic(
    conn: sqlite3.Connection,
    embedding: np.ndarray,
    limit: int = 50,
    include_private: bool = True,
    max_distance: float = 2.0,
) -> list[tuple[sqlite3.Row, float]]:
    """Semantic search across blocks via blocks_vec.

    Returns (row, distance) tuples ordered by ascending distance.
    Results with distance > max_distance are excluded.
    Cosine distance range is [0, 2] where 0 = identical.
    """
    try:
        vec_rows = conn.execute(
            """
            SELECT block_id, distance FROM blocks_vec
            WHERE embedding MATCH ?
            ORDER BY distance
            LIMIT ?
            """,
            (embedding.astype(np.float32).tobytes(), limit * 2),
        ).fetchall()
    except sqlite3.OperationalError:
        return []

    ids: list[str] = []
    dist_by_id: dict[str, float] = {}
    for r in vec_rows:
        dist = r["distance"]
        if dist <= max_distance:
            bid = r["block_id"]
            ids.append(bid)
            dist_by_id[bid] = dist

    if not ids:
        return []

    placeholders = ",".join("?" * len(ids))
    params: list = list(ids)
    q = f"SELECT * FROM blocks WHERE block_id IN ({placeholders})"  # noqa: S608
    if not include_private:
        q += " AND private = 0"
    q += " LIMIT ?"
    params.append(limit)
    rows = conn.execute(q, params).fetchall()

    # Re-sort by vector distance (IN-clause doesn't preserve order)
    result = [
        (row, dist_by_id[row["block_id"]])
        for row in rows
        if row["block_id"] in dist_by_id
    ]
    result.sort(key=lambda x: x[1])
    return result


# ── Stats ─────────────────────────────────────────────────────────────


def get_stats(conn: sqlite3.Connection) -> dict:
    blocks_count = conn.execute("SELECT COUNT(*) as n FROM blocks").fetchone()["n"]
    open_tasks = conn.execute(
        """
        SELECT COUNT(*) as n FROM blocks
        WHERE checkbox_state = ?
          AND (completion_status IS NULL OR completion_status = ?)
        """,
        (CHECKBOX_OPEN, COMPLETION_OPEN),
    ).fetchone()["n"]
    return {
        "blocks": blocks_count,
        "open_tasks": open_tasks,
    }


# ── Digest queries ────────────────────────────────────────────────────


def get_blocks_in_period(
    conn: sqlite3.Connection, since: str, limit: int = 200
) -> list[sqlite3.Row]:
    """All non-private blocks written in the period, oldest first."""
    return conn.execute(
        """
        SELECT * FROM blocks
        WHERE source_date >= ? AND private = 0
        ORDER BY source_date ASC, id ASC
        LIMIT ?
        """,
        (since, limit),
    ).fetchall()


def get_open_tasks(conn: sqlite3.Connection, limit: int = 500) -> list[sqlite3.Row]:
    """Open task blocks (checkbox open, not user-closed), newest first."""
    return conn.execute(
        """
        SELECT * FROM blocks
        WHERE checkbox_state = ?
          AND (completion_status IS NULL OR completion_status = ?)
          AND private = 0
        ORDER BY source_date DESC
        LIMIT ?
        """,
        (CHECKBOX_OPEN, COMPLETION_OPEN, limit),
    ).fetchall()
