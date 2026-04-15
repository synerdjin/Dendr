"""Database schema and operations for Dendr's knowledge store.

Tables:
  - block_annotations: rich per-block metadata for digest/synthesis (primary artifact)
  - block_state: tracks which blocks have been processed
  - feedback_scores: per-section digest feedback for learning
  - task_events: lifecycle events for task/plan blocks
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

from dendr.models import BlockAnnotation

# sqlite-vec must be loaded per-connection (extensions are connection-scoped).
_VEC_AVAILABLE: bool | None = None  # None = untested, True/False = cached result


def _load_vec(conn: sqlite3.Connection) -> None:
    global _VEC_AVAILABLE
    if _VEC_AVAILABLE is False:
        return
    try:
        import sqlite_vec

        conn.enable_load_extension(True)
        sqlite_vec.load(conn)
        conn.enable_load_extension(False)
        _VEC_AVAILABLE = True
    except Exception:
        _VEC_AVAILABLE = False


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
        CREATE TABLE IF NOT EXISTS block_annotations (
            id                INTEGER PRIMARY KEY AUTOINCREMENT,
            block_id          TEXT NOT NULL UNIQUE,
            source_file       TEXT NOT NULL,
            source_date       TEXT NOT NULL,
            original_text     TEXT NOT NULL,
            gist              TEXT NOT NULL DEFAULT '',
            block_type        TEXT NOT NULL DEFAULT 'observation',
            life_areas        TEXT NOT NULL DEFAULT '[]',
            emotional_valence REAL DEFAULT 0.0,
            intensity         REAL DEFAULT 0.5,
            urgency           TEXT,
            importance        TEXT,
            completion_status TEXT,
            causal_links      TEXT NOT NULL DEFAULT '[]',
            concepts          TEXT NOT NULL DEFAULT '[]',
            entities          TEXT NOT NULL DEFAULT '[]',
            private           INTEGER NOT NULL DEFAULT 0,
            model_version     TEXT NOT NULL DEFAULT '',
            prompt_version    TEXT NOT NULL DEFAULT '',
            created_at        TEXT NOT NULL,
            updated_at        TEXT NOT NULL
        );

        CREATE INDEX IF NOT EXISTS idx_annotations_date
            ON block_annotations(source_date);
        CREATE INDEX IF NOT EXISTS idx_annotations_type
            ON block_annotations(block_type);
        CREATE INDEX IF NOT EXISTS idx_annotations_intensity
            ON block_annotations(intensity);
        CREATE INDEX IF NOT EXISTS idx_annotations_completion
            ON block_annotations(completion_status);

        CREATE TABLE IF NOT EXISTS block_state (
            block_id    TEXT PRIMARY KEY,
            source_file TEXT NOT NULL,
            block_hash  TEXT NOT NULL,
            model_version TEXT NOT NULL DEFAULT '',
            prompt_version TEXT NOT NULL DEFAULT '',
            processed_at TEXT NOT NULL
        );

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

    # task_events.source was added after initial release — backfill on
    # existing DBs that predate it.
    try:
        conn.execute(
            "ALTER TABLE task_events ADD COLUMN source TEXT NOT NULL DEFAULT 'auto'"
        )
    except sqlite3.OperationalError:
        pass  # column already exists

    # FTS5 for annotations (text search surface)
    try:
        conn.execute(
            """
            CREATE VIRTUAL TABLE IF NOT EXISTS annotations_fts
            USING fts5(original_text, gist, content=block_annotations, content_rowid=id)
            """
        )
    except sqlite3.OperationalError:
        pass

    # sqlite-vec for annotation embeddings (semantic search)
    try:
        conn.execute(
            "CREATE VIRTUAL TABLE IF NOT EXISTS annotations_vec "
            "USING vec0(embedding float[768], block_id text)"
        )
    except sqlite3.OperationalError:
        pass

    # Migration: drop legacy tables from older schema versions.
    for table in ("concepts_vec", "concepts", "page_hashes", "log"):
        conn.execute(f"DROP TABLE IF EXISTS {table}")


# ── Block annotation operations ───────────────────────────────────────


def upsert_block_annotation(conn: sqlite3.Connection, ann: BlockAnnotation) -> int:
    """Insert or update a block annotation. Returns the row id."""
    now = datetime.now().isoformat()
    cur = conn.execute(
        """
        INSERT INTO block_annotations (
            block_id, source_file, source_date, original_text, gist,
            block_type, life_areas, emotional_valence,
            intensity, urgency, importance, completion_status,
            causal_links, concepts, entities, private,
            model_version, prompt_version, created_at, updated_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(block_id) DO UPDATE SET
            original_text = excluded.original_text,
            gist = excluded.gist,
            block_type = excluded.block_type,
            life_areas = excluded.life_areas,
            emotional_valence = excluded.emotional_valence,
            intensity = excluded.intensity,
            urgency = excluded.urgency,
            importance = excluded.importance,
            completion_status = excluded.completion_status,
            causal_links = excluded.causal_links,
            concepts = excluded.concepts,
            entities = excluded.entities,
            private = excluded.private,
            model_version = excluded.model_version,
            prompt_version = excluded.prompt_version,
            updated_at = excluded.updated_at
        """,
        (
            ann.block_id,
            ann.source_file,
            ann.source_date,
            ann.original_text,
            ann.gist,
            ann.block_type.value
            if hasattr(ann.block_type, "value")
            else ann.block_type,
            json.dumps(ann.life_areas),
            ann.emotional_valence,
            ann.intensity,
            ann.urgency,
            ann.importance,
            ann.completion_status,
            json.dumps(ann.causal_links),
            json.dumps(ann.concepts),
            json.dumps(ann.entities),
            int(ann.private),
            ann.model_version,
            ann.prompt_version,
            now,
            now,
        ),
    )
    ann_id = cur.lastrowid

    try:
        conn.execute(
            "INSERT OR REPLACE INTO annotations_fts(rowid, original_text, gist) VALUES (?, ?, ?)",
            (ann_id, ann.original_text, ann.gist),
        )
    except sqlite3.OperationalError:
        pass

    return ann_id


def get_block_annotation(conn: sqlite3.Connection, block_id: str) -> sqlite3.Row | None:
    """Get annotation for a specific block."""
    return conn.execute(
        "SELECT * FROM block_annotations WHERE block_id = ?", (block_id,)
    ).fetchone()


# ── Block state operations ────────────────────────────────────────────


def get_block_state(conn: sqlite3.Connection, block_id: str) -> sqlite3.Row | None:
    """Check if a block has been processed."""
    return conn.execute(
        "SELECT * FROM block_state WHERE block_id = ?", (block_id,)
    ).fetchone()


def upsert_block_state(
    conn: sqlite3.Connection,
    block_id: str,
    source_file: str,
    block_hash: str,
    model_version: str,
    prompt_version: str,
) -> None:
    """Record that a block has been processed."""
    now = datetime.now().isoformat()
    conn.execute(
        """
        INSERT INTO block_state (block_id, source_file, block_hash,
            model_version, prompt_version, processed_at)
        VALUES (?, ?, ?, ?, ?, ?)
        ON CONFLICT(block_id) DO UPDATE SET
            block_hash = excluded.block_hash,
            model_version = excluded.model_version,
            prompt_version = excluded.prompt_version,
            processed_at = excluded.processed_at
        """,
        (block_id, source_file, block_hash, model_version, prompt_version, now),
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
    """Compute per-section usefulness ratio from feedback history.

    Returns {section_name: ratio} where ratio is useful_count / total_rated.
    """
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
    source: str = "auto",
) -> None:
    """Record a task lifecycle event.

    `source` is 'auto' for events detected by the tagger and 'user' for
    events the user drove via the digest closure review flow.
    """
    conn.execute(
        """
        INSERT INTO task_events (block_id, event_type, source_date, source, created_at)
        VALUES (?, ?, ?, ?, ?)
        """,
        (block_id, event_type, source_date, source, datetime.now().isoformat()),
    )


def update_completion_status(
    conn: sqlite3.Connection, block_id: str, status: str
) -> bool:
    """Set completion_status on an existing block annotation.

    Returns True if a row was updated.
    """
    cur = conn.execute(
        """
        UPDATE block_annotations
           SET completion_status = ?, updated_at = ?
         WHERE block_id = ?
        """,
        (status, datetime.now().isoformat(), block_id),
    )
    return cur.rowcount > 0


def get_task_lifecycle_stats(conn: sqlite3.Connection) -> dict:
    """Compute task lifecycle statistics from task_events."""
    total_created = conn.execute(
        "SELECT COUNT(DISTINCT block_id) as n FROM task_events WHERE event_type = 'created'"
    ).fetchone()["n"]
    total_completed = conn.execute(
        "SELECT COUNT(DISTINCT block_id) as n FROM task_events WHERE event_type = 'completed'"
    ).fetchone()["n"]
    total_abandoned = conn.execute(
        "SELECT COUNT(DISTINCT block_id) as n FROM task_events WHERE event_type = 'abandoned'"
    ).fetchone()["n"]

    completion_rate = (
        round(total_completed / total_created, 2) if total_created > 0 else 0.0
    )

    avg_days_rows = conn.execute(
        """
        SELECT te_done.block_id,
               julianday(te_done.source_date) - julianday(te_created.source_date) as days
        FROM task_events te_done
        JOIN task_events te_created ON te_done.block_id = te_created.block_id
        WHERE te_done.event_type = 'completed'
          AND te_created.event_type = 'created'
        """
    ).fetchall()
    avg_days = (
        round(sum(r["days"] for r in avg_days_rows) / len(avg_days_rows), 1)
        if avg_days_rows
        else 0.0
    )

    return {
        "total_created": total_created,
        "total_completed": total_completed,
        "total_abandoned": total_abandoned,
        "completion_rate": completion_rate,
        "avg_days_to_completion": avg_days,
    }


# ── Annotation search ─────────────────────────────────────────────────


def insert_annotation_embedding(
    conn: sqlite3.Connection, block_id: str, embedding: np.ndarray
) -> None:
    """Insert or replace a block annotation's embedding."""
    conn.execute("DELETE FROM annotations_vec WHERE block_id = ?", (block_id,))
    conn.execute(
        "INSERT INTO annotations_vec(embedding, block_id) VALUES (?, ?)",
        (embedding.astype(np.float32).tobytes(), block_id),
    )


def search_annotations_fts(
    conn: sqlite3.Connection,
    query: str,
    limit: int = 50,
    include_private: bool = True,
) -> list[sqlite3.Row]:
    """Full-text search across block annotations."""
    q = """
        SELECT ba.* FROM block_annotations ba
        JOIN annotations_fts f ON ba.id = f.rowid
        WHERE annotations_fts MATCH ?
    """
    if not include_private:
        q += " AND ba.private = 0"
    q += " LIMIT ?"
    return conn.execute(q, (query, limit)).fetchall()


def search_annotations_semantic(
    conn: sqlite3.Connection,
    embedding: np.ndarray,
    limit: int = 50,
    include_private: bool = True,
) -> list[sqlite3.Row]:
    """Semantic search across block annotations via annotations_vec."""
    try:
        vec_rows = conn.execute(
            """
            SELECT block_id, distance FROM annotations_vec
            WHERE embedding MATCH ?
            ORDER BY distance
            LIMIT ?
            """,
            (embedding.astype(np.float32).tobytes(), limit * 2),
        ).fetchall()
    except sqlite3.OperationalError:
        return []

    if not vec_rows:
        return []

    ids = [r["block_id"] for r in vec_rows]
    placeholders = ",".join("?" * len(ids))
    params: list = list(ids)
    q = f"SELECT * FROM block_annotations WHERE block_id IN ({placeholders})"  # nosec B608  # nosemgrep: python.sqlalchemy.security.sqlalchemy-execute-raw-query.sqlalchemy-execute-raw-query
    if not include_private:
        q += " AND private = 0"
    q += " LIMIT ?"
    params.append(limit)
    return conn.execute(q, params).fetchall()


# ── Stats ─────────────────────────────────────────────────────────────


def get_stats(conn: sqlite3.Connection) -> dict:
    annotations_count = conn.execute(
        "SELECT COUNT(*) as n FROM block_annotations"
    ).fetchone()["n"]
    open_tasks = conn.execute(
        """
        SELECT COUNT(*) as n FROM block_annotations
        WHERE block_type IN ('task', 'plan')
          AND (completion_status IS NULL OR completion_status = 'open')
        """
    ).fetchone()["n"]
    return {
        "annotations": annotations_count,
        "open_tasks": open_tasks,
    }


# ── Digest queries (annotation-based) ─────────────────────────────────


def get_significant_blocks(
    conn: sqlite3.Connection, since: str, limit: int = 25
) -> list[sqlite3.Row]:
    """Get the most significant annotated blocks since a date.

    Ranked by intensity (how central the concern is), filtered non-private.
    """
    return conn.execute(
        """
        SELECT * FROM block_annotations
        WHERE source_date >= ? AND private = 0
        ORDER BY intensity DESC, source_date DESC
        LIMIT ?
        """,
        (since, limit),
    ).fetchall()


def get_recurring_topics(
    conn: sqlite3.Connection, since: str, limit: int = 15
) -> list[dict]:
    """Get concepts with mention counts and average emotional valence since a date."""
    rows = conn.execute(
        """
        SELECT ba.concepts, ba.emotional_valence, ba.source_date
        FROM block_annotations ba
        WHERE ba.source_date >= ? AND ba.private = 0
        ORDER BY ba.source_date
        """,
        (since,),
    ).fetchall()

    topic_data: dict[str, list[tuple[float, str]]] = {}
    for r in rows:
        concepts = json.loads(r["concepts"])
        for slug in concepts:
            if slug:
                if slug not in topic_data:
                    topic_data[slug] = []
                topic_data[slug].append((r["emotional_valence"], r["source_date"]))

    results = []
    for slug, entries in sorted(topic_data.items(), key=lambda x: -len(x[1])):
        if len(entries) < 1:
            continue
        valences = [v for v, _ in entries]
        avg_valence = sum(valences) / len(valences)
        # Trend: compare first half vs second half
        mid = len(valences) // 2
        if mid > 0:
            first_half = sum(valences[:mid]) / mid
            second_half = sum(valences[mid:]) / len(valences[mid:])
            diff = second_half - first_half
            trend = (
                "improving"
                if diff > 0.15
                else ("worsening" if diff < -0.15 else "stable")
            )
        else:
            trend = "stable"
        results.append(
            {
                "concept": slug,
                "mentions": len(entries),
                "avg_valence": round(avg_valence, 2),
                "trend": trend,
            }
        )
    return results[:limit]


def get_life_area_distribution(conn: sqlite3.Connection, since: str) -> dict[str, int]:
    """Get percentage breakdown of life areas from recent annotations."""
    rows = conn.execute(
        """
        SELECT life_areas FROM block_annotations
        WHERE source_date >= ? AND private = 0
        """,
        (since,),
    ).fetchall()

    counts: dict[str, int] = {}
    for r in rows:
        areas = json.loads(r["life_areas"])
        for area in areas:
            counts[area] = counts.get(area, 0) + 1

    total = sum(counts.values()) or 1
    return {
        area: round(100 * count / total)
        for area, count in sorted(counts.items(), key=lambda x: -x[1])
    }


def get_emotional_trajectory(conn: sqlite3.Connection, weeks: int = 4) -> list[dict]:
    """Get average emotional valence per week for the past N weeks."""
    results = []
    now = datetime.now()
    for i in range(weeks - 1, -1, -1):
        week_start = (now - timedelta(weeks=i + 1)).strftime("%Y-%m-%d")
        week_end = (now - timedelta(weeks=i)).strftime("%Y-%m-%d")
        row = conn.execute(
            """
            SELECT AVG(emotional_valence) as avg_val, COUNT(*) as n
            FROM block_annotations
            WHERE source_date >= ? AND source_date < ? AND private = 0
            """,
            (week_start, week_end),
        ).fetchone()
        results.append(
            {
                "week_start": week_start,
                "avg_valence": round(row["avg_val"], 2)
                if row["avg_val"] is not None
                else 0.0,
                "block_count": row["n"],
            }
        )
    return results


def get_open_tasks_annotated(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    """Get annotation blocks with open tasks/plans, ordered by importance."""
    return conn.execute(
        """
        SELECT * FROM block_annotations
        WHERE block_type IN ('task', 'plan')
          AND (completion_status IS NULL OR completion_status = 'open')
          AND private = 0
        ORDER BY
            CASE importance WHEN 'high' THEN 0 WHEN 'medium' THEN 1 ELSE 2 END,
            CASE urgency WHEN 'today' THEN 0 WHEN 'this_week' THEN 1 ELSE 2 END,
            source_date DESC
        """
    ).fetchall()


def get_completed_tasks(conn: sqlite3.Connection, since: str) -> list[sqlite3.Row]:
    """Get blocks whose completion_status is 'done' since a date."""
    return conn.execute(
        """
        SELECT * FROM block_annotations
        WHERE completion_status = 'done'
          AND source_date >= ?
          AND private = 0
        ORDER BY source_date DESC
        """,
        (since,),
    ).fetchall()


def get_stale_tasks(
    conn: sqlite3.Connection, stale_days: int = 14
) -> list[sqlite3.Row]:
    """Get open tasks not updated in stale_days."""
    cutoff = (datetime.now() - timedelta(days=stale_days)).isoformat()
    return conn.execute(
        """
        SELECT * FROM block_annotations
        WHERE block_type IN ('task', 'plan')
          AND (completion_status IS NULL OR completion_status = 'open')
          AND updated_at < ?
          AND private = 0
        ORDER BY source_date ASC
        """,
        (cutoff,),
    ).fetchall()
