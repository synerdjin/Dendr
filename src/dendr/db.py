"""Database schema and operations for Dendr's claim store."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from pathlib import Path

import numpy as np

from dendr.models import Claim, Concept

# sqlite-vec is loaded as an extension at runtime
_VEC_LOADED = False


def _load_vec(conn: sqlite3.Connection) -> None:
    global _VEC_LOADED
    if _VEC_LOADED:
        return
    try:
        import sqlite_vec

        conn.enable_load_extension(True)
        sqlite_vec.load(conn)
        conn.enable_load_extension(False)
        _VEC_LOADED = True
    except Exception:
        # sqlite-vec not available — vector search disabled
        pass


def connect(db_path: Path) -> sqlite3.Connection:
    """Open (or create) the Dendr state database."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path), isolation_level=None)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.row_factory = sqlite3.Row
    _load_vec(conn)
    return conn


def init_schema(conn: sqlite3.Connection) -> None:
    """Create all tables if they don't exist."""
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS claims (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            text            TEXT NOT NULL,
            subject         TEXT NOT NULL DEFAULT '',
            predicate       TEXT NOT NULL DEFAULT '',
            object          TEXT NOT NULL DEFAULT '',
            subject_predicate TEXT NOT NULL DEFAULT '',
            concept_slug    TEXT NOT NULL DEFAULT '',
            source_block_ref TEXT NOT NULL,
            source_file_hash TEXT NOT NULL DEFAULT '',
            created_at      TEXT NOT NULL,
            updated_at      TEXT NOT NULL,
            confidence      REAL NOT NULL DEFAULT 0.5,
            status          TEXT NOT NULL DEFAULT 'created',
            superseded_by   INTEGER REFERENCES claims(id),
            kind            TEXT NOT NULL DEFAULT 'statement',
            private         INTEGER NOT NULL DEFAULT 0,
            model_version   TEXT NOT NULL DEFAULT '',
            prompt_version  TEXT NOT NULL DEFAULT ''
        );

        CREATE INDEX IF NOT EXISTS idx_claims_subject_predicate
            ON claims(subject_predicate);
        CREATE INDEX IF NOT EXISTS idx_claims_concept
            ON claims(concept_slug);
        CREATE INDEX IF NOT EXISTS idx_claims_status
            ON claims(status);
        CREATE INDEX IF NOT EXISTS idx_claims_block
            ON claims(source_block_ref);
        CREATE INDEX IF NOT EXISTS idx_claims_kind
            ON claims(kind);

        CREATE TABLE IF NOT EXISTS concepts (
            slug        TEXT PRIMARY KEY,
            title       TEXT NOT NULL,
            page_type   TEXT NOT NULL DEFAULT 'concept',
            created_at  TEXT NOT NULL,
            updated_at  TEXT NOT NULL,
            page_path   TEXT NOT NULL DEFAULT ''
        );

        CREATE TABLE IF NOT EXISTS log (
            id      INTEGER PRIMARY KEY AUTOINCREMENT,
            ts      TEXT NOT NULL,
            kind    TEXT NOT NULL,
            payload TEXT NOT NULL DEFAULT '{}'
        );

        CREATE TABLE IF NOT EXISTS block_state (
            block_id    TEXT PRIMARY KEY,
            source_file TEXT NOT NULL,
            block_hash  TEXT NOT NULL,
            model_version TEXT NOT NULL DEFAULT '',
            prompt_version TEXT NOT NULL DEFAULT '',
            processed_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS page_hashes (
            page_path   TEXT PRIMARY KEY,
            llm_hash    TEXT NOT NULL,
            updated_at  TEXT NOT NULL
        );
        """
    )

    # Migrate: add 'kind' column if upgrading from older schema
    try:
        conn.execute("SELECT kind FROM claims LIMIT 1")
    except sqlite3.OperationalError:
        conn.execute(
            "ALTER TABLE claims ADD COLUMN kind TEXT NOT NULL DEFAULT 'statement'"
        )

    # FTS5 virtual table for full-text search over claims
    try:
        conn.execute(
            """
            CREATE VIRTUAL TABLE IF NOT EXISTS claims_fts
            USING fts5(text, concept_slug, content=claims, content_rowid=id)
            """
        )
    except sqlite3.OperationalError:
        pass  # already exists

    # sqlite-vec virtual table for vector search
    try:
        conn.execute(
            """
            CREATE VIRTUAL TABLE IF NOT EXISTS claims_vec
            USING vec0(embedding float[768], claim_id integer)
            """
        )
    except sqlite3.OperationalError:
        pass  # already exists or sqlite-vec not loaded

    try:
        conn.execute(
            """
            CREATE VIRTUAL TABLE IF NOT EXISTS concepts_vec
            USING vec0(embedding float[768], concept_slug text)
            """
        )
    except sqlite3.OperationalError:
        pass


# --- Claim operations ---


def insert_claim(conn: sqlite3.Connection, claim: Claim) -> int:
    """Insert a new claim and return its id."""
    now = datetime.now().isoformat()
    cur = conn.execute(
        """
        INSERT INTO claims (text, subject, predicate, object, subject_predicate,
            concept_slug, source_block_ref, source_file_hash,
            created_at, updated_at, confidence, status, kind, private,
            model_version, prompt_version)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            claim.text,
            claim.subject,
            claim.predicate,
            claim.object,
            claim.subject_predicate,
            claim.concept_slug,
            claim.source_block_ref,
            claim.source_file_hash,
            claim.created_at.isoformat() if claim.created_at else now,
            now,
            claim.confidence,
            claim.status.value,
            claim.kind.value,
            int(claim.private),
            claim.model_version,
            claim.prompt_version,
        ),
    )
    claim_id = cur.lastrowid

    # Sync FTS
    conn.execute(
        "INSERT INTO claims_fts(rowid, text, concept_slug) VALUES (?, ?, ?)",
        (claim_id, claim.text, claim.concept_slug),
    )
    return claim_id


def insert_claim_embedding(
    conn: sqlite3.Connection, claim_id: int, embedding: np.ndarray
) -> None:
    """Insert a vector embedding for a claim."""
    conn.execute(
        "INSERT INTO claims_vec(embedding, claim_id) VALUES (?, ?)",
        (embedding.astype(np.float32).tobytes(), claim_id),
    )


def find_contradictions(
    conn: sqlite3.Connection, subject_predicate: str, current_object: str
) -> list[sqlite3.Row]:
    """Find non-superseded claims with same subject_predicate but different object."""
    return conn.execute(
        """
        SELECT * FROM claims
        WHERE subject_predicate = ?
          AND object != ?
          AND status NOT IN ('superseded')
        """,
        (subject_predicate, current_object),
    ).fetchall()


def reinforce_claim(conn: sqlite3.Connection, claim_id: int) -> None:
    """Mark an existing claim as reinforced, bump confidence."""
    now = datetime.now().isoformat()
    conn.execute(
        """
        UPDATE claims
        SET status = 'reinforced', updated_at = ?,
            confidence = MIN(confidence + 0.05, 1.0)
        WHERE id = ?
        """,
        (now, claim_id),
    )


def supersede_claim(conn: sqlite3.Connection, old_id: int, new_id: int) -> None:
    """Mark old claim as superseded by new one."""
    now = datetime.now().isoformat()
    conn.execute(
        """
        UPDATE claims
        SET status = 'superseded', superseded_by = ?, updated_at = ?
        WHERE id = ?
        """,
        (new_id, now, old_id),
    )


def challenge_claim(conn: sqlite3.Connection, claim_id: int) -> None:
    """Mark a claim as challenged (contradiction detected)."""
    now = datetime.now().isoformat()
    conn.execute(
        "UPDATE claims SET status = 'challenged', updated_at = ? WHERE id = ?",
        (now, claim_id),
    )


def find_similar_claim(
    conn: sqlite3.Connection,
    subject_predicate: str,
    object_val: str,
) -> sqlite3.Row | None:
    """Find an existing non-superseded claim with matching SPO."""
    return conn.execute(
        """
        SELECT * FROM claims
        WHERE subject_predicate = ? AND object = ?
          AND status NOT IN ('superseded')
        LIMIT 1
        """,
        (subject_predicate, object_val),
    ).fetchone()


def get_claims_for_concept(
    conn: sqlite3.Connection, slug: str, include_private: bool = True
) -> list[sqlite3.Row]:
    """Get all active claims for a concept."""
    q = "SELECT * FROM claims WHERE concept_slug = ? AND status != 'superseded'"
    if not include_private:
        q += " AND private = 0"
    q += " ORDER BY confidence DESC"
    return conn.execute(q, (slug,)).fetchall()


# --- Concept operations ---


def upsert_concept(conn: sqlite3.Connection, concept: Concept) -> None:
    """Insert or update a concept."""
    now = datetime.now().isoformat()
    conn.execute(
        """
        INSERT INTO concepts (slug, title, page_type, created_at, updated_at, page_path)
        VALUES (?, ?, ?, ?, ?, ?)
        ON CONFLICT(slug) DO UPDATE SET
            title = excluded.title,
            updated_at = excluded.updated_at,
            page_path = excluded.page_path
        """,
        (
            concept.slug,
            concept.title,
            concept.page_type.value,
            concept.created_at.isoformat() if concept.created_at else now,
            now,
            concept.page_path,
        ),
    )


def insert_concept_embedding(
    conn: sqlite3.Connection, slug: str, embedding: np.ndarray
) -> None:
    """Insert or replace a concept's embedding."""
    # Delete old if exists
    conn.execute("DELETE FROM concepts_vec WHERE concept_slug = ?", (slug,))
    conn.execute(
        "INSERT INTO concepts_vec(embedding, concept_slug) VALUES (?, ?)",
        (embedding.astype(np.float32).tobytes(), slug),
    )


def find_nearest_concept(
    conn: sqlite3.Connection, embedding: np.ndarray, top_k: int = 5
) -> list[tuple[str, float]]:
    """Find nearest concept slugs by embedding. Returns (slug, distance) pairs."""
    try:
        rows = conn.execute(
            """
            SELECT concept_slug, distance
            FROM concepts_vec
            WHERE embedding MATCH ?
            ORDER BY distance
            LIMIT ?
            """,
            (embedding.astype(np.float32).tobytes(), top_k),
        ).fetchall()
        return [(r["concept_slug"], r["distance"]) for r in rows]
    except sqlite3.OperationalError:
        return []


def get_all_concepts(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    """Get all concepts."""
    return conn.execute("SELECT * FROM concepts ORDER BY slug").fetchall()


# --- Block state operations ---


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


# --- Page hash operations ---


def get_page_hash(conn: sqlite3.Connection, page_path: str) -> str | None:
    """Get the last LLM-written hash for a wiki page."""
    row = conn.execute(
        "SELECT llm_hash FROM page_hashes WHERE page_path = ?", (page_path,)
    ).fetchone()
    return row["llm_hash"] if row else None


def set_page_hash(conn: sqlite3.Connection, page_path: str, llm_hash: str) -> None:
    """Record the LLM-written hash for a page."""
    now = datetime.now().isoformat()
    conn.execute(
        """
        INSERT INTO page_hashes (page_path, llm_hash, updated_at)
        VALUES (?, ?, ?)
        ON CONFLICT(page_path) DO UPDATE SET
            llm_hash = excluded.llm_hash,
            updated_at = excluded.updated_at
        """,
        (page_path, llm_hash, now),
    )


# --- Log operations ---


def append_log(conn: sqlite3.Connection, kind: str, payload: dict) -> None:
    """Append an entry to the activity log."""
    conn.execute(
        "INSERT INTO log (ts, kind, payload) VALUES (?, ?, ?)",
        (datetime.now().isoformat(), kind, json.dumps(payload)),
    )


# --- FTS search ---


def search_claims_fts(
    conn: sqlite3.Connection, query: str, limit: int = 50, include_private: bool = True
) -> list[sqlite3.Row]:
    """Full-text search over claims."""
    q = """
        SELECT c.* FROM claims c
        JOIN claims_fts f ON c.id = f.rowid
        WHERE claims_fts MATCH ?
          AND c.status != 'superseded'
    """
    if not include_private:
        q += " AND c.private = 0"
    q += " LIMIT ?"
    return conn.execute(q, (query, limit)).fetchall()


def search_claims_semantic(
    conn: sqlite3.Connection,
    embedding: np.ndarray,
    limit: int = 50,
    include_private: bool = True,
) -> list[sqlite3.Row]:
    """Semantic search over claims using vector similarity."""
    try:
        vec_rows = conn.execute(
            """
            SELECT claim_id, distance FROM claims_vec
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

    ids = [r["claim_id"] for r in vec_rows]
    placeholders = ",".join("?" * len(ids))
    params: list = list(ids)
    q = f"SELECT * FROM claims WHERE id IN ({placeholders}) AND status != 'superseded'"
    if not include_private:
        q += " AND private = 0"
    q += " LIMIT ?"
    params.append(limit)
    return conn.execute(q, params).fetchall()  # nosemgrep: sqlalchemy-execute-raw-query


# --- Stats ---


def get_stats(conn: sqlite3.Connection) -> dict:
    """Get overview statistics."""
    claims_count = conn.execute(
        "SELECT COUNT(*) as n FROM claims WHERE status != 'superseded'"
    ).fetchone()["n"]
    concepts_count = conn.execute("SELECT COUNT(*) as n FROM concepts").fetchone()["n"]
    challenged = conn.execute(
        "SELECT COUNT(*) as n FROM claims WHERE status = 'challenged'"
    ).fetchone()["n"]
    return {
        "active_claims": claims_count,
        "concepts": concepts_count,
        "challenged_claims": challenged,
    }


# --- Digest queries ---


def get_recent_claims(
    conn: sqlite3.Connection, since: str, limit: int = 200
) -> list[sqlite3.Row]:
    """Get non-superseded claims created after `since` (ISO timestamp)."""
    return conn.execute(
        """
        SELECT * FROM claims
        WHERE created_at >= ? AND status != 'superseded' AND private = 0
        ORDER BY created_at DESC
        LIMIT ?
        """,
        (since, limit),
    ).fetchall()


def get_open_tasks(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    """Get claims of kind task/intention that are not superseded."""
    return conn.execute(
        """
        SELECT * FROM claims
        WHERE kind IN ('task', 'intention')
          AND status NOT IN ('superseded')
          AND private = 0
        ORDER BY created_at DESC
        """,
    ).fetchall()


def get_concept_frequencies(
    conn: sqlite3.Connection, since: str, limit: int = 20
) -> list[tuple[str, int]]:
    """Get concept slugs ranked by claim count since a date."""
    rows = conn.execute(
        """
        SELECT concept_slug, COUNT(*) as n FROM claims
        WHERE created_at >= ? AND status != 'superseded' AND private = 0
          AND concept_slug != ''
        GROUP BY concept_slug
        ORDER BY n DESC
        LIMIT ?
        """,
        (since, limit),
    ).fetchall()
    return [(r["concept_slug"], r["n"]) for r in rows]


def get_all_contradictions(conn: sqlite3.Connection) -> list[dict]:
    """Get all active contradiction pairs (non-superseded, differing objects)."""
    rows = conn.execute(
        """
        SELECT c1.id as id1, c1.text as text1, c1.subject_predicate,
               c1.object as obj1, c1.confidence as conf1,
               c1.created_at as created1,
               c2.id as id2, c2.text as text2, c2.object as obj2,
               c2.confidence as conf2, c2.created_at as created2
        FROM claims c1
        JOIN claims c2 ON c1.subject_predicate = c2.subject_predicate
            AND c1.id < c2.id
            AND c1.object != c2.object
        WHERE c1.status != 'superseded' AND c2.status != 'superseded'
          AND c1.private = 0 AND c2.private = 0
        ORDER BY c2.created_at DESC
        LIMIT 30
        """
    ).fetchall()
    return [
        {
            "subject_predicate": r["subject_predicate"],
            "claim_a": {
                "id": r["id1"],
                "text": r["text1"],
                "object": r["obj1"],
                "confidence": r["conf1"],
                "created_at": r["created1"],
            },
            "claim_b": {
                "id": r["id2"],
                "text": r["text2"],
                "object": r["obj2"],
                "confidence": r["conf2"],
                "created_at": r["created2"],
            },
        }
        for r in rows
    ]


def get_dropped_threads(
    conn: sqlite3.Connection, mentioned_once_before: str, limit: int = 10
) -> list[sqlite3.Row]:
    """Find concepts mentioned exactly once, with that mention before a date."""
    return conn.execute(
        """
        SELECT c.concept_slug, c.text, c.created_at
        FROM claims c
        WHERE c.status != 'superseded' AND c.private = 0 AND c.concept_slug != ''
        GROUP BY c.concept_slug
        HAVING COUNT(*) = 1 AND MAX(c.created_at) < ?
        ORDER BY c.created_at DESC
        LIMIT ?
        """,
        (mentioned_once_before, limit),
    ).fetchall()
