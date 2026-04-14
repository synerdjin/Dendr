"""Core data models for Dendr."""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from datetime import datetime


class BlockType(str, enum.Enum):
    REFLECTION = "reflection"
    TASK = "task"
    DECISION = "decision"
    QUESTION = "question"
    OBSERVATION = "observation"
    VENT = "vent"
    PLAN = "plan"
    LOG_ENTRY = "log_entry"


@dataclass
class Block:
    """A single block extracted from a daily note."""

    block_id: str
    source_file: str
    line_start: int
    line_end: int
    text: str
    block_hash: str
    is_attachment_ref: bool = False
    attachment_path: str | None = None
    attachment_type: str | None = None  # "pdf", "image", "audio"
    private: bool = False


@dataclass
class BlockAnnotation:
    """Rich annotation of a block — the primary data artifact for digest/synthesis."""

    block_id: str
    source_file: str
    source_date: str  # YYYY-MM-DD from filename
    original_text: str
    gist: str  # one-line summary
    block_type: BlockType
    life_areas: list[str] = field(default_factory=list)
    emotional_valence: float = 0.0  # -1.0 (distressed) to +1.0 (elated)
    intensity: float = 0.5  # 0.0 (passing mention) to 1.0 (central concern)
    urgency: str | None = None  # today, this_week, someday
    importance: str | None = None  # high, medium, low
    completion_status: str | None = None  # open, done, blocked, abandoned
    causal_links: list[str] = field(default_factory=list)
    concepts: list[str] = field(default_factory=list)
    entities: list[str] = field(default_factory=list)
    private: bool = False
    model_version: str = ""
    prompt_version: str = ""


@dataclass
class QueueItem:
    """An item in the processing queue."""

    block_id: str
    source_file: str
    block_hash: str
    block_text: str
    private: bool = False
    attachment_path: str | None = None
    attachment_type: str | None = None
    created_at: datetime = field(default_factory=datetime.now)
