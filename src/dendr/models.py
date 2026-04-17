"""Core data models for Dendr."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

# Checkbox states derived from Markdown structure. Tasks are identified by
# `- [ ]` (open) and `- [x]` (closed); everything else is `none`.
CHECKBOX_OPEN = "open"
CHECKBOX_CLOSED = "closed"
CHECKBOX_NONE = "none"


@dataclass
class Block:
    """A single block extracted from a daily note."""

    block_id: str
    source_file: str
    line_start: int
    line_end: int
    text: str
    block_hash: str
    checkbox_state: str = CHECKBOX_NONE
    is_attachment_ref: bool = False
    attachment_path: str | None = None
    attachment_type: str | None = None  # "pdf", "image", "audio"
    private: bool = False


@dataclass
class QueueItem:
    """An item in the processing queue."""

    block_id: str
    source_file: str
    block_hash: str
    block_text: str
    checkbox_state: str = CHECKBOX_NONE
    private: bool = False
    attachment_path: str | None = None
    attachment_type: str | None = None
    created_at: datetime = field(default_factory=datetime.now)
