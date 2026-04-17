"""Core data models for Dendr."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

# Checkbox states derived from Markdown structure. Tasks are identified by
# `- [ ]` (open) and `- [x]` (closed); everything else is `none`.
CHECKBOX_OPEN = "open"
CHECKBOX_CLOSED = "closed"
CHECKBOX_NONE = "none"

# Values stored in `blocks.completion_status` — only set when the user closes
# a task via the digest review flow (otherwise NULL).
COMPLETION_OPEN = "open"
COMPLETION_DONE = "done"
COMPLETION_ABANDONED = "abandoned"
COMPLETION_SNOOZED = "snoozed"

# Values accepted in digest closure markers (`<!-- closure:... status:X -->`).
# Maps to a completion_status + a task_event reason at reconcile time.
CLOSURE_OPEN = "open"
CLOSURE_DONE = "done"
CLOSURE_ABANDONED = "abandoned"
CLOSURE_SNOOZED = "snoozed"
CLOSURE_STILL_LIVE = "still-live"
CLOSURE_STATUSES = frozenset(
    {CLOSURE_OPEN, CLOSURE_DONE, CLOSURE_ABANDONED, CLOSURE_SNOOZED, CLOSURE_STILL_LIVE}
)

# `task_events.event_type` values.
EVENT_CREATED = "created"
EVENT_CLOSED = "closed"

# `task_events.reason` values (only meaningful for EVENT_CLOSED and for the
# "reopened" reason on an EVENT_CREATED row following a user reopen).
REASON_DONE = "done"
REASON_ABANDONED = "abandoned"
REASON_SNOOZED = "snoozed"
REASON_REOPENED = "reopened"

# `task_events.source` — who drove the transition.
SOURCE_AUTO = "auto"  # checkbox edit in a daily note
SOURCE_USER = "user"  # digest closure marker edit


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
