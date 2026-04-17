"""Block-level parser for Obsidian daily notes.

Extracts blocks (paragraphs, list items, headings), auto-assigns
Obsidian block IDs (^dendr-<ulid>), and hashes per block for
incremental re-processing.

Also parses digest-level closure markers — the round-trip format the user
edits in Wiki/digest.md to close out open task/plan blocks.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path

import ulid

from dendr.models import (
    CHECKBOX_CLOSED,
    CHECKBOX_NONE,
    CHECKBOX_OPEN,
    CLOSURE_DONE,
    CLOSURE_OPEN,
    CLOSURE_STATUSES,
    Block,
)

# Obsidian block-ref pattern: text followed by ^identifier at end of line
_BLOCK_REF_RE = re.compile(r"\s+\^([\w-]+)\s*$")

# Markdown task checkbox at the start of a block body.
_CHECKBOX_RE = re.compile(r"^\s*\[(?P<mark>[ xX])\]\s?")

# YAML frontmatter pattern
_FRONTMATTER_RE = re.compile(r"^---\n.*?\n---\n?", re.DOTALL)

# Top-level list item (not indented sub-items)
_TOP_LEVEL_LIST_RE = re.compile(r"^- ")

# Attachment embed patterns: ![[file.pdf]], ![[image.png]], etc.
_EMBED_RE = re.compile(r"!\[\[([^\]]+)\]\]")
_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp", ".svg"}
_PDF_EXTS = {".pdf"}
_AUDIO_EXTS = {".mp3", ".wav", ".m4a", ".ogg", ".flac", ".webm"}


def _hash_text(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def _generate_block_id() -> str:
    return f"dendr-{ulid.new().str.lower()}"


def _classify_attachment(filename: str) -> tuple[str, str] | None:
    """Return (attachment_type, path) or None if not an attachment."""
    suffix = Path(filename).suffix.lower()
    if suffix in _IMAGE_EXTS:
        return ("image", filename)
    if suffix in _PDF_EXTS:
        return ("pdf", filename)
    if suffix in _AUDIO_EXTS:
        return ("audio", filename)
    return None


def _strip_frontmatter(text: str) -> tuple[str, int]:
    """Remove YAML frontmatter from text.

    Returns (stripped_text, number_of_lines_removed).
    """
    match = _FRONTMATTER_RE.match(text)
    if match:
        removed = match.group(0)
        lines_removed = removed.count("\n")
        return text[match.end() :], lines_removed
    return text, 0


def _split_into_raw_blocks(lines: list[str]) -> list[tuple[int, int, list[str]]]:
    """Split lines into blocks.

    Block boundaries are:
    - Blank lines (paragraph separator)
    - Headings (# starts a new block)
    - Top-level list items (- at column 0, each becomes its own block;
      indented continuation lines stay with the parent item)

    Returns list of (start_line_0indexed, end_line_0indexed, lines).
    """
    blocks: list[tuple[int, int, list[str]]] = []
    current_lines: list[str] = []
    start = 0

    for i, line in enumerate(lines):
        stripped = line.strip()

        if stripped.startswith("#") and current_lines:
            blocks.append((start, i - 1, current_lines))
            current_lines = [line]
            start = i
        elif stripped == "":
            if current_lines:
                blocks.append((start, i - 1, current_lines))
                current_lines = []
            start = i + 1
        elif _TOP_LEVEL_LIST_RE.match(line) and current_lines:
            blocks.append((start, i - 1, current_lines))
            current_lines = [line]
            start = i
        else:
            if not current_lines:
                start = i
            current_lines.append(line)

    if current_lines:
        blocks.append((start, len(lines) - 1, current_lines))

    return blocks


def parse_daily_note(
    file_path: Path, attachments_dir: Path | None = None
) -> list[Block]:
    """Parse a daily note into blocks.

    Each block gets a stable block_id. If the note doesn't have Obsidian
    block refs, we DO NOT modify the file here — modification (injecting
    ^dendr-xxx) is done by `inject_block_ids` separately so parsing is
    read-only.
    """
    text = file_path.read_text(encoding="utf-8")

    # Strip YAML frontmatter before block splitting, but track the line
    # offset so line_start/line_end still reference the original file.
    stripped_text, fm_lines = _strip_frontmatter(text)
    lines = stripped_text.split("\n")
    raw_blocks = _split_into_raw_blocks(lines)
    blocks: list[Block] = []

    for start, end, block_lines in raw_blocks:
        block_text = "\n".join(block_lines).strip()
        if not block_text:
            continue

        last_line = block_lines[-1]
        ref_match = _BLOCK_REF_RE.search(last_line)
        if ref_match:
            block_id = ref_match.group(1)
            clean_text = "\n".join(block_lines[:-1]).strip()
            if not clean_text:
                clean_text = _BLOCK_REF_RE.sub("", last_line).strip()
        else:
            block_id = _generate_block_id()
            clean_text = block_text

        if clean_text.startswith("- "):
            clean_text = clean_text[2:]
        elif clean_text == "-":
            continue

        if not clean_text.strip():
            continue

        checkbox_state = CHECKBOX_NONE
        cb_match = _CHECKBOX_RE.match(clean_text)
        if cb_match:
            mark = cb_match.group("mark").lower()
            checkbox_state = CHECKBOX_CLOSED if mark == "x" else CHECKBOX_OPEN

        block_hash = _hash_text(clean_text)

        is_attachment = False
        att_path = None
        att_type = None
        embed_match = _EMBED_RE.search(block_text)
        if embed_match:
            filename = embed_match.group(1).split("|")[0].strip()
            classified = _classify_attachment(filename)
            if classified:
                att_type, att_path = classified
                is_attachment = True
                if attachments_dir and not Path(att_path).is_absolute():
                    att_path = str(attachments_dir / att_path)

        blocks.append(
            Block(
                block_id=block_id,
                source_file=str(file_path),
                line_start=start + fm_lines,
                line_end=end + fm_lines,
                text=clean_text,
                block_hash=block_hash,
                checkbox_state=checkbox_state,
                is_attachment_ref=is_attachment,
                attachment_path=att_path,
                attachment_type=att_type,
            )
        )

    return blocks


def inject_block_ids(file_path: Path, blocks: list[Block]) -> bool:
    """Write block IDs back into the daily note for blocks that lack them.

    Returns True if the file was modified.
    """
    text = file_path.read_text(encoding="utf-8")
    lines = text.split("\n")
    modified = False

    for block in blocks:
        last_line_idx = block.line_end
        if last_line_idx >= len(lines):
            continue
        last_line = lines[last_line_idx]
        if _BLOCK_REF_RE.search(last_line):
            continue

        lines[last_line_idx] = last_line.rstrip() + f" ^{block.block_id}"
        modified = True

    if modified:
        file_path.write_text("\n".join(lines), encoding="utf-8")

    return modified


def get_file_hash(file_path: Path) -> str:
    """Hash the entire file for change detection."""
    return hashlib.sha256(file_path.read_bytes()).hexdigest()[:16]


# ── Closure markers (digest review flow) ─────────────────────────────
#
# Closure lines look like:
#
#   - [ ] **Finish X** — *from 2026-03-15 (3w old)* <!-- closure:dendr-01h... status:open -->
#
# The checkbox is the ergonomic handle; the HTML comment is the source
# of truth (carries the block_id). A status word inside the comment
# wins if present; otherwise the checkbox state is authoritative
# (`[x]` → done, `[ ]` → open).

_CLOSURE_STATUS_ALT = "|".join(sorted(CLOSURE_STATUSES))
_CLOSURE_RE = re.compile(
    r"^\s*-\s*\[(?P<checkbox>[ xX])\]"  # checkbox
    r".*?"  # task label
    r"<!--\s*closure:(?P<block_id>[\w-]+)"  # block_id
    rf"(?:\s+status:(?P<status>{_CLOSURE_STATUS_ALT}))?"
    r"\s*-->",
    re.MULTILINE,
)


@dataclass
class TaskClosure:
    """A closure marker parsed from digest.md."""

    block_id: str
    status: str  # open | done | abandoned | snoozed | still-live
    checkbox_checked: bool


def parse_closures(digest_text: str) -> list[TaskClosure]:
    """Extract closure markers from a digest markdown string.

    Status resolution:
      1. If the HTML comment has `status:X` with X != open, that wins.
      2. Otherwise `[x]` → done, `[ ]` → open.
    """
    results: list[TaskClosure] = []
    seen: set[str] = set()

    for match in _CLOSURE_RE.finditer(digest_text):
        block_id = match.group("block_id")
        if block_id in seen:
            continue
        seen.add(block_id)

        checked = match.group("checkbox").lower() == "x"
        explicit_status = match.group("status")

        if explicit_status and explicit_status != CLOSURE_OPEN:
            status = explicit_status
        elif checked:
            status = CLOSURE_DONE
        else:
            status = CLOSURE_OPEN

        if status not in CLOSURE_STATUSES:
            continue

        results.append(
            TaskClosure(
                block_id=block_id,
                status=status,
                checkbox_checked=checked,
            )
        )

    return results
