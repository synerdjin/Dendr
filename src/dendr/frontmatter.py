"""Minimal YAML frontmatter read/write for Markdown notes.

Distinct from `parser.py`'s `_strip_frontmatter`, which only strips and counts
removed lines for Daily-note block offsetting — this module actually parses
the frontmatter into a dict and can render it back, for features (like the
garden) that need to read/write structured metadata on notes.
"""

from __future__ import annotations

import re

import yaml

# Shared with parser.py's _strip_frontmatter, so the delimiter pattern has
# one source of truth even though the two functions do different things with
# what they match (parse-to-dict here, strip-and-count-lines there).
FRONTMATTER_RE = re.compile(r"^---\n(.*?)\n---\n?", re.DOTALL)


def split_frontmatter(text: str) -> tuple[dict, str]:
    """Split leading YAML frontmatter from a note's body.

    Returns `({}, text)` unchanged if there's no `---`-delimited block at
    all. If the block exists but doesn't parse as a YAML mapping (invalid
    YAML, or a scalar/list instead of a mapping), the delimiters are still
    stripped from the returned body — otherwise a caller that re-renders
    `(meta, body)` via `render_frontmatter` would prepend a fresh frontmatter
    block onto the old unparseable one still sitting in `body`, corrupting
    the file with two stacked `---` blocks.
    """
    match = FRONTMATTER_RE.match(text)
    if not match:
        return {}, text
    body = text[match.end() :]
    try:
        meta = yaml.safe_load(match.group(1))
    except yaml.YAMLError:
        return {}, body
    if not isinstance(meta, dict):
        return {}, body
    return meta, body


def render_frontmatter(meta: dict, body: str) -> str:
    """Serialize `meta` as a YAML frontmatter block prepended to `body`."""
    if not meta:
        return body
    yaml_text = yaml.dump(
        meta, sort_keys=False, default_flow_style=False, allow_unicode=True
    )
    return f"---\n{yaml_text}---\n{body}"
