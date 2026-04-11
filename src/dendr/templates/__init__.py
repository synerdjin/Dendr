"""Static template files shipped with Dendr (schema, Claude prompts)."""

from __future__ import annotations

from pathlib import Path

TEMPLATES_DIR = Path(__file__).parent


def read(name: str) -> str:
    """Return the text of a bundled template by filename."""
    return (TEMPLATES_DIR / name).read_text(encoding="utf-8")
