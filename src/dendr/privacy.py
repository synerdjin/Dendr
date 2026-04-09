"""Privacy filter for Dendr.

Tags blocks as private if they contain sensitive information that
should never be sent to Claude. Runs locally, before any external call.
"""

from __future__ import annotations

import re

from dendr.models import Block

# --- Patterns for common secrets ---

_PATTERNS: list[tuple[str, re.Pattern]] = [
    # API keys / tokens (generic)
    ("api_key", re.compile(
        r"""(?:api[_-]?key|api[_-]?secret|access[_-]?token|auth[_-]?token|bearer)"""
        r"""\s*[:=]\s*['"]?[A-Za-z0-9\-_\.]{20,}""",
        re.IGNORECASE,
    )),
    # AWS keys
    ("aws_key", re.compile(r"AKIA[0-9A-Z]{16}")),
    # GitHub tokens
    ("github_token", re.compile(r"gh[pousr]_[A-Za-z0-9_]{36,}")),
    # Generic secret assignment
    ("secret_assign", re.compile(
        r"""(?:secret|password|passwd|pwd|token|credential)"""
        r"""\s*[:=]\s*['"]?[^\s'"]{8,}""",
        re.IGNORECASE,
    )),
    # Private keys
    ("private_key", re.compile(r"-----BEGIN (?:RSA |EC |DSA )?PRIVATE KEY-----")),
    # Connection strings with passwords
    ("connection_string", re.compile(
        r"""(?:postgres|mysql|mongodb|redis|amqp)://[^:]+:[^@]+@""",
        re.IGNORECASE,
    )),
    # SSN-like patterns (US)
    ("ssn", re.compile(r"\b\d{3}-\d{2}-\d{4}\b")),
    # Credit card numbers (basic)
    ("credit_card", re.compile(r"\b(?:4\d{15}|5[1-5]\d{14}|3[47]\d{13})\b")),
    # Email + password combos
    ("email_password", re.compile(
        r"""[\w.+-]+@[\w-]+\.[\w.]+\s*[:\/]\s*\S{6,}""",
        re.IGNORECASE,
    )),
]

# User-defined redaction tag in daily notes
_REDACT_TAG_RE = re.compile(r"#dendr[/-]private|#private|#redact", re.IGNORECASE)


def is_private(block: Block) -> bool:
    """Check if a block contains sensitive content.

    Returns True if any privacy pattern matches or if the user
    has tagged the block with #dendr-private / #private / #redact.
    """
    text = block.text

    # User-explicit redaction tag
    if _REDACT_TAG_RE.search(text):
        return True

    # Pattern matching
    for _name, pattern in _PATTERNS:
        if pattern.search(text):
            return True

    return False


def filter_blocks(blocks: list[Block]) -> list[Block]:
    """Tag blocks as private in-place and return the full list."""
    for block in blocks:
        if is_private(block):
            block.private = True
    return blocks
