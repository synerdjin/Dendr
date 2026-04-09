"""Wiki page updater — creates and maintains concept/entity/summary pages.

Respects the LLM-zone rule: human edits above <!-- llm-zone --> are sacred.
"""

from __future__ import annotations

import hashlib
import logging
import re
from datetime import datetime
from pathlib import Path

from dendr.config import Config
from dendr.db import (
    append_log,
    get_page_hash,
    set_page_hash,
    upsert_concept,
)
from dendr.llm import LLMClient
from dendr.models import Concept, PageType

logger = logging.getLogger(__name__)

_FRONTMATTER_RE = re.compile(r"^---\n(.*?)\n---\n", re.DOTALL)
_LLM_ZONE_START = "<!-- llm-zone -->"
_LLM_ZONE_END = "<!-- /llm-zone -->"
_HUMAN_ZONE_START = "<!-- human-zone -->"
_HUMAN_ZONE_END = "<!-- /human-zone -->"


def _hash_content(content: str) -> str:
    return hashlib.sha256(content.encode()).hexdigest()[:16]


def _page_template(
    slug: str, title: str, page_type: PageType
) -> str:
    """Generate a new wiki page with the standard template."""
    now = datetime.now().strftime("%Y-%m-%d")
    return f"""---
type: {page_type.value}
slug: {slug}
human_touched: false
last_llm_hash: ""
created: {now}
updated: {now}
---

# {title}

{_HUMAN_ZONE_START}

{_HUMAN_ZONE_END}

{_LLM_ZONE_START}

{_LLM_ZONE_END}
"""


def _extract_llm_zone(content: str) -> str:
    """Extract the content between llm-zone markers."""
    start = content.find(_LLM_ZONE_START)
    end = content.find(_LLM_ZONE_END)
    if start == -1 or end == -1:
        return ""
    zone_start = start + len(_LLM_ZONE_START)
    return content[zone_start:end].strip()


def _replace_llm_zone(content: str, new_zone: str) -> str:
    """Replace the content between llm-zone markers."""
    start = content.find(_LLM_ZONE_START)
    end = content.find(_LLM_ZONE_END)
    if start == -1 or end == -1:
        return content
    zone_start = start + len(_LLM_ZONE_START)
    return content[:zone_start] + "\n" + new_zone + "\n" + content[end:]


def _is_human_touched(content: str, stored_hash: str | None) -> bool:
    """Check if a page was edited by a human since last LLM write."""
    if stored_hash is None:
        return False
    current_hash = _hash_content(content)
    return current_hash != stored_hash


def _update_frontmatter(content: str, key: str, value: str) -> str:
    """Update a single frontmatter field."""
    match = _FRONTMATTER_RE.match(content)
    if not match:
        return content
    fm = match.group(1)
    pattern = re.compile(rf"^{re.escape(key)}:\s*.*$", re.MULTILINE)
    if pattern.search(fm):
        fm = pattern.sub(f"{key}: {value}", fm)
    else:
        fm += f"\n{key}: {value}"
    return f"---\n{fm}\n---\n" + content[match.end():]


def get_page_path(config: Config, slug: str, page_type: PageType) -> Path:
    """Determine the file path for a concept/entity page."""
    if page_type == PageType.ENTITY:
        return config.entities_dir / f"{slug}.md"
    elif page_type == PageType.SUMMARY:
        return config.summaries_dir / f"{slug}.md"
    return config.concepts_dir / f"{slug}.md"


def ensure_page(
    config: Config,
    conn,
    slug: str,
    title: str,
    page_type: PageType = PageType.CONCEPT,
) -> Path:
    """Create a wiki page if it doesn't exist. Returns the path."""
    page_path = get_page_path(config, slug, page_type)

    if not page_path.exists():
        page_path.parent.mkdir(parents=True, exist_ok=True)
        content = _page_template(slug, title, page_type)
        page_path.write_text(content, encoding="utf-8")
        content_hash = _hash_content(content)
        set_page_hash(conn, str(page_path.relative_to(config.vault_path)), content_hash)

        # Register in concept store
        upsert_concept(
            conn,
            Concept(
                slug=slug,
                title=title,
                page_type=page_type,
                created_at=datetime.now(),
                updated_at=datetime.now(),
                page_path=str(page_path.relative_to(config.vault_path)),
            ),
        )
        append_log(
            conn,
            "page_created",
            {"slug": slug, "type": page_type.value, "path": str(page_path)},
        )
        logger.info("Created wiki page: %s", page_path)

    return page_path


def append_evidence(
    config: Config,
    conn,
    llm: LLMClient,
    slug: str,
    title: str,
    new_evidence: list[str],
    source_ref: str,
    page_type: PageType = PageType.CONCEPT,
) -> None:
    """Append new evidence to a wiki page's LLM zone.

    Respects human_touched: if the page was edited by the user, we only
    append (never rewrite the LLM zone).
    """
    page_path = ensure_page(config, conn, slug, title, page_type)
    rel_path = str(page_path.relative_to(config.vault_path))
    content = page_path.read_text(encoding="utf-8")

    stored_hash = get_page_hash(conn, rel_path)
    human_touched = _is_human_touched(content, stored_hash)

    if human_touched:
        content = _update_frontmatter(content, "human_touched", "true")
        logger.info("Page %s is human-touched, append-only mode", slug)

    # Get existing LLM zone content
    existing_zone = _extract_llm_zone(content)

    # Generate new section via local LLM
    now = datetime.now().strftime("%Y-%m-%d")
    section_header = f"\n### Evidence ({now}, from {source_ref})\n\n"
    section_body = llm.generate_wiki_section(title, new_evidence, existing_zone)

    if human_touched:
        # Append-only: add to end of LLM zone
        new_zone = existing_zone + section_header + section_body
    else:
        # Can rewrite — but we still just append for incremental coherence
        new_zone = existing_zone + section_header + section_body

    # Replace the LLM zone
    new_content = _replace_llm_zone(content, new_zone)
    new_content = _update_frontmatter(new_content, "updated", now)

    # Write and record hash
    page_path.write_text(new_content, encoding="utf-8")
    new_hash = _hash_content(new_content)
    set_page_hash(conn, rel_path, new_hash)
    new_content = _update_frontmatter(new_content, "last_llm_hash", new_hash)
    page_path.write_text(new_content, encoding="utf-8")

    append_log(
        conn,
        "evidence_added",
        {"slug": slug, "source": source_ref, "claims_count": len(new_evidence)},
    )
    logger.info("Appended %d evidence items to %s", len(new_evidence), slug)


def update_index(config: Config, conn) -> None:
    """Regenerate Wiki/index.md from the concepts table."""
    concepts = conn.execute(
        "SELECT slug, title, page_type, updated_at FROM concepts ORDER BY page_type, slug"
    ).fetchall()

    lines = ["---", "type: index", f"updated: {datetime.now().strftime('%Y-%m-%d')}", "---", "", "# Dendr Knowledge Index", ""]

    current_type = None
    for c in concepts:
        if c["page_type"] != current_type:
            current_type = c["page_type"]
            lines.append(f"\n## {current_type.title()}s\n")
        lines.append(f"- [[{c['slug']}]] — {c['title']} *(updated {c['updated_at'][:10]})*")

    stats = conn.execute(
        "SELECT COUNT(*) as n FROM claims WHERE status != 'superseded'"
    ).fetchone()
    lines.extend([
        "",
        "---",
        f"*{len(concepts)} pages, {stats['n']} active claims*",
    ])

    index_path = config.wiki_dir / "index.md"
    index_path.write_text("\n".join(lines), encoding="utf-8")


def append_activity_log(config: Config, entry: str) -> None:
    """Append to Wiki/log.md."""
    log_path = config.wiki_dir / "log.md"
    if not log_path.exists():
        log_path.write_text("---\ntype: log\n---\n\n# Activity Log\n\n", encoding="utf-8")

    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(f"- `{now}` {entry}\n")
