"""Lint system — periodic health checks on the knowledge base.

Identifies: orphan pages, missing cross-refs.
Produces a markdown report in Wiki/_lint/.
"""

from __future__ import annotations

import logging
import sqlite3
from datetime import datetime

from dendr import db
from dendr.config import Config
from dendr.wiki import append_activity_log

logger = logging.getLogger(__name__)


def _find_orphan_pages(config: Config, conn: sqlite3.Connection) -> list[str]:
    """Find wiki pages that no annotation references."""
    orphans: list[str] = []
    for page_dir in [config.concepts_dir, config.entities_dir]:
        if not page_dir.exists():
            continue
        for page_path in page_dir.glob("*.md"):
            slug = page_path.stem
            # JSON array text search — slugs are stored as JSON arrays in
            # block_annotations.concepts / .entities.
            needle = f'"{slug}"'
            row = conn.execute(
                """
                SELECT COUNT(*) as n FROM block_annotations
                WHERE concepts LIKE ? OR entities LIKE ?
                """,
                (f"%{needle}%", f"%{needle}%"),
            ).fetchone()
            if row["n"] == 0:
                orphans.append(slug)
    return orphans


def _find_missing_crossrefs(config: Config, conn: sqlite3.Connection) -> list[str]:
    """Find concept pages that reference slugs with no page."""
    missing: list[str] = []
    for page_dir in [config.concepts_dir, config.entities_dir]:
        if not page_dir.exists():
            continue
        for page_path in page_dir.glob("*.md"):
            content = page_path.read_text(encoding="utf-8")
            import re

            refs = re.findall(r"\[\[([^\]|]+?)(?:\|[^\]]+)?\]\]", content)
            for ref in refs:
                ref_slug = ref.strip().lower().replace(" ", "-")
                exists = conn.execute(
                    "SELECT 1 FROM concepts WHERE slug = ?", (ref_slug,)
                ).fetchone()
                if not exists:
                    missing.append(f"{page_path.stem} → [[{ref}]]")
    return missing


def run_lint(config: Config, conn: sqlite3.Connection) -> str:
    """Run lint checks and produce a markdown report."""
    logger.info("Running lint checks...")

    orphans = _find_orphan_pages(config, conn)
    missing_refs = _find_missing_crossrefs(config, conn)
    stats = db.get_stats(conn)

    now = datetime.now()
    report_lines = [
        "---",
        "type: lint-report",
        f"date: {now.strftime('%Y-%m-%d')}",
        "---",
        "",
        f"# Lint Report — {now.strftime('%Y-%m-%d %H:%M')}",
        "",
        f"**Stats:** {stats['annotations']} annotations, "
        f"{stats['concepts']} concepts, "
        f"{stats['open_tasks']} open tasks",
        "",
    ]

    report_lines.append(f"## Orphan Pages ({len(orphans)})")
    report_lines.append("*Pages not referenced by any annotation.*")
    report_lines.append("")
    for o in orphans:
        report_lines.append(f"- [[{o}]]")
    report_lines.append("")

    report_lines.append(f"## Missing Cross-References ({len(missing_refs)})")
    report_lines.append("")
    for m in missing_refs[:30]:
        report_lines.append(f"- {m}")
    if len(missing_refs) > 30:
        report_lines.append(f"- *...and {len(missing_refs) - 30} more*")
    report_lines.append("")

    report = "\n".join(report_lines)

    config.lint_dir.mkdir(parents=True, exist_ok=True)
    report_path = config.lint_dir / f"lint-{now.strftime('%Y-%m-%d')}.md"
    report_path.write_text(report, encoding="utf-8")

    append_activity_log(
        config,
        f"LINT {len(orphans)} orphans, {len(missing_refs)} missing refs",
    )

    logger.info(
        "Lint complete: %d orphans, %d missing refs",
        len(orphans),
        len(missing_refs),
    )

    return report
