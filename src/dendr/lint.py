"""Lint system — periodic health checks on the knowledge base.

Identifies: orphan pages, stale claims, contradictions, missing cross-refs.
Produces a markdown report in Wiki/_lint/.
"""

from __future__ import annotations

import logging
import sqlite3
from datetime import datetime, timedelta

from dendr import db
from dendr.config import Config
from dendr.wiki import append_activity_log

logger = logging.getLogger(__name__)


def _find_orphan_pages(config: Config, conn: sqlite3.Connection) -> list[str]:
    """Find wiki pages that have no claims pointing to them."""
    orphans: list[str] = []
    for page_dir in [config.concepts_dir, config.entities_dir]:
        if not page_dir.exists():
            continue
        for page_path in page_dir.glob("*.md"):
            slug = page_path.stem
            count = conn.execute(
                "SELECT COUNT(*) as n FROM claims WHERE concept_slug = ? AND status != 'superseded'",
                (slug,),
            ).fetchone()["n"]
            if count == 0:
                orphans.append(slug)
    return orphans


def _find_stale_claims(config: Config, conn: sqlite3.Connection) -> list[sqlite3.Row]:
    """Find claims not reinforced within the staleness window."""
    cutoff = (datetime.now() - timedelta(weeks=config.stale_claim_weeks)).isoformat()
    return conn.execute(
        """
        SELECT * FROM claims
        WHERE status IN ('created', 'reinforced')
          AND updated_at < ?
        ORDER BY updated_at ASC
        LIMIT 100
        """,
        (cutoff,),
    ).fetchall()


def _find_contradictions(conn: sqlite3.Connection) -> list[dict]:
    """Find all challenged claims (contradictions detected by semantic dedup)."""
    rows = conn.execute(
        """
        SELECT id, text, concept_slug, confidence, created_at
        FROM claims
        WHERE status = 'challenged'
          AND private = 0
        ORDER BY created_at DESC
        LIMIT 50
        """
    ).fetchall()

    return [
        {
            "id": r["id"],
            "text": r["text"],
            "concept_slug": r["concept_slug"],
            "confidence": r["confidence"],
        }
        for r in rows
    ]


def _find_missing_crossrefs(config: Config, conn: sqlite3.Connection) -> list[str]:
    """Find concept pages that reference slugs with no page."""
    missing: list[str] = []
    for page_dir in [config.concepts_dir, config.entities_dir]:
        if not page_dir.exists():
            continue
        for page_path in page_dir.glob("*.md"):
            content = page_path.read_text(encoding="utf-8")
            # Find [[slug]] references
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
    """Run all lint checks and produce a markdown report.

    Returns the report content and writes it to Wiki/_lint/.
    """
    logger.info("Running lint checks...")

    orphans = _find_orphan_pages(config, conn)
    stale = _find_stale_claims(config, conn)
    contradictions = _find_contradictions(conn)
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
        f"**Stats:** {stats['active_claims']} active claims, "
        f"{stats['concepts']} concepts, "
        f"{stats['challenged_claims']} challenged",
        "",
    ]

    # Contradictions (highest priority)
    report_lines.append(f"## Contradictions ({len(contradictions)})")
    report_lines.append("")
    if contradictions:
        report_lines.append("*These need Claude adjudication in the weekly synthesis.*")
        report_lines.append("")
        for c in contradictions:
            report_lines.append(
                f"- (id={c['id']}, c={c['confidence']:.2f}) "
                f"[{c['concept_slug']}] {c['text']}"
            )
            report_lines.append("")
    else:
        report_lines.append("None found.")
        report_lines.append("")

    # Stale claims
    report_lines.append(f"## Stale Claims ({len(stale)})")
    report_lines.append(f"*Not reinforced in {config.stale_claim_weeks} weeks.*")
    report_lines.append("")
    for s in stale[:20]:
        report_lines.append(
            f"- [{s['concept_slug']}] {s['text'][:80]}... "
            f"(c={s['confidence']:.2f}, last={s['updated_at'][:10]})"
        )
    if len(stale) > 20:
        report_lines.append(f"- *...and {len(stale) - 20} more*")
    report_lines.append("")

    # Orphan pages
    report_lines.append(f"## Orphan Pages ({len(orphans)})")
    report_lines.append("*Pages with no active claims.*")
    report_lines.append("")
    for o in orphans:
        report_lines.append(f"- [[{o}]]")
    report_lines.append("")

    # Missing cross-references
    report_lines.append(f"## Missing Cross-References ({len(missing_refs)})")
    report_lines.append("")
    for m in missing_refs[:30]:
        report_lines.append(f"- {m}")
    if len(missing_refs) > 30:
        report_lines.append(f"- *...and {len(missing_refs) - 30} more*")
    report_lines.append("")

    report = "\n".join(report_lines)

    # Write report
    config.lint_dir.mkdir(parents=True, exist_ok=True)
    report_path = config.lint_dir / f"lint-{now.strftime('%Y-%m-%d')}.md"
    report_path.write_text(report, encoding="utf-8")

    append_activity_log(
        config,
        f"LINT {len(contradictions)} contradictions, {len(stale)} stale, "
        f"{len(orphans)} orphans, {len(missing_refs)} missing refs",
    )

    logger.info(
        "Lint complete: %d contradictions, %d stale, %d orphans, %d missing refs",
        len(contradictions),
        len(stale),
        len(orphans),
        len(missing_refs),
    )

    return report
