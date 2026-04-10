"""Weekly digest generator — assembles a data payload and renders a briefing page.

Queries the claim store for the past week's activity, open tasks, contradictions,
emerging themes, and dropped threads. Outputs Wiki/digest.md with quality-ranked
sections. Sections with no insights are omitted.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path

from dendr import db
from dendr.config import Config
from dendr.wiki import append_activity_log

logger = logging.getLogger(__name__)


def _gather_digest_data(
    config: Config, conn: sqlite3.Connection, weeks: int = 1
) -> dict:
    """Query the claim store and assemble raw data for the digest.

    Returns a dict with all the data sections needed for rendering.
    """
    now = datetime.now()
    since = (now - timedelta(weeks=weeks)).isoformat()
    # "Dropped threads" = mentioned once, and that mention was before
    # the digest window — things you touched once and never returned to
    dropped_before = (now - timedelta(weeks=2)).isoformat()

    recent_claims = db.get_recent_claims(conn, since)
    open_tasks = db.get_open_tasks(conn)
    contradictions = db.get_all_contradictions(conn)
    concept_freq = db.get_concept_frequencies(conn, since)
    dropped = db.get_dropped_threads(conn, dropped_before)
    stats = db.get_stats(conn)

    return {
        "generated_at": now.isoformat(),
        "period_start": since,
        "period_end": now.isoformat(),
        "stats": stats,
        "recent_claims": [
            {
                "id": r["id"],
                "text": r["text"],
                "subject": r["subject"],
                "predicate": r["predicate"],
                "object": r["object"],
                "kind": r["kind"],
                "concept_slug": r["concept_slug"],
                "confidence": r["confidence"],
                "created_at": r["created_at"],
                "source_block_ref": r["source_block_ref"],
            }
            for r in recent_claims
        ],
        "open_tasks": [
            {
                "id": r["id"],
                "text": r["text"],
                "kind": r["kind"],
                "concept_slug": r["concept_slug"],
                "confidence": r["confidence"],
                "created_at": r["created_at"],
                "status": r["status"],
            }
            for r in open_tasks
        ],
        "contradictions": contradictions,
        "emerging_themes": [
            {"concept": slug, "mentions": count} for slug, count in concept_freq
        ],
        "dropped_threads": [
            {
                "concept_slug": r["concept_slug"],
                "text": r["text"],
                "created_at": r["created_at"],
            }
            for r in dropped
        ],
    }


def build_synthesis_prompt(data: dict) -> str:
    """Build the Claude synthesis prompt from gathered data.

    Returns a prompt string that can be passed to Claude for generating
    the final digest with actionable insights.
    """
    data_json = json.dumps(data, indent=2, default=str)

    return f"""You are Dendr's weekly digest synthesizer. Your job is to produce a
concise, actionable weekly briefing from the user's knowledge base activity.

## Raw data from the past week

```json
{data_json}
```

## Output format

Write a markdown document with ONLY the sections that have genuine insights.
Skip any section where you have nothing meaningful to say. Use neutral, direct tone.

### Available sections (include only if substantive):

**Contradictions** — Pair up conflicting claims. State both sides neutrally.
Ask whether the change was intentional or a genuine conflict.

**Open Loops** — Tasks and intentions from notes that have no follow-up.
Group related ones. For old items, ask whether they're still relevant.

**Emerging Themes** — Topics gaining frequency. Note the pattern, don't
over-interpret. Only mention themes with 3+ mentions.

**Reframes & Next Steps** — The highest-value section. Look for:
- Circling patterns (same problem approached repeatedly without resolution)
- Implicit priorities (what keeps coming up reveals what matters)
- Concrete next actions derivable from the user's own stated intentions
Be specific and actionable. Reference the user's own words.

**Dropped Threads** — Concepts mentioned once and never revisited. Only
surface ones that seem genuinely interesting or unfinished (skip trivial mentions).

## Rules
- Lead with the most important insight, not a summary of what you did.
- Be specific — quote or paraphrase the user's actual claims.
- Do NOT pad sections. If a section has only one item, that's fine.
- Do NOT add generic productivity advice. Every suggestion must be grounded
  in the user's actual notes.
- Use `[[concept-slug]]` for cross-references to wiki pages.
- Use `[c:0.82]` confidence pills when referencing specific claims.
- Keep the total output under 3000 words.
- Do NOT include a preamble or sign-off. Start directly with content.
"""


def render_local_digest(data: dict) -> str:
    """Render a data-only digest using local processing (no Claude).

    This is a fallback that presents the raw data in a readable format
    without the synthesis/reframing that Claude provides.
    """
    now_str = datetime.fromisoformat(data["generated_at"]).strftime("%Y-%m-%d %H:%M")
    period_start = datetime.fromisoformat(data["period_start"]).strftime("%Y-%m-%d")
    lines = [
        "---",
        "type: digest",
        f"generated: {data['generated_at']}",
        f"period: {period_start} to {now_str[:10]}",
        "---",
        "",
        f"# Weekly Digest — {now_str[:10]}",
        "",
        f"**Period:** {period_start} → {now_str[:10]}  ",
        f"**Active claims:** {data['stats']['active_claims']} | "
        f"**Concepts:** {data['stats']['concepts']} | "
        f"**Challenged:** {data['stats']['challenged_claims']}",
        "",
    ]

    # Contradictions
    if data["contradictions"]:
        lines.append(f"## Contradictions ({len(data['contradictions'])})")
        lines.append("")
        for c in data["contradictions"]:
            lines.append(f"### `{c['subject_predicate']}`")
            lines.append(
                f"- **A** [c:{c['claim_a']['confidence']:.2f}]: "
                f"{c['claim_a']['text']}"
            )
            lines.append(
                f"- **B** [c:{c['claim_b']['confidence']:.2f}]: "
                f"{c['claim_b']['text']}"
            )
            lines.append("- *Was this change intentional?*")
            lines.append("")

    # Open loops
    if data["open_tasks"]:
        lines.append(f"## Open Loops ({len(data['open_tasks'])})")
        lines.append("")
        for t in data["open_tasks"]:
            age = ""
            try:
                created = datetime.fromisoformat(t["created_at"])
                days = (datetime.now() - created).days
                if days > 14:
                    age = f" *(~{days}d ago — still relevant?)*"
                elif days > 0:
                    age = f" *({days}d ago)*"
            except (ValueError, TypeError):
                pass
            kind_label = t["kind"]
            slug = f" [[{t['concept_slug']}]]" if t["concept_slug"] else ""
            lines.append(f"- [{kind_label}]{slug} {t['text']}{age}")
        lines.append("")

    # Emerging themes
    themes = [t for t in data["emerging_themes"] if t["mentions"] >= 2]
    if themes:
        lines.append(f"## Emerging Themes ({len(themes)})")
        lines.append("")
        for t in themes:
            lines.append(f"- [[{t['concept']}]] — {t['mentions']} mentions this week")
        lines.append("")

    # Dropped threads
    if data["dropped_threads"]:
        lines.append(f"## Dropped Threads ({len(data['dropped_threads'])})")
        lines.append("*Mentioned once, never revisited.*")
        lines.append("")
        for d in data["dropped_threads"]:
            lines.append(f"- [[{d['concept_slug']}]]: {d['text'][:100]}")
        lines.append("")

    # Recent activity summary
    if data["recent_claims"]:
        kind_counts: dict[str, int] = {}
        for c in data["recent_claims"]:
            k = c.get("kind", "statement")
            kind_counts[k] = kind_counts.get(k, 0) + 1
        lines.append(f"## This Week's Activity ({len(data['recent_claims'])} claims)")
        lines.append("")
        for kind, count in sorted(kind_counts.items(), key=lambda x: -x[1]):
            lines.append(f"- **{kind}**: {count}")
        lines.append("")

    if not any([
        data["contradictions"],
        data["open_tasks"],
        themes,
        data["dropped_threads"],
    ]):
        lines.append("*No notable insights this week. Keep writing!*")
        lines.append("")

    return "\n".join(lines)


def generate_digest(
    config: Config,
    conn: sqlite3.Connection,
    weeks: int = 1,
    use_claude: bool = False,
) -> str:
    """Generate the weekly digest and write it to Wiki/digest.md.

    If use_claude=True, outputs the synthesis prompt for Claude to process.
    Otherwise, renders a local data-only digest.

    Returns the path to the written digest file.
    """
    data = _gather_digest_data(config, conn, weeks=weeks)

    if use_claude:
        # Write the synthesis prompt to a file for Claude Code to pick up
        prompt = build_synthesis_prompt(data)
        prompt_path = config.wiki_dir / "_digest_prompt.md"
        prompt_path.write_text(prompt, encoding="utf-8")

        # Also render the local version as a baseline
        content = render_local_digest(data)
        logger.info(
            "Claude synthesis prompt written to %s. "
            "Run a Claude Code session with this prompt for the full digest.",
            prompt_path,
        )
    else:
        content = render_local_digest(data)

    digest_path = config.wiki_dir / "digest.md"
    digest_path.write_text(content, encoding="utf-8")

    append_activity_log(
        config,
        f"DIGEST generated ({len(data['recent_claims'])} claims, "
        f"{len(data['open_tasks'])} open tasks, "
        f"{len(data['contradictions'])} contradictions)",
    )

    logger.info("Digest written to %s", digest_path)
    return str(digest_path)
