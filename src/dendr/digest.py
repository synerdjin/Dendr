"""Weekly digest generator — three-layer context assembly for actionable advice.

Layer 1: Narrative Blocks — top annotated blocks with original text + metadata
Layer 2: Pattern Summaries — recurring topics, life area distribution, emotional trajectory
Layer 3: Claim-level Data — contradictions, dropped threads

Supports a feedback loop via per-section comment blocks in the rendered digest.
"""

from __future__ import annotations

import json
import logging
import re
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timedelta

from dendr import db
from dendr.config import Config
from dendr.models import Claim, ClaimKind, ClaimStatus
from dendr.wiki import append_activity_log

logger = logging.getLogger(__name__)

SECTION_IDS = [
    "narrative",
    "patterns",
    "open-loops",
    "contradictions",
    "dropped-threads",
    "activity",
]

_FEEDBACK_RE = re.compile(
    r"<!--\s*feedback:(\S+)\s*\n(.*?)-->",
    re.DOTALL,
)


@dataclass
class SectionFeedback:
    """Parsed feedback for one digest section."""

    section: str
    useful: bool | None = None
    note: str = ""


def _render_feedback_block(section_id: str) -> str:
    return f"<!-- feedback:{section_id}\nuseful: \nnote: \n-->"


def parse_feedback(digest_text: str) -> list[SectionFeedback]:
    """Parse all feedback comment blocks from a digest markdown string."""
    results: list[SectionFeedback] = []
    for match in _FEEDBACK_RE.finditer(digest_text):
        section = match.group(1)
        body = match.group(2)

        useful = None
        note = ""

        for line in body.splitlines():
            line = line.strip()
            if line.lower().startswith("useful:"):
                val = line.split(":", 1)[1].strip().lower()
                if val in ("yes", "true", "1", "y"):
                    useful = True
                elif val in ("no", "false", "0", "n"):
                    useful = False
            elif line.lower().startswith("note:"):
                note = line.split(":", 1)[1].strip()

        if useful is not None or note:
            results.append(SectionFeedback(section=section, useful=useful, note=note))

    return results


def ingest_feedback(
    conn: sqlite3.Connection,
    feedback: list[SectionFeedback],
    digest_date: str,
) -> dict:
    """Ingest feedback into feedback_scores table and optionally as claims."""
    ingested_claims = 0
    logged_ratings = 0

    for fb in feedback:
        db.upsert_feedback_score(conn, digest_date, fb.section, fb.useful, fb.note)
        logged_ratings += 1

        if fb.note:
            claim = Claim(
                id=None,
                text=fb.note,
                concept_slug="",
                source_block_ref=f"digest-feedback-{digest_date}",
                source_file_hash="",
                created_at=datetime.now(),
                updated_at=datetime.now(),
                confidence=0.9,
                status=ClaimStatus.CREATED,
                kind=ClaimKind.STATEMENT,
            )
            db.insert_claim(conn, claim)
            ingested_claims += 1

    return {"ingested_claims": ingested_claims, "logged_ratings": logged_ratings}


def _annotation_to_dict(row: sqlite3.Row) -> dict:
    """Convert a block_annotations row to a dict for JSON serialization."""
    return {
        "block_id": row["block_id"],
        "source_date": row["source_date"],
        "original_text": row["original_text"],
        "gist": row["gist"],
        "block_type": row["block_type"],
        "life_areas": json.loads(row["life_areas"]),
        "emotional_valence": row["emotional_valence"],
        "emotional_labels": json.loads(row["emotional_labels"]),
        "intensity": row["intensity"],
        "urgency": row["urgency"],
        "importance": row["importance"],
        "completion_status": row["completion_status"],
        "epistemic_status": row["epistemic_status"],
        "causal_links": json.loads(row["causal_links"]),
        "concepts": json.loads(row["concepts"]),
        "entities": json.loads(row["entities"]),
    }


def _gather_digest_data(
    config: Config, conn: sqlite3.Connection, weeks: int = 1
) -> dict:
    """Assemble three-layer digest data from the knowledge store."""
    now = datetime.now()
    since = (now - timedelta(weeks=weeks)).strftime("%Y-%m-%d")
    since_4w = (now - timedelta(weeks=4)).strftime("%Y-%m-%d")
    dropped_before = (now - timedelta(weeks=2)).isoformat()

    # Layer 1: Narrative blocks (original text + annotation metadata)
    significant_rows = db.get_significant_blocks(conn, since, limit=25)
    narrative_blocks = [_annotation_to_dict(r) for r in significant_rows]

    # Layer 2: Pattern summaries
    patterns = {
        "recurring_topics": db.get_recurring_topics(conn, since_4w),
        "life_area_distribution": db.get_life_area_distribution(conn, since),
        "emotional_trajectory": db.get_emotional_trajectory(conn, weeks=4),
        "open_tasks": [
            _annotation_to_dict(r) for r in db.get_open_tasks_annotated(conn)
        ],
        "completed_recently": [
            _annotation_to_dict(r) for r in db.get_completed_tasks(conn, since)
        ],
        "stale_tasks": [_annotation_to_dict(r) for r in db.get_stale_tasks(conn)],
    }

    # Layer 3: Claim-level data
    contradictions = db.get_all_contradictions(conn)
    dropped_threads = [
        {
            "concept_slug": r["concept_slug"],
            "text": r["text"],
            "created_at": r["created_at"],
        }
        for r in db.get_dropped_threads(conn, dropped_before)
    ]

    # Feedback effectiveness
    section_scores = db.get_section_effectiveness(conn)

    stats = db.get_stats(conn)

    return {
        "generated_at": now.isoformat(),
        "period_start": since,
        "period_end": now.strftime("%Y-%m-%d"),
        "stats": stats,
        "narrative_blocks": narrative_blocks,
        "patterns": patterns,
        "contradictions": contradictions,
        "dropped_threads": dropped_threads,
        "section_effectiveness": section_scores,
    }


def build_synthesis_prompt(data: dict) -> str:
    """Build the Claude synthesis prompt for actionable advice."""
    data_json = json.dumps(data, indent=2, default=str)

    return f"""You are Dendr's weekly advisor. Your job is to produce actionable,
specific advice grounded in the user's actual notes and patterns.

## Data from the past week

The data has three layers:

1. **narrative_blocks** — the user's original text with rich annotations
   (emotional valence, intensity, life areas, causal links). READ THESE FIRST
   to understand what the user is actually going through.

2. **patterns** — aggregated trends over 4 weeks: recurring topics with
   emotional trajectory, life area distribution, open/completed/stale tasks.

3. **contradictions** and **dropped_threads** — claim-level signals.

```json
{data_json}
```

## Section effectiveness

The `section_effectiveness` scores show which sections the user has found
useful in past digests (1.0 = always useful, 0.0 = never useful).
Spend more depth on high-scoring sections. Skip or minimize low-scoring ones.

## Output format

Write markdown with ONLY sections that have genuine, specific insights.
Use neutral, direct tone. Every piece of advice MUST reference the user's
actual words or patterns — no generic productivity advice.

### Available sections (include only if substantive):

**What's on your mind** — Synthesize the narrative blocks into a brief
picture of the user's current state. What are they focused on? What's
weighing on them? Use their own language.

**Open Loops** — Tasks and plans still open. Group by urgency/importance.
For stale items (open > 2 weeks), ask directly: still relevant?

**Reframes & Next Steps** — THE HIGHEST VALUE SECTION. Look for:
- Circling patterns: same problem approached repeatedly without resolution
- Emotional signals: high-intensity blocks reveal what actually matters
- Causal links the user stated: use their own reasoning to suggest next steps
- Implicit priorities: what keeps coming up reveals what matters most
Be specific. Quote the user's words. Suggest concrete actions.

**Emerging Patterns** — Topics gaining frequency or shifting emotional valence.
Note trends, don't over-interpret. Mention if a topic is trending more negative.

**Contradictions** — Conflicting claims. State both sides neutrally.
Ask whether the change was intentional.

**Dropped Threads** — Mentioned once, never revisited. Only surface
interesting/unfinished ones.

## Rules
- Lead with the most important insight.
- Be specific — quote or paraphrase the user's actual words.
- Use `[[concept-slug]]` for cross-references.
- Keep total output under 3000 words.
- Do NOT include a preamble or sign-off.
- Do NOT pad sections. If empty, skip entirely.
"""


def render_local_digest(data: dict) -> str:
    """Render an annotation-based digest using local processing (no Claude)."""
    now_str = data["period_end"]
    period_start = data["period_start"]
    lines = [
        "---",
        "type: digest",
        f"generated: {data['generated_at']}",
        f"period: {period_start} to {now_str}",
        "---",
        "",
        f"# Weekly Digest — {now_str}",
        "",
        f"**Period:** {period_start} → {now_str}  ",
        f"**Active claims:** {data['stats']['active_claims']} | "
        f"**Concepts:** {data['stats']['concepts']} | "
        f"**Annotations:** {data['stats'].get('annotations', 0)}",
        "",
    ]

    has_content = False

    # What's on your mind — top narrative blocks
    if data["narrative_blocks"]:
        has_content = True
        top = data["narrative_blocks"][:10]
        lines.append(f"## What's On Your Mind ({len(top)} key blocks)")
        lines.append("")
        for b in top:
            valence_indicator = ""
            v = b.get("emotional_valence", 0)
            if v <= -0.3:
                valence_indicator = " [negative]"
            elif v >= 0.3:
                valence_indicator = " [positive]"
            areas = ", ".join(b.get("life_areas", []))
            area_tag = f" ({areas})" if areas else ""
            lines.append(
                f"- **{b['source_date']}**{area_tag}{valence_indicator}: {b['gist']}"
            )
            if b.get("causal_links"):
                for link in b["causal_links"]:
                    lines.append(f"  - *Cause:* {link}")
        lines.append("")
        lines.append(_render_feedback_block("narrative"))
        lines.append("")

    # Open loops from annotations
    open_tasks = data["patterns"].get("open_tasks", [])
    stale_tasks = data["patterns"].get("stale_tasks", [])
    if open_tasks or stale_tasks:
        has_content = True
        lines.append(
            f"## Open Loops ({len(open_tasks)} active, {len(stale_tasks)} stale)"
        )
        lines.append("")
        for t in open_tasks[:15]:
            urgency = f" [{t['urgency']}]" if t.get("urgency") else ""
            importance = f" [{t['importance']}]" if t.get("importance") else ""
            lines.append(f"- {t['gist']}{urgency}{importance}")
        if stale_tasks:
            lines.append("")
            lines.append("**Stale (> 2 weeks, no update):**")
            for t in stale_tasks[:10]:
                lines.append(f"- {t['gist']} *({t['source_date']} — still relevant?)*")
        lines.append("")
        lines.append(_render_feedback_block("open-loops"))
        lines.append("")

    # Patterns
    topics = data["patterns"].get("recurring_topics", [])
    topics_3plus = [t for t in topics if t["mentions"] >= 2]
    trajectory = data["patterns"].get("emotional_trajectory", [])
    life_areas = data["patterns"].get("life_area_distribution", {})

    if topics_3plus or trajectory or life_areas:
        has_content = True
        lines.append("## Patterns")
        lines.append("")

        if topics_3plus:
            lines.append("**Recurring topics:**")
            for t in topics_3plus[:10]:
                trend_arrow = (
                    " ↑"
                    if t["trend"] == "improving"
                    else (" ↓" if t["trend"] == "worsening" else "")
                )
                lines.append(
                    f"- [[{t['concept']}]] — {t['mentions']} mentions, "
                    f"valence {t['avg_valence']:+.1f}{trend_arrow}"
                )
            lines.append("")

        if life_areas:
            lines.append("**Life area focus:**")
            for area, pct in life_areas.items():
                lines.append(f"- {area}: {pct}%")
            lines.append("")

        if trajectory:
            lines.append("**Emotional trajectory (4 weeks):**")
            for w in trajectory:
                bar = "█" * max(1, int(abs(w["avg_valence"]) * 10))
                sign = "+" if w["avg_valence"] >= 0 else ""
                lines.append(
                    f"- {w['week_start']}: {sign}{w['avg_valence']:.1f} {bar} "
                    f"({w['block_count']} blocks)"
                )
            lines.append("")

        lines.append(_render_feedback_block("patterns"))
        lines.append("")

    # Contradictions
    if data["contradictions"]:
        has_content = True
        lines.append(f"## Contradictions ({len(data['contradictions'])})")
        lines.append("")
        for c in data["contradictions"]:
            lines.append(
                f"- [c:{c['confidence']:.2f}] {c['text'][:120]} "
                f"([[{c['concept_slug']}]])"
            )
        lines.append("")
        lines.append(_render_feedback_block("contradictions"))
        lines.append("")

    # Dropped threads
    if data["dropped_threads"]:
        has_content = True
        lines.append(f"## Dropped Threads ({len(data['dropped_threads'])})")
        lines.append("*Mentioned once, never revisited.*")
        lines.append("")
        for d in data["dropped_threads"]:
            lines.append(f"- [[{d['concept_slug']}]]: {d['text'][:100]}")
        lines.append("")
        lines.append(_render_feedback_block("dropped-threads"))
        lines.append("")

    # Completed recently
    completed = data["patterns"].get("completed_recently", [])
    if completed:
        has_content = True
        lines.append(f"## Completed ({len(completed)})")
        lines.append("")
        for c in completed[:10]:
            lines.append(f"- ~~{c['gist']}~~ ({c['source_date']})")
        lines.append("")

    if not has_content:
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

    Before generating, parses feedback from the previous digest.
    """
    digest_path = config.wiki_dir / "digest.md"

    # Ingest feedback from previous digest
    feedback_stats = {"ingested_claims": 0, "logged_ratings": 0}
    if digest_path.exists():
        old_content = digest_path.read_text(encoding="utf-8")
        feedback = parse_feedback(old_content)
        if feedback:
            date_match = re.search(r"generated:\s*(\S+)", old_content)
            digest_date = date_match.group(1)[:10] if date_match else "unknown"
            feedback_stats = ingest_feedback(conn, feedback, digest_date)
            if feedback_stats["logged_ratings"] > 0:
                logger.info(
                    "Ingested feedback: %d ratings, %d claims",
                    feedback_stats["logged_ratings"],
                    feedback_stats["ingested_claims"],
                )

    data = _gather_digest_data(config, conn, weeks=weeks)

    if use_claude:
        prompt = build_synthesis_prompt(data)
        prompt_path = config.wiki_dir / "_digest_prompt.md"
        prompt_path.write_text(prompt, encoding="utf-8")
        content = render_local_digest(data)
        logger.info("Claude synthesis prompt written to %s", prompt_path)
    else:
        content = render_local_digest(data)

    digest_path.write_text(content, encoding="utf-8")

    n_blocks = len(data.get("narrative_blocks", []))
    n_tasks = len(data.get("patterns", {}).get("open_tasks", []))
    n_contras = len(data.get("contradictions", []))
    append_activity_log(
        config,
        f"DIGEST generated ({n_blocks} blocks, {n_tasks} open tasks, "
        f"{n_contras} contradictions, {feedback_stats['logged_ratings']} feedback)",
    )

    logger.info("Digest written to %s", digest_path)
    return str(digest_path)
