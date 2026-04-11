"""Weekly digest generator — period-scoped context assembly for actionable advice.

The synthesis payload is split by time:
- `this_period`    — narrative_blocks + new_open_tasks written in the digest window
- `carried_forward` — open_tasks + stale_tasks from BEFORE the window
- `patterns`       — 4-week aggregates (topics, trajectory, life areas, lifecycle)

Persistent user context (`Wiki/_user_context.md`) is injected into the Claude
synthesis prompt so the reviewer has stable background on who the user is.
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
from dendr.wiki import append_activity_log

logger = logging.getLogger(__name__)

SECTION_IDS = [
    "narrative",
    "task-review",
    "patterns",
    "open-loops",
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


def _task_review_bucket(days: int) -> str:
    """Pick an age bucket for a stale open task."""
    if days < 14:
        return "1-2w"
    if days < 30:
        return "2-4w"
    return "1m+"


_BUCKET_ORDER = ["1m+", "2-4w", "1-2w"]


def _render_task_review(tasks: list[dict]) -> str:
    """Render the Task Review section with closure markers.

    Each task gets a round-trip marker the user can edit in place:

        - [ ] **gist** — *written 3w ago (work)* <!-- closure:BLOCK_ID status:open -->

    Users flip `[ ]` → `[x]`, or change `status:open` to `done`,
    `abandoned`, `snoozed`, or `still-live`. The next ingest reconciles.
    """
    by_bucket: dict[str, list[dict]] = {}
    for t in tasks:
        bucket = _task_review_bucket(_age_days(t.get("source_date", "")))
        by_bucket.setdefault(bucket, []).append(t)

    lines = [f"## Task Review ({len(tasks)} open, >1 week old)"]
    lines.append("")
    lines.append(
        "*Flip `[ ]` to `[x]` to close, or edit `status:` to "
        "`done`, `abandoned`, `snoozed`, or `still-live`. "
        "The next ingest will reconcile.*"
    )
    lines.append("")

    for bucket in _BUCKET_ORDER:
        items = by_bucket.get(bucket)
        if not items:
            continue
        lines.append(f"### {bucket} old")
        lines.append("")
        for t in items:
            gist = t.get("gist") or "(no gist)"
            areas = t.get("life_areas") or []
            area_tag = f", {', '.join(areas)}" if areas else ""
            age = _age_suffix(t.get("source_date", ""))
            block_id = t.get("block_id", "")
            lines.append(
                f"- [ ] **{gist}** — *{age}{area_tag}* "
                f"<!-- closure:{block_id} status:open -->"
            )
        lines.append("")

    return "\n".join(lines).rstrip()


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
    """Ingest feedback into feedback_scores table."""
    logged_ratings = 0
    for fb in feedback:
        db.upsert_feedback_score(conn, digest_date, fb.section, fb.useful, fb.note)
        logged_ratings += 1
    return {"logged_ratings": logged_ratings}


def _age_days(source_date: str) -> int:
    """Days between source_date (YYYY-MM-DD) and today. 0 for today/malformed."""
    try:
        d = datetime.strptime(source_date, "%Y-%m-%d").date()
    except (ValueError, TypeError):
        return 0
    delta = datetime.now().date() - d
    return max(0, delta.days)


def _age_suffix(source_date: str) -> str:
    """Human-readable 'written Nw ago' suffix."""
    days = _age_days(source_date)
    if days == 0:
        return "written today"
    if days == 1:
        return "written 1d ago"
    if days < 7:
        return f"written {days}d ago"
    if days < 30:
        weeks = days // 7
        return f"written {weeks}w ago"
    if days < 365:
        months = days // 30
        return f"written {months}mo ago"
    return f"written {days // 365}y ago"


def _load_user_context(config: Config) -> str:
    """Read `Wiki/_user_context.md` if present — free-form markdown.

    Injected verbatim into the Claude synthesis prompt so the reviewer
    has persistent background on role, life situation, active goals, and
    stable constraints. Absent file returns empty string.
    """
    path = config.wiki_dir / "_user_context.md"
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8").strip()


def _annotation_to_dict(row: sqlite3.Row) -> dict:
    """Convert a block_annotations row to a dict for JSON serialization.

    `age_days` is computed at render time so Claude (and the local
    renderer) can distinguish urgency-when-written from urgency-now.
    """
    source_date = row["source_date"]
    return {
        "block_id": row["block_id"],
        "source_date": source_date,
        "age_days": _age_days(source_date),
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
    """Assemble period-scoped digest data from the knowledge store.

    The payload separates `this_period` (blocks written in the digest
    window) from `carried_forward` (still-open work from before the
    window). Aggregated `patterns` span a fixed 4-week lookback.
    """
    now = datetime.now()
    since = (now - timedelta(weeks=weeks)).strftime("%Y-%m-%d")
    since_4w = (now - timedelta(weeks=4)).strftime("%Y-%m-%d")

    # Narrative blocks: get_significant_blocks is already period-scoped.
    narrative_blocks = [
        _annotation_to_dict(r) for r in db.get_significant_blocks(conn, since, limit=25)
    ]

    # Open tasks: split into new (this period) vs carried-forward (older).
    all_open = [_annotation_to_dict(r) for r in db.get_open_tasks_annotated(conn)]
    new_open_tasks = [t for t in all_open if (t.get("source_date") or "") >= since]
    carried_open_tasks = [t for t in all_open if (t.get("source_date") or "") < since]

    stale_tasks = [_annotation_to_dict(r) for r in db.get_stale_tasks(conn)]

    patterns = {
        "recurring_topics": db.get_recurring_topics(conn, since_4w),
        "life_area_distribution": db.get_life_area_distribution(conn, since),
        "emotional_trajectory": db.get_emotional_trajectory(conn, weeks=4),
        "completed_recently": [
            _annotation_to_dict(r) for r in db.get_completed_tasks(conn, since)
        ],
        "task_lifecycle": db.get_task_lifecycle_stats(conn),
    }

    return {
        "generated_at": now.isoformat(),
        "period_start": since,
        "period_end": now.strftime("%Y-%m-%d"),
        "stats": db.get_stats(conn),
        "user_context": _load_user_context(config),
        "this_period": {
            "narrative_blocks": narrative_blocks,
            "new_open_tasks": new_open_tasks,
        },
        "carried_forward": {
            "open_tasks": carried_open_tasks,
            "stale_tasks": stale_tasks,
        },
        "patterns": patterns,
        "section_effectiveness": db.get_section_effectiveness(conn),
    }


def build_synthesis_prompt(data: dict) -> str:
    """Build the Claude synthesis prompt for actionable advice."""
    data_json = json.dumps(data, indent=2, default=str)

    user_context = (data.get("user_context") or "").strip()
    if user_context:
        context_section = f"\n## Who the user is\n\n{user_context}\n"
    else:
        context_section = (
            "\n## Who the user is\n\n"
            "*(No `Wiki/_user_context.md` file found. The user can create one "
            "with free-form background on their role, life situation, active "
            "goals, and stable constraints to give you better grounding.)*\n"
        )

    return f"""You are reviewing a week of the user's daily notes. You've been doing
this with them for a while — you know what matters to them, what they circle
back to, what they let slide. Your goal is not to summarize their week.
Summaries are cheap. Your goal is to notice what they might be missing in
their own writing and name it plainly.

Write like a thoughtful friend who actually read the notes — not a productivity
coach, not a therapist, not a life consultant. Direct, specific, and short.
{context_section}
## How the data is shaped

The payload is split by time, which matters a lot:

- `this_period.narrative_blocks` — what the user wrote THIS PAST WEEK. Read
  these first. This is the current shape of their attention.
- `this_period.new_open_tasks` — tasks they wrote this week that are still
  open. Normal open-loop territory: help them decide what to prioritize.
- `carried_forward.open_tasks` — tasks from BEFORE this period that are still
  unresolved. These are stuck, standing concerns, or abandoned in practice.
  Treat them differently from new tasks: the question isn't "what should I do
  about this" — it's "is this still alive, or do I need to let it go?"
- `carried_forward.stale_tasks` — same idea, filtered to the oldest ones.
- `patterns` — aggregates over 4 weeks: recurring topics, emotional trajectory,
  life area distribution, task lifecycle. Use sparingly; aggregates over small
  samples are noise.

## Critical reading rule — urgency is historical

Every block has `source_date` and `age_days`. The `urgency` (today / this_week /
someday) and `importance` (high / medium / low) fields reflect the user's state
**at `source_date`, not today**.

- A block from 3 weeks ago tagged `urgency: today` means "the user felt it was
  urgent 3 weeks ago". That is a SIGNAL (they cared a lot), not a CURRENT
  deadline.
- Anything with `age_days > 14` and `completion_status != 'done'` is either
  stale, abandoned, or a standing concern the user never resolved. Do not
  present it as if it's due this week.
- Prefer "3 weeks ago you flagged X as urgent — is it still live?" over
  "you need to do X today".

## Data

```json
{data_json}
```

## Section effectiveness

The `section_effectiveness` scores show which sections the user has found
useful in past digests (1.0 = always useful, 0.0 = never useful).
Spend depth on high-scoring sections. Skip or minimize low-scoring ones.

## Output format

Write markdown with ONLY sections that have genuine, specific insights. Every
piece of advice MUST reference the user's actual words or patterns — no
generic productivity advice.

### Sections (include only if substantive):

**What's on your mind** — Synthesize `this_period.narrative_blocks` into a
short picture of the user's current state. What are they focused on? What's
weighing on them? Use their own language.

**Still hanging** — For `carried_forward.open_tasks`: which of these are still
alive, which look abandoned, which deserve a direct "is this still live?"
question. Don't list everything. Pick the 3-5 that matter most.

**Reframes & next steps** — THE HIGHEST VALUE SECTION. Look for:
- Circling patterns: same problem approached repeatedly without resolution
- Emotional signals: high-intensity blocks reveal what actually matters
- Causal links the user stated: use their reasoning to suggest next steps
- Implicit priorities: what keeps coming up reveals what matters most
Be specific. Quote the user's words. Suggest concrete actions.

**Emerging patterns** — Topics gaining frequency or shifting emotional valence.
Note trends, don't over-interpret. Skip if sample sizes are tiny.

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
    this_period = data.get("this_period", {})
    carried_forward = data.get("carried_forward", {})
    narrative_blocks = this_period.get("narrative_blocks", [])
    fresh_tasks = this_period.get("new_open_tasks", [])
    review_tasks = carried_forward.get("open_tasks", [])

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
        f"**Annotations:** {data['stats']['annotations']} | "
        f"**Concepts:** {data['stats']['concepts']} | "
        f"**Open tasks:** {data['stats']['open_tasks']}",
        "",
    ]

    has_content = False

    # What's on your mind — top narrative blocks from this period.
    if narrative_blocks:
        has_content = True
        top = narrative_blocks[:10]
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

    # Task Review — carried-forward open tasks with closure markers.
    if review_tasks:
        has_content = True
        review_block = _render_task_review(review_tasks)
        lines.append(review_block)
        lines.append("")
        lines.append(_render_feedback_block("task-review"))
        lines.append("")

    # Open Loops — fresh tasks only (<7d). Stale ones live in Task Review.
    if fresh_tasks:
        has_content = True
        lines.append(f"## Open Loops ({len(fresh_tasks)} fresh)")
        lines.append("")
        for t in fresh_tasks[:15]:
            suffix = ""
            if t.get("urgency"):
                suffix += f" [{t['urgency']} when written]"
            if t.get("importance"):
                suffix += f" [{t['importance']}]"
            lines.append(f"- {t['gist']}{suffix}")
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

    # Completed recently
    completed = data["patterns"].get("completed_recently", [])
    if completed:
        has_content = True
        lines.append(f"## Completed ({len(completed)})")
        lines.append("")
        for c in completed[:10]:
            lines.append(f"- ~~{c['gist']}~~ ({c['source_date']})")
        lines.append("")

    # Task lifecycle stats
    lifecycle = data["patterns"].get("task_lifecycle", {})
    if lifecycle.get("total_created", 0) > 0:
        lines.append("## Task Lifecycle")
        lines.append("")
        lines.append(
            f"- **Created:** {lifecycle['total_created']} | "
            f"**Completed:** {lifecycle['total_completed']} | "
            f"**Abandoned:** {lifecycle['total_abandoned']}"
        )
        lines.append(f"- **Completion rate:** {lifecycle['completion_rate']:.0%}")
        if lifecycle.get("avg_days_to_completion", 0) > 0:
            lines.append(
                f"- **Avg days to completion:** {lifecycle['avg_days_to_completion']}"
            )
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
    feedback_stats = {"logged_ratings": 0}
    if digest_path.exists():
        old_content = digest_path.read_text(encoding="utf-8")
        feedback = parse_feedback(old_content)
        if feedback:
            date_match = re.search(r"generated:\s*(\S+)", old_content)
            digest_date = date_match.group(1)[:10] if date_match else "unknown"
            feedback_stats = ingest_feedback(conn, feedback, digest_date)
            if feedback_stats["logged_ratings"] > 0:
                logger.info(
                    "Ingested feedback: %d ratings",
                    feedback_stats["logged_ratings"],
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

    this_period = data.get("this_period", {})
    carried_forward = data.get("carried_forward", {})
    n_blocks = len(this_period.get("narrative_blocks", []))
    n_new = len(this_period.get("new_open_tasks", []))
    n_carried = len(carried_forward.get("open_tasks", []))
    append_activity_log(
        config,
        f"DIGEST generated ({n_blocks} blocks, {n_new} new tasks, "
        f"{n_carried} carried-forward, "
        f"{feedback_stats['logged_ratings']} feedback)",
    )

    logger.info("Digest written to %s", digest_path)
    return str(digest_path)
