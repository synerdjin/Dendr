"""Weekly digest generator — raw-text context assembly for Claude synthesis.

The synthesis payload splits by time:
- `this_period`     — non-private blocks written in the digest window
- `carried_forward` — open tasks from BEFORE the window (still unresolved)

Persistent user context (`Wiki/_user_context.md`) is injected into the prompt.
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

logger = logging.getLogger(__name__)

SECTION_IDS = [
    "narrative",
    "task-review",
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

        - [ ] **first line of block text** — *written 3w ago* <!-- closure:BLOCK_ID status:open -->

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
            label = _task_label(t.get("text", ""))
            age = _age_suffix(t.get("source_date", ""))
            block_id = t.get("block_id", "")
            lines.append(
                f"- [ ] **{label}** — *{age}* <!-- closure:{block_id} status:open -->"
            )
        lines.append("")

    return "\n".join(lines).rstrip()


def _task_label(text: str, max_len: int = 80) -> str:
    """First meaningful line of a block, stripped of checkbox markup."""
    first = text.strip().splitlines()[0] if text.strip() else ""
    first = re.sub(r"^\[[ xX]\]\s*", "", first).strip()
    if len(first) > max_len:
        first = first[: max_len - 1].rstrip() + "…"
    return first or "(no text)"


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


def _block_to_dict(row: sqlite3.Row) -> dict:
    """Convert a blocks row to a dict for JSON serialization."""
    source_date = row["source_date"]
    return {
        "block_id": row["block_id"],
        "source_date": source_date,
        "age_days": _age_days(source_date),
        "text": row["text"],
        "checkbox_state": row["checkbox_state"],
        "completion_status": row["completion_status"],
    }


def _gather_digest_data(
    config: Config, conn: sqlite3.Connection, weeks: int = 1
) -> dict:
    """Assemble period-scoped digest data from the knowledge store.

    The payload separates `this_period` (blocks written in the digest
    window) from `carried_forward` (still-open work from before).
    """
    now = datetime.now()
    since = (now - timedelta(weeks=weeks)).strftime("%Y-%m-%d")

    period_blocks = [
        _block_to_dict(r) for r in db.get_blocks_in_period(conn, since, limit=500)
    ]

    all_open = [_block_to_dict(r) for r in db.get_open_tasks(conn)]
    new_open_tasks = [t for t in all_open if (t.get("source_date") or "") >= since]
    carried_open_tasks = [t for t in all_open if (t.get("source_date") or "") < since]

    return {
        "generated_at": now.isoformat(),
        "period_start": since,
        "period_end": now.strftime("%Y-%m-%d"),
        "stats": db.get_stats(conn),
        "user_context": _load_user_context(config),
        "this_period": {
            "blocks": period_blocks,
            "new_open_tasks": new_open_tasks,
        },
        "carried_forward": {
            "open_tasks": carried_open_tasks,
        },
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

Each block is raw Markdown the user wrote, plus minimal structural metadata:

- `block_id`, `source_date`, `age_days`
- `text` — the block's raw content
- `checkbox_state` — `open` (`- [ ]`), `closed` (`- [x]`), or `none`
- `completion_status` — only set when the user closed a task via a digest
  review; `null` for everything else

You do the classification, affect reading, and clustering yourself — nothing
has been pre-tagged for you. Read the raw text.

The payload is split by time:

- `this_period.blocks` — what the user wrote THIS PAST WEEK. Read these first.
  This is the current shape of their attention.
- `this_period.new_open_tasks` — open-checkbox tasks from this week.
- `carried_forward.open_tasks` — open-checkbox tasks from BEFORE this period
  that are still unresolved. These are stuck, standing concerns, or abandoned
  in practice. The question isn't "what should I do about this" — it's "is
  this still alive, or do I need to let it go?"

## Critical reading rule — urgency is historical

Every block has `source_date` and `age_days`. Anything the user described as
urgent 3 weeks ago was urgent *then*, not today. Prefer
"3 weeks ago you flagged X as urgent — is it still live?" over
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

**What's on your mind** — Synthesize `this_period.blocks` into a short picture
of the user's current state. What are they focused on? What's weighing on
them? Use their own language.

**Still hanging** — For `carried_forward.open_tasks`: which of these are still
alive, which look abandoned, which deserve a direct "is this still live?"
question. Don't list everything. Pick the 3-5 that matter most.

**Reframes & next steps** — THE HIGHEST VALUE SECTION. Look for:
- Circling patterns: same problem approached repeatedly without resolution
- Implicit priorities: what keeps coming up reveals what matters most
- Causal reasoning stated in the text: use their own reasoning to suggest
  next steps
Be specific. Quote the user's words. Suggest concrete actions.

## Rules
- Lead with the most important insight.
- Be specific — quote or paraphrase the user's actual words.
- Keep total output under 3000 words.
- Do NOT include a preamble or sign-off.
- Do NOT pad sections. If empty, skip entirely.
"""


def render_local_digest(data: dict) -> str:
    """Render a minimal local digest (no Claude). Mostly the task review."""
    now_str = data["period_end"]
    period_start = data["period_start"]
    this_period = data.get("this_period", {})
    carried_forward = data.get("carried_forward", {})
    period_blocks = this_period.get("blocks", [])
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
        f"**Blocks:** {data['stats']['blocks']} | "
        f"**Open tasks:** {data['stats']['open_tasks']}",
        "",
    ]

    has_content = False

    # Recent writing — just the raw blocks from this period.
    if period_blocks:
        has_content = True
        lines.append(f"## This Week ({len(period_blocks)} blocks)")
        lines.append("")
        for b in period_blocks[:25]:
            lines.append(
                f"- **{b['source_date']}**: {_task_label(b.get('text', ''), max_len=140)}"
            )
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
            lines.append(f"- {_task_label(t.get('text', ''))}")
        lines.append("")
        lines.append(_render_feedback_block("open-loops"))
        lines.append("")

    if not has_content:
        lines.append("*No notable activity this week. Keep writing!*")
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
    n_blocks = len(this_period.get("blocks", []))
    n_new = len(this_period.get("new_open_tasks", []))
    n_carried = len(carried_forward.get("open_tasks", []))
    config.append_activity_log(
        f"DIGEST generated ({n_blocks} blocks, {n_new} new tasks, "
        f"{n_carried} carried-forward, "
        f"{feedback_stats['logged_ratings']} feedback)",
    )

    logger.info("Digest written to %s", digest_path)
    return str(digest_path)
