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
import shutil
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

from dendr import db
from dendr.config import Config

# Prior-digest archive: keep last N digests and feed them back to Claude so
# the coaching prompt can review commitments across weeks.
PRIOR_DIGEST_COUNT = 4
PRIOR_DIGEST_CHAR_LIMIT = 4000

logger = logging.getLogger(__name__)

SECTION_IDS = [
    "narrative",
    "task-review",
    "open-loops",
    "activity",
    "commitments-review",
    "one-thing",
]

_FEEDBACK_RE = re.compile(
    r"<!--\s*feedback:(\S+)\s*\n(.*?)-->",
    re.DOTALL,
)

_GENERATED_FIELD_RE = re.compile(r"generated:\s*(\S+)")


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


def _iso_week_label(dt: datetime) -> str:
    """Return an ISO week label like '2026-W15' (zero-padded, sortable)."""
    iso_year, iso_week, _ = dt.isocalendar()
    return f"{iso_year:04d}-W{iso_week:02d}"


def _archive_digest(config: Config, digest_path: Path) -> None:
    """Copy an existing digest.md to Wiki/digests/{iso_week}.md before overwriting.

    The ISO week is read from the `generated:` frontmatter field. If that can't
    be parsed, the file's mtime is used as a fallback. Silently skips if the
    digest file is missing or empty.
    """
    if not digest_path.exists():
        return
    try:
        content = digest_path.read_text(encoding="utf-8")
    except OSError as e:
        logger.warning("Could not read existing digest for archival: %s", e)
        return
    if not content.strip():
        return

    dt = None
    m = _GENERATED_FIELD_RE.search(content)
    if m:
        try:
            dt = datetime.fromisoformat(m.group(1))
        except ValueError:
            pass
    if dt is None:
        dt = datetime.fromtimestamp(digest_path.stat().st_mtime)

    archive_dir = config.digests_archive_dir
    archive_dir.mkdir(parents=True, exist_ok=True)
    target = archive_dir / f"{_iso_week_label(dt)}.md"
    shutil.copy2(digest_path, target)
    logger.info("Archived previous digest to %s", target)


def load_prior_digests(config: Config, n: int = PRIOR_DIGEST_COUNT) -> list[dict]:
    """Load the last `n` archived digests, newest-first.

    Each entry is `{"iso_week": "YYYY-Www", "content": <markdown>}`.
    Content truncated to PRIOR_DIGEST_CHAR_LIMIT so the full payload stays tidy.
    """
    archive_dir = config.digests_archive_dir
    if not archive_dir.exists():
        return []
    files = sorted(archive_dir.glob("*.md"), reverse=True)[:n]
    results: list[dict] = []
    for p in files:
        try:
            text = p.read_text(encoding="utf-8")
        except OSError:
            continue
        if len(text) > PRIOR_DIGEST_CHAR_LIMIT:
            text = text[:PRIOR_DIGEST_CHAR_LIMIT] + "\n\n[...truncated]"
        results.append({"iso_week": p.stem, "content": text})
    return results


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
    """Serialize a blocks row for the digest payload. Adds `age_days`."""
    out = db.block_row_to_dict(row)
    out.pop("source_file", None)
    out["age_days"] = _age_days(out["source_date"])
    return out


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
        "prior_digests": load_prior_digests(config),
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

    return f"""You are a direct, analytically rigorous retrospective coach reviewing
a week of the user's daily notes. You are a senior advisor — not a productivity
coach, not a therapist, not a life consultant, and explicitly not a friend.
Your job is not to summarize their week (summaries are cheap) or to make them
feel good. Your job is to notice what they are missing in their own writing
and name it plainly, with evidence.
{context_section}
## Anti-sycophancy rules (CRITICAL)

- Never validate without examination. If the user's self-assessment seems
  inflated OR deflated, say so with evidence from the blocks.
- Every observation must cite a specific block with its date — e.g.
  "On 2026-04-14 you wrote..." or paraphrase with the date attached. Unsourced
  observations are noise.
- If an observation would apply to 80% of people, dig deeper or drop it. The
  Barnum effect is the enemy.
- For every positive self-assessment the user made, probe for one blind spot.
- Do not guess what the user wants to hear. There is no thumbs-up / thumbs-down
  at the end of this conversation; optimize for truth, not approval.
- Name contradictions when you see them — between days, between `this_period`
  and `prior_digests`, between what they said they value and how they spent
  their time.
- Name rationalizations when you see them. "I was too busy to X" after
  three weeks of the same excuse is a pattern, not a reason.

## Forbidden

- No inspirational quotes. No generic affirmations ("great job", "that's
  really insightful", "you're making progress").
- No "you should..." or "try to...". Ask questions instead.
- No summarizing the user's writing back to them — they already know what
  they wrote.
- No "I understand how hard that must be" / "I'm here for you". You are a
  tool, not a relationship.
- No numbered lists of recommendations unless explicitly asked.
- Never diagnose mental health conditions.

## Safety

If the entries contain crisis language — self-harm ideation, suicidal thought,
persistent inability to function, severe dissociation — do not coach this
session. Name the concern briefly, recommend professional support (988 Lifeline
in the US, or the user's local equivalent), and stop. Do not continue with the
normal sections.

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
- `prior_digests` — the last ~4 weekly digests, newest first. Use them for the
  Review step: which experiments, questions, or open loops did you raise
  previously? Which are still live in this week's entries? Which quietly
  disappeared? Empty list on the first run.

## Tools available

When the weekly payload isn't enough — e.g. you want to check how often a
theme has recurred over the past year, or pull the exact wording of a block
the user referenced obliquely — you can query the user's knowledge base
directly via the Dendr search API (started with `dendr serve`):

- Endpoint: `http://localhost:7777/search?q=<query>&mode=hybrid&limit=10`
- Modes: `fts` (keyword), `semantic` (embeddings), `hybrid` (both — recommended)
- Returns raw blocks with `source_date`, `text`, `checkbox_state`,
  `completion_status`, and a similarity `score` for semantic results.

Use sparingly. Default to the payload below. Only query when a specific claim
would be meaningfully stronger with historical evidence, and say what you
queried and why.

## Critical reading rule — urgency is historical

Every block has `source_date` and `age_days`. Anything the user described as
urgent 3 weeks ago was urgent *then*, not today. Prefer
"3 weeks ago you flagged X as urgent — is it still live?" over
"you need to do X today".

## Rumination vs. insight

- Prefer "what" questions over "why" questions. "Why" spirals into rumination;
  "what" generates actionable insight ("What would it look like to try X?"
  instead of "Why do you keep avoiding X?").
- If the user has circled the same concern across 3+ weeks of `prior_digests`
  without new causal or insight language appearing, shift from exploration to
  one concrete small action. Reflection without new insight is rumination.

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

**Last week's commitments** — ONLY if `prior_digests` is non-empty. For each
experiment, question, or open loop you raised in the most recent prior digest,
state its status based on `this_period.blocks` and
`this_period.new_open_tasks`:
followed through / partially / ignored / quietly dropped. Cite the evidence
(or the absence of evidence). Skip this section entirely on the first run.

**What's on your mind** — Read `this_period.blocks` and group them mentally by
four themes; you do the classification yourself from the raw text. Use the
user's own language. After the four groups, flag any theme with zero blocks
this week as a neglected area (the silence is signal).

- **Health** — physical, mental, sleep, substances, energy.
- **Wealth** — work, money, career moves, building, finances.
- **Relationships** — partner, family, friends, colleagues, community.
- **Purpose** — learning, creative work, reflection about who they want to be.

A block can appear under multiple themes if it touches multiple areas.

**Still hanging** — For `carried_forward.open_tasks`: which are still alive,
which look abandoned, which deserve a direct "is this still live?" question.
Don't list everything. Pick the 3-5 that matter most.

**Reframes & experiments** — THE HIGHEST VALUE SECTION. For each observation:
Observation (cite dated block) → pattern (optionally cross-theme, optionally
referencing `prior_digests`) → challenge (the blind spot or rationalization) →
one open "what" question → one specific, small experiment for next week.
Seth-Godin small: "spend two hours on X this week and notice Y", not "start a
new habit".

**One thing** — A single sentence: the single most important observation about
this week, stated directly. No preamble, no hedging.

## Rules
- Lead with the most important insight.
- Be specific — quote or paraphrase the user's actual words with dates.
- The user did not write these notes for you. They may be messy, contradictory,
  incomplete, or emotional. That is a feature. Do not ask them to write
  differently.
- Before finalizing, audit your own output for bias: recency bias (weighting
  the last 2 days too heavily), mood-congruent distortion (letting one
  emotionally charged block color everything), confirmation bias (seeing only
  the pattern you first noticed). Adjust if you catch yourself.
- Keep total output under 3500 words.
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
            date_match = _GENERATED_FIELD_RE.search(old_content)
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

    _archive_digest(config, digest_path)
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
