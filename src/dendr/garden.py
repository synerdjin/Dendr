"""Digital-garden support for `Pages/` — growth-stage tracking over the
user's hand-written topic notes.

Deliberately outside the ingest pipeline: `Pages/` is user-authored (imported
from Logseq, predates Dendr), not Daily/-derived, and never touches
`state.sqlite`. This module is a pure filesystem scan of `Pages/*.md`
frontmatter, safe to re-run any time, and gated off gracefully wherever
`Pages/` doesn't exist — the same posture `digest.py` already takes with
optional Wiki files like `_user_context.md`/`_intentions.md`.

Growth stages (seedling → budding → evergreen) and the `planted`/`tended`
dates are stored as plain frontmatter on each page, so they're visible and
editable directly in Obsidian — this module reads/backfills them, it doesn't
own them.
"""

from __future__ import annotations

import hashlib
import logging
import re
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path

from dendr.config import Config
from dendr.frontmatter import FRONTMATTER_RE, render_frontmatter, split_frontmatter
from dendr.fsutil import (
    atomic_write_text,
    is_conflicted_copy,
    iso_week_label,
    scan_order,
)

logger = logging.getLogger(__name__)

STAGES = ("seedling", "budding", "evergreen")
_STAGE_ORDER = {stage: i for i, stage in enumerate(STAGES)}

# One-time initial guess for notes that have never been staged before — not a
# quality judgment, just a starting point the user is free to override by
# hand. backfill_frontmatter() never touches a note that already has a
# `stage` key, no matter what it's set to.
_BUDDING_WORDS = 50
_EVERGREEN_WORDS = 300
_BUDDING_LINKS = 2
_EVERGREEN_LINKS = 5

_WIKILINK_RE = re.compile(r"\[\[([^\]|#]+)")

STALE_EVERGREEN_LIMIT = 5
RESURFACE_COUNT = 2


@dataclass
class PageNote:
    path: Path
    title: str
    stage: str
    planted: str  # YYYY-MM-DD
    tended: str  # YYYY-MM-DD
    word_count: int
    link_count: int

    @property
    def slug(self) -> str:
        return self.path.stem


def _coerce_date_str(value: object, fallback: str) -> str:
    """Normalize a frontmatter date field to YYYY-MM-DD.

    `yaml.safe_load` parses unquoted dates (`planted: 2026-04-11`) into
    `datetime.date`/`datetime.datetime` objects rather than strings.
    """
    if isinstance(value, datetime):
        return value.date().isoformat()
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, str) and value.strip():
        return value.strip()[:10]
    return fallback


def _file_dates(path: Path) -> tuple[str, str]:
    """Best-effort (planted, tended) from filesystem metadata.

    `planted` uses birth time where available (macOS `st_birthtime` — Dendr
    targets macOS only, see repo CLAUDE.md's v6 upgrade note), falling back
    to mtime. `tended` is always mtime.
    """
    st = path.stat()
    tended = datetime.fromtimestamp(st.st_mtime).strftime("%Y-%m-%d")
    birth = getattr(st, "st_birthtime", None)
    planted = (
        datetime.fromtimestamp(birth).strftime("%Y-%m-%d")
        if birth is not None
        else tended
    )
    return planted, tended


def _guess_stage(word_count: int, link_count: int) -> str:
    if word_count >= _EVERGREEN_WORDS or link_count >= _EVERGREEN_LINKS:
        return "evergreen"
    if word_count >= _BUDDING_WORDS or link_count >= _BUDDING_LINKS:
        return "budding"
    return "seedling"


def _note_title(path: Path, body: str) -> str:
    for line in body.splitlines():
        stripped = line.strip()
        if stripped.startswith("# "):
            return stripped[2:].strip()
    return path.stem


@dataclass
class _LoadedPage:
    meta: dict
    body: str
    planted_fs: str
    tended_fs: str
    word_count: int
    link_count: int
    frontmatter_unparsed: bool


def _load_page(path: Path) -> _LoadedPage | None:
    """Read and parse one Pages/ note. `None` if it can't be read as text."""
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return None
    meta, body = split_frontmatter(text)
    # split_frontmatter returns meta={} both when there's genuinely no
    # frontmatter block AND when a block exists but fails to parse as a YAML
    # mapping — the two need to be told apart here so backfill_frontmatter
    # doesn't mistake "unparseable" for "empty" and overwrite real content.
    frontmatter_unparsed = not meta and FRONTMATTER_RE.match(text) is not None
    planted_fs, tended_fs = _file_dates(path)
    return _LoadedPage(
        meta=meta,
        body=body,
        planted_fs=planted_fs,
        tended_fs=tended_fs,
        word_count=len(body.split()),
        link_count=len(_WIKILINK_RE.findall(body)),
        frontmatter_unparsed=frontmatter_unparsed,
    )


def _iter_page_paths(pages_dir: Path):
    """Pages/*.md in canonical-first order, sync-conflict copies excluded.

    Pages/ has no `^dendr-<ulid>` block-ref mechanism to fall back on the way
    Daily/ does (see pipeline.scan_daily_notes) to confirm a conflict-shaped
    name is *actually* a duplicate of something — a conflict-shaped name
    (`fsutil.is_conflicted_copy`) is skipped outright on pattern match alone.
    That's usually right (sync artifacts), but a real page titled e.g.
    "Draft (2).md" would false-positive and silently vanish from the garden;
    logging a warning here at least makes that discoverable (rename the file)
    instead of a page disappearing with zero trace.
    """
    for path in sorted(pages_dir.glob("*.md"), key=scan_order):
        if is_conflicted_copy(path):
            logger.warning(
                "Skipping %s: looks like a sync-conflict copy. If this is a "
                "real page title, rename it so it doesn't match.",
                path.name,
            )
            continue
        yield path


def scan_pages(config: Config) -> list[PageNote]:
    """Read every `Pages/*.md` note's frontmatter into a PageNote.

    Returns `[]` if `Pages/` doesn't exist. Read-only — never writes.
    """
    pages_dir = config.pages_dir
    if not pages_dir.exists():
        return []

    notes = []
    for path in _iter_page_paths(pages_dir):
        loaded = _load_page(path)
        if loaded is None:
            continue

        stage = loaded.meta.get("stage")
        if stage is None:
            stage = _guess_stage(loaded.word_count, loaded.link_count)

        notes.append(
            PageNote(
                path=path,
                title=_note_title(path, loaded.body),
                stage=stage,
                planted=_coerce_date_str(loaded.meta.get("planted"), loaded.planted_fs),
                tended=_coerce_date_str(loaded.meta.get("tended"), loaded.tended_fs),
                word_count=loaded.word_count,
                link_count=loaded.link_count,
            )
        )
    return notes


def backfill_frontmatter(
    config: Config, dry_run: bool = False
) -> list[tuple[Path, dict]]:
    """Fill in missing stage/planted/tended frontmatter on `Pages/` notes.

    Only ever fills keys that are entirely absent from a note's frontmatter
    — an existing `stage` (even one outside STAGES) is left alone, since a
    hand-edit always wins over the heuristic guess. Safe to re-run: notes
    that already have all three keys are untouched. Returns the list of
    `(path, added_fields)` for reporting, whether or not `dry_run` is set.
    """
    pages_dir = config.pages_dir
    if not pages_dir.exists():
        return []

    changes: list[tuple[Path, dict]] = []
    for path in _iter_page_paths(pages_dir):
        loaded = _load_page(path)
        if loaded is None:
            continue
        if loaded.frontmatter_unparsed:
            # A frontmatter block exists but didn't parse as a YAML mapping —
            # meta came back {} not because the note has no frontmatter, but
            # because we couldn't read what's there. Writing render_frontmatter
            # over it would silently discard whatever fields it actually has.
            logger.warning(
                "Skipping %s: frontmatter exists but isn't valid YAML (or "
                "isn't a mapping). Fix it by hand, then rerun.",
                path.name,
            )
            continue
        meta = loaded.meta

        added: dict = {}
        if "stage" not in meta:
            added["stage"] = _guess_stage(loaded.word_count, loaded.link_count)
        if "planted" not in meta:
            added["planted"] = loaded.planted_fs
        if "tended" not in meta:
            added["tended"] = loaded.tended_fs

        if not added:
            continue

        changes.append((path, added))
        if not dry_run:
            atomic_write_text(path, render_frontmatter({**meta, **added}, loaded.body))

    return changes


def compute_garden_summary(
    pages: list[PageNote],
    period_start: str,
    period_end: str,
    iso_week: str,
) -> dict:
    """Summarize garden state for the digest payload and `Wiki/garden.md`.

    `tended_this_period` uses the same `[period_start, period_end]` window as
    the rest of the digest. `resurface` is a stable hash of `(iso_week,
    slug)` so the same note(s) get surfaced all week and change the next —
    deterministic, no `random` needed.
    """
    if not pages:
        return {}

    counts = {stage: 0 for stage in STAGES}
    for p in pages:
        counts[p.stage] = counts.get(p.stage, 0) + 1

    evergreens = [p for p in pages if p.stage == "evergreen"]
    stalest = sorted(evergreens, key=lambda p: p.tended)[:STALE_EVERGREEN_LIMIT]

    tended_this_period = sorted(
        (p for p in pages if period_start <= p.tended <= period_end),
        key=lambda p: p.tended,
    )

    ranked = sorted(
        pages,
        key=lambda p: hashlib.sha256(f"{iso_week}:{p.slug}".encode()).hexdigest(),
    )
    resurface = ranked[:RESURFACE_COUNT]

    return {
        "counts": counts,
        "total": len(pages),
        "stalest_evergreens": [
            {"title": p.title, "slug": p.slug, "tended": p.tended} for p in stalest
        ],
        "tended_this_period": [
            {"title": p.title, "slug": p.slug, "tended": p.tended}
            for p in tended_this_period
        ],
        "resurface": [{"title": p.title, "slug": p.slug} for p in resurface],
    }


def summary_for_now(pages: list[PageNote], weeks: int = 1) -> dict:
    """compute_garden_summary for the trailing `weeks`-wide window ending now.

    Convenience for callers (like `dendr garden status`) that only want the
    summary and don't need the raw period bounds `_gather_digest_data` also
    returns for the rest of its payload.
    """
    now = datetime.now()
    since = (now - timedelta(weeks=weeks)).strftime("%Y-%m-%d")
    period_end = now.strftime("%Y-%m-%d")
    return compute_garden_summary(pages, since, period_end, iso_week_label(now))


def format_counts(counts: dict) -> str:
    """'N seedling, N budding, N evergreen' — one shared phrasing for every
    render site, built from STAGES so a future stage can't be left out of
    one of them by a copy-paste miss."""
    return ", ".join(f"{counts.get(stage, 0)} {stage}" for stage in STAGES)


def _join_slugs(items: list[dict]) -> str:
    return ", ".join(f"[[{p['slug']}]]" for p in items)


def render_garden_dashboard(pages: list[PageNote], summary: dict) -> str:
    """Render the full `Wiki/garden.md` dashboard."""
    if not pages or not summary:
        return (
            "---\ntype: garden\n---\n\n# Garden\n\n"
            "*No `Pages/` notes found yet — this dashboard fills in once "
            "you have topic notes there.*\n"
        )

    lines = [
        "---",
        "type: garden",
        f"generated: {datetime.now().isoformat()}",
        "---",
        "",
        "# Garden",
        "",
        f"**{summary['total']} pages** — {format_counts(summary['counts'])}",
        "",
    ]

    sections = [
        (
            f"Grown this period ({len(summary['tended_this_period'])})",
            summary["tended_this_period"],
            lambda p: f"- [[{p['slug']}]] — tended {p['tended']}",
        ),
        (
            "Resurfaced this week",
            summary["resurface"],
            lambda p: f"- [[{p['slug']}]]",
        ),
        (
            "Stalest evergreens (tend these first)",
            summary["stalest_evergreens"],
            lambda p: f"- [[{p['slug']}]] — last tended {p['tended']}",
        ),
    ]
    for header, items, fmt in sections:
        if not items:
            continue
        lines.append(f"## {header}")
        lines.append("")
        lines.extend(fmt(p) for p in items)
        lines.append("")

    lines.append("## All pages")
    lines.append("")

    def _stage_order(p: PageNote) -> tuple[int, str]:
        return (_STAGE_ORDER.get(p.stage, len(STAGES)), p.tended)

    for p in sorted(pages, key=_stage_order):
        lines.append(f"- [[{p.slug}]] — *{p.stage}*, tended {p.tended}")
    lines.append("")

    return "\n".join(lines)


def write_dashboard(config: Config, pages: list[PageNote], summary: dict) -> Path:
    """Render and atomically write `Wiki/garden.md`. Returns the path."""
    atomic_write_text(
        config.garden_dashboard_path, render_garden_dashboard(pages, summary)
    )
    return config.garden_dashboard_path


def render_garden_digest_section(summary: dict) -> str:
    """Short deterministic Garden block for the non-Claude digest render."""
    if not summary:
        return ""

    lines = [
        f"## Garden ({summary['total']} pages: {format_counts(summary['counts'])})",
        "",
    ]
    if summary["tended_this_period"]:
        lines.append(f"Grown this period: {_join_slugs(summary['tended_this_period'])}")
    if summary["stalest_evergreens"]:
        stalest = summary["stalest_evergreens"][0]
        lines.append(
            f"Stalest evergreen: [[{stalest['slug']}]] "
            f"(last tended {stalest['tended']})"
        )
    if summary["resurface"]:
        lines.append(f"Resurfaced: {_join_slugs(summary['resurface'])}")
    lines.append("")
    return "\n".join(lines)


def render_garden_prompt_section(summary: dict) -> str:
    """'## Garden state' block for the Claude synthesis prompt.

    Mirrors render_garden_digest_section/render_garden_dashboard: garden.py
    owns all Garden rendering, digest.py just calls in.
    """
    if not summary:
        return (
            "\n## Garden state\n\n"
            "*(No `Pages/` notes found, so there's no digital-garden state to "
            "report this week.)*\n"
        )

    lines = [
        "\n## Garden state\n\n",
        f"`Pages/` has {summary['total']} notes — {format_counts(summary['counts'])}.\n",
    ]
    if summary["tended_this_period"]:
        lines.append(
            f"Tended this period: {_join_slugs(summary['tended_this_period'])}.\n"
        )
    if summary["stalest_evergreens"]:
        lines.append(
            "Stalest evergreens (candidates to revisit): "
            f"{_join_slugs(summary['stalest_evergreens'])}.\n"
        )
    if summary["resurface"]:
        lines.append(
            f"Resurfaced pick for this week: {_join_slugs(summary['resurface'])}.\n"
        )
    return "".join(lines)
