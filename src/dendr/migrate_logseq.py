"""Migrate LogSeq vault data into an Obsidian vault.

Converts journals -> Daily/, pages -> Pages/, copies assets,
and transforms LogSeq markdown syntax to Obsidian-compatible format.
"""

from __future__ import annotations

import logging
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path

log = logging.getLogger(__name__)

# LogSeq system pages to skip (not user content)
SYSTEM_PAGES = frozenset({
    "a", "b", "c", "todo", "done", "doing", "now", "later",
    "wait", "waiting", "canceled", "cancelled", "in-progress",
    "favorites", "contents",
})


# ---------------------------------------------------------------------------
# EDN metadata parsing
# ---------------------------------------------------------------------------

def parse_pages_metadata(edn_path: Path) -> dict[str, dict[str, datetime]]:
    """Parse logseq/pages-metadata.edn into {page_name: {created, updated}}."""
    text = edn_path.read_text(encoding="utf-8")
    result: dict[str, dict[str, datetime]] = {}

    # Match each entry block
    for m in re.finditer(
        r':block/name\s+"([^"]+)".*?'
        r':block/created-at\s+(\d+).*?'
        r':block/updated-at\s+(\d+)',
        text,
        re.DOTALL,
    ):
        name = m.group(1)
        created = datetime.fromtimestamp(int(m.group(2)) / 1000, tz=timezone.utc)
        updated = datetime.fromtimestamp(int(m.group(3)) / 1000, tz=timezone.utc)
        result[name] = {"created": created, "updated": updated}

    return result


# ---------------------------------------------------------------------------
# Markdown conversion
# ---------------------------------------------------------------------------

def convert_logseq_markdown(text: str) -> str:
    """Convert LogSeq markdown syntax to Obsidian-compatible markdown."""
    lines = text.split("\n")
    out_lines: list[str] = []

    for line in lines:
        line = _convert_line(line)
        out_lines.append(line)

    return "\n".join(out_lines)


def _convert_line(line: str) -> str:
    stripped = line.lstrip()
    indent = line[: len(line) - len(stripped)]

    # Remove block IDs (id:: uuid)
    if re.match(r"id::\s+[0-9a-f-]{36}", stripped):
        return ""

    # Remove trailing block IDs on content lines
    stripped = re.sub(r"\s*\n?\s*id::\s+[0-9a-f-]{36}\s*$", "", stripped)

    # Convert TODO/DONE/DOING/NOW/LATER at start of bullet
    stripped = _convert_task_keywords(stripped)

    # Convert image references: ![alt](../assets/file) -> ![[file]]
    stripped = re.sub(
        r"!\[([^\]]*)\]\(\.\./assets/([^)]+)\)",
        lambda m: f"![[{m.group(2)}]]",
        stripped,
    )

    # Convert block embeds: {{embed ((uuid))}} -> %%logseq-embed: uuid%%
    stripped = re.sub(
        r"\{\{embed\s+\(\(([0-9a-f-]{36})\)\)\}\}",
        r"%%logseq-embed: \1%%",
        stripped,
    )

    # Convert inline block references: ((uuid)) -> %%logseq-ref: uuid%%
    stripped = re.sub(
        r"\(\(([0-9a-f-]{36})\)\)",
        r"%%logseq-ref: \1%%",
        stripped,
    )

    return indent + stripped


def _convert_task_keywords(text: str) -> str:
    """Convert LogSeq task keywords to Obsidian checkboxes."""
    # Match at start of line or after bullet marker
    for pattern, replacement in [
        (r"^- DONE\s+", "- [x] "),
        (r"^- TODO\s+", "- [ ] "),
        (r"^- DOING\s+", "- [ ] "),
        (r"^- NOW\s+", "- [ ] "),
        (r"^- LATER\s+", "- [ ] "),
        (r"^DONE\s+", "- [x] "),
        (r"^TODO\s+", "- [ ] "),
        (r"^DOING\s+", "- [ ] "),
        (r"^NOW\s+", "- [ ] "),
        (r"^LATER\s+", "- [ ] "),
    ]:
        text = re.sub(pattern, replacement, text)
    return text


# ---------------------------------------------------------------------------
# Frontmatter generation
# ---------------------------------------------------------------------------

def _make_frontmatter(**fields: str | None) -> str:
    lines = ["---"]
    for key, val in fields.items():
        if val is not None:
            lines.append(f"{key}: {val}")
    lines.append("---")
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Migration orchestrator
# ---------------------------------------------------------------------------

class MigrationResult:
    def __init__(self) -> None:
        self.journals_created: list[str] = []
        self.journals_skipped: list[str] = []
        self.pages_created: list[str] = []
        self.pages_skipped: list[str] = []
        self.assets_copied: list[str] = []
        self.assets_skipped: list[str] = []
        self.errors: list[str] = []

    def summary(self) -> str:
        lines = [
            f"Journals: {len(self.journals_created)} created, {len(self.journals_skipped)} skipped",
            f"Pages:    {len(self.pages_created)} created, {len(self.pages_skipped)} skipped",
            f"Assets:   {len(self.assets_copied)} copied, {len(self.assets_skipped)} skipped",
        ]
        if self.errors:
            lines.append(f"Errors:   {len(self.errors)}")
            for e in self.errors:
                lines.append(f"  - {e}")
        return "\n".join(lines)


def migrate(
    logseq_dir: Path,
    obsidian_vault: Path,
    *,
    dry_run: bool = True,
) -> MigrationResult:
    """Run the full LogSeq -> Obsidian migration."""
    result = MigrationResult()

    journals_dir = logseq_dir / "journals"
    pages_dir = logseq_dir / "pages"
    assets_dir = logseq_dir / "assets"
    metadata_path = logseq_dir / "logseq" / "pages-metadata.edn"

    daily_dir = obsidian_vault / "Daily"
    target_pages_dir = obsidian_vault / "Pages"
    target_assets_dir = obsidian_vault / "assets"

    # Load metadata
    page_meta: dict[str, dict[str, datetime]] = {}
    if metadata_path.exists():
        page_meta = parse_pages_metadata(metadata_path)
        log.info("Loaded metadata for %d pages", len(page_meta))

    # --- Journals ---
    if journals_dir.exists():
        for journal_file in sorted(journals_dir.glob("*.md")):
            try:
                _migrate_journal(journal_file, daily_dir, dry_run, result)
            except Exception as e:
                result.errors.append(f"Journal {journal_file.name}: {e}")
                log.error("Error migrating journal %s: %s", journal_file.name, e)

    # --- Pages ---
    if pages_dir.exists():
        for page_file in sorted(pages_dir.glob("*.md")):
            try:
                _migrate_page(page_file, target_pages_dir, page_meta, dry_run, result)
            except Exception as e:
                result.errors.append(f"Page {page_file.name}: {e}")
                log.error("Error migrating page %s: %s", page_file.name, e)

    # --- Assets ---
    if assets_dir.exists():
        for asset_file in sorted(assets_dir.iterdir()):
            if asset_file.is_file():
                try:
                    _copy_asset(asset_file, target_assets_dir, dry_run, result)
                except Exception as e:
                    result.errors.append(f"Asset {asset_file.name}: {e}")
                    log.error("Error copying asset %s: %s", asset_file.name, e)

    return result


def _migrate_journal(
    src: Path,
    daily_dir: Path,
    dry_run: bool,
    result: MigrationResult,
) -> None:
    # Parse date from filename: YYYY_MM_DD.md
    # Skip iCloud conflict duplicates (e.g. "2026_03_24 2.md", "2026_03_28(1).md")
    stem = src.stem
    if re.search(r"[\s(]", stem):
        result.journals_skipped.append(f"{src.name} (iCloud duplicate)")
        log.debug("Skipping iCloud duplicate: %s", src.name)
        return

    parts = stem.split("_")
    if len(parts) != 3:
        result.errors.append(f"Journal {src.name}: unexpected filename format")
        return

    year, month, day = int(parts[0]), int(parts[1]), int(parts[2])
    date_str = f"{year:04d}-{month:02d}-{day:02d}"
    target = daily_dir / f"{date_str}.md"

    if target.exists():
        result.journals_skipped.append(f"{date_str} (already exists)")
        log.debug("Skipping journal %s: target exists", date_str)
        return

    content = src.read_text(encoding="utf-8")
    converted = convert_logseq_markdown(content)

    frontmatter = _make_frontmatter(
        created=date_str,
        source="logseq",
    )

    output = frontmatter + converted

    if dry_run:
        log.info("[DRY RUN] Would create %s", target)
    else:
        daily_dir.mkdir(parents=True, exist_ok=True)
        target.write_text(output, encoding="utf-8")
        log.info("Created %s", target)

    result.journals_created.append(date_str)


def _migrate_page(
    src: Path,
    pages_dir: Path,
    page_meta: dict[str, dict[str, datetime]],
    dry_run: bool,
    result: MigrationResult,
) -> None:
    page_name = src.stem
    # Check if system page
    if page_name.lower() in SYSTEM_PAGES:
        result.pages_skipped.append(f"{page_name} (system page)")
        log.debug("Skipping system page: %s", page_name)
        return

    target = pages_dir / src.name

    if target.exists():
        result.pages_skipped.append(f"{page_name} (already exists)")
        log.debug("Skipping page %s: target exists", page_name)
        return

    content = src.read_text(encoding="utf-8")
    converted = convert_logseq_markdown(content)

    # Build frontmatter with metadata
    meta = page_meta.get(page_name.lower(), {})
    created_str = meta["created"].strftime("%Y-%m-%d") if "created" in meta else None
    updated_str = meta["updated"].strftime("%Y-%m-%d") if "updated" in meta else None

    frontmatter = _make_frontmatter(
        created=created_str,
        updated=updated_str,
        source="logseq",
        aliases=f'"{page_name}"',
    )

    output = frontmatter + converted

    if dry_run:
        log.info("[DRY RUN] Would create %s", target)
    else:
        pages_dir.mkdir(parents=True, exist_ok=True)
        target.write_text(output, encoding="utf-8")
        log.info("Created %s", target)

    result.pages_created.append(page_name)


def _copy_asset(
    src: Path,
    assets_dir: Path,
    dry_run: bool,
    result: MigrationResult,
) -> None:
    target = assets_dir / src.name

    if target.exists():
        result.assets_skipped.append(f"{src.name} (already exists)")
        log.debug("Skipping asset %s: target exists", src.name)
        return

    if dry_run:
        log.info("[DRY RUN] Would copy %s -> %s", src, target)
    else:
        assets_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, target)
        log.info("Copied %s -> %s", src, target)

    result.assets_copied.append(src.name)
