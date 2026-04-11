"""CLI entry point for Dendr.

Commands:
  dendr init <vault_path>    Initialize a vault for Dendr
  dendr daemon               Run the watcher daemon
  dendr ingest               Run a single ingest cycle
  dendr search <query>       Search the knowledge base
  dendr lint                 Run lint checks
  dendr serve                Start the search server
  dendr stats                Show knowledge base statistics
  dendr models pull          Download models from manifest
  dendr models verify        Verify model integrity
  dendr models list          Show model status
  dendr models lock          Pin SHA256 hashes in manifest
"""

from __future__ import annotations

import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import click

from dendr import __version__


class _JsonFormatter(logging.Formatter):
    """Emit structured JSON log lines for Docker / log aggregation."""

    def format(self, record: logging.LogRecord) -> str:
        entry = {
            "ts": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        if record.exc_info and record.exc_info[1]:
            entry["exc"] = self.formatException(record.exc_info)
        return json.dumps(entry, default=str)


def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    if os.environ.get("DENDR_LOG_JSON"):
        handler = logging.StreamHandler()
        handler.setFormatter(_JsonFormatter())
        logging.root.addHandler(handler)
        logging.root.setLevel(level)
    else:
        logging.basicConfig(
            level=level,
            format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
            datefmt="%H:%M:%S",
        )


@click.group()
@click.version_option(__version__)
@click.option("-v", "--verbose", is_flag=True, help="Enable debug logging")
def main(verbose: bool) -> None:
    """Dendr — Personal knowledge compiler for Obsidian Daily Notes."""
    _setup_logging(verbose)


@main.command()
@click.argument("vault_path", type=click.Path(exists=True, file_okay=False))
def init(vault_path: str) -> None:
    """Initialize a vault for Dendr processing."""
    from dendr.config import Config
    from dendr.db import connect, init_schema

    vault = Path(vault_path).resolve()
    config = Config(vault_path=vault)
    config.ensure_dirs()
    config.write_vault_marker()
    config.save()

    # Initialize database
    conn = connect(config.db_path)
    init_schema(conn)
    conn.close()

    # Write the schema.md spec into the wiki
    schema_path = config.wiki_dir / "schema.md"
    if not schema_path.exists():
        _write_default_schema(schema_path)

    # Write Claude Code prompts
    claude_dir = vault / ".claude"
    claude_dir.mkdir(exist_ok=True)
    _write_claude_prompts(claude_dir, config)

    click.echo(f"Dendr initialized for vault: {vault}")
    click.echo(f"  Data directory: {config.data_dir}")
    click.echo(f"  Database: {config.db_path}")
    click.echo(f"  Vault ID: {config.vault_id}")


@main.command()
@click.option(
    "--data-dir", type=click.Path(), default=None, help="Override data directory"
)
@click.option(
    "--vault",
    type=click.Path(exists=True, file_okay=False),
    default=None,
    help="Override vault path",
)
def daemon(data_dir: str | None, vault: str | None) -> None:
    """Run the watcher daemon (blocks until Ctrl-C)."""
    from dendr.config import Config
    from dendr.watcher import run_daemon

    dd = Path(data_dir) if data_dir else None
    config = Config.load(dd)
    if vault:
        config.vault_path = Path(vault).resolve()
    click.echo(f"Starting daemon for vault: {config.vault_path}")
    run_daemon(config)


@main.command()
@click.option("--data-dir", type=click.Path(), default=None)
def ingest(data_dir: str | None) -> None:
    """Run a single ingest cycle."""
    from dendr.config import Config
    from dendr.db import connect, init_schema
    from dendr.llm import LLMClient
    from dendr.pipeline import run_ingest

    dd = Path(data_dir) if data_dir else None
    config = Config.load(dd)
    conn = connect(config.db_path)
    init_schema(conn)
    llm = LLMClient(config)

    stats = run_ingest(config, conn, llm)
    conn.close()
    click.echo(json.dumps(stats, indent=2))


@main.command()
@click.option("--data-dir", type=click.Path(), default=None)
@click.option(
    "--vault",
    type=click.Path(exists=True, file_okay=False),
    default=None,
    help="Override vault path",
)
@click.option(
    "--run", is_flag=True, default=False, help="Immediately run ingest after reset"
)
@click.confirmation_option(
    prompt="This will reset all block state and reprocess everything. Continue?"
)
def reprocess(data_dir: str | None, vault: str | None, run: bool) -> None:
    """Reset block state and reprocess all daily notes from scratch.

    Clears the block_state table and the done queue so every block is
    treated as new on the next ingest cycle. Existing claims, concepts,
    and wiki pages are preserved — blocks that produce identical claims
    will simply reinforce them.
    """
    # TODO: Think through and test reprocess edge cases:
    #   - LLM non-determinism: same block may produce slightly different claims,
    #     causing duplicates or false contradictions even with source_block_ref guard
    #   - Concept evidence duplication: append_evidence adds sections even if
    #     identical evidence already exists in the LLM zone
    #   - Entity dedup: no embedding-based canonicalization for entities yet,
    #     so "Sarah M." and "Sarah" create separate pages
    #   - Consider adding a --entities-only flag that skips enrichment entirely
    #     and only runs entity page creation from existing claims in the DB
    #   - Need integration tests covering the full reprocess flow
    import shutil

    from dendr.config import Config
    from dendr.db import connect, init_schema
    from dendr.llm import LLMClient
    from dendr.pipeline import run_ingest

    dd = Path(data_dir) if data_dir else None
    config = Config.load(dd)
    if vault:
        config.vault_path = Path(vault).resolve()

    conn = connect(config.db_path)
    init_schema(conn)

    # Clear block state so all blocks appear dirty
    count = conn.execute("SELECT COUNT(*) as n FROM block_state").fetchone()["n"]
    conn.execute("DELETE FROM block_state")
    conn.commit()
    click.echo(f"Cleared {count} block state entries")

    # Clear done queue
    done_count = 0
    if config.done_dir.exists():
        done_count = len(list(config.done_dir.glob("*.json")))
        shutil.rmtree(config.done_dir)
        config.done_dir.mkdir(parents=True, exist_ok=True)
    click.echo(f"Cleared {done_count} done queue items")

    # Clear processing queue (stale from prior runs)
    if config.processing_dir.exists():
        proc_count = len(list(config.processing_dir.glob("*.json")))
        if proc_count:
            shutil.rmtree(config.processing_dir)
            config.processing_dir.mkdir(parents=True, exist_ok=True)
            click.echo(f"Cleared {proc_count} stale processing items")

    click.echo("Block state reset. All blocks will be reprocessed on next ingest.")

    if run:
        click.echo("Starting ingest...")
        llm = LLMClient(config)
        stats = run_ingest(config, conn, llm)
        click.echo(json.dumps(stats, indent=2))

    conn.close()


@main.command()
@click.argument("query")
@click.option(
    "--mode", type=click.Choice(["fts", "semantic", "hybrid"]), default="hybrid"
)
@click.option("--limit", type=int, default=10)
@click.option("--data-dir", type=click.Path(), default=None)
def search(query: str, mode: str, limit: int, data_dir: str | None) -> None:
    """Search the knowledge base."""
    from dendr import db as dendr_db
    from dendr.config import Config
    from dendr.llm import LLMClient

    dd = Path(data_dir) if data_dir else None
    config = Config.load(dd)
    conn = dendr_db.connect(config.db_path)
    dendr_db.init_schema(conn)

    results = []

    if mode in ("fts", "hybrid"):
        fts = dendr_db.search_claims_fts(conn, query, limit=limit)
        for r in fts:
            results.append(
                {
                    "id": r["id"],
                    "text": r["text"],
                    "concept": r["concept_slug"],
                    "confidence": r["confidence"],
                    "source": r["source_block_ref"],
                    "type": "fts",
                }
            )

    if mode in ("semantic", "hybrid"):
        try:
            llm = LLMClient(config)
            emb = llm.embed(query)
            sem = dendr_db.search_claims_semantic(conn, emb, limit=limit)
            for r in sem:
                if not any(x["id"] == r["id"] for x in results):
                    results.append(
                        {
                            "id": r["id"],
                            "text": r["text"],
                            "concept": r["concept_slug"],
                            "confidence": r["confidence"],
                            "source": r["source_block_ref"],
                            "type": "semantic",
                        }
                    )
        except Exception as e:
            click.echo(f"Semantic search unavailable: {e}", err=True)

    conn.close()

    if not results:
        click.echo("No results found.")
        return

    for r in results[:limit]:
        click.echo(
            f"  [{r['type']:8s}] (c={r['confidence']:.2f}) [{r['concept']}] {r['text'][:100]}"
        )


@main.command()
@click.option("--data-dir", type=click.Path(), default=None)
def lint(data_dir: str | None) -> None:
    """Run lint checks on the knowledge base."""
    from dendr.config import Config
    from dendr.db import connect, init_schema
    from dendr.lint import run_lint

    dd = Path(data_dir) if data_dir else None
    config = Config.load(dd)
    conn = connect(config.db_path)
    init_schema(conn)

    report = run_lint(config, conn)
    conn.close()
    click.echo(report)


@main.command()
@click.option("--data-dir", type=click.Path(), default=None)
@click.option(
    "--vault",
    type=click.Path(exists=True, file_okay=False),
    default=None,
    help="Override vault path",
)
def serve(data_dir: str | None, vault: str | None) -> None:
    """Start the search HTTP server."""
    from dendr.config import Config
    from dendr.search import run_server

    dd = Path(data_dir) if data_dir else None
    config = Config.load(dd)
    if vault:
        config.vault_path = Path(vault).resolve()
    click.echo(f"Starting search server on http://127.0.0.1:{config.search_port}")
    run_server(config)


@main.command()
@click.option("--data-dir", type=click.Path(), default=None)
@click.option(
    "--vault",
    type=click.Path(exists=True, file_okay=False),
    default=None,
    help="Override vault path (needed when config.json has a stale path, e.g. inside containers)",
)
@click.option("--weeks", type=int, default=1, help="Number of weeks to cover")
@click.option("--claude", is_flag=True, help="Also generate Claude synthesis prompt")
def digest(
    data_dir: str | None, vault: str | None, weeks: int, claude: bool
) -> None:
    """Generate a weekly digest briefing."""
    from dendr.config import Config
    from dendr.db import connect, init_schema
    from dendr.digest import generate_digest

    dd = Path(data_dir) if data_dir else None
    config = Config.load(dd)
    if vault:
        config.vault_path = Path(vault).resolve()
    conn = connect(config.db_path)
    init_schema(conn)

    path = generate_digest(config, conn, weeks=weeks, use_claude=claude)
    conn.close()

    click.echo(f"Digest written to: {path}")
    if claude:
        prompt_path = config.wiki_dir / "_digest_prompt.md"
        click.echo(f"Claude prompt at: {prompt_path}")
        click.echo("Run a Claude Code session with .claude/digest.md to synthesize.")


@main.command()
@click.option("--data-dir", type=click.Path(), default=None)
def stats(data_dir: str | None) -> None:
    """Show knowledge base statistics."""
    from dendr import db as dendr_db
    from dendr.config import Config
    from dendr import queue

    dd = Path(data_dir) if data_dir else None
    config = Config.load(dd)
    conn = dendr_db.connect(config.db_path)
    dendr_db.init_schema(conn)

    s = dendr_db.get_stats(conn)
    pending = queue.pending_count(config)
    conn.close()

    click.echo(f"Active claims:     {s['active_claims']}")
    click.echo(f"Concepts:          {s['concepts']}")
    click.echo(f"Challenged claims: {s['challenged_claims']}")
    click.echo(f"Pending queue:     {pending}")


# --- LogSeq migration ---


@main.command("migrate-logseq")
@click.argument("logseq_dir", type=click.Path(exists=True, file_okay=False))
@click.option(
    "--vault",
    type=click.Path(exists=True, file_okay=False),
    default=None,
    help="Obsidian vault path (defaults to configured vault)",
)
@click.option(
    "--execute", is_flag=True, help="Actually perform migration (default is dry-run)"
)
def migrate_logseq(logseq_dir: str, vault: str | None, execute: bool) -> None:
    """Migrate a LogSeq vault into Obsidian.

    LOGSEQ_DIR is the path to the LogSeq graph directory (containing journals/, pages/, assets/).
    """
    from dendr.migrate_logseq import migrate

    src = Path(logseq_dir).resolve()
    if vault:
        dst = Path(vault).resolve()
    else:
        from dendr.config import Config

        config = Config.load(None)
        dst = config.vault_path

    dry_run = not execute
    if dry_run:
        click.echo("=== DRY RUN (pass --execute to write files) ===\n")

    click.echo(f"Source:  {src}")
    click.echo(f"Target:  {dst}")
    click.echo()

    result = migrate(src, dst, dry_run=dry_run)
    click.echo(result.summary())


@main.command("reformat-logseq")
@click.option(
    "--vault",
    type=click.Path(exists=True, file_okay=False),
    default=None,
    help="Obsidian vault path (defaults to configured vault)",
)
@click.option("--dry-run", is_flag=True, help="Show what would change without writing")
def reformat_logseq(vault: str | None, dry_run: bool) -> None:
    """Reformat migrated LogSeq notes: top-level bullets become paragraphs.

    Processes both Daily/ and Pages/ directories.
    """
    from dendr.migrate_logseq import reformat_logseq_bullets

    if vault:
        vault_path = Path(vault).resolve()
    else:
        from dendr.config import Config

        config = Config.load(None)
        vault_path = config.vault_path

    dirs = [vault_path / "Daily", vault_path / "Pages"]
    modified = 0
    skipped = 0

    for target_dir in dirs:
        if not target_dir.exists():
            click.echo(f"  skipping (not found): {target_dir}")
            continue
        click.echo(f"Processing {target_dir.name}/")
        for md_file in sorted(target_dir.glob("*.md")):
            if dry_run:
                text = md_file.read_text(encoding="utf-8")
                if "source: logseq" in text:
                    click.echo(f"  would reformat: {md_file.name}")
                    modified += 1
                else:
                    skipped += 1
            else:
                if reformat_logseq_bullets(md_file):
                    click.echo(f"  reformatted: {md_file.name}")
                    modified += 1
                else:
                    skipped += 1

    click.echo(
        f"\n{'Would reformat' if dry_run else 'Reformatted'}: {modified}, skipped: {skipped}"
    )


# --- Model management ---


def _find_manifest() -> Path:
    """Find dendr-models.yaml, searching cwd and up."""
    check = Path.cwd()
    for _ in range(10):
        candidate = check / "dendr-models.yaml"
        if candidate.exists():
            return candidate
        parent = check.parent
        if parent == check:
            break
        check = parent
    raise click.ClickException(
        "dendr-models.yaml not found. Are you in the Dendr repo?"
    )


@main.group()
def models() -> None:
    """Manage local model weights."""


@models.command("pull")
@click.option(
    "--role", type=str, default=None, help="Download only this role (e.g. enrichment)"
)
@click.option("--force", is_flag=True, help="Re-download even if present")
@click.option("--data-dir", type=click.Path(), default=None)
def models_pull(role: str | None, force: bool, data_dir: str | None) -> None:
    """Download models declared in dendr-models.yaml."""
    import os

    from dendr.config import Config
    from dendr.model_manager import ModelManifest, pull_all_models

    manifest_path = _find_manifest()
    manifest = ModelManifest.load(manifest_path)

    dd = Path(data_dir) if data_dir else None
    config = Config.load(dd)
    config.models_dir.mkdir(parents=True, exist_ok=True)

    token = os.environ.get("HF_TOKEN")
    roles = [role] if role else None

    click.echo(f"Manifest: {manifest_path}")
    click.echo(f"Models dir: {config.models_dir}")
    if roles:
        click.echo(f"Pulling role: {role}")
    else:
        click.echo(f"Pulling all {len(manifest.specs)} models...")

    results = pull_all_models(
        config.models_dir, manifest, roles=roles, force=force, token=token
    )

    click.echo(f"\nDownloaded {len(results)} model(s):")
    for r, p in results.items():
        click.echo(f"  [{r}] {p.name} ({p.stat().st_size / 1e9:.1f} GB)")


@models.command("verify")
@click.option("--data-dir", type=click.Path(), default=None)
def models_verify(data_dir: str | None) -> None:
    """Verify SHA256 integrity of downloaded models."""
    from dendr.config import Config
    from dendr.model_manager import ModelManifest, check_all_models, sha256_file

    manifest_path = _find_manifest()
    manifest = ModelManifest.load(manifest_path)

    dd = Path(data_dir) if data_dir else None
    config = Config.load(dd)

    statuses = check_all_models(config.models_dir, manifest)
    all_ok = True

    for role, status in statuses.items():
        if not status.present:
            click.echo(f"  MISSING  [{role}] {status.spec.filename}")
            all_ok = False
        elif status.hash_match is False:
            actual = sha256_file(config.models_dir / status.spec.filename)
            click.echo(f"  MISMATCH [{role}] {status.spec.filename}")
            click.echo(f"           expected: {status.spec.sha256}")
            click.echo(f"           actual:   {actual}")
            all_ok = False
        elif status.hash_match is True:
            click.echo(f"  OK       [{role}] {status.spec.filename}")
        else:
            click.echo(
                f"  NO HASH  [{role}] {status.spec.filename} (run `dendr models lock` to pin)"
            )

    if all_ok:
        click.echo("\nAll models verified.")
    else:
        click.echo("\nSome models need attention. Run `dendr models pull` to fix.")
        sys.exit(1)


@models.command("list")
@click.option("--data-dir", type=click.Path(), default=None)
def models_list(data_dir: str | None) -> None:
    """Show model status table."""
    from dendr.config import Config
    from dendr.model_manager import ModelManifest, check_all_models

    manifest_path = _find_manifest()
    manifest = ModelManifest.load(manifest_path)

    dd = Path(data_dir) if data_dir else None
    config = Config.load(dd)

    statuses = check_all_models(config.models_dir, manifest)

    click.echo(f"{'Role':<14} {'Filename':<42} {'Size':>8} {'Status':<12}")
    click.echo("-" * 80)
    for role, status in statuses.items():
        size_str = (
            f"{status.spec.size_bytes / 1e9:.1f} GB" if status.spec.size_bytes else "?"
        )
        if not status.present:
            state = "MISSING"
        elif status.hash_match is False:
            state = "MISMATCH"
        elif status.hash_match is True:
            state = "OK"
        else:
            state = "UNVERIFIED"
        click.echo(f"{role:<14} {status.spec.filename:<42} {size_str:>8} {state:<12}")

    click.echo(f"\nModels dir: {config.models_dir}")


@models.command("lock")
@click.option("--data-dir", type=click.Path(), default=None)
def models_lock(data_dir: str | None) -> None:
    """Compute SHA256 of present models and write to manifest."""
    from dendr.config import Config
    from dendr.model_manager import ModelManifest, lock_models

    manifest_path = _find_manifest()
    manifest = ModelManifest.load(manifest_path)

    dd = Path(data_dir) if data_dir else None
    config = Config.load(dd)

    hashes = lock_models(config.models_dir, manifest, manifest_path)

    click.echo(f"Locked {len(hashes)} model(s) in {manifest_path}:")
    for role, h in hashes.items():
        click.echo(f"  [{role}] {h[:16]}...")
    click.echo("\nCommit dendr-models.yaml to pin these versions.")


# --- Schema and prompt generation ---


def _write_default_schema(path: Path) -> None:
    """Write the default Wiki/schema.md."""
    path.write_text(
        """---
type: schema
version: "1.0"
---

# Dendr Wiki Schema

This document defines how the Dendr system maintains the knowledge wiki.
Both the local LLM and Claude read this on every session.

## Page Types

### Concept Page (`concepts/<slug>.md`)
- Tracks a single concept, idea, or topic
- Frontmatter: type, slug, human_touched, last_llm_hash, created, updated
- Structure: human-zone (user edits) + llm-zone (system-managed evidence)

### Entity Page (`entities/<slug>.md`)
- Tracks a named entity (person, project, tool, organization)
- Same structure as concept pages

### Summary (`summaries/weekly-YYYY-Www.md`)
- Weekly synthesis produced by Claude
- Contains: key themes, notable changes, contradictions resolved

## Claim Format

Every factual statement is stored as a claim with:
- `text`: the atomic statement
- `confidence`: 0.0-1.0 (inline as `[c:0.82]` in markdown)
- `kind`: statement, task, intention, question, belief
- `status`: created → reinforced → challenged → superseded
- `source_block_ref`: link back to the daily note block
- Deduplication via embedding similarity (semantic matching)

## Conventions

- Slugs: lowercase, hyphens, no spaces (`machine-learning`, not `Machine Learning`)
- Cross-references: use `[[slug]]` Obsidian wikilinks
- Confidence pills: `[c:0.82]` after factual claims in markdown
- Citations: `(from YYYY-MM-DD ^block-id)` after evidence
- LLM zone: everything between `<!-- llm-zone -->` markers is system-managed
- Human zone: everything between `<!-- human-zone -->` markers is sacred

## Lint Rules

- Orphan pages: concept pages with zero active claims
- Stale claims: not reinforced in 8+ weeks
- Contradictions: semantically similar claims with conflicting content
- Missing cross-refs: `[[slug]]` links pointing to non-existent pages

## Privacy

- Blocks tagged `#dendr-private`, `#private`, or `#redact` are never sent to Claude
- Blocks matching secret patterns (API keys, passwords) are auto-tagged private
- Private claims are stored locally for search but excluded from Claude payloads
""",
        encoding="utf-8",
    )


def _write_claude_prompts(claude_dir: Path, config) -> None:
    """Write Claude Code session prompts."""
    # Weekly synthesis prompt
    (claude_dir / "weekly.md").write_text(
        f"""# Weekly Synthesis Session

You are Dendr's knowledge synthesizer. Your job is to maintain and improve
a personal knowledge wiki based on this week's activity.

## Your inputs (already loaded)
- `Wiki/schema.md` — the wiki spec (READ THIS FIRST)
- `Wiki/weekly-digest.md` — the rolling digest (UPDATE THIS)
- `Wiki/log.md` — this week's activity log
- Top concept pages touched this week

## Your tasks
1. Read the activity log for the past week
2. Review any new contradictions flagged in the lint report
3. For each contradiction: decide which claim to supersede, or flag as genuinely unresolved
4. Update `weekly-digest.md` with a ≤8000 token synthesis of this week's knowledge gains
5. Revise any stale summaries for heavily-updated concepts
6. Write `summaries/weekly-YYYY-Www.md` for this week

## Rules
- NEVER read or reference raw daily notes — only the distilled wiki pages
- NEVER modify the human-zone of any page
- Keep the weekly digest under 8000 tokens
- Use `[c:0.82]` confidence pills for factual claims
- Use `[[concept-slug]]` for cross-references
- Cite sources as `(from YYYY-MM-DD)`
- Private claims (private=true) must NEVER appear in your output

## Vault path: {config.vault_path}
""",
        encoding="utf-8",
    )

    # Q&A session prompt
    (claude_dir / "qa.md").write_text(
        f"""# Q&A Session

You are Dendr's knowledge assistant. Answer questions using the wiki's
accumulated knowledge, with citations.

## Your inputs
- `Wiki/schema.md` — the wiki spec
- Search results from `localhost:{config.search_port}/search` (pre-loaded)
- Relevant concept pages

## Rules
- Ground answers in wiki claims with citations: `(from YYYY-MM-DD ^block-id)`
- Include confidence levels: "According to your notes [c:0.82], ..."
- Flag contradictions: "Note: your notes contain conflicting information about..."
- If the answer would make a good wiki page, say so and offer to create it
- NEVER include private claims in your answers
- If you don't have enough information, say so honestly

## Vault path: {config.vault_path}
""",
        encoding="utf-8",
    )

    # Digest synthesis prompt
    (claude_dir / "digest.md").write_text(
        f"""# Weekly Digest Synthesis

You are Dendr's weekly digest synthesizer. Run `dendr digest --claude` first
to generate the data payload, then process the prompt at
`Wiki/_digest_prompt.md` to produce the final briefing.

## Your workflow
1. Read `Wiki/_digest_prompt.md` (generated by `dendr digest --claude`)
2. Follow the instructions in that prompt to synthesize the data
3. Write the result to `Wiki/digest.md`

## Tone
- Neutral and direct. No cheerleading, no nagging.
- Every suggestion must be grounded in the user's actual notes.
- Omit sections with nothing substantive to say.

## Vault path: {config.vault_path}
""",
        encoding="utf-8",
    )

    # Schema review prompt
    (claude_dir / "schema-review.md").write_text(
        f"""# Monthly Schema Review

You are reviewing the Dendr wiki schema for potential improvements.

## Your inputs
- `Wiki/schema.md` — current schema
- Recent lint reports from `Wiki/_lint/`
- Current `Wiki/index.md`

## Your tasks
1. Review lint patterns: are the same issues recurring?
2. Propose schema amendments if needed (new page types, new fields, rule changes)
3. Identify any emerging categories that deserve their own page type
4. Check if the canonicalization is working (look for near-duplicate concepts)

## Rules
- Propose changes as diffs to schema.md — don't rewrite from scratch
- Explain the rationale for each proposed change
- Flag any concepts that should be merged
- Keep the schema minimal — add structure only where it solves a real problem

## Vault path: {config.vault_path}
""",
        encoding="utf-8",
    )
