"""CLI entry point for Dendr.

Commands:
  dendr init <vault_path>    Initialize a vault for Dendr
  dendr daemon               Run the watcher daemon
  dendr ingest               Run a single ingest cycle
  dendr search <query>       Search the knowledge base
  dendr lint                 Run lint checks
  dendr serve                Start the search server
  dendr stats                Show knowledge base statistics
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import click

from dendr import __version__


def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
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
@click.option("--data-dir", type=click.Path(), default=None, help="Override data directory")
def daemon(data_dir: str | None) -> None:
    """Run the watcher daemon (blocks until Ctrl-C)."""
    from dendr.config import Config
    from dendr.watcher import run_daemon

    dd = Path(data_dir) if data_dir else None
    config = Config.load(dd)
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
@click.argument("query")
@click.option("--mode", type=click.Choice(["fts", "semantic", "hybrid"]), default="hybrid")
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
            results.append({
                "id": r["id"],
                "text": r["text"],
                "concept": r["concept_slug"],
                "confidence": r["confidence"],
                "source": r["source_block_ref"],
                "type": "fts",
            })

    if mode in ("semantic", "hybrid"):
        try:
            llm = LLMClient(config)
            emb = llm.embed(query)
            sem = dendr_db.search_claims_semantic(conn, emb, limit=limit)
            for r in sem:
                if not any(x["id"] == r["id"] for x in results):
                    results.append({
                        "id": r["id"],
                        "text": r["text"],
                        "concept": r["concept_slug"],
                        "confidence": r["confidence"],
                        "source": r["source_block_ref"],
                        "type": "semantic",
                    })
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
def serve(data_dir: str | None) -> None:
    """Start the search HTTP server."""
    from dendr.config import Config
    from dendr.search import run_server

    dd = Path(data_dir) if data_dir else None
    config = Config.load(dd)
    click.echo(f"Starting search server on http://127.0.0.1:{config.search_port}")
    run_server(config)


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
- `subject`, `predicate`, `object`: SPO triple
- `confidence`: 0.0-1.0 (inline as `[c:0.82]` in markdown)
- `status`: created → reinforced → challenged → superseded
- `source_block_ref`: link back to the daily note block

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
- Contradictions: same subject+predicate, different object, both non-superseded
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
