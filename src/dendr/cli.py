"""CLI entry point for Dendr.

Commands:
  dendr init <vault_path>    Initialize a vault for Dendr
  dendr ingest               Run a single ingest cycle
  dendr search <query>       Search the knowledge base
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
    """Emit structured JSON log lines for log aggregation."""

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
@click.option("--data-dir", type=click.Path(), default=None)
@click.option(
    "--vault",
    type=click.Path(exists=True, file_okay=False),
    default=None,
    help="Override vault path (e.g. if config.json has a stale path)",
)
def ingest(data_dir: str | None, vault: str | None) -> None:
    """Run a single ingest cycle."""
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
    """Mark every block dirty and replay the ingest pipeline.

    Blanks `block_hash` on every row so the scan detects them as changed,
    and clears the done/processing queue directories. User-set
    `completion_status` values are preserved.
    """
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

    # Blank the hash so every block is treated as dirty on next ingest.
    # completion_status is preserved so user closures survive re-ingest.
    count = conn.execute("SELECT COUNT(*) as n FROM blocks").fetchone()["n"]
    conn.execute("UPDATE blocks SET block_hash = ''")
    click.echo(f"Marked {count} blocks as dirty")

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

    pool = limit * 2 if mode == "hybrid" else limit

    fts_rows: list = []
    if mode in ("fts", "hybrid"):
        fts_rows = dendr_db.search_blocks_fts(conn, query, limit=pool)

    sem_pairs: list = []
    if mode in ("semantic", "hybrid"):
        try:
            llm = LLMClient(config)
            emb = llm.embed(query, kind="query")
            sem_pairs = dendr_db.search_blocks_semantic(conn, emb, limit=pool)
        except Exception as e:
            click.echo(f"Semantic search unavailable: {e}", err=True)

    results: list[tuple[str, object]] = []  # (score_type, row)
    if mode == "hybrid":
        results = [
            ("hybrid", row)
            for row, _score, _sim in dendr_db.rrf_fuse(fts_rows, sem_pairs, limit)
        ]
    elif mode == "semantic":
        results = [("semantic", row) for row, _sim in sem_pairs[:limit]]
    else:  # fts
        results = [("fts", row) for row in fts_rows[:limit]]

    conn.close()

    if not results:
        click.echo("No results found.")
        return

    for score_type, row in results:
        snippet = row["text"].splitlines()[0][:120] if row["text"] else ""
        click.echo(f"  [{score_type:8s}] {row['source_date']}  {snippet}")


@main.command()
@click.option("--data-dir", type=click.Path(), default=None)
@click.option(
    "--vault",
    type=click.Path(exists=True, file_okay=False),
    default=None,
    help="Override vault path",
)
@click.option(
    "--host",
    default="127.0.0.1",
    help="Bind address (use 0.0.0.0 to expose on the LAN)",
)
def serve(data_dir: str | None, vault: str | None, host: str) -> None:
    """Start the search HTTP server."""
    from dendr.config import Config
    from dendr.search import run_server

    dd = Path(data_dir) if data_dir else None
    config = Config.load(dd)
    if vault:
        config.vault_path = Path(vault).resolve()
    run_server(config, host=host)


@main.command()
@click.option("--data-dir", type=click.Path(), default=None)
@click.option(
    "--vault",
    type=click.Path(exists=True, file_okay=False),
    default=None,
    help="Override vault path (needed when config.json has a stale path)",
)
@click.option("--weeks", type=int, default=1, help="Number of weeks to cover")
@click.option("--claude", is_flag=True, help="Also generate Claude synthesis prompt")
def digest(data_dir: str | None, vault: str | None, weeks: int, claude: bool) -> None:
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

    from dendr.metrics import DIGEST_RUNS

    path = generate_digest(config, conn, weeks=weeks, use_claude=claude)
    conn.close()
    DIGEST_RUNS.labels(mode="claude" if claude else "local").inc()

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

    click.echo(f"Blocks:            {s['blocks']}")
    click.echo(f"Open tasks:        {s['open_tasks']}")
    click.echo(f"Pending queue:     {pending}")


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
    "--role", type=str, default=None, help="Download only this role (e.g. embedding)"
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


@main.group()
def autostart() -> None:
    """Manage the login LaunchAgent that runs ingest on a schedule (macOS)."""


def _require_macos() -> None:
    if sys.platform != "darwin":
        raise click.ClickException(
            "`dendr autostart` uses macOS launchd and only works on macOS."
        )


@autostart.command("install")
@click.option("--data-dir", type=click.Path(), default=None)
@click.option(
    "--interval-minutes",
    type=int,
    default=15,
    help="How often to run `dendr ingest` (default: every 15 minutes)",
)
def autostart_install(data_dir: str | None, interval_minutes: int) -> None:
    """Install + load a LaunchAgent that runs ingest on a schedule."""
    from dendr import autostart as agent
    from dendr.config import Config

    _require_macos()

    dd = Path(data_dir) if data_dir else None
    config = Config.load(dd)
    config.logs_dir.mkdir(parents=True, exist_ok=True)

    if not config.config_file_path.exists():
        click.echo(
            f"⚠  No saved config at {config.config_file_path}. "
            "Run `dendr init <vault>` first, or ingest will fail to start.",
            err=True,
        )

    # Clean up a pre-v8 watcher-daemon agent if it's still installed, so it
    # doesn't keep running (KeepAlive) alongside the new scheduled one.
    legacy_path = agent.remove_legacy_agent()
    if legacy_path:
        click.echo(f"✓ Removed legacy LaunchAgent: {legacy_path}")

    working_dir = str(config.vault_path) if config.vault_path.exists() else None
    plist = agent.render_plist(
        agent.program_args(config.data_dir),
        interval_seconds=interval_minutes * 60,
        stdout_path=str(config.logs_dir / "ingest.out.log"),
        stderr_path=str(config.logs_dir / "ingest.err.log"),
        working_dir=working_dir,
    )

    path = agent.plist_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(plist)

    # Reload cleanly if a previous agent is already loaded.
    agent.unload_agent(path)
    rc, out = agent.load_agent(path)
    if rc != 0:
        raise click.ClickException(
            f"Wrote {path} but `launchctl` failed to load it:\n{out}"
        )

    click.echo(f"✓ Installed LaunchAgent: {path}")
    click.echo(f"  Runs: {' '.join(agent.program_args(config.data_dir))}")
    click.echo(f"  Every {interval_minutes} minutes, plus once at login.")
    click.echo(f"  Logs: {config.logs_dir}/ingest.{{out,err}}.log")
    click.echo("  Manage it with `dendr autostart status` / `... uninstall`.")


@autostart.command("uninstall")
def autostart_uninstall() -> None:
    """Stop + remove the login LaunchAgent."""
    from dendr import autostart as agent

    _require_macos()

    path = agent.plist_path()
    agent.unload_agent(path)
    if path.exists():
        path.unlink()
        click.echo(f"✓ Removed LaunchAgent: {path}")
    else:
        click.echo(f"No LaunchAgent found at {path}; nothing to remove.")

    legacy_path = agent.remove_legacy_agent()
    if legacy_path:
        click.echo(f"✓ Removed legacy LaunchAgent: {legacy_path}")


@autostart.command("status")
def autostart_status() -> None:
    """Show whether the login LaunchAgent is installed and loaded."""
    from dendr import autostart as agent

    _require_macos()

    path = agent.plist_path()
    installed = path.exists()
    loaded = agent.is_loaded()
    click.echo(f"Plist:  {path}  ({'present' if installed else 'absent'})")
    click.echo(f"Loaded: {'yes (running / scheduled)' if loaded else 'no'}")
    if installed and not loaded:
        click.echo("  Installed but not loaded — try `dendr autostart install`.")

    legacy_path = agent.plist_path(agent.LEGACY_LAUNCH_AGENT_LABEL)
    if legacy_path.exists() or agent.is_loaded(agent.LEGACY_LAUNCH_AGENT_LABEL):
        click.echo(
            f"⚠  Legacy pre-v8 agent ({agent.LEGACY_LAUNCH_AGENT_LABEL}) still "
            "present — run `dendr autostart install` or `... uninstall` to remove it."
        )


# --- Schema and prompt generation ---


def _write_default_schema(path: Path) -> None:
    from dendr.templates import read

    path.write_text(read("schema.md"), encoding="utf-8")


def _write_claude_prompts(claude_dir: Path, config) -> None:
    from dendr.templates import read

    (claude_dir / "digest.md").write_text(read("claude_digest.md"), encoding="utf-8")
    (claude_dir / "qa.md").write_text(
        read("claude_qa.md").format(
            search_port=config.search_port,
            vault_path=config.vault_path,
        ),
        encoding="utf-8",
    )
    (claude_dir / "schema-review.md").write_text(
        read("claude_schema_review.md").format(
            vault_path=config.vault_path,
        ),
        encoding="utf-8",
    )

    # Clean up stale prompt files from older versions
    stale = claude_dir / "weekly.md"
    if stale.exists():
        stale.unlink()
