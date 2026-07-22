# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Workflow rule

Every time you change code in this repo, run `/simplify` and then `/code-review` on the diff before considering the work done, and apply any warranted fixes they surface.

## What is Dendr

A personal knowledge compiler that ingests Obsidian Daily Notes on a schedule, stores each block as raw text in SQLite with FTS and vector search, and generates weekly digests. Claude (via Claude Code) reads the raw blocks directly and does classification, affect reading, and narrative synthesis in one pass. Local models handle only embeddings (for semantic search).

> **Upgrade note (v3):** the text tagger and `block_annotations` schema were removed. If you have a pre-existing `state.sqlite`, rebuild it: `rm state.sqlite && dendr ingest`. The schema no longer contains `block_annotations`, `annotations_fts`, `annotations_vec`, or `block_state`. Claude now reads raw block text at digest synthesis time instead of pre-computed annotations.
>
> **Upgrade note (v4):** the embedding model changed from nomic-embed-text-v1.5 to **embeddinggemma-300m** (still 768d, so `blocks_vec` is unchanged), and embeddings now carry task-instruction prompts. The vector space is different, so stored embeddings must be regenerated: `dendr models pull` then `rm state.sqlite && dendr ingest`. Hybrid search now fuses FTS + semantic with Reciprocal Rank Fusion instead of concatenation.
>
> **Upgrade note (v5):** the regex privacy filter (`privacy.py`, `#dendr-private`/`#private`/`#redact` tags, the `blocks.private` column) was removed — it added noise without meaningfully protecting anything a determined regex couldn't miss. All blocks are now stored, searched, and sent to Claude at digest time the same way. No rebuild needed: `init_schema()` drops the leftover `private` column from `state.sqlite` automatically on next connect (SQLite ≥3.35; older versions just leave it in place, harmless either way).
>
> **Upgrade note (v6):** Docker support was dropped — `Dockerfile`, `docker-compose.yaml`, and the Prometheus/Grafana/GPU-exporter monitoring stack are gone. The Docker image was built on `nvidia/cuda` and was never actually usable on Apple Silicon anyway (no Metal passthrough in a Linux VM); this MacBook Air M4 is now the only platform Dendr targets. `dendr serve`'s `/metrics` and the daemon's `:9100/metrics` endpoints still exist in code — nothing scrapes them by default, no dashboard is provided. Day-to-day tasks now go through the `Makefile` (`make help`) instead of `docker compose`.
>
> **Upgrade note (v7):** dependency management switched from bare `pip` to [`uv`](https://docs.astral.sh/uv/), with a committed `uv.lock` for reproducible installs. Dev tools moved from a `dev` extra to PEP 735 `[dependency-groups]` (`lint` = ruff, `test` = pytest, `dev` = both — `dev` is uv's default group, so a bare `uv sync` still gets everything locally). `make install` / `scripts/update.sh` now run `UV_PROJECT_ENVIRONMENT=~/.dendr-venv uv sync` instead of `pip install -e .[dev]` — same venv location, no rebuild needed. Requires `uv` on `PATH` (`brew install uv`). CI (`pr-checks.yml`, `security-scan.yml`) installs via `astral-sh/setup-uv` + `uv sync --locked` variants scoped per job (`--only-group lint --no-install-project` for the lint job so it skips building `llama-cpp-python` just to run ruff; `--group test --no-default-groups` for the tests job), so a stale lockfile now fails the build instead of silently resolving something different than what's pinned.
>
> **Upgrade note (v8):** the live file-watcher daemon (`watcher.py`, `dendr daemon`, the `watchdog` dependency) was removed in favor of scheduled ingest — a live watcher with a 5s debounce meant editing a block mid-draft could trigger a full re-embed on every short pause. `dendr autostart install` now writes a `~/Library/LaunchAgents/com.dendr.ingest.plist` (label `com.dendr.ingest`, replacing the old `com.dendr.daemon`) that runs `dendr ingest` once at login (`RunAtLoad`) and then every N minutes via `StartInterval` (`--interval-minutes`, default 15) — each run is a single ingest cycle that exits, not a long-lived process. `dendr autostart install` on the new code automatically unloads and removes a lingering `com.dendr.daemon` agent (`LEGACY_LAUNCH_AGENT_LABEL` in `autostart.py`) before installing the new one, so re-running install after upgrading is enough — no manual `launchctl` cleanup needed. The Prometheus queue/DB gauges that only the watcher's periodic loop ever set (`dendr_queue_pending`, `dendr_queue_processing`, `dendr_blocks_total`, `dendr_open_tasks`) were removed along with it; `dendr serve`'s `/metrics` still exposes pipeline/search metrics that are set inline during a request or ingest run.

## Commands

```bash
# Install (editable, from uv.lock — dev group included by default)
uv sync

# Update a local install after pulling changes
# (git pull + refresh deps + verify models + restart the scheduled ingest agent).
# Editable install means pure-Python changes need only `git pull`; this
# script also covers new deps, model-manifest changes, and kicking the agent.
scripts/update.sh

# Or via the Makefile, which wraps the above (and more) around ~/.dendr-venv —
# see `make help` for the full target list.
make update

# Lint & format (run before pushing)
ruff check --fix src/ tests/           # auto-fixes unused imports (F401 only)
ruff format --check src/ tests/        # CI runs format --check, not just check
make check                             # lint + format-check + test in one shot

# Run tests
pytest
pytest tests/test_db.py                # single file

# Run the CLI
dendr init /path/to/vault
dendr ingest                          # single ingest cycle
dendr search "query" --mode hybrid
dendr serve                           # search server on :7777
dendr stats
dendr digest                          # generate weekly briefing
dendr digest --claude                 # also generate Claude synthesis prompt
dendr models pull                     # download all models from manifest
dendr models verify                   # check SHA256 integrity

# Run ingest on a schedule (macOS launchd LaunchAgent, every 15 min by default)
dendr autostart install               # write ~/Library/LaunchAgents/com.dendr.ingest.plist + load it
dendr autostart status                # is the agent installed / loaded?
dendr autostart uninstall             # stop + remove the agent

# Search API (requires dendr serve)
curl "http://localhost:7777/search?q=your+query&mode=hybrid&limit=10"
curl "http://localhost:7777/search?q=your+query&mode=semantic&limit=5"
curl "http://localhost:7777/search?q=your+query&mode=fts&limit=10"
curl "http://localhost:7777/stats"

# Structured JSON logging
DENDR_LOG_JSON=1 dendr ingest
```

## Architecture

### Data flow

```
Daily/*.md  →  parser (block_id, source_date, checkbox_state, hash)
           →  queue (pending/processing/done)
           →  embed raw text (EmbeddingGemma, model stays loaded)
           →  commit: blocks upsert + blocks_fts + blocks_vec
                     + task_events for checkbox transitions

reconcile_closures (pre-ingest)
    →  scan Wiki/digest.md for `<!-- closure:<block_id> status:... -->` markers
    →  write completion_status on the block + log task_events (source='user')

dendr digest --claude
    →  archive previous Wiki/digest.md to Wiki/digests/{ISO-week}.md
    →  load last 4 archived digests into the synthesis payload (prior_digests)
    →  write templates/synthesis_prompt.md filled with raw blocks + prior digests
       to Wiki/_digest_prompt.md for Claude Code to execute
```

### Key modules (src/dendr/)

- **cli.py** — Click CLI entry point, registered as `dendr` console script
- **config.py** — `Config` dataclass with all paths and `append_activity_log()`. Vault content lives in iCloud-synced `vault_path`; state/models/queue live in `data_dir` (`~/.local/share/dendr` on macOS)
- **parser.py** — Block-level parser for Obsidian daily notes. Assigns `^dendr-<ulid>` block IDs, hashes blocks for incremental processing, detects `- [ ]` / `- [x]` checkbox state
- **queue.py** — File-based two-phase commit queue (pending → processing → done). On crash, items in processing/ are recovered on restart
- **db.py** — SQLite with FTS5 and `sqlite-vec`. Tables: `blocks`, `blocks_fts`, `blocks_vec`, `feedback_scores`, `task_events`. Uses WAL mode
- **llm.py** — `LLMClient` wrapper around llama-cpp-python. Methods: `embed()` / `embed_batch()` for semantic search
- **model_manager.py** — Declarative model manifest (`dendr-models.yaml`), HuggingFace download, SHA256 verification and locking
- **pipeline.py** — Ingest pipeline: parse → queue → embed → commit. Runs `reconcile_closures` first so user digest edits are in place before re-parse. Checkbox transitions (open→closed, closed→open) are logged as `task_events`
- **metrics.py** — Prometheus counters/gauges/histograms for pipeline and search observability, exposed via `dendr serve`'s `/metrics`
- **search.py** — FastAPI server on port 7777 with `/search` (FTS + semantic + hybrid over raw block text), `/stats`, and `/metrics` endpoints. Semantic and hybrid results carry a 0-1 `score` (cosine similarity); `min_score` filters out weak matches. Uses per-request DB connections and a threading lock around LLM inference for thread safety under uvicorn's worker pool
- **autostart.py** — macOS launchd LaunchAgent generation (`dendr autostart install/uninstall/status`). Renders a `~/Library/LaunchAgents/com.dendr.ingest.plist` that runs `<python> -m dendr ingest` with `RunAtLoad` (once at login) + `StartInterval` (every N seconds, 15 min by default via `--interval-minutes`) — each run is a single ingest cycle that exits. Pure plist rendering is unit-tested; launchctl bootstrap/bootout is wrapped with legacy load/unload fallback
- **digest.py** — Weekly digest generator. Assembles a raw-text payload (this-period blocks + carried-forward open tasks + user context + per-period intentions + last 4 archived digests) and either writes a Claude synthesis prompt to `Wiki/_digest_prompt.md` or renders a minimal local digest. The prompt body lives in `templates/synthesis_prompt.md`. The `## Task Review` section carries age-bucketed closure markers. Before overwriting, the prior `digest.md` is archived to `Wiki/digests/{ISO-week}.md` for cross-week context

### Models

Declared in `dendr-models.yaml`. One GGUF model runs via llama-cpp-python:

- **embedding** (embeddinggemma-300m, QAT Q8_0) — 768d Matryoshka embeddings (2048-token context) for semantic search over raw block text. Queries and documents are embedded with EmbeddingGemma's task-instruction prompts (`task: search result | query: …` / `title: none | text: …`); skipping these degrades retrieval, so the prompt layer lives in `llm.py` and is keyed on the active model family

### Storage split

- **Vault** (iCloud-synced): `Daily/`, `Wiki/`, `Attachments/` — markdown files the user reads on any device
- **Data dir** (local only): `state.sqlite`, `queue/`, `models/` — never synced

## Key design patterns

- **Raw-text first**: The `blocks` table stores the block's original Markdown verbatim. All search, digest, and synthesis work reads raw text directly; Claude does classification and affect reading on the fly
- **Checkbox-driven task lifecycle**: Tasks are identified by `- [ ]` / `- [x]` checkboxes. Checkbox transitions log `task_events` (source='auto'). User-driven closures via the digest review flow log events with source='user' and set `completion_status`
- **Closure review loop**: Digest `## Task Review` section lists open tasks >1 week old with checkbox + `<!-- closure:<block_id> status:open -->` markers. User edits the checkbox or status value in `digest.md`; on next ingest, `reconcile_closures` parses the markers and writes `completion_status` + user-sourced task events. User closures take precedence — re-parsing the source file doesn't clobber them, because the upsert preserves `completion_status`. **Source write-back**: closing a task `done`/`abandoned` in the digest also flips the checkbox in the original Daily note via `parser.close_task_in_source` — `- [x]` + `✅ <date>` (done) or `- [-]` + `❌ <date>` (cancelled), Tasks-plugin format — so the user never has to hunt down the source line. `snoozed`/`still-live` leave source untouched. The write-back's re-parse echo (open→closed) is suppressed in `_track_checkbox_transition` when `completion_status` is already a user-set terminal state, so the close isn't double-logged
- **Claude-at-synthesis**: The `--claude` digest path writes a prompt to `Wiki/_digest_prompt.md` containing raw blocks + minimal structural metadata (block_id, source_date, age_days, checkbox_state, completion_status) + user context + per-period intentions + the last 4 archived digests. Claude does the classification, clustering, affect reading, and narrative synthesis at read time. The prompt framing is a retrospective-coach with anti-sycophancy / safety / rumination-vs-insight rules. Stated intentions (`Wiki/_intentions.md`) drive an **intention-vs-attention drift** lens: Claude judges where the week's attention actually went against what the user said it would, and is told to examine the intention rather than rubber-stamp it; the full template lives in `src/dendr/templates/synthesis_prompt.md`
- **Prior-digest archive**: Before each `--claude` run overwrites `Wiki/digest.md`, the old content is copied to `Wiki/digests/{ISO-week}.md`. The next run loads the last `PRIOR_DIGEST_COUNT` archives into the synthesis payload so Claude can do a Review step — which experiments / questions / open loops from prior weeks are still live, which quietly disappeared. Archives are re-ingested verbatim, so manual user edits to `Wiki/digests/*.md` DO influence future Claude output
- **Feedback loop**: Digest sections include feedback comment blocks. User ratings are stored in `feedback_scores` and surfaced via `get_section_effectiveness()` so Claude can weight sections by past usefulness
- **Two-phase queue**: Crash-safe processing via file-based pending → processing → done transitions
- **Age-is-explicit**: Each block carries `source_date` and a computed `age_days`, and the synthesis prompt warns Claude that anything the user flagged as urgent N days ago was urgent *then*, not today
