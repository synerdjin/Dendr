# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Workflow rule

Every time you change code in this repo, run `/simplify` and then `/code-review` on the diff before considering the work done, and apply any warranted fixes they surface.

## What is Dendr

A personal knowledge compiler that watches Obsidian Daily Notes, stores each block as raw text in SQLite with FTS and vector search, and generates weekly digests. Claude (via Claude Code) reads the raw blocks directly and does classification, affect reading, and narrative synthesis in one pass. Local models handle only embeddings (for semantic search).

> **Upgrade note (v3):** the text tagger and `block_annotations` schema were removed. If you have a pre-existing `state.sqlite`, rebuild it: `rm state.sqlite && dendr ingest`. The schema no longer contains `block_annotations`, `annotations_fts`, `annotations_vec`, or `block_state`. Claude now reads raw block text at digest synthesis time instead of pre-computed annotations.
>
> **Upgrade note (v4):** the embedding model changed from nomic-embed-text-v1.5 to **embeddinggemma-300m** (still 768d, so `blocks_vec` is unchanged), and embeddings now carry task-instruction prompts. The vector space is different, so stored embeddings must be regenerated: `dendr models pull` then `rm state.sqlite && dendr ingest`. Hybrid search now fuses FTS + semantic with Reciprocal Rank Fusion instead of concatenation.

## Commands

```bash
# Install (editable)
pip install -e .

# Update a local install after pulling changes
# (git pull + refresh deps + verify models + restart the launchd daemon).
# Editable install means pure-Python changes need only `git pull`; this
# script also covers new deps, model-manifest changes, and the long-lived daemon.
scripts/update.sh

# Lint & format (run before pushing)
ruff check --fix src/ tests/           # auto-fixes unused imports (F401 only)
ruff format src/ tests/                # apply formatting

# Run tests
pytest
pytest tests/test_db.py                # single file

# Run the CLI
dendr init /path/to/vault
dendr daemon                          # watch Daily/ for changes
dendr ingest                          # single ingest cycle
dendr search "query" --mode hybrid
dendr serve                           # search server on :7777
dendr serve --host 0.0.0.0            # bind all interfaces (Docker)
dendr mcp                             # MCP server (stdio) for Claude clients; needs dendr[mcp]
dendr stats
dendr digest                          # generate weekly briefing
dendr digest --claude                 # also generate Claude synthesis prompt
dendr models pull                     # download all models from manifest
dendr models verify                   # check SHA256 integrity

# Run the daemon on login (macOS launchd LaunchAgent)
dendr autostart install               # write ~/Library/LaunchAgents/com.dendr.daemon.plist + load it
dendr autostart status                # is the agent installed / loaded?
dendr autostart uninstall             # stop + remove the agent

# Search API (requires dendr serve or Docker search container)
curl "http://localhost:7777/search?q=your+query&mode=hybrid&limit=10"
curl "http://localhost:7777/search?q=your+query&mode=semantic&limit=5"
curl "http://localhost:7777/search?q=your+query&mode=fts&limit=10"
curl "http://localhost:7777/stats"

# Docker (requires nvidia-container-toolkit)
docker compose up -d                  # daemon + search + monitoring
docker compose run daemon dendr models pull  # first-time model download

# Structured JSON logging
DENDR_LOG_JSON=1 dendr daemon
```

## Architecture

### Data flow

```
Daily/*.md  →  parser (block_id, source_date, checkbox_state, hash)
           →  privacy filter
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
- **config.py** — `Config` dataclass with all paths and `append_activity_log()`. Vault content lives in iCloud-synced `vault_path`; state/models/queue live in `data_dir` (`%LOCALAPPDATA%\Dendr` on Windows)
- **parser.py** — Block-level parser for Obsidian daily notes. Assigns `^dendr-<ulid>` block IDs, hashes blocks for incremental processing, detects `- [ ]` / `- [x]` checkbox state
- **privacy.py** — Regex-based filter that tags blocks containing secrets or `#dendr-private`/`#private`/`#redact` tags. Private blocks are stored locally but never sent to Claude
- **queue.py** — File-based two-phase commit queue (pending → processing → done). On crash, items in processing/ are recovered on restart
- **db.py** — SQLite with FTS5 and `sqlite-vec`. Tables: `blocks`, `blocks_fts`, `blocks_vec`, `feedback_scores`, `task_events`. Uses WAL mode
- **llm.py** — `LLMClient` wrapper around llama-cpp-python. Methods: `embed()` / `embed_batch()` for semantic search
- **model_manager.py** — Declarative model manifest (`dendr-models.yaml`), HuggingFace download, SHA256 verification and locking
- **pipeline.py** — Ingest pipeline: parse → privacy → queue → embed → commit. Runs `reconcile_closures` first so user digest edits are in place before re-parse. Checkbox transitions (open→closed, closed→open) are logged as `task_events`
- **metrics.py** — Prometheus counters/gauges/histograms for pipeline and search observability
- **search.py** — FastAPI server on port 7777 with `/search` (FTS + semantic + hybrid over raw block text), `/stats`, and `/metrics` endpoints. Semantic and hybrid results carry a 0-1 `score` (cosine similarity); `min_score` filters out weak matches. Uses per-request DB connections and a threading lock around LLM inference for thread safety under uvicorn's worker pool
- **watcher.py** — `watchdog`-based filesystem watcher that triggers ingest on Daily/ changes
- **autostart.py** — macOS launchd LaunchAgent generation (`dendr autostart install/uninstall/status`). Renders a `~/Library/LaunchAgents/com.dendr.daemon.plist` that runs `<python> -m dendr daemon` with `RunAtLoad` + `KeepAlive` so the daemon starts on login and respawns on crash. Pure plist rendering is unit-tested; launchctl bootstrap/bootout is wrapped with legacy load/unload fallback
- **digest.py** — Weekly digest generator. Assembles a raw-text payload (this-period blocks + carried-forward open tasks + user context + last 4 archived digests) and either writes a Claude synthesis prompt to `Wiki/_digest_prompt.md` or renders a minimal local digest. The prompt body lives in `templates/synthesis_prompt.md`. The `## Task Review` section carries age-bucketed closure markers. Before overwriting, the prior `digest.md` is archived to `Wiki/digests/{ISO-week}.md` for cross-week context

### Models

Declared in `dendr-models.yaml`. One GGUF model runs via llama-cpp-python:

- **embedding** (embeddinggemma-300m, QAT Q8_0) — 768d Matryoshka embeddings (2048-token context) for semantic search over raw block text. Queries and documents are embedded with EmbeddingGemma's task-instruction prompts (`task: search result | query: …` / `title: none | text: …`); skipping these degrades retrieval, so the prompt layer lives in `llm.py` and is keyed on the active model family

### Storage split

- **Vault** (iCloud-synced): `Daily/`, `Wiki/`, `Attachments/` — markdown files the user reads on any device
- **Data dir** (local only): `state.sqlite`, `queue/`, `models/` — never synced

### Monitoring

Prometheus + Grafana stack via `docker compose up prometheus grafana gpu-exporter`. Dashboard auto-provisions with pipeline, model, and GPU metrics.

## Key design patterns

- **Raw-text first**: The `blocks` table stores the block's original Markdown verbatim. All search, digest, and synthesis work reads raw text directly; Claude does classification and affect reading on the fly
- **Checkbox-driven task lifecycle**: Tasks are identified by `- [ ]` / `- [x]` checkboxes. Checkbox transitions log `task_events` (source='auto'). User-driven closures via the digest review flow log events with source='user' and set `completion_status`
- **Closure review loop**: Digest `## Task Review` section lists open tasks >1 week old with checkbox + `<!-- closure:<block_id> status:open -->` markers. User edits the checkbox or status value in `digest.md`; on next ingest, `reconcile_closures` parses the markers and writes `completion_status` + user-sourced task events. User closures take precedence — re-parsing the source file doesn't clobber them, because the upsert preserves `completion_status`. **Source write-back**: closing a task `done`/`abandoned` in the digest also flips the checkbox in the original Daily note via `parser.close_task_in_source` — `- [x]` + `✅ <date>` (done) or `- [-]` + `❌ <date>` (cancelled), Tasks-plugin format — so the user never has to hunt down the source line. `snoozed`/`still-live` leave source untouched. The write-back's re-parse echo (open→closed) is suppressed in `_track_checkbox_transition` when `completion_status` is already a user-set terminal state, so the close isn't double-logged
- **Claude-at-synthesis**: The `--claude` digest path writes a prompt to `Wiki/_digest_prompt.md` containing raw blocks + minimal structural metadata (block_id, source_date, age_days, checkbox_state, completion_status) + user context + the last 4 archived digests. Claude does the classification, clustering, affect reading, and narrative synthesis at read time. The prompt framing is a retrospective-coach with anti-sycophancy / safety / rumination-vs-insight rules; the full template lives in `src/dendr/templates/synthesis_prompt.md`
- **Prior-digest archive**: Before each `--claude` run overwrites `Wiki/digest.md`, the old content is copied to `Wiki/digests/{ISO-week}.md`. The next run loads the last `PRIOR_DIGEST_COUNT` archives into the synthesis payload so Claude can do a Review step — which experiments / questions / open loops from prior weeks are still live, which quietly disappeared. Archives are re-ingested verbatim, so manual user edits to `Wiki/digests/*.md` DO influence future Claude output
- **Feedback loop**: Digest sections include feedback comment blocks. User ratings are stored in `feedback_scores` and surfaced via `get_section_effectiveness()` so Claude can weight sections by past usefulness
- **Two-phase queue**: Crash-safe processing via file-based pending → processing → done transitions
- **Age-is-explicit**: Each block carries `source_date` and a computed `age_days`, and the synthesis prompt warns Claude that anything the user flagged as urgent N days ago was urgent *then*, not today
- **Privacy-first**: Blocks pass through a regex privacy filter before embedding/storage. Private blocks are stored for local search but excluded from Claude payloads
