# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What is Dendr

A personal knowledge compiler that watches Obsidian Daily Notes, annotates blocks with rich metadata via local LLMs (llama-cpp-python), and stores them in SQLite with FTS and vector search. Weekly digests surface actionable advice with task lifecycle tracking, closure review, and pattern detection. Claude (via Claude Code) handles synthesis and Q&A directly over annotations; all mechanical work uses local models.

> **Upgrade note:** the concepts/entities wiki layer and claims layer were removed. If you have a pre-existing `state.sqlite`, rebuild it: `rm state.sqlite && dendr ingest`. The schema no longer contains `concepts`, `concepts_vec`, `claims`, `claims_fts`, or `claims_vec`.

## Commands

```bash
# Install (editable)
pip install -e .

# Lint & format (run before pushing)
ruff check --fix src/ tests/           # auto-fixes unused imports (F401 only)
ruff format src/ tests/                # apply formatting

# Run tests
pytest
pytest tests/test_db.py                # single file
pytest tests/test_db.py::test_upsert_block_annotation  # single test

# Run the CLI
dendr init /path/to/vault
dendr daemon                          # watch Daily/ for changes
dendr ingest                          # single ingest cycle
dendr search "query" --mode hybrid
dendr serve                           # search server on :7777
dendr serve --host 0.0.0.0            # bind all interfaces (Docker)
dendr stats
dendr digest                          # generate weekly briefing
dendr digest --claude                 # also generate Claude synthesis prompt
dendr models pull                     # download all models from manifest
dendr models verify                   # check SHA256 integrity

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
Daily/*.md  →  parser  →  privacy filter  →  queue (pending/processing/done)
    →  Phase 1: annotate all blocks (tagger model stays loaded)
        →  BlockAnnotation (original text + rich metadata)
    →  Phase 2: embed all annotations (embedding model stays loaded)
        →  annotations_vec (semantic search index)
    →  Phase 3: commit in one transaction per block
        →  task lifecycle tracking (open → done/abandoned, sticky closures)
        →  block_annotations upsert + annotations_fts
reconcile_closures (pre-ingest)
    →  scan Wiki/digest.md for `<!-- closure:... status:... -->` markers
    →  apply user-driven status changes to block_annotations + task_events
```

### Key modules (src/dendr/)

- **cli.py** — Click CLI entry point, registered as `dendr` console script
- **config.py** — `Config` dataclass with all paths and `append_activity_log()`. Vault content lives in iCloud-synced `vault_path`; state/models/queue live in `data_dir` (`%LOCALAPPDATA%\Dendr` on Windows)
- **parser.py** — Block-level parser for Obsidian daily notes. Assigns `^dendr-<ulid>` block IDs and hashes blocks for incremental processing
- **privacy.py** — Regex-based filter that tags blocks containing secrets or `#dendr-private`/`#private`/`#redact` tags. Private blocks are stored locally but never sent to Claude
- **queue.py** — File-based two-phase commit queue (pending → processing → done). On crash, items in processing/ are recovered on restart
- **db.py** — SQLite with FTS5 and `sqlite-vec`. Tables: `block_annotations`, `annotations_fts`, `annotations_vec`, `block_state`, `feedback_scores`, `task_events`. Uses WAL mode
- **llm.py** — `LLMClient` wrapper around llama-cpp-python. Key methods: `annotate_block()` (tagger, temp=0.0) for rich block annotation, `embed()` for semantic search
- **model_manager.py** — Declarative model manifest (`dendr-models.yaml`), HuggingFace download, SHA256 verification and locking
- **pipeline.py** — Three-phase ingest pipeline: annotate → embed → commit. Batches by model to avoid VRAM thrashing. Runs `reconcile_closures` first to apply user-driven closures from digest.md. Task lifecycle tracking detects open→done/abandoned transitions and preserves user-sourced closures on re-annotation (sticky rule)
- **metrics.py** — Prometheus counters/gauges/histograms for pipeline and search observability
- **search.py** — FastAPI server on port 7777 with `/search` (FTS + semantic + hybrid over block_annotations), `/stats`, and `/metrics` endpoints. Uses per-request DB connections and a threading lock around LLM inference for thread safety under uvicorn's worker pool
- **watcher.py** — `watchdog`-based filesystem watcher that triggers ingest on Daily/ changes
- **digest.py** — Weekly digest generator with two-layer context: narrative blocks (original text + annotations) and pattern summaries (topics, life areas, emotional trajectory, task lifecycle). Renders a `## Task Review` section with age-bucketed closure markers; per-section feedback markers feed `feedback_scores` for effectiveness tracking
- **migrate_logseq.py** — One-shot LogSeq-to-Obsidian vault migration

### Models

Declared in `dendr-models.yaml`. Two GGUF models run via llama-cpp-python:
- **tagger** (Gemma 4 E4B, Q4_K_M) — block annotation (emotional signals, life areas, urgency, causal links, concept tagging) + vision/OCR via mmproj. Single model handles both text annotation and image extraction
- **embedding** (nomic-embed-text-v1.5, FP16) — 768d Matryoshka embeddings for semantic search over annotations

### Storage split

- **Vault** (iCloud-synced): `Daily/`, `Wiki/`, `Attachments/` — markdown files the user reads on any device
- **Data dir** (local only): `state.sqlite`, `queue/`, `models/`, `ft-pairs.jsonl` — never synced

### Monitoring

Prometheus + Grafana stack via `docker compose up prometheus grafana gpu-exporter`. Dashboard auto-provisions with pipeline, model, knowledge quality, and GPU metrics.

## Key design patterns

- **Annotation-first**: `block_annotations` is the single source of truth — original text preserved alongside rich metadata (emotional valence, life areas, urgency/importance, causal links, completion status). All search, digest, and synthesis work reads annotations directly
- **Task lifecycle tracking**: Task/plan blocks have `completion_status` tracked across annotations. Status transitions (open→done, open→abandoned) are logged as `task_events` for lifecycle statistics. Sticky-closure rule: if an existing annotation is `done`/`abandoned`/`snoozed` and re-annotation returns `None`/`open`, the existing status is preserved (prevents re-annotation from clobbering user closures)
- **Closure review loop**: Digest `## Task Review` section lists open tasks >7d old with checkbox + `<!-- closure:<block_id> status:open -->` markers. User edits the checkbox and status value in `digest.md`; on next ingest, `reconcile_closures` parses the markers and writes status changes + user-sourced task events
- **Two-layer digest**: Narrative blocks (original text + annotations) → Pattern summaries (recurring topics, emotional trajectory, life areas, task lifecycle)
- **Feedback loop**: Digest sections include feedback comment blocks. User ratings are stored in `feedback_scores` and aggregated via `get_section_effectiveness()` to calibrate future digests
- **Two-phase queue**: Crash-safe processing via file-based pending → processing → done transitions
- **Urgency-is-frozen**: `urgency`/`importance` tags reflect the user's state at `source_date`, not today. Local digest renders age suffixes (`[today when written]`, `written 3w ago`); synthesis prompt passes `age_days` so Claude can reason about staleness
- **Privacy-first**: All blocks pass through the privacy filter before any LLM call. Private blocks are stored for local search but excluded from Claude payloads
