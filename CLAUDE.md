# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What is Dendr

A personal knowledge compiler that watches Obsidian Daily Notes, extracts claims via local LLMs (llama-cpp-python), and maintains a structured wiki with confidence-scored SPO triples, contradiction detection, and concept canonicalization. Claude (via Claude Code) handles weekly synthesis and Q&A; all mechanical work uses local models.

## Commands

```bash
# Install (editable)
pip install -e .

# Run tests
pytest
pytest tests/test_parser.py           # single file
pytest tests/test_parser.py::test_parse_single_paragraph  # single test

# Run the CLI
dendr init /path/to/vault
dendr daemon                          # watch Daily/ for changes
dendr ingest                          # single ingest cycle
dendr search "query" --mode hybrid
dendr lint
dendr serve                           # search server on :7777
dendr stats
dendr models pull                     # download all models from manifest
dendr models verify                   # check SHA256 integrity

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
    →  enrichment (local LLM)  →  canonicalization  →  claim store (SQLite)
    →  wiki page update  →  Wiki/concepts/*.md, Wiki/entities/*.md
```

### Key modules (src/dendr/)

- **cli.py** — Click CLI entry point, registered as `dendr` console script
- **config.py** — `Config` dataclass with all paths. Vault content lives in iCloud-synced `vault_path`; state/models/queue live in `data_dir` (`%LOCALAPPDATA%\Dendr` on Windows)
- **parser.py** — Block-level parser for Obsidian daily notes. Assigns `^dendr-<ulid>` block IDs and hashes blocks for incremental processing. Parsing is read-only; `inject_block_ids` writes back separately
- **privacy.py** — Regex-based filter that tags blocks containing secrets or `#dendr-private`/`#private`/`#redact` tags. Private blocks are stored locally but never sent to Claude
- **queue.py** — File-based two-phase commit queue (pending → processing → done). On crash, items in processing/ are recovered on restart
- **enrichment.py** — Calls `LLMClient` to extract claims from blocks. Supports backpressure mode (shallow tagging when queue is deep)
- **canonicalize.py** — Embedding-based concept deduplication using `sqlite-vec` nearest-neighbor search against a configurable threshold (default 0.86)
- **db.py** — SQLite with FTS5 for text search and `sqlite-vec` for vector search. Tables: `claims`, `concepts`, `block_state`, `page_hashes`, `log`. Uses WAL mode
- **wiki.py** — Creates/updates concept and entity pages. Enforces the LLM-zone rule: content between `<!-- llm-zone -->` markers is system-managed; content between `<!-- human-zone -->` markers is never touched. Detects human edits via content hash and switches to append-only mode
- **llm.py** — `LLMClient` wrapper around llama-cpp-python for enrichment, tagging, embedding, OCR, and wiki section generation
- **model_manager.py** — Declarative model manifest (`dendr-models.yaml`), HuggingFace download, SHA256 verification and locking
- **pipeline.py** — Orchestrates the full ingest cycle: scan → queue → process → wiki update. Manages transactions with explicit BEGIN/COMMIT/ROLLBACK
- **metrics.py** — Prometheus counters/gauges/histograms for pipeline and search observability
- **search.py** — FastAPI server on port 7777 with `/search` (FTS + semantic + hybrid) and `/metrics` endpoints
- **watcher.py** — `watchdog`-based filesystem watcher that triggers ingest on Daily/ changes
- **lint.py** — Health checks: orphan pages, stale claims, contradictions, missing cross-references. Outputs markdown reports to `Wiki/_lint/`
- **migrate_logseq.py** — One-shot LogSeq-to-Obsidian vault migration

### Models

Declared in `dendr-models.yaml`. Four GGUF models run via llama-cpp-python:
- **enrichment** (Phi-4 14B) — claim extraction, wiki section generation
- **tagger** (Gemma 3 4B) — fast concept/entity tagging, privacy NER
- **vlm** (Llama 3.2 Vision 11B) — screenshot OCR, PDF extraction (gated, needs HF_TOKEN)
- **embedding** (nomic-embed-text-v1.5) — 768d Matryoshka embeddings for canonicalization and semantic search

### Storage split

- **Vault** (iCloud-synced): `Daily/`, `Wiki/`, `Attachments/` — markdown files the user reads on any device
- **Data dir** (local only): `state.sqlite`, `queue/`, `models/`, `ft-pairs.jsonl` — never synced

### Monitoring

Prometheus + Grafana stack via `docker compose up prometheus grafana gpu-exporter`. Dashboard auto-provisions with pipeline, model, knowledge quality, and GPU metrics.

## Key design patterns

- **Two-phase queue**: Crash-safe processing via file-based pending → processing → done transitions
- **LLM-zone rule**: Wiki pages split into human-zone (sacred) and llm-zone (system-managed). Human edits detected via content hash comparison trigger append-only mode
- **Backpressure**: When queue exceeds `backpressure_days` threshold, switches to shallow tagging (fast tagger model only, no full claim extraction)
- **Concept canonicalization**: New concepts are matched against existing ones by embedding similarity; concepts within the threshold are merged to the existing slug
- **Privacy-first**: All blocks pass through the privacy filter before any LLM call. Private blocks are stored for local search but excluded from Claude payloads
