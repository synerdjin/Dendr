# Dendr

A personal knowledge compiler that watches Obsidian Daily Notes, annotates blocks with rich metadata via local LLMs, and stores them in SQLite with FTS and vector search. Weekly digests surface actionable advice with task lifecycle tracking, closure review, and pattern detection.

**Local LLM handles all mechanical work** — block annotation, tagging, embeddings, vision/OCR. **Claude (via Claude Code)** is reserved for weekly synthesis and on-demand Q&A, so a Pro/Max subscription is enough.

## How it works

```
Daily Notes (you write here)
    ↓ watcher detects changes
    ↓ block-level parsing + privacy filter
    ↓ Phase 1: local LLM annotates blocks (gist, type, emotional signals, urgency, concepts)
    ↓ Phase 2: embed annotation text for semantic search
    ↓ Phase 3: commit to SQLite (annotations + FTS + vector index + task events)
Wiki/digest.md (you read here, on any device)
```

## Requirements

- Python 3.11+
- GPU with 12GB+ VRAM (tested on RTX 4070)
- Local model weights (see [Model Setup](#model-setup))
- Obsidian vault synced via iCloud (or any sync)

## Install

```bash
git clone https://github.com/synerdjin/Dendr.git
cd Dendr
pip install -e .
```

## Quick start

```bash
# 1. Initialize your vault
dendr init /path/to/your/obsidian/vault

# 2. Download models
dendr models pull

# 3. Run the daemon (watches Daily/ for changes)
dendr daemon

# 4. Or run a single ingest cycle
dendr ingest

# 5. Search your knowledge base
dendr search "machine learning"

# 6. Generate a weekly digest
dendr digest --claude

# 7. Start the search HTTP server (for Claude Code integration)
dendr serve
```

## CLI commands

| Command | Description |
|---------|-------------|
| `dendr init <vault>` | Initialize vault structure + database + Claude prompts |
| `dendr daemon` | Watch `Daily/` and auto-ingest on changes |
| `dendr ingest` | Run a single ingest cycle |
| `dendr reprocess` | Reset block state and reprocess everything from scratch |
| `dendr search <query>` | Search annotations (FTS, semantic, or hybrid) |
| `dendr serve` | Start search server on `localhost:7777` |
| `dendr stats` | Show knowledge base statistics |
| `dendr digest` | Generate weekly digest briefing |
| `dendr digest --claude` | Also generate Claude synthesis prompt |
| `dendr models pull` | Download all models from manifest |
| `dendr models verify` | Check SHA256 integrity of models |
| `dendr models list` | Show model status table |
| `dendr models lock` | Pin SHA256 hashes into manifest |

## Model setup

Models are declared in `dendr-models.yaml` (version-controlled). Download them all with one command:

```bash
dendr models pull
dendr models verify
dendr models lock    # pin hashes for reproducibility
```

| Role | Model | Size | Purpose |
|------|-------|------|---------|
| Tagger + Vision | [Gemma 4 E4B Q4_K_M](https://huggingface.co/unsloth/gemma-4-E4B-it-GGUF) | ~5 GB | Block annotation, concept tagging, vision/OCR |
| Vision projector | mmproj-BF16 (same repo) | ~1 GB | Bridges vision encoder for image input |
| Embeddings | [nomic-embed-text-v1.5 FP16](https://huggingface.co/nomic-ai/nomic-embed-text-v1.5-GGUF) | ~0.3 GB | 768d Matryoshka embeddings for semantic search |

Models are stored in `%LOCALAPPDATA%\Dendr\models\` (Windows) or `~/.local/share/dendr/models/` (macOS/Linux). Only one model is loaded in VRAM at a time.

## Docker

For reproducible deployment with GPU access:

```bash
# Start daemon + search server + monitoring
docker compose up -d

# Check it's working
curl http://localhost:7777/stats
```

Requires [nvidia-container-toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) for GPU passthrough.

## Architecture

### Storage layout

```
Vault/                          (iCloud-synced)
  Daily/YYYY-MM-DD.md           your raw notes (never modified by Dendr)
  Wiki/
    schema.md                   annotation spec
    log.md                      append-only activity log
    digest.md                   weekly digest (with closure markers)
    _digest_prompt.md           Claude synthesis input
    _user_context.md            stable user background for Claude

%LOCALAPPDATA%\Dendr\           (PC-only, never synced)
  state.sqlite                  annotations + FTS5 + vector search
  queue/                        two-phase commit processing queue
  models/                       GGUF model weights
  ft-pairs.jsonl                fine-tuning corpus (logged automatically)
```

### Block annotation

Every block from a daily note is annotated with structured metadata:
- `gist` — one-line summary
- `block_type` — reflection, task, decision, question, observation, vent, plan, log_entry
- `life_areas` — work, health, relationships, finance, learning, creative, meta
- `emotional_valence` / `intensity` — emotional signals (-1.0 to +1.0, 0.0 to 1.0)
- `urgency` / `importance` — frozen at source date, not today
- `completion_status` — open, done, blocked, abandoned (tracked across re-annotations)
- `concepts` — topic tags for recurring theme detection
- `causal_links` — extracted cause-effect relationships

### Task lifecycle

Task/plan blocks have `completion_status` tracked across annotations. Status transitions are logged as `task_events`. The sticky-closure rule prevents re-annotation from clobbering user-driven closures: if you mark a task done via the digest, a re-ingest won't reopen it.

### Privacy

Blocks containing API keys, passwords, SSNs, or tagged with `#dendr-private` / `#private` / `#redact` are stored locally but **never sent to Claude**.

## Observability

Dendr includes a Prometheus + Grafana monitoring stack:

```bash
docker compose up -d prometheus grafana gpu-exporter
# Grafana:    http://localhost:3000
# Prometheus: http://localhost:9090
```

The daemon exposes metrics on `localhost:9100/metrics` and the search server on `localhost:7777/metrics`.

| Category | Metrics |
|----------|---------|
| **Models** | Which model is loaded, load time, inference latency, tokens/sec |
| **Pipeline** | Blocks processed/hour, ingest cycle duration, queue depth |
| **Search** | Request latency by mode (FTS/semantic/hybrid) |
| **GPU** | Utilization %, VRAM used/free, temperature, power draw |

### Structured logging

```bash
DENDR_LOG_JSON=1 dendr daemon
```

## Claude Code integration

`dendr init` generates three Claude Code session prompts in `.claude/`:

- **digest.md** — weekly synthesis (read `_digest_prompt.md`, write `digest.md`)
- **qa.md** — on-demand Q&A grounded in your annotations
- **schema-review.md** — monthly annotation schema evolution review

## Future

- **Fine-tuning** — all LLM calls are logged as training pairs in `ft-pairs.jsonl`, ready for QLoRA
- **Multi-vault** support via `vault_id` binding

## License

MIT
