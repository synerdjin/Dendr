# Dendr

A personal knowledge compiler that watches Obsidian Daily Notes, stores each block as raw text in SQLite with FTS and vector search, and generates weekly digests. Claude (via Claude Code) reads the raw blocks directly at digest time and does classification, affect reading, and narrative synthesis in one pass.

**Local models do only what Claude can't or shouldn't**: embeddings for semantic search, and vision/OCR for image and scanned-PDF attachments. **Claude (via Claude Code)** handles weekly synthesis and on-demand Q&A, so a Pro/Max subscription is enough.

## How it works

```
Daily Notes (you write here)
    ↓ watcher detects changes
    ↓ block-level parsing + privacy filter
    ↓ embed raw block text (Nomic)
    ↓ commit to SQLite (blocks + FTS + vector index)
    ↓ log task_events on checkbox transitions
Wiki/digest.md (you read here, on any device)
```

## Requirements

- Python 3.11+
- GPU with 6GB+ VRAM (for embeddings; more if you use vision/OCR)
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
| `dendr reprocess` | Mark all blocks dirty and re-embed from scratch |
| `dendr search <query>` | Search blocks (FTS, semantic, or hybrid) |
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
| Vision | [Gemma 4 E4B Q4_K_M](https://huggingface.co/unsloth/gemma-4-E4B-it-GGUF) | ~5 GB | On-demand vision/OCR for image and scanned-PDF attachments |
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
    schema.md                   block schema (raw text + minimal metadata)
    log.md                      append-only activity log
    digest.md                   current weekly digest (with closure markers)
    digests/YYYY-Www.md         archived prior digests (re-fed to Claude)
    _digest_prompt.md           Claude synthesis input
    _user_context.md            stable user background for Claude

%LOCALAPPDATA%\Dendr\           (PC-only, never synced)
  state.sqlite                  blocks + FTS5 + vector search
  queue/                        two-phase commit processing queue
  models/                       GGUF model weights
```

### Block storage

Each block is stored with:
- `text` — the raw Markdown, as written
- `source_date` — date from the daily note filename
- `checkbox_state` — `open` (`- [ ]`), `closed` (`- [x]`), or `none`
- `completion_status` — only set when the user closes a task via the digest review flow
- `private` — true if the privacy filter flagged the block

Claude reads raw text directly during digest synthesis; there is no pre-computed annotation layer.

### Task lifecycle

Tasks are identified by Markdown checkboxes. Checkbox transitions (open → closed) are logged as `task_events` with source='auto'. When you close a task through the digest review flow, `completion_status` is set and the event is logged with source='user'. User closures take precedence — re-parsing the source file doesn't clobber them.

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
- **qa.md** — on-demand Q&A grounded in your blocks
- **schema-review.md** — monthly review of the block schema

## License

MIT
