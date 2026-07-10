# Dendr

A personal knowledge compiler that watches Obsidian Daily Notes, stores each block as raw text in SQLite with FTS and vector search, and generates weekly digests. Claude (via Claude Code) reads the raw blocks directly at digest time and does classification, affect reading, and narrative synthesis in one pass.

**Local models do only what Claude can't or shouldn't**: embeddings for semantic search. **Claude (via Claude Code)** handles weekly synthesis and on-demand Q&A, so a Pro/Max subscription is enough.

## How it works

```
Daily Notes (you write here)
    ↓ watcher detects changes
    ↓ block-level parsing
    ↓ embed raw block text (EmbeddingGemma)
    ↓ commit to SQLite (blocks + FTS + vector index)
    ↓ log task_events on checkbox transitions
Wiki/digest.md (you read here, on any device)
```

## Requirements

Dendr targets Apple Silicon Macs — this is the only platform it's built and run against.

- macOS on Apple Silicon (embeddings run on the Metal GPU via `llama-cpp-python`)
- Python 3.11+
- Local model weights (see [Model Setup](#model-setup))
- Obsidian vault synced via iCloud (or any sync)

## Install

Runs natively — no containers. A dedicated venv (rather than the ambient
`pip install -e .` into whatever Python happens to be active) gives the
autostart agent a stable interpreter to pin, real macOS FSEvents for the file
watcher, and lets `llama-cpp-python` build against Metal so embeddings
actually use the GPU.

```bash
git clone https://github.com/synerdjin/Dendr.git
cd Dendr

python3 -m venv ~/.dendr-venv
~/.dendr-venv/bin/pip install -e .          # builds llama-cpp-python with Metal
~/.dendr-venv/bin/dendr init /path/to/vault
~/.dendr-venv/bin/dendr models pull && ~/.dendr-venv/bin/dendr models lock

# Run the daemon on every login (writes a launchd LaunchAgent):
~/.dendr-venv/bin/dendr autostart install
```

## Quick start

The commands below assume `~/.dendr-venv/bin` is on your `PATH` (or prefix
each with `~/.dendr-venv/bin/`):

```bash
# Search your knowledge base
dendr search "machine learning"

# Generate a weekly digest
dendr digest --claude

# Start the search HTTP server (for Claude Code integration)
dendr serve

# Run a single ingest cycle (the daemon from `autostart install` already does this on change)
dendr ingest
```

See [Regular tasks](#regular-tasks) below for the `Makefile` wrapping these
(and more) so you don't have to type the venv path each time.

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
| `dendr autostart install` | Run the daemon on login via a macOS LaunchAgent |
| `dendr autostart status` | Show whether the login agent is installed / loaded |
| `dendr autostart uninstall` | Stop and remove the login agent |

## Model setup

Models are declared in `dendr-models.yaml` (version-controlled). Download them all with one command:

```bash
dendr models pull
dendr models verify
dendr models lock    # pin hashes for reproducibility
```

| Role | Model | Size | Purpose |
|------|-------|------|---------|
| Embeddings | [embeddinggemma-300m QAT Q8_0](https://huggingface.co/ggml-org/embeddinggemma-300m-qat-q8_0-GGUF) | ~0.3 GB | 768d Matryoshka embeddings for semantic search |

Models are stored in `~/.local/share/dendr/models/`. Only one model is loaded in VRAM at a time.

## Regular tasks

A `Makefile` wraps the commands above (and `scripts/update.sh`) around the
`~/.dendr-venv` venv, so day-to-day operation doesn't require typing the venv path
each time:

```bash
make help          # list every target
make update        # git pull + refresh deps + verify models + restart the daemon
make ingest         # one ingest cycle
make digest         # weekly digest + Claude synthesis prompt
make stats          # knowledge base statistics
make serve          # search server on :7777
make check          # lint + format-check + test — same as CI, run before pushing
```

Override the venv location with `make DENDR_VENV=~/other-venv <target>` if you're not
using the default.

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
    _intentions.md              per-period stated intentions (drift lens)

~/.local/share/dendr/           (local only, never synced)
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

Claude reads raw text directly during digest synthesis; there is no pre-computed annotation layer.

### Task lifecycle

Tasks are identified by Markdown checkboxes. Checkbox transitions (open → closed) are logged as `task_events` with source='auto'. When you close a task through the digest review flow, `completion_status` is set and the event is logged with source='user'. User closures take precedence — re-parsing the source file doesn't clobber them.

## Structured logging

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
