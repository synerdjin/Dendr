# Dendr

A personal knowledge compiler that turns Obsidian Daily Notes into a persistent, interlinked, confidence-scored knowledge base — without you ever doing bookkeeping.

Inspired by [Karpathy's LLM-wiki pattern](https://gist.github.com/karpathy/442a6bf555914893e9891c11519de94f): raw sources flow in, an LLM maintains a structured wiki of concepts, entities, and claims with cross-references, contradiction detection, and temporal metadata.

## How it works

```
Daily Notes (you write here)
    ↓ watcher detects changes
    ↓ block-level parsing + privacy filter
    ↓ local LLM extracts claims, concepts, entities (SPO triples)
    ↓ embedding-based concept canonicalization
    ↓ wiki pages created/updated (LLM-zone rule protects your edits)
    ↓ claims stored with {confidence, source, date, status}
Wiki/ (you read here, on any device)
```

**Local LLM handles all mechanical work** — claim extraction, tagging, embeddings, wiki page updates, nightly lint. **Claude (via Claude Code)** is reserved for weekly synthesis and on-demand Q&A, so a Pro/Max subscription is enough.

## Requirements

- Python 3.11+
- GPU with 12GB+ VRAM (tested on RTX 4070)
- Local model weights (see [Model Setup](#model-setup))
- Obsidian vault synced via iCloud (or any sync)

## Install

```bash
git clone https://github.com/YOUR_USERNAME/Dendr.git
cd Dendr
pip install -e .
```

## Quick start

```bash
# 1. Initialize your vault
dendr init /path/to/your/obsidian/vault

# 2. Download models (see Model Setup below)

# 3. Run the daemon (watches Daily/ for changes)
dendr daemon

# 4. Or run a single ingest cycle
dendr ingest

# 5. Search your knowledge base
dendr search "machine learning"

# 6. Start the search HTTP server (for Claude Code integration)
dendr serve
```

## CLI commands

| Command | Description |
|---------|-------------|
| `dendr init <vault>` | Initialize vault structure + database + Claude prompts |
| `dendr daemon` | Watch `Daily/` and auto-ingest on changes |
| `dendr ingest` | Run a single ingest cycle |
| `dendr search <query>` | Search claims (FTS, semantic, or hybrid) |
| `dendr lint` | Run health checks (contradictions, stale claims, orphans) |
| `dendr serve` | Start search server on `localhost:7777` |
| `dendr stats` | Show knowledge base statistics |

## Model setup

Download these weights into `%LOCALAPPDATA%\Dendr\models\` (Windows) or `~/.local/share/dendr/models/` (macOS/Linux):

| Role | Model | Size |
|------|-------|------|
| Enrichment | [Phi-4 14B Q4_K_M](https://huggingface.co/microsoft/phi-4-gguf) | ~9 GB |
| Fast tagger | [Gemma 3 4B Instruct Q4_K_M](https://huggingface.co/google/gemma-3-4b-it-gguf) | ~3 GB |
| Vision/OCR | [Llama 3.2 Vision 11B Q4_K_M](https://huggingface.co/meta-llama/Llama-3.2-11B-Vision-Instruct-GGUF) | ~7 GB |
| Embeddings | [nomic-embed-text-v1.5 Q8_0](https://huggingface.co/nomic-ai/nomic-embed-text-v1.5-GGUF) | ~0.3 GB |

## Architecture

### Storage layout

```
Vault/                          (iCloud-synced)
  Daily/YYYY-MM-DD.md           your raw notes (never modified by Dendr)
  Wiki/
    schema.md                   the spec that governs the wiki
    index.md                    auto-generated catalog
    log.md                      append-only activity log
    concepts/<slug>.md          LLM-maintained concept pages
    entities/<slug>.md          LLM-maintained entity pages
    summaries/weekly-*.md       Claude-produced weekly syntheses

%LOCALAPPDATA%\Dendr\           (PC-only, never synced)
  state.sqlite                  claims + FTS5 + vector search
  queue/                        two-phase commit processing queue
  ft-pairs.jsonl                fine-tuning corpus (logged automatically)
```

### Claim lifecycle

Every factual statement extracted from your notes becomes a **claim** with:
- SPO triple (`subject`, `predicate`, `object`)
- `confidence` score (0.0–1.0)
- `status`: created → reinforced → challenged → superseded
- Source provenance back to the exact daily note block

Contradictions are detected automatically when two non-superseded claims share the same `subject|predicate` but differ in `object`.

### The LLM-zone rule

Wiki pages have two zones: a **human-zone** (your edits, never touched) and an **llm-zone** (system-managed). If you edit a page, Dendr detects the drift and switches to append-only mode for that page.

### Privacy

Blocks containing API keys, passwords, SSNs, or tagged with `#dendr-private` are stored locally but **never sent to Claude**. You can also use `#private` or `#redact` tags.

## Claude Code integration

`dendr init` generates three Claude Code session prompts in `.claude/`:

- **weekly.md** — weekly synthesis (resolve contradictions, update digest)
- **qa.md** — on-demand Q&A grounded in your wiki
- **schema-review.md** — monthly schema evolution review

Token budget: ~30k input per weekly session, ~20k per Q&A. Fits comfortably in a Max subscription.

## Future

- **Voice transcription** (whisper.cpp) — deferred to v2
- **Fine-tuning** — all LLM calls are logged as training pairs in `ft-pairs.jsonl`, ready for QLoRA on the local model
- **Multi-vault** support via `vault_id` binding

## License

MIT
