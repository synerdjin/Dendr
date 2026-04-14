---
type: schema
version: "2.0"
---

# Dendr Annotation Schema

This document defines how the Dendr system annotates and stores knowledge.
Both the local LLM and Claude read this on every session.

## Block Annotation Format

Every block from a daily note is annotated with:
- `gist`: one-line summary of what the block is about
- `block_type`: reflection, task, decision, question, observation, vent, plan, log_entry
- `life_areas`: which domains the block touches (work, health, etc.)
- `emotional_valence`, `intensity`: emotional signals
- `urgency`, `importance`: only for tasks/plans, reflect state at source_date
- `completion_status`: open, done, blocked, abandoned — updated via digest closure flow
- `concepts`, `entities`: tag slugs assigned by the tagger for topic aggregation

## Conventions

- Slugs: lowercase, hyphens, no spaces (`machine-learning`, not `Machine Learning`)
- Cross-references: use `[[slug]]` Obsidian wikilinks in daily notes

## Privacy

- Blocks tagged `#dendr-private`, `#private`, or `#redact` are never sent to Claude
- Blocks matching secret patterns (API keys, passwords) are auto-tagged private
- Private annotations are stored locally for search but excluded from Claude payloads
