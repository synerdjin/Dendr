---
type: schema
version: "1.0"
---

# Dendr Wiki Schema

This document defines how the Dendr system maintains the knowledge wiki.
Both the local LLM and Claude read this on every session.

## Page Types

### Concept Page (`concepts/<slug>.md`)
- Tracks a single concept, idea, or topic
- Frontmatter: type, slug, human_touched, last_llm_hash, created, updated
- Structure: human-zone (user edits) + llm-zone (system-managed evidence)

### Entity Page (`entities/<slug>.md`)
- Tracks a named entity (person, project, tool, organization)
- Same structure as concept pages

## Block Annotation Format

Every block from a daily note is annotated with:
- `gist`: one-line summary of what the block is about
- `block_type`: reflection, task, decision, question, observation, vent, plan, log_entry
- `life_areas`: which domains the block touches (work, health, etc.)
- `emotional_valence`, `emotional_labels`, `intensity`: emotional signals
- `urgency`, `importance`: only for tasks/plans, reflect state at source_date
- `completion_status`: open, done, blocked, abandoned — updated via digest closure flow
- `concepts`, `entities`: canonicalized slugs for cross-referencing

## Conventions

- Slugs: lowercase, hyphens, no spaces (`machine-learning`, not `Machine Learning`)
- Cross-references: use `[[slug]]` Obsidian wikilinks
- LLM zone: everything between `<!-- llm-zone -->` markers is system-managed
- Human zone: everything between `<!-- human-zone -->` markers is sacred

## Lint Rules

- Orphan pages: concept/entity pages referenced by zero annotations
- Missing cross-refs: `[[slug]]` links pointing to non-existent pages

## Privacy

- Blocks tagged `#dendr-private`, `#private`, or `#redact` are never sent to Claude
- Blocks matching secret patterns (API keys, passwords) are auto-tagged private
- Private annotations are stored locally for search but excluded from Claude payloads
