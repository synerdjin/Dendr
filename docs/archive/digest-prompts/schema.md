---
type: schema
version: "3.1"
---

# Dendr Block Store Schema

Dendr stores raw block text and minimal structural metadata. Claude reads the
raw text directly during digest synthesis and does classification, affect
reading, and clustering itself.

## Block Fields

- `block_id` — stable Obsidian block ref (`^dendr-<ulid>`)
- `source_file`, `source_date` — where and when the block was written
- `text` — raw Markdown, as written
- `checkbox_state` — `open` (`- [ ]`), `closed` (`- [x]`), or `none`
- `completion_status` — set only when the user closes a task via the digest
  review flow (`done` | `abandoned` | `snoozed` | `open`); `null` otherwise
