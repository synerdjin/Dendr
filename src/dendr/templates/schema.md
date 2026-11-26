---
type: schema
version: "3.1"
---

# Dendr Block Store Schema

Dendr stores raw block text and minimal structural metadata. Claude reads the
raw text directly during digest synthesis and does classification, affect
reading, and clustering itself.

## Raw source vs. workspace

- `Daily/` is the **immutable source of truth** — your own daily notes. Dendr
  only ever appends a `^dendr-<ulid>` block ref; it never rewrites your prose.
- `Wiki/` is the **generated/editable workspace** — `digest.md`, archived
  `digests/`, `log.md`, and the synthesis prompt all live here. Edit these
  freely; the digest's closure markers and `Wiki/digests/*.md` round-trip back
  into Dendr on the next ingest.
- `Wiki/_user_context.md` is the **hand-seeded anchor** you curate by hand
  (role, life situation, active goals, stable constraints). It is injected into
  every synthesis so Claude stays grounded instead of inventing structure.

## Block Fields

- `block_id` — stable Obsidian block ref (`^dendr-<ulid>`)
- `source_file`, `source_date` — where and when the block was written
- `text` — raw Markdown, as written
- `checkbox_state` — `open` (`- [ ]`), `closed` (`- [x]`), or `none`
- `completion_status` — set only when the user closes a task via the digest
  review flow (`done` | `abandoned` | `snoozed` | `open`); `null` otherwise
