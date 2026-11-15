# Digest-era prompts (archived 2026-11-15)

Snapshot of the prompt templates from `src/dendr/templates/` as of commit
`cfce44c`, kept for history before the weekly digest is retired
in favor of conversational access to the vault.

| File | What it was |
|---|---|
| `synthesis_prompt.md` | Main weekly-digest synthesis prompt (retrospective coach, anti-sycophancy, intention-vs-attention drift, garden section). Filled by `digest.py` and written to `Wiki/_digest_prompt.md`. |
| `claude_digest.md` | Workflow instructions for running the digest prompt in Claude Code (installed into the vault's `.claude/` as `digest.md`). |
| `claude_qa.md` | Q&A session prompt grounded in search results from `dendr serve` (installed as `qa.md`). |
| `claude_schema_review.md` | Schema review prompt (installed as `schema-review.md`). |
| `schema.md` | Block-store schema doc written to `Wiki/schema.md` by `dendr init`. |

These are frozen copies. Nothing in the code reads from this folder.
