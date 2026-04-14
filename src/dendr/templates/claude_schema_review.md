# Monthly Schema Review

Review the Dendr wiki schema and knowledge health for potential improvements.

## Your inputs
- `Wiki/schema.md` — current wiki spec
- Latest lint report in `Wiki/_lint/` (orphan pages, missing cross-refs)
- `dendr stats` output (annotation count, concept count, open tasks)
- Sample concept and entity pages from `Wiki/concepts/` and `Wiki/entities/`

## Your tasks
1. Run `dendr lint` and `dendr stats` to get current health metrics
2. Review the latest lint report — are the same orphan pages or broken links recurring?
3. Check concept canonicalization quality: look for near-duplicate concept slugs that
   should have been merged (embedding similarity threshold is 0.86)
4. Review block annotation field usage patterns — are any fields consistently empty or
   under-used? (e.g., `causal_links`)
5. Check task lifecycle health: ratio of open vs completed vs abandoned tasks, average
   age of open tasks, closure rate trends
6. Identify any emerging categories that deserve their own page type or life area

## Rules
- Propose changes as diffs to `Wiki/schema.md` — don't rewrite from scratch
- Explain the rationale for each proposed change
- Flag any concept slugs that should be merged
- Keep the schema minimal — add structure only where it solves a real problem
- NEVER modify the human-zone of any wiki page

## Vault path: {vault_path}
