# Monthly Schema Review

Review the Dendr annotation schema and knowledge health for potential improvements.

## Your inputs
- `Wiki/schema.md` — current annotation spec
- `dendr stats` output (annotation count, open tasks)
- `dendr search` to sample annotations across time ranges

## Your tasks
1. Run `dendr stats` to get current health metrics
2. Review block annotation field usage patterns — are any fields consistently empty or
   under-used? (e.g., `causal_links`)
3. Check task lifecycle health: ratio of open vs completed vs abandoned tasks, average
   age of open tasks, closure rate trends
4. Identify any emerging categories that deserve their own block type or life area

## Rules
- Propose changes as diffs to `Wiki/schema.md` — don't rewrite from scratch
- Explain the rationale for each proposed change
- Keep the schema minimal — add structure only where it solves a real problem

## Vault path: {vault_path}
