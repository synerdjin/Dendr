# Monthly Schema Review

Review the Dendr block schema and knowledge health for potential improvements.

## Your inputs
- `Wiki/schema.md` — current block schema
- `dendr stats` output (block count, open tasks)
- `dendr search` to sample blocks across time ranges

## Your tasks
1. Run `dendr stats` to get current health metrics
2. Sample blocks across recent time ranges and check whether the raw-text storage
   is serving digest synthesis well — are there fields you wish you had that
   the user could be writing down in a structured way?
3. Check task lifecycle health: ratio of open vs closed tasks, average age of
   open tasks, closure rate trends. Query `task_events` directly via sqlite3 if
   needed.
4. Identify any repeated themes the user might benefit from capturing more
   deliberately in `_user_context.md`.

## Rules
- Propose changes as diffs to `Wiki/schema.md` — don't rewrite from scratch
- Explain the rationale for each proposed change
- Keep the schema minimal — add structure only where it solves a real problem
  in the digest synthesis flow

## Vault path: {vault_path}
