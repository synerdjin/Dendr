# Digest Synthesis

Dendr's weekly digest is assembled by `dendr digest --claude`, which writes a
self-contained prompt to `Wiki/_digest_prompt.md`. Your job is to execute that
prompt and write the result to `Wiki/digest.md`.

## Workflow
1. Run `dendr digest --claude` if `Wiki/_digest_prompt.md` is missing or stale.
2. Read `Wiki/_digest_prompt.md` — it contains full instructions, the user's
   context, the period-scoped data, and carried-forward items.
3. Follow the instructions in that prompt verbatim. Do not improvise framing.
4. Write the synthesis output to `Wiki/digest.md`, preserving the closure
   markers (`<!-- closure:... -->`) so they round-trip on next ingest.
