# TODO

## LLM vs Hardcoded Logic — Should We Replace More Pipeline Code with LLM Calls?

**Context:** Currently the pipeline uses hardcoded logic for parsing, privacy, queue, dedup, canonicalization, and wiki templating. LLMs handle claim extraction, tagging, wiki prose, and embeddings.

### Good candidates for LLM replacement
- **Block splitting** — LLM could understand semantic boundaries better than `- ` regex (e.g., split a paragraph about two unrelated topics). Tradeoff: slow and expensive for a step that runs on every file change.
- **Concept canonicalization** — LLM could reason about whether "machine-learning" and "ML" are the same concept with more nuance than a fixed 0.86 embedding threshold.
- **Contradiction detection** — currently checks same subject-predicate with different object. LLM could catch semantic contradictions across different phrasings.

### Keep as hardcoded logic
- **Queue/state management** — infrastructure, not intelligence
- **Privacy filtering** — must be deterministic and auditable
- **File I/O, hashing, dedup** — mechanical operations, faster and correct by construction
- **Wiki page structure** — templating is better done deterministically

### Recommendation
The current split is roughly right. Highest-value upgrade path: improve LLM prompts (already tightened) or fine-tune models using the `ft-pairs.jsonl` training data we're collecting — not replace code with more LLM passes. At ~1 min/block, adding more LLM steps would multiply an already long pipeline.
