# TODO

## Insight & Digest Enhancements (from review/ brainstorm)

### 1. Attention Flow Table in Digest
Track how much attention goes to each life area over a time window and surface the balance/imbalance in digests as a quantified table. Data already exists in `life_areas` from block_annotations — needs aggregation, week-over-week comparison, and trend indicators. Directly implements the Tim Urban "octopus tentacles" metaphor: which areas are getting fed and which are starving.

Target format for the digest:

```
| Life Area          | This Week | Last Week | Trend |
|--------------------|-----------|-----------|-------|
| Mental Health      | ████████  | ██████    | ↑     |
| Career & Work      | ██████    | ████████  | ↓     |
| Family             | ████      | ████      | →     |
| Fitness            | ██        | ██████    | ↓↓    |
```

### 2. Mood/Energy Time Series
Add explicit `energy_level` (numeric) and `mood` (categorical) as first-class fields in tagger output, beyond the current `emotional_valence`. Plot these as time series in digests to show trajectories, not just prose summaries.

### 3. Cross-Domain Correlation Detection
Compute correlations between life areas in the digest (e.g., substance use vs. focus, sleep vs. productivity, fitness vs. energy). Currently digest does per-area pattern summaries but doesn't explicitly surface cross-domain relationships.

### 4. "What HAS Worked" Tracking
Mine positive reinforcement patterns from the journal — things the user tried that actually helped, with evidence. Distinct from contradiction/dropped-thread detection. Surface these in digests as "proven strategies" with date references.

### 5. Domain-Specific Semantic Retrieval Queries
Curate per-domain query sets (mental health, substances, career, family, etc.) to seed smarter digest retrieval instead of relying only on recency or generic topic matching. See `review/daily-index-prompts.md` Phase 4 for examples.

### 6. Substances as a First-Class Domain
Elevate substance tracking from a generic life_area tag to its own explicitly tracked domain with dedicated queries, trend detection, and correlation analysis against other areas (focus, mood, energy).

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


Change logseq data from bullet points to paragraphs unles it is indented bullet points list.
