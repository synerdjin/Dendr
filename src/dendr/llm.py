"""LLM client abstraction for local inference via llama-cpp-python.

Manages model lifecycle, provides typed completion methods,
and logs all calls to ft-pairs.jsonl for future fine-tuning.
"""

from __future__ import annotations

import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from dendr.config import Config
from dendr.metrics import (
    INFERENCE_JSON_FAILURES,
    INFERENCE_SECONDS,
    INFERENCE_TOKENS,
    MODEL_LOAD_SECONDS,
    MODEL_LOADED,
)

logger = logging.getLogger(__name__)

# Lazy-loaded model instances
_models: dict[str, Any] = {}


def _model_role_from_path(model_path: Path) -> str:
    """Derive a short role label from a model filename for metrics."""
    name = model_path.stem.lower()
    if "phi" in name:
        return "enrichment"
    if "gemma" in name:
        return "tagger"
    if "llama" in name and "vision" in name:
        return "vision"
    if "nomic" in name or "embed" in name:
        return "embedding"
    return "unknown"


def _unload_all_except(keep_key: str | None = None) -> None:
    """Unload all models except the one with the given key to free VRAM."""
    keys_to_remove = [k for k in _models if k != keep_key]
    for key in keys_to_remove:
        role = _model_role_from_path(Path(key))
        MODEL_LOADED.labels(model_role=role).set(0)
        del _models[key]
    if keys_to_remove:
        import gc

        gc.collect()


def _get_model(
    model_path: Path, n_ctx: int = 4096, n_gpu_layers: int = -1, embedding: bool = False
) -> Any:
    """Get or create a llama-cpp-python model instance.

    Only one model is kept in VRAM at a time to fit within GPU memory.
    """
    from llama_cpp import Llama

    key = str(model_path)
    if key not in _models:
        # Unload other models to free VRAM before loading a new one
        _unload_all_except(None)
        role = _model_role_from_path(model_path)
        logger.info("Loading model: %s (ctx=%d)", model_path.name, n_ctx)
        t0 = time.monotonic()
        _models[key] = Llama(
            model_path=str(model_path),
            n_ctx=n_ctx,
            n_gpu_layers=n_gpu_layers,
            verbose=False,
            embedding=embedding,
        )
        MODEL_LOAD_SECONDS.labels(model_role=role).observe(time.monotonic() - t0)
        MODEL_LOADED.labels(model_role=role).set(1)
    return _models[key]


def unload_all() -> None:
    """Release all loaded models."""
    for key in _models:
        role = _model_role_from_path(Path(key))
        MODEL_LOADED.labels(model_role=role).set(0)
    _models.clear()


def _log_ft_pair(
    config: Config, prompt: str, response: str, model: str, task: str
) -> None:
    """Append a (prompt, response) pair for future fine-tuning."""
    entry = {
        "prompt": prompt,
        "response": response,
        "model_version": model,
        "prompt_version": "v1",
        "task": task,
        "ts": datetime.now().isoformat(),
    }
    with open(config.ft_pairs_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")


class LLMClient:
    """Unified interface to local LLM models."""

    ENRICHMENT_PROMPT_VERSION = "v2"
    ANNOTATION_PROMPT_VERSION = "v1"

    def __init__(self, config: Config, skip_preflight: bool = False):
        self.config = config
        if not skip_preflight:
            self._preflight()

    def _preflight(self) -> None:
        """Check that required model files exist. Fail fast with actionable message."""
        try:
            from dendr.model_manager import ModelManifest, preflight_check

            manifest_path = self.config.manifest_path
            if manifest_path.exists():
                manifest = ModelManifest.load(manifest_path)
                errors = preflight_check(self.config.models_dir, manifest)
                if errors:
                    msg = "Model preflight check failed:\n" + "\n".join(
                        f"  - {e}" for e in errors
                    )
                    raise RuntimeError(msg)
        except FileNotFoundError:
            # No manifest — check files directly by config filenames
            missing = []
            for name in [
                self.config.models.enrichment_model,
                self.config.models.tagger_model,
                self.config.models.embedding_model,
            ]:
                if not (self.config.models_dir / name).exists():
                    missing.append(name)
            if missing:
                msg = (
                    "Missing model files:\n"
                    + "\n".join(f"  - {m}" for m in missing)
                    + f"\n\nPlace them in: {self.config.models_dir}"
                    + "\nOr run: dendr models pull"
                )
                raise RuntimeError(msg)

    def _model_path(self, filename: str) -> Path:
        return self.config.models_dir / filename

    def _enrichment_model(self) -> Any:
        return _get_model(
            self._model_path(self.config.models.enrichment_model),
            n_ctx=self.config.models.enrichment_ctx,
        )

    def _tagger_model(self) -> Any:
        return _get_model(
            self._model_path(self.config.models.tagger_model),
            n_ctx=self.config.models.tagger_ctx,
        )

    def _embedding_model(self) -> Any:
        return _get_model(
            self._model_path(self.config.models.embedding_model),
            n_ctx=512,
            embedding=True,
        )

    def annotate_block(self, block_text: str) -> dict:
        """Rich annotation of a block — the primary extraction step.

        Uses the tagger model (fast) to classify block type, emotional signals,
        life areas, urgency/importance, and extract a one-line gist.

        Returns a dict matching BlockAnnotation fields.
        """
        prompt = f"""Annotate this personal daily note block with structured metadata.

TEXT BLOCK:
{block_text}

Return ONLY valid JSON:
{{
  "gist": "one-line summary of what this block is about",
  "block_type": "reflection|task|decision|question|observation|vent|plan|log_entry",
  "life_areas": ["work", "health", "relationships", "finance", "learning", "creative", "meta"],
  "emotional_valence": -1.0 to 1.0,
  "emotional_labels": ["frustrated", "excited", "anxious", "relieved", "burned_out", "curious", "conflicted", "satisfied", "overwhelmed"],
  "intensity": 0.0 to 1.0,
  "urgency": "today|this_week|someday|null",
  "importance": "high|medium|low|null",
  "completion_status": "open|done|blocked|abandoned|null",
  "epistemic_status": "certain|likely|exploring|questioning|venting",
  "causal_links": ["cause -> effect"],
  "concepts": ["concept-slug"],
  "entities": ["entity name"]
}}

Rules:
- gist: one sentence, neutral tone, captures the core meaning.
- block_type: classify the primary purpose of this block.
- life_areas: which domains does this touch? Include ALL that apply. Leave empty if unclear.
- emotional_valence: -1.0 = very negative, 0.0 = neutral, 1.0 = very positive.
- emotional_labels: only include labels that clearly apply. Empty array if neutral.
- intensity: 0.0 = passing mention, 1.0 = this is a central concern right now.
- urgency/importance: null if not applicable (e.g. a reflection has no urgency).
- completion_status: only for tasks/plans. null for reflections/observations.
- causal_links: extract "X -> Y" relationships if the text states or implies causality.
- concepts: lowercase slugs with hyphens (e.g. "machine-learning").
- entities: proper names of people, projects, tools, organizations.
"""

        model = self._tagger_model()
        t0 = time.monotonic()
        response = model.create_chat_completion(
            messages=[
                {
                    "role": "system",
                    "content": "You are a personal note analyst. Output ONLY valid JSON.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
            max_tokens=1024,
            response_format={"type": "json_object"},
        )
        INFERENCE_SECONDS.labels(model_role="tagger", task="annotate").observe(
            time.monotonic() - t0
        )
        usage = response.get("usage", {})
        if usage:
            INFERENCE_TOKENS.labels(model_role="tagger", direction="prompt").inc(
                usage.get("prompt_tokens", 0)
            )
            INFERENCE_TOKENS.labels(model_role="tagger", direction="completion").inc(
                usage.get("completion_tokens", 0)
            )

        raw = response["choices"][0]["message"]["content"]
        _log_ft_pair(
            self.config, prompt, raw, self.config.models.tagger_model, "annotate"
        )

        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            INFERENCE_JSON_FAILURES.labels(task="annotate").inc()
            logger.warning("Failed to parse annotation JSON, returning defaults")
            return {
                "gist": "",
                "block_type": "observation",
                "life_areas": [],
                "emotional_valence": 0.0,
                "emotional_labels": [],
                "intensity": 0.5,
                "urgency": None,
                "importance": None,
                "completion_status": None,
                "epistemic_status": "certain",
                "causal_links": [],
                "concepts": [],
                "entities": [],
            }

    def enrich_block(self, block_text: str, existing_concepts: list[str]) -> dict:
        """Extract atomic claims from a text block (simplified — no SPO).

        Returns a dict with keys: claims, concepts, entities, related_slugs.
        Each claim has: text, confidence, kind.
        """
        concept_list = (
            ", ".join(existing_concepts[:50]) if existing_concepts else "none yet"
        )

        prompt = f"""Extract atomic claims from the following text block.

Existing concepts in the knowledge base: [{concept_list}]

TEXT BLOCK:
{block_text}

Return ONLY valid JSON:
{{
  "claims": [
    {{
      "text": "one atomic factual statement, task, or intention",
      "confidence": 0.0 to 1.0,
      "kind": "statement|task|intention|question|belief"
    }}
  ],
  "concepts": ["concept-slug-1", "concept-slug-2"],
  "entities": ["entity name 1"],
  "related_slugs": ["existing-concept-slug"]
}}

Rules:
- Each claim must be a single atomic statement in natural language.
- Confidence: 1.0 = stated as fact, 0.5 = implied, 0.3 = speculative/hedged.
- Kind: "statement" for facts, "task" for action items, "intention" for goals/plans, "question" for open questions, "belief" for opinions.
- Concept slugs: lowercase, hyphens (e.g. "machine-learning"). Reuse existing slugs.
- If the text has no extractable claims, return empty arrays.
"""

        model = self._enrichment_model()
        t0 = time.monotonic()
        response = model.create_chat_completion(
            messages=[
                {
                    "role": "system",
                    "content": "You are a precise knowledge extraction system. Output ONLY valid JSON.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.1,
            max_tokens=2048,
            response_format={"type": "json_object"},
        )
        INFERENCE_SECONDS.labels(model_role="enrichment", task="enrich").observe(
            time.monotonic() - t0
        )
        usage = response.get("usage", {})
        if usage:
            INFERENCE_TOKENS.labels(model_role="enrichment", direction="prompt").inc(
                usage.get("prompt_tokens", 0)
            )
            INFERENCE_TOKENS.labels(
                model_role="enrichment", direction="completion"
            ).inc(usage.get("completion_tokens", 0))

        raw = response["choices"][0]["message"]["content"]
        _log_ft_pair(
            self.config, prompt, raw, self.config.models.enrichment_model, "enrich"
        )

        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            INFERENCE_JSON_FAILURES.labels(task="enrich").inc()
            logger.warning("Failed to parse enrichment JSON, returning empty result")
            return {"claims": [], "concepts": [], "entities": [], "related_slugs": []}

    def tag_block(self, block_text: str) -> dict:
        """Quick tagging pass using the smaller/faster model.

        Returns: {"concepts": [...], "entities": [...], "is_task": bool}
        """
        prompt = f"""Tag this text block with concepts and entities.

TEXT: {block_text}

Return ONLY valid JSON:
{{"concepts": ["slug-1"], "entities": ["Name"], "is_task": false}}
"""
        model = self._tagger_model()
        t0 = time.monotonic()
        response = model.create_chat_completion(
            messages=[
                {"role": "system", "content": "Output ONLY valid JSON."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
            max_tokens=512,
            response_format={"type": "json_object"},
        )
        INFERENCE_SECONDS.labels(model_role="tagger", task="tag").observe(
            time.monotonic() - t0
        )
        usage = response.get("usage", {})
        if usage:
            INFERENCE_TOKENS.labels(model_role="tagger", direction="prompt").inc(
                usage.get("prompt_tokens", 0)
            )
            INFERENCE_TOKENS.labels(model_role="tagger", direction="completion").inc(
                usage.get("completion_tokens", 0)
            )

        raw = response["choices"][0]["message"]["content"]
        _log_ft_pair(self.config, prompt, raw, self.config.models.tagger_model, "tag")

        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            INFERENCE_JSON_FAILURES.labels(task="tag").inc()
            return {"concepts": [], "entities": [], "is_task": False}

    def embed(self, text: str) -> np.ndarray:
        """Generate an embedding vector for text."""
        model = self._embedding_model()
        t0 = time.monotonic()
        result = model.embed(text)
        INFERENCE_SECONDS.labels(model_role="embedding", task="embed").observe(
            time.monotonic() - t0
        )
        # llama-cpp-python returns list or list-of-lists
        if isinstance(result[0], list):
            vec = np.array(result[0], dtype=np.float32)
        else:
            vec = np.array(result, dtype=np.float32)
        return vec

    def embed_batch(self, texts: list[str]) -> list[np.ndarray]:
        """Embed multiple texts efficiently."""
        model = self._embedding_model()
        t0 = time.monotonic()
        results = model.embed(texts)
        INFERENCE_SECONDS.labels(model_role="embedding", task="embed_batch").observe(
            time.monotonic() - t0
        )
        out = []
        for r in results:
            if isinstance(r, list):
                out.append(np.array(r, dtype=np.float32))
            else:
                out.append(np.array([r], dtype=np.float32))
        return out

    def generate_wiki_section(
        self, concept_title: str, new_evidence: list[str], existing_content: str
    ) -> str:
        """Generate a new evidence section for a wiki page."""
        evidence_text = "\n".join(f"- {e}" for e in new_evidence)

        # Truncate existing content to avoid blowing the context window.
        # We only need recent sections for de-dup context — the prompt tells
        # the model to summarize ONLY the new evidence, not the history.
        # ~4000 chars ≈ 1000 tokens, leaving headroom under the 8192 ctx.
        MAX_EXISTING_CHARS = 4000
        truncated = existing_content or ""
        if len(truncated) > MAX_EXISTING_CHARS:
            tail = truncated[-MAX_EXISTING_CHARS:]
            # Snap to the next section boundary so we don't start mid-sentence
            boundary = tail.find("### Evidence")
            if boundary != -1:
                tail = tail[boundary:]
            truncated = "(… earlier sections truncated …)\n\n" + tail

        prompt = f"""You are maintaining a knowledge base wiki page for "{concept_title}".

EXISTING PAGE CONTENT (LLM zone only, most recent sections):
{truncated or "(empty — this is a new page)"}

NEW EVIDENCE to integrate:
{evidence_text}

Summarize ONLY what the evidence above actually says. Rules:
- Do NOT invent, infer, or add any information not present in the evidence
- Do NOT speculate about implications, trends, or future developments
- Stick strictly to the facts stated in the evidence
- Use inline confidence pills like [c:0.82] after factual claims
- Use [[concept-slug]] Obsidian links for cross-references when relevant
- Write plain markdown — no code fences, no ```markdown``` blocks
- Be concise: 2-4 sentences maximum

Output ONLY the section content (no page frontmatter, no heading).
"""
        model = self._enrichment_model()
        t0 = time.monotonic()
        response = model.create_chat_completion(
            messages=[
                {
                    "role": "system",
                    "content": "You are a wiki maintainer. Summarize ONLY what the evidence says. Never add information beyond what is provided. Write plain markdown, never wrap output in code fences.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.1,
            max_tokens=1024,
        )
        INFERENCE_SECONDS.labels(model_role="enrichment", task="wiki_section").observe(
            time.monotonic() - t0
        )
        usage = response.get("usage", {})
        if usage:
            INFERENCE_TOKENS.labels(model_role="enrichment", direction="prompt").inc(
                usage.get("prompt_tokens", 0)
            )
            INFERENCE_TOKENS.labels(
                model_role="enrichment", direction="completion"
            ).inc(usage.get("completion_tokens", 0))
        raw = response["choices"][0]["message"]["content"]
        _log_ft_pair(
            self.config,
            prompt,
            raw,
            self.config.models.enrichment_model,
            "wiki_section",
        )
        return raw.strip()

    def extract_text_from_image(self, image_path: str) -> str:
        """OCR / caption an image using the VLM.

        Falls back to empty string if VLM is not available.
        """
        try:
            import base64

            with open(image_path, "rb") as f:
                img_b64 = base64.b64encode(f.read()).decode()

            model = _get_model(
                self._model_path(self.config.models.vlm_model),
                n_ctx=self.config.models.vlm_ctx,
            )
            t0 = time.monotonic()
            response = model.create_chat_completion(
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": "Describe this image in detail. If it contains text, transcribe all visible text.",
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{img_b64}"
                                },
                            },
                        ],
                    }
                ],
                temperature=0.1,
                max_tokens=2048,
            )
            INFERENCE_SECONDS.labels(model_role="vision", task="ocr").observe(
                time.monotonic() - t0
            )
            return response["choices"][0]["message"]["content"].strip()
        except Exception as e:
            logger.warning("VLM extraction failed for %s: %s", image_path, e)
            return ""

    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extract text from a PDF. Uses PyMuPDF; falls back to VLM for scans."""
        import fitz  # pymupdf

        doc = fitz.open(pdf_path)
        text_parts: list[str] = []
        for page in doc:
            page_text = page.get_text().strip()
            if page_text:
                text_parts.append(page_text)
        doc.close()

        full_text = "\n\n".join(text_parts)
        if len(full_text) < 50 and Path(pdf_path).stat().st_size > 10000:
            # Likely a scanned PDF — try VLM on first page image
            logger.info("PDF appears scanned, attempting VLM OCR: %s", pdf_path)
            doc = fitz.open(pdf_path)
            if doc.page_count > 0:
                page = doc[0]
                pix = page.get_pixmap(dpi=200)
                img_path = str(Path(pdf_path).with_suffix(".tmp.png"))
                pix.save(img_path)
                doc.close()
                text = self.extract_text_from_image(img_path)
                Path(img_path).unlink(missing_ok=True)
                return text
            doc.close()

        return full_text
