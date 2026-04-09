"""LLM client abstraction for local inference via llama-cpp-python.

Manages model lifecycle, provides typed completion methods,
and logs all calls to ft-pairs.jsonl for future fine-tuning.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from dendr.config import Config

logger = logging.getLogger(__name__)

# Lazy-loaded model instances
_models: dict[str, Any] = {}


def _get_model(model_path: Path, n_ctx: int = 4096, n_gpu_layers: int = -1, embedding: bool = False) -> Any:
    """Get or create a llama-cpp-python model instance."""
    from llama_cpp import Llama

    key = str(model_path)
    if key not in _models:
        logger.info("Loading model: %s (ctx=%d)", model_path.name, n_ctx)
        _models[key] = Llama(
            model_path=str(model_path),
            n_ctx=n_ctx,
            n_gpu_layers=n_gpu_layers,
            verbose=False,
            embedding=embedding,
        )
    return _models[key]


def unload_all() -> None:
    """Release all loaded models."""
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

    ENRICHMENT_PROMPT_VERSION = "v1"

    def __init__(self, config: Config):
        self.config = config

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

    def enrich_block(self, block_text: str, existing_concepts: list[str]) -> dict:
        """Extract claims, concepts, entities from a text block.

        Returns a dict with keys: claims, concepts, entities, related_slugs.
        Each claim has: text, subject, predicate, object, confidence.
        """
        concept_list = ", ".join(existing_concepts[:50]) if existing_concepts else "none yet"

        prompt = f"""Extract structured knowledge from the following text block.

Existing concepts in the knowledge base: [{concept_list}]

TEXT BLOCK:
{block_text}

Return ONLY valid JSON with this exact schema:
{{
  "claims": [
    {{
      "text": "atomic factual statement",
      "subject": "the subject entity",
      "predicate": "the relationship or property",
      "object": "the value or target",
      "confidence": 0.0 to 1.0
    }}
  ],
  "concepts": ["concept-slug-1", "concept-slug-2"],
  "entities": ["entity name 1", "entity name 2"],
  "related_slugs": ["existing-concept-slug that relates"]
}}

Rules:
- Claims must be atomic (one fact each), in SPO form.
- Confidence: 1.0 = stated as fact, 0.5 = implied, 0.3 = speculative/hedged.
- Concept slugs: lowercase, hyphens, no spaces (e.g. "machine-learning").
- Reuse existing concept slugs when the text refers to an existing concept.
- If the text is a task/todo, extract the intent as a claim with low confidence.
- If the text is purely conversational with no extractable claims, return empty arrays.
"""

        model = self._enrichment_model()
        response = model.create_chat_completion(
            messages=[
                {"role": "system", "content": "You are a precise knowledge extraction system. Output ONLY valid JSON."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.1,
            max_tokens=2048,
            response_format={"type": "json_object"},
        )

        raw = response["choices"][0]["message"]["content"]
        _log_ft_pair(self.config, prompt, raw, self.config.models.enrichment_model, "enrich")

        try:
            return json.loads(raw)
        except json.JSONDecodeError:
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
        response = model.create_chat_completion(
            messages=[
                {"role": "system", "content": "Output ONLY valid JSON."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
            max_tokens=512,
            response_format={"type": "json_object"},
        )

        raw = response["choices"][0]["message"]["content"]
        _log_ft_pair(self.config, prompt, raw, self.config.models.tagger_model, "tag")

        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return {"concepts": [], "entities": [], "is_task": False}

    def embed(self, text: str) -> np.ndarray:
        """Generate an embedding vector for text."""
        model = self._embedding_model()
        result = model.embed(text)
        # llama-cpp-python returns list or list-of-lists
        if isinstance(result[0], list):
            vec = np.array(result[0], dtype=np.float32)
        else:
            vec = np.array(result, dtype=np.float32)
        return vec

    def embed_batch(self, texts: list[str]) -> list[np.ndarray]:
        """Embed multiple texts efficiently."""
        model = self._embedding_model()
        results = model.embed(texts)
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
        prompt = f"""You are maintaining a knowledge base wiki page for "{concept_title}".

EXISTING PAGE CONTENT (LLM zone only):
{existing_content or "(empty — this is a new page)"}

NEW EVIDENCE to integrate:
{evidence_text}

Write a concise new evidence section in markdown. Include:
- A brief synthesis of what the new evidence adds
- Inline confidence pills like [c:0.82] after factual claims
- Cross-references as [[concept-slug]] Obsidian links where relevant

Output ONLY the new section content (no page frontmatter, no heading).
"""
        model = self._enrichment_model()
        response = model.create_chat_completion(
            messages=[
                {"role": "system", "content": "You are a wiki maintainer. Write clear, concise markdown."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.3,
            max_tokens=1024,
        )
        raw = response["choices"][0]["message"]["content"]
        _log_ft_pair(self.config, prompt, raw, self.config.models.enrichment_model, "wiki_section")
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
            response = model.create_chat_completion(
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "Describe this image in detail. If it contains text, transcribe all visible text."},
                            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}},
                        ],
                    }
                ],
                temperature=0.1,
                max_tokens=2048,
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
