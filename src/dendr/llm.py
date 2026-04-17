"""LLM client abstraction for local inference via llama-cpp-python.

Manages model lifecycle for the embedding model and the vision/OCR model.
Text annotation has been removed — Claude handles classification, affect
reading, and narrative synthesis directly from raw block text.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

import numpy as np

from dendr.config import Config
from dendr.metrics import (
    INFERENCE_SECONDS,
    MODEL_LOAD_SECONDS,
    MODEL_LOADED,
)

logger = logging.getLogger(__name__)

# Lazy-loaded model instances
_models: dict[str, Any] = {}


def _model_role_from_path(model_path: Path) -> str:
    """Derive a short role label from a model filename for metrics."""
    name = model_path.stem.lower()
    if "gemma" in name:
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
    model_path: Path,
    n_ctx: int = 4096,
    n_gpu_layers: int = -1,
    embedding: bool = False,
    chat_handler: Any = None,
) -> Any:
    """Get or create a llama-cpp-python model instance.

    Only one model is kept in VRAM at a time to fit within GPU memory.
    Pass chat_handler for vision models that need an mmproj projector.
    """
    from llama_cpp import Llama

    # Use a different cache key for vision vs text-only mode of the same model
    key = str(model_path) + (":vision" if chat_handler else "")
    if key not in _models:
        _unload_all_except(None)
        role = _model_role_from_path(model_path)
        if chat_handler:
            role = "vision"
        logger.info("Loading model: %s (ctx=%d)", model_path.name, n_ctx)
        t0 = time.monotonic()
        kwargs: dict[str, Any] = {
            "model_path": str(model_path),
            "n_ctx": n_ctx,
            "n_gpu_layers": n_gpu_layers,
            "verbose": False,
            "embedding": embedding,
        }
        if chat_handler:
            kwargs["chat_handler"] = chat_handler
        _models[key] = Llama(**kwargs)
        MODEL_LOAD_SECONDS.labels(model_role=role).observe(time.monotonic() - t0)
        MODEL_LOADED.labels(model_role=role).set(1)
    return _models[key]


def unload_all() -> None:
    """Release all loaded models."""
    for key in _models:
        role = _model_role_from_path(Path(key))
        MODEL_LOADED.labels(model_role=role).set(0)
    _models.clear()


class LLMClient:
    """Local-inference surface: embeddings + vision/OCR."""

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
            missing = []
            for name in [
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

    def _embedding_model(self) -> Any:
        return _get_model(
            self._model_path(self.config.models.embedding_model),
            n_ctx=512,
            embedding=True,
        )

    def embed(self, text: str) -> np.ndarray:
        """Generate an embedding vector for text."""
        model = self._embedding_model()
        t0 = time.monotonic()
        result = model.embed(text)
        INFERENCE_SECONDS.labels(model_role="embedding", task="embed").observe(
            time.monotonic() - t0
        )
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

    def extract_text_from_image(self, image_path: str) -> str:
        """OCR / caption an image using the VLM.

        Returns an empty string on failure. Loaded on demand — the VLM
        stays off the GPU during normal text ingest.
        """
        try:
            import base64

            with open(image_path, "rb") as f:
                img_b64 = base64.b64encode(f.read()).decode()

            from llama_cpp.llama_chat_format import Llava15ChatHandler

            mmproj_path = self._model_path(self.config.models.vlm_mmproj)
            handler = Llava15ChatHandler(clip_model_path=str(mmproj_path))
            model = _get_model(
                self._model_path(self.config.models.vlm_model),
                n_ctx=self.config.models.vlm_ctx,
                chat_handler=handler,
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
