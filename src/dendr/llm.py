"""Local-inference surface: embeddings (Nomic)."""

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
) -> Any:
    """Get or create a llama-cpp-python model instance.

    Only one model is kept in VRAM at a time to fit within GPU memory.
    """
    from llama_cpp import Llama

    key = str(model_path)
    if key not in _models:
        _unload_all_except(None)
        role = _model_role_from_path(model_path)
        logger.info("Loading model: %s (ctx=%d)", model_path.name, n_ctx)
        t0 = time.monotonic()
        kwargs: dict[str, Any] = {
            "model_path": str(model_path),
            "n_ctx": n_ctx,
            "n_gpu_layers": n_gpu_layers,
            "verbose": False,
            "embedding": embedding,
        }
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
    """Local-inference surface: embeddings."""

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
