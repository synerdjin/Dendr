"""Configuration management for Dendr."""

from __future__ import annotations

import json
import os
import uuid
from dataclasses import dataclass, field
from pathlib import Path


def _default_data_dir() -> Path:
    """Platform-appropriate local data directory (never synced to iCloud)."""
    local = os.environ.get("LOCALAPPDATA")
    if local:
        return Path(local) / "Dendr"
    # macOS / Linux fallback
    return Path.home() / ".local" / "share" / "dendr"


@dataclass
class ModelConfig:
    """Local model configuration."""

    enrichment_model: str = "phi-4-Q4_K_M.gguf"
    tagger_model: str = "gemma-3-4b-it-Q4_K_M.gguf"
    vlm_model: str = "Llama-3.2-11B-Vision-Q4_K_M.gguf"
    embedding_model: str = "nomic-embed-text-v1.5.Q8_0.gguf"
    # Context sizes
    enrichment_ctx: int = 8192
    tagger_ctx: int = 4096
    vlm_ctx: int = 4096
    embedding_dim: int = 768
    embedding_dim_short: int = 256  # Matryoshka truncation for ANN


@dataclass
class Config:
    """Top-level Dendr configuration."""

    vault_path: Path = field(default_factory=lambda: Path.cwd())
    data_dir: Path = field(default_factory=_default_data_dir)
    vault_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    models: ModelConfig = field(default_factory=ModelConfig)

    # Pipeline settings
    canonicalization_threshold: float = 0.86
    backpressure_days: int = 7  # switch to shallow mode after N queued days
    search_port: int = 7777
    stale_claim_weeks: int = 8  # lint: flag claims not reinforced in N weeks

    # Paths derived from vault_path
    @property
    def daily_dir(self) -> Path:
        return self.vault_path / "Daily"

    @property
    def attachments_dir(self) -> Path:
        return self.vault_path / "Attachments"

    @property
    def wiki_dir(self) -> Path:
        return self.vault_path / "Wiki"

    @property
    def concepts_dir(self) -> Path:
        return self.wiki_dir / "concepts"

    @property
    def entities_dir(self) -> Path:
        return self.wiki_dir / "entities"

    @property
    def summaries_dir(self) -> Path:
        return self.wiki_dir / "summaries"

    @property
    def lint_dir(self) -> Path:
        return self.wiki_dir / "_lint"

    # Paths derived from data_dir
    @property
    def db_path(self) -> Path:
        return self.data_dir / "state.sqlite"

    @property
    def queue_dir(self) -> Path:
        return self.data_dir / "queue"

    @property
    def pending_dir(self) -> Path:
        return self.queue_dir / "pending"

    @property
    def processing_dir(self) -> Path:
        return self.queue_dir / "processing"

    @property
    def done_dir(self) -> Path:
        return self.queue_dir / "done"

    @property
    def logs_dir(self) -> Path:
        return self.data_dir / "logs"

    @property
    def models_dir(self) -> Path:
        return self.data_dir / "models"

    @property
    def ft_pairs_path(self) -> Path:
        return self.data_dir / "ft-pairs.jsonl"

    @property
    def dendr_marker_path(self) -> Path:
        return self.vault_path / ".dendr"

    @property
    def config_file_path(self) -> Path:
        return self.data_dir / "config.json"

    def ensure_dirs(self) -> None:
        """Create all necessary directories."""
        for d in [
            self.daily_dir,
            self.attachments_dir,
            self.wiki_dir,
            self.concepts_dir,
            self.entities_dir,
            self.summaries_dir,
            self.lint_dir,
            self.data_dir,
            self.queue_dir,
            self.pending_dir,
            self.processing_dir,
            self.done_dir,
            self.logs_dir,
            self.models_dir,
        ]:
            d.mkdir(parents=True, exist_ok=True)

    def write_vault_marker(self) -> None:
        """Write the .dendr marker file to the vault."""
        import socket

        marker = {
            "vault_id": self.vault_id,
            "hostname": socket.gethostname(),
            "created": __import__("datetime").datetime.now().isoformat(),
        }
        self.dendr_marker_path.write_text(json.dumps(marker, indent=2))

    def save(self) -> None:
        """Persist config to disk."""
        self.data_dir.mkdir(parents=True, exist_ok=True)
        data = {
            "vault_path": str(self.vault_path),
            "vault_id": self.vault_id,
            "models": {
                "enrichment_model": self.models.enrichment_model,
                "tagger_model": self.models.tagger_model,
                "vlm_model": self.models.vlm_model,
                "embedding_model": self.models.embedding_model,
                "enrichment_ctx": self.models.enrichment_ctx,
                "tagger_ctx": self.models.tagger_ctx,
                "vlm_ctx": self.models.vlm_ctx,
                "embedding_dim": self.models.embedding_dim,
                "embedding_dim_short": self.models.embedding_dim_short,
            },
            "canonicalization_threshold": self.canonicalization_threshold,
            "backpressure_days": self.backpressure_days,
            "search_port": self.search_port,
            "stale_claim_weeks": self.stale_claim_weeks,
        }
        self.config_file_path.write_text(json.dumps(data, indent=2))

    @classmethod
    def load(cls, data_dir: Path | None = None) -> Config:
        """Load config from disk, or return defaults."""
        dd = data_dir or _default_data_dir()
        config_path = dd / "config.json"
        if not config_path.exists():
            return cls(data_dir=dd)
        data = json.loads(config_path.read_text())
        models = ModelConfig(**data.get("models", {}))
        return cls(
            vault_path=Path(data["vault_path"]),
            data_dir=dd,
            vault_id=data.get("vault_id", str(uuid.uuid4())),
            models=models,
            canonicalization_threshold=data.get("canonicalization_threshold", 0.86),
            backpressure_days=data.get("backpressure_days", 7),
            search_port=data.get("search_port", 7777),
            stale_claim_weeks=data.get("stale_claim_weeks", 8),
        )
