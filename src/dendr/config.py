"""Configuration management for Dendr."""

from __future__ import annotations

import json
import os
import uuid
from dataclasses import dataclass, field
from datetime import datetime
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

    vlm_model: str = "gemma-4-E4B-it-Q4_K_M.gguf"
    vlm_mmproj: str = "mmproj-BF16.gguf"
    embedding_model: str = "nomic-embed-text-v1.5.f16.gguf"
    vlm_ctx: int = 4096
    embedding_dim: int = 768


@dataclass
class Config:
    """Top-level Dendr configuration."""

    vault_path: Path = field(default_factory=lambda: Path.cwd())
    data_dir: Path = field(default_factory=_default_data_dir)
    vault_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    models: ModelConfig = field(default_factory=ModelConfig)

    # Pipeline settings
    search_port: int = 7777

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
    def dead_dir(self) -> Path:
        return self.queue_dir / "dead"

    @property
    def logs_dir(self) -> Path:
        return self.data_dir / "logs"

    @property
    def models_dir(self) -> Path:
        return self.data_dir / "models"

    @property
    def manifest_path(self) -> Path:
        """Path to dendr-models.yaml (repo root, vault root, or cwd)."""
        # Check several locations
        for candidate in [
            self.vault_path / "dendr-models.yaml",
            Path.cwd() / "dendr-models.yaml",
        ]:
            if candidate.exists():
                return candidate
        # Default to cwd even if missing
        return Path.cwd() / "dendr-models.yaml"

    @property
    def dendr_marker_path(self) -> Path:
        return self.vault_path / ".dendr"

    @property
    def config_file_path(self) -> Path:
        return self.data_dir / "config.json"

    def append_activity_log(self, entry: str) -> None:
        """Append to Wiki/log.md."""
        log_path = self.wiki_dir / "log.md"
        if not log_path.exists():
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log_path.write_text(
                "---\ntype: log\n---\n\n# Activity Log\n\n", encoding="utf-8"
            )

        now = datetime.now().strftime("%Y-%m-%d %H:%M")
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(f"- `{now}` {entry}\n")

    def ensure_dirs(self) -> None:
        """Create all necessary directories."""
        for d in [
            self.daily_dir,
            self.attachments_dir,
            self.wiki_dir,
            self.data_dir,
            self.queue_dir,
            self.pending_dir,
            self.processing_dir,
            self.done_dir,
            self.dead_dir,
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
            "created": datetime.now().isoformat(),
        }
        self.dendr_marker_path.write_text(json.dumps(marker, indent=2))

    def save(self) -> None:
        """Persist config to disk."""
        self.data_dir.mkdir(parents=True, exist_ok=True)
        data = {
            "vault_path": str(self.vault_path),
            "vault_id": self.vault_id,
            "models": {
                "vlm_model": self.models.vlm_model,
                "vlm_mmproj": self.models.vlm_mmproj,
                "embedding_model": self.models.embedding_model,
                "vlm_ctx": self.models.vlm_ctx,
                "embedding_dim": self.models.embedding_dim,
            },
            "search_port": self.search_port,
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
        model_data = data.get("models", {})
        models = ModelConfig(**model_data)
        return cls(
            vault_path=Path(data["vault_path"]),
            data_dir=dd,
            vault_id=data.get("vault_id", str(uuid.uuid4())),
            models=models,
            search_port=data.get("search_port", 7777),
        )
