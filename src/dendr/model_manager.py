"""Model manager — declarative model manifest, download, verify, lock.

Reads dendr-models.yaml, downloads GGUF weights from HuggingFace,
verifies SHA256 integrity, and provides preflight checks.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)

_CHUNK_SIZE = 8 * 1024 * 1024  # 8MB chunks for hashing


@dataclass
class ModelSpec:
    """A single model declaration from the manifest."""

    role: str
    repo: str
    filename: str
    sha256: str
    size_bytes: int
    context: int
    gpu_layers: int
    gated: bool
    description: str  # human-readable role description


@dataclass
class ModelStatus:
    """Status of a model on disk."""

    spec: ModelSpec
    present: bool
    hash_match: bool | None  # None if no expected hash or file missing
    actual_size: int | None


class ModelManifest:
    """Parsed model manifest."""

    def __init__(self, specs: dict[str, ModelSpec], version: int = 1):
        self.specs = specs
        self.version = version

    @classmethod
    def load(cls, manifest_path: Path) -> ModelManifest:
        """Load manifest from YAML file."""
        if not manifest_path.exists():
            raise FileNotFoundError(
                f"Model manifest not found: {manifest_path}\n"
                "Create one with `dendr-models.yaml` in the repo root."
            )

        with open(manifest_path, encoding="utf-8") as f:
            data = yaml.safe_load(f)

        version = data.get("version", 1)
        specs: dict[str, ModelSpec] = {}

        for role, model_data in data.get("models", {}).items():
            specs[role] = ModelSpec(
                role=role,
                repo=model_data["repo"],
                filename=model_data["filename"],
                sha256=model_data.get("sha256", ""),
                size_bytes=model_data.get("size_bytes", 0),
                context=model_data.get("context", 4096),
                gpu_layers=model_data.get("gpu_layers", -1),
                gated=model_data.get("gated", False),
                description=model_data.get("role", ""),
            )

        return cls(specs=specs, version=version)

    def save(self, manifest_path: Path) -> None:
        """Write manifest back to YAML (e.g. after lock)."""
        data: dict = {"version": self.version, "models": {}}
        for role, spec in self.specs.items():
            data["models"][role] = {
                "repo": spec.repo,
                "filename": spec.filename,
                "sha256": spec.sha256,
                "size_bytes": spec.size_bytes,
                "role": spec.description,
                "context": spec.context,
                "gpu_layers": spec.gpu_layers,
                "gated": spec.gated,
            }

        with open(manifest_path, "w", encoding="utf-8") as f:
            f.write(
                "# Dendr Model Manifest — version-controlled model stack definition.\n"
                "# Run `dendr models pull` to download, `dendr models lock` to pin SHA256 hashes.\n"
            )
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)


def sha256_file(path: Path) -> str:
    """Compute SHA256 of a file without loading it all into memory."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(_CHUNK_SIZE)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def check_model(models_dir: Path, spec: ModelSpec) -> ModelStatus:
    """Check status of a single model on disk."""
    model_path = models_dir / spec.filename

    if not model_path.exists():
        return ModelStatus(spec=spec, present=False, hash_match=None, actual_size=None)

    actual_size = model_path.stat().st_size
    hash_match = None

    if spec.sha256:
        actual_hash = sha256_file(model_path)
        hash_match = actual_hash == spec.sha256

    return ModelStatus(
        spec=spec, present=True, hash_match=hash_match, actual_size=actual_size
    )


def check_all_models(
    models_dir: Path, manifest: ModelManifest
) -> dict[str, ModelStatus]:
    """Check status of all models."""
    return {
        role: check_model(models_dir, spec) for role, spec in manifest.specs.items()
    }


def preflight_check(models_dir: Path, manifest: ModelManifest) -> list[str]:
    """Run preflight check. Returns list of error messages (empty = OK)."""
    errors: list[str] = []
    statuses = check_all_models(models_dir, manifest)

    for role, status in statuses.items():
        if not status.present:
            errors.append(
                f"Missing model [{role}]: {status.spec.filename}. "
                f"Run `dendr models pull` to download."
            )
        elif status.hash_match is False:
            errors.append(
                f"Hash mismatch [{role}]: {status.spec.filename}. "
                f"Run `dendr models verify` for details or `dendr models pull` to re-download."
            )

    return errors


def pull_model(
    models_dir: Path,
    spec: ModelSpec,
    force: bool = False,
    token: str | None = None,
) -> Path:
    """Download a single model from HuggingFace.

    Skips download if file exists with matching SHA256 (unless force=True).
    Returns the local path.
    """
    from huggingface_hub import hf_hub_download

    model_path = models_dir / spec.filename

    # Skip if already present and verified
    if not force and model_path.exists():
        if spec.sha256:
            actual_hash = sha256_file(model_path)
            if actual_hash == spec.sha256:
                logger.info(
                    "[%s] Already present and verified: %s", spec.role, spec.filename
                )
                return model_path
            logger.warning(
                "[%s] Hash mismatch, re-downloading: %s", spec.role, spec.filename
            )
        else:
            logger.info(
                "[%s] Already present (no hash to verify): %s", spec.role, spec.filename
            )
            return model_path

    logger.info("[%s] Downloading %s from %s...", spec.role, spec.filename, spec.repo)

    downloaded_path = hf_hub_download(
        repo_id=spec.repo,
        filename=spec.filename,
        local_dir=str(models_dir),
        token=token,
        revision="main",  # pin to branch; post-download SHA256 check provides integrity
    )

    # huggingface_hub may place the file in a subdirectory or use its cache;
    # ensure it's at the expected flat path
    dl_path = Path(downloaded_path)
    if dl_path != model_path:
        import shutil

        shutil.move(str(dl_path), str(model_path))

    # Verify after download
    if spec.sha256:
        actual_hash = sha256_file(model_path)
        if actual_hash != spec.sha256:
            raise RuntimeError(
                f"Downloaded file hash mismatch for {spec.filename}: "
                f"expected {spec.sha256}, got {actual_hash}"
            )
        logger.info("[%s] Verified: %s", spec.role, spec.filename)
    else:
        logger.info("[%s] Downloaded (no hash to verify): %s", spec.role, spec.filename)

    return model_path


def pull_all_models(
    models_dir: Path,
    manifest: ModelManifest,
    roles: list[str] | None = None,
    force: bool = False,
    token: str | None = None,
) -> dict[str, Path]:
    """Download all (or specified) models.

    Returns mapping of role → local path.
    """
    models_dir.mkdir(parents=True, exist_ok=True)

    results: dict[str, Path] = {}
    specs = manifest.specs

    if roles:
        specs = {r: s for r, s in specs.items() if r in roles}

    for role, spec in specs.items():
        try:
            path = pull_model(models_dir, spec, force=force, token=token)
            results[role] = path
        except Exception as e:
            logger.error("[%s] Failed to download %s: %s", role, spec.filename, e)
            if spec.gated:
                logger.error(
                    "  This is a gated model. Set HF_TOKEN env var or run "
                    "`huggingface-cli login` first."
                )

    return results


def lock_models(
    models_dir: Path, manifest: ModelManifest, manifest_path: Path
) -> dict[str, str]:
    """Compute SHA256 of all present models and write back to manifest.

    Returns mapping of role → sha256.
    """
    hashes: dict[str, str] = {}

    for role, spec in manifest.specs.items():
        model_path = models_dir / spec.filename
        if not model_path.exists():
            logger.warning("[%s] Not present, skipping: %s", role, spec.filename)
            continue

        h = sha256_file(model_path)
        spec.sha256 = h
        spec.size_bytes = model_path.stat().st_size
        hashes[role] = h
        logger.info("[%s] Locked: %s → %s", role, spec.filename, h[:16] + "...")

    manifest.save(manifest_path)
    return hashes
