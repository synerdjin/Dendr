"""Tests for the model manager."""

import tempfile
from pathlib import Path

import yaml

from dendr.model_manager import (
    ModelManifest,
    check_all_models,
    lock_models,
    preflight_check,
    sha256_file,
)


def _write_manifest(path: Path, models: dict | None = None) -> Path:
    """Write a test manifest."""
    data = {
        "version": 1,
        "models": models
        or {
            "embedding": {
                "repo": "test/repo",
                "filename": "test-model.gguf",
                "sha256": "",
                "size_bytes": 100,
                "role": "Test model",
                "gated": False,
            }
        },
    }
    manifest_path = path / "dendr-models.yaml"
    with open(manifest_path, "w") as f:
        yaml.dump(data, f)
    return manifest_path


def test_load_manifest():
    with tempfile.TemporaryDirectory() as td:
        mp = _write_manifest(Path(td))
        manifest = ModelManifest.load(mp)
        assert "embedding" in manifest.specs
        assert manifest.specs["embedding"].filename == "test-model.gguf"


def test_check_missing_model():
    with tempfile.TemporaryDirectory() as td:
        mp = _write_manifest(Path(td))
        manifest = ModelManifest.load(mp)
        models_dir = Path(td) / "models"
        models_dir.mkdir()

        statuses = check_all_models(models_dir, manifest)
        assert not statuses["embedding"].present


def test_check_present_model():
    with tempfile.TemporaryDirectory() as td:
        mp = _write_manifest(Path(td))
        manifest = ModelManifest.load(mp)
        models_dir = Path(td) / "models"
        models_dir.mkdir()

        # Create a fake model file
        (models_dir / "test-model.gguf").write_bytes(b"fake model data")

        statuses = check_all_models(models_dir, manifest)
        assert statuses["embedding"].present
        assert statuses["embedding"].hash_match is None  # no expected hash


def test_sha256_file():
    with tempfile.NamedTemporaryFile(delete=False) as f:
        f.write(b"test content for hashing")
        f.flush()
        h = sha256_file(Path(f.name))
        assert len(h) == 64  # full sha256 hex


def test_preflight_missing():
    with tempfile.TemporaryDirectory() as td:
        mp = _write_manifest(Path(td))
        manifest = ModelManifest.load(mp)
        models_dir = Path(td) / "models"
        models_dir.mkdir()

        errors = preflight_check(models_dir, manifest)
        assert len(errors) == 1
        assert "Missing" in errors[0]


def test_preflight_ok():
    with tempfile.TemporaryDirectory() as td:
        mp = _write_manifest(Path(td))
        manifest = ModelManifest.load(mp)
        models_dir = Path(td) / "models"
        models_dir.mkdir()
        (models_dir / "test-model.gguf").write_bytes(b"fake")

        errors = preflight_check(models_dir, manifest)
        assert len(errors) == 0


def test_lock_models():
    with tempfile.TemporaryDirectory() as td:
        mp = _write_manifest(Path(td))
        manifest = ModelManifest.load(mp)
        models_dir = Path(td) / "models"
        models_dir.mkdir()
        (models_dir / "test-model.gguf").write_bytes(b"model content")

        hashes = lock_models(models_dir, manifest, mp)
        assert "embedding" in hashes
        assert len(hashes["embedding"]) == 64

        # Reload and check hash was persisted
        manifest2 = ModelManifest.load(mp)
        assert manifest2.specs["embedding"].sha256 == hashes["embedding"]


def test_preflight_hash_mismatch():
    with tempfile.TemporaryDirectory() as td:
        mp = _write_manifest(
            Path(td),
            models={
                "embedding": {
                    "repo": "test/repo",
                    "filename": "test-model.gguf",
                    "sha256": "0000000000000000000000000000000000000000000000000000000000000000",
                    "size_bytes": 100,
                    "role": "Test",
                    "gated": False,
                }
            },
        )
        manifest = ModelManifest.load(mp)
        models_dir = Path(td) / "models"
        models_dir.mkdir()
        (models_dir / "test-model.gguf").write_bytes(b"model content")

        errors = preflight_check(models_dir, manifest)
        assert len(errors) == 1
        assert "mismatch" in errors[0].lower()
