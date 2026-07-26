"""Tests for the ingest/reprocess advisory lock (F12)."""

from __future__ import annotations

import pytest

from dendr.config import Config
from dendr.locking import LockHeld, ingest_lock


def _config(tmp_path) -> Config:
    return Config(vault_path=tmp_path / "vault", data_dir=tmp_path / "data")


def test_lock_is_exclusive(tmp_path):
    config = _config(tmp_path)
    with ingest_lock(config):
        with pytest.raises(LockHeld):
            with ingest_lock(config):
                pass


def test_lock_released_after_use(tmp_path):
    config = _config(tmp_path)
    with ingest_lock(config):
        pass
    # A second, sequential acquisition succeeds — the lock was released.
    with ingest_lock(config):
        pass


def test_lock_creates_data_dir_and_file(tmp_path):
    config = _config(tmp_path)
    with ingest_lock(config) as lock_path:
        assert lock_path.exists()
        assert lock_path == config.data_dir / "ingest.lock"
