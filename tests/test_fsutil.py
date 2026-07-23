"""Tests for atomic file writes (F8)."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from dendr.fsutil import atomic_write_text


def test_atomic_write_creates_and_reads_back(tmp_path: Path):
    target = tmp_path / "note.md"
    atomic_write_text(target, "hello world\n")
    assert target.read_text(encoding="utf-8") == "hello world\n"


def test_atomic_write_overwrites_existing(tmp_path: Path):
    target = tmp_path / "note.md"
    target.write_text("old content", encoding="utf-8")
    atomic_write_text(target, "new content")
    assert target.read_text(encoding="utf-8") == "new content"


def test_atomic_write_creates_missing_parent_dirs(tmp_path: Path):
    target = tmp_path / "nested" / "sub" / "note.md"
    atomic_write_text(target, "x")
    assert target.read_text(encoding="utf-8") == "x"


def test_atomic_write_leaves_no_temp_files(tmp_path: Path):
    target = tmp_path / "note.md"
    atomic_write_text(target, "content")
    # Only the target should remain — no leftover .note.md.*.tmp files.
    assert [p.name for p in tmp_path.iterdir()] == ["note.md"]


def test_failed_write_preserves_original_and_cleans_up(tmp_path: Path, monkeypatch):
    """If the replace step fails, the original file must survive intact and no
    temp file may be left behind."""
    target = tmp_path / "note.md"
    target.write_text("ORIGINAL", encoding="utf-8")

    def boom(src, dst):
        raise OSError("simulated replace failure")

    monkeypatch.setattr(os, "replace", boom)
    with pytest.raises(OSError):
        atomic_write_text(target, "SHOULD NOT LAND")

    assert target.read_text(encoding="utf-8") == "ORIGINAL"
    assert [p.name for p in tmp_path.iterdir()] == ["note.md"]
