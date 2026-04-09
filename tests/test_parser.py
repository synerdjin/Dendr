"""Tests for the block parser."""

import tempfile
from pathlib import Path

from dendr.parser import parse_daily_note, inject_block_ids, get_file_hash


def test_parse_empty_note():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False, encoding="utf-8") as f:
        f.write("")
        f.flush()
        blocks = parse_daily_note(Path(f.name))
        assert blocks == []


def test_parse_single_paragraph():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False, encoding="utf-8") as f:
        f.write("This is a test paragraph about machine learning.")
        f.flush()
        blocks = parse_daily_note(Path(f.name))
        assert len(blocks) == 1
        assert "machine learning" in blocks[0].text
        assert blocks[0].block_hash  # has a hash


def test_parse_multiple_blocks():
    content = """# Heading

First paragraph about Python.

Second paragraph about Rust.

- A list item about Go
"""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False, encoding="utf-8") as f:
        f.write(content)
        f.flush()
        blocks = parse_daily_note(Path(f.name))
        assert len(blocks) >= 3  # heading, para1, para2, list


def test_parse_existing_block_ref():
    content = "Some text about databases ^my-block-id"
    with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False, encoding="utf-8") as f:
        f.write(content)
        f.flush()
        blocks = parse_daily_note(Path(f.name))
        assert len(blocks) == 1
        assert blocks[0].block_id == "my-block-id"


def test_parse_attachment_embed():
    content = "![[screenshot.png]]"
    with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False, encoding="utf-8") as f:
        f.write(content)
        f.flush()
        blocks = parse_daily_note(Path(f.name))
        assert len(blocks) == 1
        assert blocks[0].is_attachment_ref
        assert blocks[0].attachment_type == "image"


def test_inject_block_ids():
    content = "First paragraph.\n\nSecond paragraph."
    with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False, encoding="utf-8") as f:
        f.write(content)
        f.flush()
        path = Path(f.name)
        blocks = parse_daily_note(path)
        modified = inject_block_ids(path, blocks)
        assert modified is True
        # Re-read and check block IDs are present
        new_content = path.read_text(encoding="utf-8")
        assert "^dendr-" in new_content


def test_block_hash_stability():
    content = "Stable content for hashing."
    with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False, encoding="utf-8") as f:
        f.write(content)
        f.flush()
        blocks1 = parse_daily_note(Path(f.name))
        blocks2 = parse_daily_note(Path(f.name))
        assert blocks1[0].block_hash == blocks2[0].block_hash


def test_file_hash():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False, encoding="utf-8") as f:
        f.write("test content")
        f.flush()
        h1 = get_file_hash(Path(f.name))
        h2 = get_file_hash(Path(f.name))
        assert h1 == h2
        assert len(h1) == 16
