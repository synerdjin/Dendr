"""Tests for the block parser."""

import tempfile
from pathlib import Path

from dendr.parser import (
    get_file_hash,
    inject_block_ids,
    parse_closures,
    parse_daily_note,
)


def test_parse_empty_note():
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".md", delete=False, encoding="utf-8"
    ) as f:
        f.write("")
        f.flush()
        blocks = parse_daily_note(Path(f.name))
        assert blocks == []


def test_parse_single_paragraph():
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".md", delete=False, encoding="utf-8"
    ) as f:
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
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".md", delete=False, encoding="utf-8"
    ) as f:
        f.write(content)
        f.flush()
        blocks = parse_daily_note(Path(f.name))
        assert len(blocks) >= 3  # heading, para1, para2, list


def test_parse_existing_block_ref():
    content = "Some text about databases ^my-block-id"
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".md", delete=False, encoding="utf-8"
    ) as f:
        f.write(content)
        f.flush()
        blocks = parse_daily_note(Path(f.name))
        assert len(blocks) == 1
        assert blocks[0].block_id == "my-block-id"


def test_parse_attachment_embed():
    content = "![[screenshot.png]]"
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".md", delete=False, encoding="utf-8"
    ) as f:
        f.write(content)
        f.flush()
        blocks = parse_daily_note(Path(f.name))
        assert len(blocks) == 1
        assert blocks[0].is_attachment_ref
        assert blocks[0].attachment_type == "image"


def test_inject_block_ids():
    content = "First paragraph.\n\nSecond paragraph."
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".md", delete=False, encoding="utf-8"
    ) as f:
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
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".md", delete=False, encoding="utf-8"
    ) as f:
        f.write(content)
        f.flush()
        blocks1 = parse_daily_note(Path(f.name))
        blocks2 = parse_daily_note(Path(f.name))
        assert blocks1[0].block_hash == blocks2[0].block_hash


def test_file_hash():
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".md", delete=False, encoding="utf-8"
    ) as f:
        f.write("test content")
        f.flush()
        h1 = get_file_hash(Path(f.name))
        h2 = get_file_hash(Path(f.name))
        assert h1 == h2
        assert len(h1) == 16


# ── Closure markers ────────────────────────────────────────────────


def test_parse_closures_unchecked_open():
    text = "- [ ] **Do X** — *3w ago* <!-- closure:dendr-abc status:open -->"
    closures = parse_closures(text)
    assert len(closures) == 1
    assert closures[0].block_id == "dendr-abc"
    assert closures[0].status == "open"
    assert closures[0].checkbox_checked is False


def test_parse_closures_checked_becomes_done():
    text = "- [x] **Done thing** — *1w ago* <!-- closure:dendr-xyz status:open -->"
    closures = parse_closures(text)
    assert len(closures) == 1
    assert closures[0].status == "done"
    assert closures[0].checkbox_checked is True


def test_parse_closures_explicit_status_wins():
    # User typed "abandoned" even though checkbox is still unchecked
    text = "- [ ] **Old plan** — *2mo ago* <!-- closure:dendr-111 status:abandoned -->"
    closures = parse_closures(text)
    assert len(closures) == 1
    assert closures[0].status == "abandoned"


def test_parse_closures_still_live():
    text = "- [ ] **Keep** — *6w ago* <!-- closure:dendr-222 status:still-live -->"
    closures = parse_closures(text)
    assert len(closures) == 1
    assert closures[0].status == "still-live"


def test_parse_closures_snoozed():
    text = "- [ ] **Later** — *3w ago* <!-- closure:dendr-333 status:snoozed -->"
    closures = parse_closures(text)
    assert len(closures) == 1
    assert closures[0].status == "snoozed"


def test_parse_closures_multiple_and_dedup():
    text = """
## Task Review

### 1-2w old

- [ ] **First** — *1w ago* <!-- closure:dendr-a status:open -->
- [x] **Second** — *2w ago* <!-- closure:dendr-b status:open -->

### 1m+ old

- [ ] **Third** — *6w ago* <!-- closure:dendr-c status:abandoned -->
- [ ] **Dup** — *6w ago* <!-- closure:dendr-c status:open -->
"""
    closures = parse_closures(text)
    # Three unique block_ids, dendr-c dedups to the first occurrence
    assert len(closures) == 3
    ids = [c.block_id for c in closures]
    assert ids == ["dendr-a", "dendr-b", "dendr-c"]
    assert closures[0].status == "open"
    assert closures[1].status == "done"
    assert closures[2].status == "abandoned"


def test_parse_closures_ignores_unrelated_comments():
    text = """
- [ ] Regular task (no marker)
<!-- feedback:narrative
useful: yes
-->
- [ ] **Real one** — *8d ago* <!-- closure:dendr-real status:open -->
"""
    closures = parse_closures(text)
    assert len(closures) == 1
    assert closures[0].block_id == "dendr-real"


def test_parse_closures_empty():
    assert parse_closures("") == []
    assert parse_closures("# Just a heading\n\nNo closures here.") == []


def test_parse_closures_missing_status_defaults_to_checkbox():
    # No status: in the comment — checkbox drives it
    text_unchecked = "- [ ] **A** <!-- closure:dendr-q -->"
    text_checked = "- [x] **B** <!-- closure:dendr-r -->"
    assert parse_closures(text_unchecked)[0].status == "open"
    assert parse_closures(text_checked)[0].status == "done"
