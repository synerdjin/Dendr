"""Tests for the block parser."""

import tempfile
from pathlib import Path

from dendr.models import CHECKBOX_CLOSED, CHECKBOX_NONE, CHECKBOX_OPEN
from dendr.parser import (
    close_task_in_source,
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


# ── Checkbox state ────────────────────────────────────────────────────


def test_checkbox_state_open():
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".md", delete=False, encoding="utf-8"
    ) as f:
        f.write("- [ ] open task\n")
        f.flush()
        blocks = parse_daily_note(Path(f.name))
        assert len(blocks) == 1
        assert blocks[0].checkbox_state == CHECKBOX_OPEN


def test_checkbox_state_closed():
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".md", delete=False, encoding="utf-8"
    ) as f:
        f.write("- [x] done thing\n")
        f.flush()
        blocks = parse_daily_note(Path(f.name))
        assert len(blocks) == 1
        assert blocks[0].checkbox_state == CHECKBOX_CLOSED


def test_checkbox_state_none_for_prose():
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".md", delete=False, encoding="utf-8"
    ) as f:
        f.write("Just a reflection about today.\n")
        f.flush()
        blocks = parse_daily_note(Path(f.name))
        assert len(blocks) == 1
        assert blocks[0].checkbox_state == CHECKBOX_NONE


def test_cancelled_checkbox_parses_as_none():
    # Contract relied on by the abandoned write-back: `- [-]` (Tasks "cancelled")
    # is NOT a recognised checkbox, so re-parse yields checkbox_state=none and
    # logs no transition; completion_status='abandoned' stays authoritative.
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".md", delete=False, encoding="utf-8"
    ) as f:
        f.write("- [-] abandoned task ^dendr-cancel\n")
        f.flush()
        blocks = parse_daily_note(Path(f.name))
        assert len(blocks) == 1
        assert blocks[0].checkbox_state == CHECKBOX_NONE


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


# ── close_task_in_source ──────────────────────────────────────────────


def _write_note(text: str) -> Path:
    f = tempfile.NamedTemporaryFile(suffix=".md", delete=False, mode="w")
    f.write(text)
    f.close()
    return Path(f.name)


def test_close_task_in_source_marks_done_with_date():
    note = _write_note("- [ ] finish the report ^dendr-aaa\n")
    assert close_task_in_source(note, "dendr-aaa", "x", "2026-06-26") is True
    assert note.read_text() == "- [x] finish the report ✅ 2026-06-26 ^dendr-aaa\n"


def test_close_task_in_source_cancelled_uses_dash_and_cross():
    note = _write_note("- [ ] maybe later ^dendr-bbb\n")
    assert close_task_in_source(note, "dendr-bbb", "-", "2026-06-26") is True
    assert note.read_text() == "- [-] maybe later ❌ 2026-06-26 ^dendr-bbb\n"


def test_close_task_in_source_preserves_indent_and_neighbors():
    note = _write_note(
        "# Tasks\n\n    - [ ] nested item ^dendr-ccc\n- [ ] other ^dendr-ddd\n"
    )
    assert close_task_in_source(note, "dendr-ccc", "x", "2026-06-26") is True
    lines = note.read_text().split("\n")
    assert lines[2] == "    - [x] nested item ✅ 2026-06-26 ^dendr-ccc"
    assert lines[3] == "- [ ] other ^dendr-ddd"  # untouched


def test_close_task_in_source_idempotent_when_already_closed():
    note = _write_note("- [x] done ✅ 2026-06-26 ^dendr-eee\n")
    assert close_task_in_source(note, "dendr-eee", "x", "2026-06-26") is False


def test_close_task_in_source_refreshes_stale_date():
    note = _write_note("- [ ] redo ✅ 2025-01-01 ^dendr-fff\n")
    assert close_task_in_source(note, "dendr-fff", "x", "2026-06-26") is True
    assert note.read_text() == "- [x] redo ✅ 2026-06-26 ^dendr-fff\n"


def test_close_task_in_source_missing_ref_returns_false():
    note = _write_note("- [ ] no ref here\n")
    assert close_task_in_source(note, "dendr-ghost", "x", "2026-06-26") is False


def test_close_task_in_source_missing_file_returns_false():
    assert (
        close_task_in_source(Path("/no/such/file.md"), "dendr-x", "x", "2026-06-26")
        is False
    )


def test_close_task_in_source_multiline_block_checkbox_above_ref():
    note = _write_note("- [ ] multi line task\n  continued line ^dendr-hhh\n")
    assert close_task_in_source(note, "dendr-hhh", "x", "2026-06-26") is True
    lines = note.read_text().split("\n")
    assert lines[0] == "- [x] multi line task ✅ 2026-06-26"
    assert lines[1] == "  continued line ^dendr-hhh"


def test_close_task_in_source_does_not_cross_block_boundary():
    # The ref'd block is prose with no checkbox; the walk-up must stop at the
    # blank line and NOT grab the unrelated task above.
    note = _write_note("- [ ] unrelated task\n\njust a note ^dendr-iii\n")
    assert close_task_in_source(note, "dendr-iii", "x", "2026-06-26") is False
    assert note.read_text() == "- [ ] unrelated task\n\njust a note ^dendr-iii\n"


def test_close_task_in_source_exact_block_id_match():
    # A different block whose id is a suffix of another must not be matched.
    note = _write_note("- [ ] a ^dendr-xwb\n- [ ] b ^dendr-wb\n")
    assert close_task_in_source(note, "dendr-wb", "x", "2026-06-26") is True
    lines = note.read_text().split("\n")
    assert lines[0] == "- [ ] a ^dendr-xwb"  # untouched
    assert lines[1] == "- [x] b ✅ 2026-06-26 ^dendr-wb"


# ── Fenced code blocks (F3) ───────────────────────────────────────────


def test_fenced_code_block_stays_one_block():
    """A blank line inside a ``` fence must not split it into pieces."""
    content = (
        "Debugging the deploy script:\n"
        "\n"
        "```bash\n"
        "export FOO=1\n"
        "\n"
        "./deploy.sh --prod\n"
        "```\n"
    )
    note = _write_note(content)
    blocks = parse_daily_note(note)
    # One prose block + exactly one code block (not three fragments).
    code_blocks = [b for b in blocks if "deploy.sh" in b.text]
    assert len(code_blocks) == 1
    assert "export FOO=1" in code_blocks[0].text
    assert "./deploy.sh --prod" in code_blocks[0].text


def test_inject_never_writes_id_inside_a_fence():
    """Block-ref injection must not land inside fenced code, and must not
    break the closing fence — it goes on its own line just after it."""
    content = "```bash\nexport FOO=1\n\n./deploy.sh --prod\n```\n"
    note = _write_note(content)
    blocks = parse_daily_note(note)
    inject_block_ids(note, blocks)
    out = note.read_text()

    # The code lines are untouched — no ^dendr-… injected mid-fence.
    assert "export FOO=1\n" in out
    assert "./deploy.sh --prod\n" in out
    assert "^dendr-" not in out.split("```", 2)[1]  # nothing inside the fence
    # A closing fence line is exactly ``` (no trailing ref appended).
    assert any(line.strip() == "```" for line in out.split("\n"))

    # Re-parsing round-trips: the ref is recognized, so no second id is added.
    blocks2 = parse_daily_note(note)
    injected_again = inject_block_ids(note, blocks2)
    assert injected_again is False
    assert note.read_text() == out
