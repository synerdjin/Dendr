"""Tests for the Pages/ digital garden feature."""

from dendr import garden
from dendr.config import Config
from dendr.frontmatter import render_frontmatter, split_frontmatter


def _config(tmp_path):
    config = Config(vault_path=tmp_path, data_dir=tmp_path / "data")
    config.pages_dir.mkdir(parents=True, exist_ok=True)
    config.wiki_dir.mkdir(parents=True, exist_ok=True)
    return config


def _write_page(config, name, text):
    path = config.pages_dir / name
    path.write_text(text, encoding="utf-8")
    return path


# ── frontmatter split/render ───────────────────────────────────────────


def test_split_frontmatter_present():
    text = "---\nstage: evergreen\nplanted: '2026-01-01'\n---\n\nBody text.\n"
    meta, body = split_frontmatter(text)
    assert meta == {"stage": "evergreen", "planted": "2026-01-01"}
    assert body == "\nBody text.\n"


def test_split_frontmatter_absent():
    text = "Just a note, no frontmatter.\n"
    meta, body = split_frontmatter(text)
    assert meta == {}
    assert body == text


def test_split_frontmatter_unquoted_date_becomes_date_object():
    """A bare `key: 2026-01-01` is parsed by YAML as a date, not a string —
    garden.py's callers must coerce this back to a string themselves."""
    import datetime

    meta, _body = split_frontmatter("---\nplanted: 2026-01-01\n---\nBody\n")
    assert meta["planted"] == datetime.date(2026, 1, 1)


def test_render_frontmatter_round_trip():
    meta = {"stage": "budding", "tended": "2026-04-01"}
    rendered = render_frontmatter(meta, "Body content.\n")
    parsed_meta, parsed_body = split_frontmatter(rendered)
    assert parsed_meta == meta
    assert parsed_body == "Body content.\n"


def test_render_frontmatter_empty_meta_is_noop():
    assert render_frontmatter({}, "just body\n") == "just body\n"


def test_split_frontmatter_strips_block_even_when_yaml_is_malformed():
    """Regression: if a note's frontmatter fails to parse, the delimited
    block must still be stripped from body — otherwise re-rendering
    (meta={}, body) via render_frontmatter prepends a fresh frontmatter
    block onto the old broken one still sitting in body, stacking two
    `---` blocks in the file."""
    text = "---\nstage: [unterminated\n---\n\nReal body.\n"
    meta, body = split_frontmatter(text)
    assert meta == {}
    assert body == "\nReal body.\n"
    assert "---" not in body


def test_split_frontmatter_strips_block_when_yaml_is_not_a_mapping():
    text = "---\n- just\n- a\n- list\n---\n\nReal body.\n"
    meta, body = split_frontmatter(text)
    assert meta == {}
    assert body == "\nReal body.\n"


# ── scan_pages ──────────────────────────────────────────────────────────


def test_scan_pages_missing_dir(tmp_path):
    config = Config(vault_path=tmp_path, data_dir=tmp_path / "data")
    assert garden.scan_pages(config) == []


def test_scan_pages_reads_existing_frontmatter(tmp_path):
    config = _config(tmp_path)
    _write_page(
        config,
        "stoicism.md",
        "---\nstage: evergreen\nplanted: '2025-01-01'\ntended: '2026-04-01'\n"
        "---\n\n# Stoicism is a practice, not a philosophy\n\nSome body text.\n",
    )
    notes = garden.scan_pages(config)
    assert len(notes) == 1
    note = notes[0]
    assert note.stage == "evergreen"
    assert note.planted == "2025-01-01"
    assert note.tended == "2026-04-01"
    assert note.title == "Stoicism is a practice, not a philosophy"
    assert note.slug == "stoicism"


def test_scan_pages_coerces_unquoted_date_to_string(tmp_path):
    config = _config(tmp_path)
    _write_page(
        config,
        "loose-dates.md",
        "---\nstage: seedling\ntended: 2026-04-01\n---\nBody\n",
    )
    notes = garden.scan_pages(config)
    assert notes[0].tended == "2026-04-01"
    assert isinstance(notes[0].tended, str)


def test_scan_pages_guesses_stage_when_absent(tmp_path):
    config = _config(tmp_path)
    _write_page(config, "tiny.md", "Just a couple words.\n")
    _write_page(
        config,
        "meaty.md",
        "# Meaty note\n\n"
        + ("word " * 400)
        + "\n[[a]] [[b]] [[c]] [[d]] [[e]] [[f]]\n",
    )
    notes = {n.slug: n for n in garden.scan_pages(config)}
    assert notes["tiny"].stage == "seedling"
    assert notes["meaty"].stage == "evergreen"


def test_scan_pages_never_overrides_explicit_stage(tmp_path):
    """A hand-set stage always wins, even one the heuristic would disagree with."""
    config = _config(tmp_path)
    _write_page(config, "short-but-evergreen.md", "---\nstage: evergreen\n---\nTiny.\n")
    notes = garden.scan_pages(config)
    assert notes[0].stage == "evergreen"


def test_scan_pages_skips_sync_conflict_copies(tmp_path):
    """Pages/ has no block-ref mechanism to dedupe on (unlike Daily/), so a
    conflict-shaped filename is skipped outright rather than becoming its own
    spurious page."""
    config = _config(tmp_path)
    _write_page(config, "stoicism.md", "---\nstage: evergreen\n---\nReal note.\n")
    _write_page(config, "stoicism (1).md", "---\nstage: evergreen\n---\nStale copy.\n")
    notes = garden.scan_pages(config)
    assert [n.slug for n in notes] == ["stoicism"]


def test_backfill_frontmatter_skips_sync_conflict_copies(tmp_path):
    config = _config(tmp_path)
    _write_page(config, "note (1).md", "No frontmatter.\n")
    assert garden.backfill_frontmatter(config) == []


def test_scan_pages_warns_when_skipping_conflict_shaped_name(tmp_path, caplog):
    """A false-positive match (a real page title that happens to look
    conflict-shaped) silently vanishing from the garden would be worse than
    a log line — this makes the skip discoverable."""
    config = _config(tmp_path)
    _write_page(config, "Draft (2).md", "A real page, not a sync artifact.\n")
    with caplog.at_level("WARNING"):
        notes = garden.scan_pages(config)
    assert notes == []
    assert "Draft (2).md" in caplog.text


# ── backfill_frontmatter ──────────────────────────────────────────────


def test_backfill_fills_only_missing_keys(tmp_path):
    config = _config(tmp_path)
    _write_page(config, "partial.md", "---\nstage: budding\n---\nBody.\n")

    changes = garden.backfill_frontmatter(config)
    assert len(changes) == 1
    path, added = changes[0]
    assert "stage" not in added  # already set — never touched
    assert "planted" in added
    assert "tended" in added

    meta, _body = split_frontmatter(path.read_text(encoding="utf-8"))
    assert meta["stage"] == "budding"


def test_backfill_dry_run_writes_nothing(tmp_path):
    config = _config(tmp_path)
    path = _write_page(config, "note.md", "No frontmatter at all.\n")
    original = path.read_text(encoding="utf-8")

    changes = garden.backfill_frontmatter(config, dry_run=True)
    assert len(changes) == 1
    assert path.read_text(encoding="utf-8") == original


def test_backfill_is_idempotent(tmp_path):
    config = _config(tmp_path)
    _write_page(config, "note.md", "No frontmatter at all.\n")

    first = garden.backfill_frontmatter(config)
    assert len(first) == 1

    second = garden.backfill_frontmatter(config)
    assert second == []


def test_backfill_missing_pages_dir(tmp_path):
    config = Config(vault_path=tmp_path, data_dir=tmp_path / "data")
    assert garden.backfill_frontmatter(config) == []


# ── compute_garden_summary ────────────────────────────────────────────


def _note(slug, stage, tended, planted="2026-01-01"):
    from pathlib import Path

    return garden.PageNote(
        path=Path(f"{slug}.md"),
        title=slug,
        stage=stage,
        planted=planted,
        tended=tended,
        word_count=10,
        link_count=1,
    )


def test_compute_garden_summary_empty_pages():
    assert (
        garden.compute_garden_summary([], "2026-04-01", "2026-04-08", "2026-W15") == {}
    )


def test_compute_garden_summary_counts():
    pages = [
        _note("a", "seedling", "2026-04-05"),
        _note("b", "budding", "2026-04-06"),
        _note("c", "evergreen", "2026-01-01"),
        _note("d", "evergreen", "2026-03-01"),
    ]
    summary = garden.compute_garden_summary(
        pages, "2026-04-01", "2026-04-08", "2026-W15"
    )
    assert summary["total"] == 4
    assert summary["counts"] == {"seedling": 1, "budding": 1, "evergreen": 2}


def test_compute_garden_summary_stalest_ordering():
    pages = [
        _note("old", "evergreen", "2026-01-01"),
        _note("newer", "evergreen", "2026-03-01"),
    ]
    summary = garden.compute_garden_summary(
        pages, "2026-04-01", "2026-04-08", "2026-W15"
    )
    assert [p["slug"] for p in summary["stalest_evergreens"]] == ["old", "newer"]


def test_compute_garden_summary_tended_this_period_filters_by_window():
    pages = [
        _note("in-window", "budding", "2026-04-05"),
        _note("before-window", "budding", "2026-03-01"),
    ]
    summary = garden.compute_garden_summary(
        pages, "2026-04-01", "2026-04-08", "2026-W15"
    )
    tended_slugs = [p["slug"] for p in summary["tended_this_period"]]
    assert tended_slugs == ["in-window"]


def test_compute_garden_summary_resurface_is_stable_within_a_week():
    pages = [_note(f"page-{i}", "seedling", "2026-04-01") for i in range(10)]
    first = garden.compute_garden_summary(pages, "2026-04-01", "2026-04-08", "2026-W15")
    second = garden.compute_garden_summary(
        pages, "2026-04-01", "2026-04-08", "2026-W15"
    )
    assert first["resurface"] == second["resurface"]
    assert len(first["resurface"]) == garden.RESURFACE_COUNT


# ── rendering ───────────────────────────────────────────────────────────


def test_render_garden_dashboard_empty():
    rendered = garden.render_garden_dashboard([], {})
    assert "# Garden" in rendered
    assert "No `Pages/` notes found" in rendered


def test_render_garden_dashboard_with_pages():
    pages = [
        _note("old-evergreen", "evergreen", "2026-01-01"),
        _note("fresh-seed", "seedling", "2026-04-05"),
    ]
    summary = garden.compute_garden_summary(
        pages, "2026-04-01", "2026-04-08", "2026-W15"
    )
    rendered = garden.render_garden_dashboard(pages, summary)
    assert "seedling" in rendered
    assert "evergreen" in rendered
    assert "[[old-evergreen]]" in rendered
    assert "[[fresh-seed]]" in rendered
    assert "Stalest evergreens" in rendered
    assert "Grown this period" in rendered


def test_render_garden_digest_section_empty():
    assert garden.render_garden_digest_section({}) == ""


def test_render_garden_digest_section_with_data():
    pages = [
        _note("old-evergreen", "evergreen", "2026-01-01"),
        _note("fresh-seed", "seedling", "2026-04-05"),
    ]
    summary = garden.compute_garden_summary(
        pages, "2026-04-01", "2026-04-08", "2026-W15"
    )
    rendered = garden.render_garden_digest_section(summary)
    assert "## Garden" in rendered
    assert "[[old-evergreen]]" in rendered
    assert "Stalest evergreen" in rendered
