"""CLI behavior tests (click-level, no real inference)."""

from __future__ import annotations

import json

from click.testing import CliRunner

from dendr.cli import main


def test_ingest_vault_flag_overrides_saved_config(tmp_path, monkeypatch):
    """`dendr ingest --vault <path>` uses the CLI vault, not config.json's."""
    host_vault = tmp_path / "host-vault"
    host_vault.mkdir()
    cli_vault = tmp_path / "cli-vault"
    (cli_vault / "Daily").mkdir(parents=True)

    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "config.json").write_text(
        json.dumps(
            {
                "vault_path": str(host_vault),
                "vault_id": "test-id",
                "models": {},
                "search_port": 7777,
            }
        ),
        encoding="utf-8",
    )

    seen: dict = {}

    class FakeLLM:
        def __init__(self, config):
            seen["config_vault"] = config.vault_path

    def fake_run_ingest(config, conn, llm):
        return {"blocks_processed": 0}

    monkeypatch.setattr("dendr.llm.LLMClient", FakeLLM)
    monkeypatch.setattr("dendr.pipeline.run_ingest", fake_run_ingest)

    result = CliRunner().invoke(
        main,
        ["ingest", "--data-dir", str(data_dir), "--vault", str(cli_vault)],
    )

    assert result.exit_code == 0, result.output
    assert seen["config_vault"] == cli_vault.resolve()


def test_ingest_without_vault_flag_uses_saved_config(tmp_path, monkeypatch):
    """Without --vault, ingest falls back to the saved config.json path."""
    saved_vault = tmp_path / "saved-vault"
    (saved_vault / "Daily").mkdir(parents=True)

    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "config.json").write_text(
        json.dumps(
            {
                "vault_path": str(saved_vault),
                "vault_id": "test-id",
                "models": {},
                "search_port": 7777,
            }
        ),
        encoding="utf-8",
    )

    seen: dict = {}

    class FakeLLM:
        def __init__(self, config):
            seen["config_vault"] = config.vault_path

    def fake_run_ingest(config, conn, llm):
        return {"blocks_processed": 0}

    monkeypatch.setattr("dendr.llm.LLMClient", FakeLLM)
    monkeypatch.setattr("dendr.pipeline.run_ingest", fake_run_ingest)

    result = CliRunner().invoke(main, ["ingest", "--data-dir", str(data_dir)])

    assert result.exit_code == 0, result.output
    assert seen["config_vault"] == saved_vault
