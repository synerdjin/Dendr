"""Tests for the privacy filter."""

from dendr.models import Block
from dendr.privacy import is_private, filter_blocks


def _make_block(text: str) -> Block:
    return Block(
        block_id="test-1",
        source_file="test.md",
        line_start=0,
        line_end=0,
        text=text,
        block_hash="abc123",
    )


def test_detects_api_key():
    block = _make_block("api_key = 'sk-1234567890abcdefghijklmnop'")
    assert is_private(block)


def test_detects_aws_key():
    block = _make_block("The key is AKIAIOSFODNN7EXAMPLE")
    assert is_private(block)


def test_detects_github_token():
    block = _make_block("ghp_1234567890abcdefghijklmnopqrstuvwxyz")
    assert is_private(block)


def test_detects_private_key():
    block = _make_block("-----BEGIN RSA PRIVATE KEY-----")
    assert is_private(block)


def test_detects_connection_string():
    block = _make_block("postgres://user:password@host:5432/db")
    assert is_private(block)


def test_detects_ssn():
    block = _make_block("My SSN is 123-45-6789")
    assert is_private(block)


def test_detects_user_redact_tag():
    block = _make_block("Some sensitive info #dendr-private")
    assert is_private(block)


def test_detects_private_tag():
    block = _make_block("Secret stuff #private here")
    assert is_private(block)


def test_normal_text_not_private():
    block = _make_block("Today I learned about machine learning transformers")
    assert not is_private(block)


def test_filter_blocks_tags_in_place():
    blocks = [
        _make_block("Normal text"),
        _make_block("api_key = 'sk-supersecretkey12345678'"),
        _make_block("More normal text"),
    ]
    filtered = filter_blocks(blocks)
    assert len(filtered) == 3
    assert not filtered[0].private
    assert filtered[1].private
    assert not filtered[2].private
