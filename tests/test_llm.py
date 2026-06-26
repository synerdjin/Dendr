"""Tests for the embedding task-prompt layer (no model load required)."""

from dendr.llm import _format_for_embedding

GEMMA = "embeddinggemma-300m-qat-Q8_0.gguf"
NOMIC = "nomic-embed-text-v1.5.f16.gguf"


def test_gemma_query_prompt():
    assert (
        _format_for_embedding("hi", "query", GEMMA) == "task: search result | query: hi"
    )


def test_gemma_document_prompt():
    assert _format_for_embedding("hi", "document", GEMMA) == "title: none | text: hi"


def test_nomic_prompts():
    assert _format_for_embedding("hi", "query", NOMIC) == "search_query: hi"
    assert _format_for_embedding("hi", "document", NOMIC) == "search_document: hi"


def test_unknown_model_passthrough():
    assert _format_for_embedding("hi", "query", "mystery-model.gguf") == "hi"
