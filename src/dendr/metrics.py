"""Prometheus metrics for Dendr observability.

All metric definitions live here. Other modules import and use them.
The search server exposes them on its existing FastAPI app at /metrics.
"""

from __future__ import annotations

import logging

from prometheus_client import Counter, Gauge, Histogram

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Model lifecycle
# ---------------------------------------------------------------------------
MODEL_LOADED = Gauge(
    "dendr_model_loaded",
    "Whether a model is currently loaded (1) or not (0)",
    ["model_role"],
)

MODEL_LOAD_SECONDS = Histogram(
    "dendr_model_load_seconds",
    "Time to load a model into memory",
    ["model_role"],
    buckets=(1, 5, 10, 30, 60, 120, 300),
)

# ---------------------------------------------------------------------------
# Embedding
# ---------------------------------------------------------------------------
EMBED_SECONDS = Histogram(
    "dendr_embed_seconds",
    "Embedding call duration in seconds",
    ["mode"],  # "single" | "batch"
    buckets=(0.1, 0.5, 1, 2, 5, 10, 30, 60, 120),
)

EMBED_THROUGHPUT = Gauge(
    "dendr_embed_blocks_per_sec",
    "Embedding throughput from the most recent ingest batch",
)

# ---------------------------------------------------------------------------
# Pipeline / queue
# ---------------------------------------------------------------------------
BLOCKS_PROCESSED = Counter(
    "dendr_blocks_processed_total",
    "Total blocks successfully committed",
)

BLOCKS_ERROR = Counter(
    "dendr_blocks_error_total",
    "Total blocks that failed to embed or commit",
)

INGEST_CYCLE_SECONDS = Histogram(
    "dendr_ingest_cycle_seconds",
    "Duration of a full ingest cycle",
    buckets=(1, 5, 10, 30, 60, 120, 300, 600),
)

# ---------------------------------------------------------------------------
# Task lifecycle
# ---------------------------------------------------------------------------
TASKS_CLOSED = Counter(
    "dendr_tasks_closed_total",
    "Total task closure events recorded",
    ["source"],  # "auto" (checkbox flip) | "user" (digest review)
)

# ---------------------------------------------------------------------------
# Search server
# ---------------------------------------------------------------------------
SEARCH_REQUEST_SECONDS = Histogram(
    "dendr_search_request_seconds",
    "Search request duration in seconds",
    ["mode"],
    buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1, 2, 5),
)

# ---------------------------------------------------------------------------
# Digest
# ---------------------------------------------------------------------------
DIGEST_RUNS = Counter(
    "dendr_digest_runs_total",
    "Total digest generation runs",
    ["mode"],  # "local" | "claude"
)
