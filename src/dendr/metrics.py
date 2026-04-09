"""Prometheus metrics for Dendr observability.

All metric definitions live here. Other modules import and use them.
The daemon exposes metrics via a standalone HTTP server on port 9100.
The search server exposes them on its existing FastAPI app at /metrics.
"""

from __future__ import annotations

import logging
from pathlib import Path

from prometheus_client import Counter, Gauge, Histogram, start_http_server

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
    "Time to load a model into GPU memory",
    ["model_role"],
    buckets=(1, 5, 10, 30, 60, 120, 300),
)

# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------
INFERENCE_SECONDS = Histogram(
    "dendr_inference_seconds",
    "LLM inference duration in seconds",
    ["model_role", "task"],
    buckets=(0.5, 1, 2, 5, 10, 30, 60, 120, 300),
)

INFERENCE_TOKENS = Counter(
    "dendr_inference_tokens_total",
    "Total tokens processed",
    ["model_role", "direction"],
)

INFERENCE_JSON_FAILURES = Counter(
    "dendr_inference_json_parse_failures_total",
    "Number of times LLM output failed JSON parsing",
    ["task"],
)

# ---------------------------------------------------------------------------
# Pipeline / queue
# ---------------------------------------------------------------------------
QUEUE_PENDING = Gauge(
    "dendr_queue_pending",
    "Number of blocks in the pending queue",
)

QUEUE_PROCESSING = Gauge(
    "dendr_queue_processing",
    "Number of blocks currently being processed",
)

BLOCKS_PROCESSED = Counter(
    "dendr_blocks_processed_total",
    "Total blocks successfully processed",
)

CLAIMS_EXTRACTED = Counter(
    "dendr_claims_extracted_total",
    "Total claims extracted from blocks",
)

CLAIMS_REINFORCED = Counter(
    "dendr_claims_reinforced_total",
    "Total claims reinforced (duplicate seen again)",
)

CONTRADICTIONS_DETECTED = Counter(
    "dendr_contradictions_detected_total",
    "Total contradictions detected between claims",
)

INGEST_CYCLE_SECONDS = Histogram(
    "dendr_ingest_cycle_seconds",
    "Duration of a full ingest cycle",
    buckets=(1, 5, 10, 30, 60, 120, 300, 600),
)

BACKPRESSURE_ACTIVE = Gauge(
    "dendr_backpressure_active",
    "1 if backpressure (shallow enrichment) mode is active",
)

# ---------------------------------------------------------------------------
# Canonicalization
# ---------------------------------------------------------------------------
CANONICALIZATION_REUSE = Counter(
    "dendr_canonicalization_reuse_total",
    "Times an existing concept slug was reused",
)

CANONICALIZATION_NEW = Counter(
    "dendr_canonicalization_new_total",
    "Times a new concept slug was created",
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
# Knowledge base gauges (updated periodically)
# ---------------------------------------------------------------------------
ACTIVE_CLAIMS = Gauge(
    "dendr_active_claims",
    "Total active claims in the knowledge base",
)

CONCEPTS_TOTAL = Gauge(
    "dendr_concepts_total",
    "Total concepts in the knowledge base",
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def start_metrics_server(port: int = 9100) -> None:
    """Start a standalone Prometheus metrics HTTP server (for the daemon)."""
    try:
        start_http_server(port)
        logger.info("Metrics server started on port %d", port)
    except OSError as e:
        logger.warning("Could not start metrics server on port %d: %s", port, e)


def collect_queue_metrics(config) -> None:
    """Read queue directories and update gauge values."""
    try:
        pending_dir: Path = config.pending_dir
        processing_dir: Path = config.processing_dir

        pending_count = (
            len(list(pending_dir.glob("*.json"))) if pending_dir.exists() else 0
        )
        processing_count = (
            len(list(processing_dir.glob("*.json"))) if processing_dir.exists() else 0
        )

        QUEUE_PENDING.set(pending_count)
        QUEUE_PROCESSING.set(processing_count)
    except Exception as e:
        logger.debug("Failed to collect queue metrics: %s", e)


def collect_db_metrics(conn) -> None:
    """Read database stats and update gauge values."""
    try:
        from dendr import db

        stats = db.get_stats(conn)
        ACTIVE_CLAIMS.set(stats.get("active_claims", 0))
        CONCEPTS_TOTAL.set(stats.get("concepts", 0))
    except Exception as e:
        logger.debug("Failed to collect DB metrics: %s", e)
