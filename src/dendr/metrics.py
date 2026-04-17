"""Prometheus metrics for Dendr observability.

All metric definitions live here. Other modules import and use them.
The daemon exposes metrics via a standalone HTTP server on port 9100.
The search server exposes them on its existing FastAPI app at /metrics.
"""

from __future__ import annotations

import logging

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

INGEST_CYCLE_SECONDS = Histogram(
    "dendr_ingest_cycle_seconds",
    "Duration of a full ingest cycle",
    buckets=(1, 5, 10, 30, 60, 120, 300, 600),
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
BLOCKS_TOTAL = Gauge(
    "dendr_blocks_total",
    "Total blocks in the knowledge base",
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
    from dendr import queue

    try:
        QUEUE_PENDING.set(queue.pending_count(config))
        QUEUE_PROCESSING.set(queue.processing_count(config))
    except Exception:
        logger.debug("Failed to collect queue metrics", exc_info=True)


def collect_db_metrics(conn) -> None:
    """Read database stats and update gauge values."""
    try:
        from dendr import db

        stats = db.get_stats(conn)
        BLOCKS_TOTAL.set(stats.get("blocks", 0))
    except Exception:
        logger.debug("Failed to collect DB metrics", exc_info=True)
