"""File watcher daemon — monitors Vault/Daily/ for changes and triggers ingest.

Uses watchdog for cross-platform filesystem events.
Debounces rapid changes to avoid re-processing mid-edit.
"""

from __future__ import annotations

import logging
import sqlite3
import threading
import time
from pathlib import Path

from watchdog.events import FileSystemEvent, FileSystemEventHandler
from watchdog.observers import Observer

from dendr.config import Config
from dendr.db import connect, init_schema
from dendr.llm import LLMClient
from dendr.pipeline import run_ingest

logger = logging.getLogger(__name__)


class _DebouncedHandler(FileSystemEventHandler):
    """Collects file events and triggers ingest after a quiet period."""

    def __init__(self, config: Config, debounce_seconds: float = 5.0):
        self.config = config
        self.debounce_seconds = debounce_seconds
        self._dirty = False
        self._lock = threading.Lock()
        self._timer: threading.Timer | None = None

    def _schedule_ingest(self) -> None:
        with self._lock:
            if self._timer:
                self._timer.cancel()
            self._timer = threading.Timer(self.debounce_seconds, self._run_ingest)
            self._timer.daemon = True
            self._timer.start()

    def _run_ingest(self) -> None:
        with self._lock:
            self._dirty = False
        try:
            logger.info("Watcher triggered ingest cycle")
            conn = connect(self.config.db_path)
            init_schema(conn)
            llm = LLMClient(self.config)
            stats = run_ingest(self.config, conn, llm)
            logger.info("Ingest complete: %s", stats)
            conn.close()
        except Exception as e:
            logger.error("Ingest failed: %s", e, exc_info=True)

    def on_modified(self, event: FileSystemEvent) -> None:
        if event.is_directory:
            return
        if not event.src_path.endswith(".md"):
            return
        logger.debug("File modified: %s", event.src_path)
        self._schedule_ingest()

    def on_created(self, event: FileSystemEvent) -> None:
        if event.is_directory:
            return
        if not event.src_path.endswith(".md"):
            return
        logger.debug("File created: %s", event.src_path)
        self._schedule_ingest()


def run_daemon(config: Config) -> None:
    """Run the watcher daemon. Blocks until interrupted."""
    daily_dir = config.daily_dir
    if not daily_dir.exists():
        daily_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Starting Dendr daemon, watching: %s", daily_dir)

    # Run an initial ingest on startup
    try:
        conn = connect(config.db_path)
        init_schema(conn)
        llm = LLMClient(config)
        stats = run_ingest(config, conn, llm)
        logger.info("Startup ingest: %s", stats)
        conn.close()
    except Exception as e:
        logger.error("Startup ingest failed: %s", e, exc_info=True)

    handler = _DebouncedHandler(config, debounce_seconds=5.0)
    observer = Observer()
    observer.schedule(handler, str(daily_dir), recursive=False)
    observer.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("Daemon shutting down...")
        observer.stop()
    observer.join()
