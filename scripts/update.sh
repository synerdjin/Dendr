#!/usr/bin/env bash
#
# Update a local Dendr install after pulling new changes.
#
# Because Dendr is installed editable (via `uv sync`), pure-Python changes
# take effect on `git pull` alone. This script handles the cases that need
# more than that: new dependencies (resolved from uv.lock), model-manifest
# changes, and restarting the long-lived launchd daemon so it runs the new
# code.
#
# Usage:
#   scripts/update.sh                 # pull + reinstall deps + restart daemon
#   DENDR_VENV=~/.dendr-venv scripts/update.sh
#
set -euo pipefail

# --- Resolve paths -----------------------------------------------------------
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="${DENDR_VENV:-$HOME/.dendr-venv}"
DENDR="$VENV/bin/dendr"
LABEL="com.dendr.daemon"

cd "$REPO_DIR"

if ! command -v uv >/dev/null 2>&1; then
  echo "error: uv not found on PATH (install: https://docs.astral.sh/uv/)" >&2
  exit 1
fi

if [[ ! -x "$DENDR" ]]; then
  echo "error: no venv at $VENV (set DENDR_VENV to override; run 'make install' first)" >&2
  exit 1
fi

# --- 1. Pull latest code -----------------------------------------------------
echo "==> git pull (fast-forward only)"
before="$(git rev-parse HEAD)"
git pull --ff-only
after="$(git rev-parse HEAD)"

if [[ "$before" == "$after" ]]; then
  echo "    already up to date ($after)"
else
  echo "    $before -> $after"
fi

# --- 2. Reinstall (picks up new deps / entry points; cheap if unchanged) -----
# Installs the dev group too (ruff, pytest) since this venv also backs
# `make check` — same venv, dev and runtime aren't split for a single-user tool.
echo "==> uv sync (refresh dependencies from uv.lock)"
UV_PROJECT_ENVIRONMENT="$VENV" uv sync

# --- 3. Verify models; pull if the manifest changed --------------------------
echo "==> dendr models verify"
if ! "$DENDR" models verify; then
  echo "    model mismatch — pulling"
  "$DENDR" models pull
fi

# --- 4. Restart the daemon so it runs the new code ---------------------------
if launchctl list "$LABEL" >/dev/null 2>&1; then
  echo "==> restarting daemon ($LABEL)"
  launchctl kickstart -k "gui/$(id -u)/$LABEL"
  echo "    daemon restarted"
else
  echo "==> daemon not loaded (skip restart)"
  echo "    start it with: dendr autostart install"
fi

echo "==> done"
