#!/usr/bin/env bash
#
# Update a local Dendr install after pulling new changes.
#
# Because Dendr is installed editable (via `uv sync`), pure-Python changes
# take effect on `git pull` alone. This script handles the cases that need
# more than that: new dependencies (resolved from uv.lock), model-manifest
# changes, and kicking the scheduled launchd ingest agent so its next run
# picks up the new code.
#
# Usage:
#   scripts/update.sh                 # pull + reinstall deps + restart agent
#   DENDR_VENV=~/.dendr-venv scripts/update.sh
#
set -euo pipefail

# --- Resolve paths -----------------------------------------------------------
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="${DENDR_VENV:-$HOME/.dendr-venv}"
DENDR="$VENV/bin/dendr"
LABEL="com.dendr.ingest"
SERVE_LABEL="com.dendr.serve"

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

# --- 4. Kick the agents so they pick up the new code -------------------------
kick_agent() {
  local label="$1" name="$2" install_hint="$3"
  if launchctl list "$label" >/dev/null 2>&1; then
    echo "==> restarting $name agent ($label)"
    launchctl kickstart -k "gui/$(id -u)/$label"
    echo "    agent restarted"
  else
    echo "==> $name agent not loaded (skip restart)"
    echo "    start it with: $install_hint"
  fi
}

kick_agent "$LABEL" "ingest" "dendr autostart install"
kick_agent "$SERVE_LABEL" "search server" "dendr autostart install-serve"

echo "==> done"
