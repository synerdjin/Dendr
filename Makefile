.DEFAULT_GOAL := help

# Same venv scripts/update.sh manages and the launchd ingest agent runs from.
# Override with `make DENDR_VENV=~/other-venv <target>`.
DENDR_VENV ?= $(HOME)/.dendr-venv
DENDR      := $(DENDR_VENV)/bin/dendr
PYTEST     := $(DENDR_VENV)/bin/pytest
RUFF       := $(DENDR_VENV)/bin/ruff

# Points every `uv` invocation in this file at the shared venv above instead
# of uv's default project-local .venv.
export UV_PROJECT_ENVIRONMENT := $(DENDR_VENV)

.PHONY: help
help: ## Show this list
	@grep -hE '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}'

## --- Setup -----------------------------------------------------------------

.PHONY: install
install: ## Create the venv (if missing) and install Dendr + dev tools into it, editable, from uv.lock
	@command -v uv >/dev/null 2>&1 || { echo "error: uv not found on PATH (install: https://docs.astral.sh/uv/)" >&2; exit 1; }
	uv sync

.PHONY: update
update: ## Pull latest, refresh deps, verify models, restart the scheduled ingest agent
	DENDR_VENV=$(DENDR_VENV) scripts/update.sh

## --- Day to day --------------------------------------------------------------

.PHONY: ingest
ingest: ## Run a single ingest cycle
	$(DENDR) ingest

.PHONY: serve
serve: ## Start the search server on localhost:7777
	$(DENDR) serve

.PHONY: digest
digest: ## Generate the weekly digest + Claude synthesis prompt
	$(DENDR) digest --claude

.PHONY: stats
stats: ## Show knowledge base statistics
	$(DENDR) stats

## --- Models ------------------------------------------------------------------

.PHONY: models-pull
models-pull: ## Download all models from dendr-models.yaml
	$(DENDR) models pull

.PHONY: models-verify
models-verify: ## Check SHA256 integrity of downloaded models
	$(DENDR) models verify

.PHONY: models-list
models-list: ## Show model status table
	$(DENDR) models list

## --- Scheduled ingest (launchd) -------------------------------------------------

.PHONY: autostart-install
autostart-install: ## Run ingest on a schedule via a macOS LaunchAgent
	$(DENDR) autostart install

.PHONY: autostart-status
autostart-status: ## Show whether the login agent is installed / loaded
	$(DENDR) autostart status

.PHONY: autostart-uninstall
autostart-uninstall: ## Stop and remove the login agent
	$(DENDR) autostart uninstall

## --- Development ---------------------------------------------------------------

.PHONY: test
test: ## Run the test suite (same invocation as CI)
	$(PYTEST) tests/ -v

.PHONY: lint
lint: ## Lint (ruff check)
	$(RUFF) check src/ tests/

.PHONY: format
format: ## Auto-format (ruff format)
	$(RUFF) format src/ tests/

.PHONY: format-check
format-check: ## Check formatting without changing files (what CI runs)
	$(RUFF) format --check src/ tests/

.PHONY: check
check: lint format-check test ## Everything CI + the CLAUDE.md workflow rule require before pushing

.PHONY: clean
clean: ## Remove caches and build artifacts
	rm -rf .pytest_cache .ruff_cache build dist *.egg-info src/*.egg-info
	find . -name '__pycache__' -not -path './.venv/*' -exec rm -rf {} +
