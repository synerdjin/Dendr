.DEFAULT_GOAL := help

# Same venv scripts/update.sh manages and the launchd daemon runs from.
# Override with `make DENDR_VENV=~/other-venv <target>`.
DENDR_VENV ?= $(HOME)/.dendr-venv
DENDR      := $(DENDR_VENV)/bin/dendr
PYTHON     := $(DENDR_VENV)/bin/python
PIP        := $(DENDR_VENV)/bin/pip
PYTEST     := $(DENDR_VENV)/bin/pytest
RUFF       := $(DENDR_VENV)/bin/ruff

.PHONY: help
help: ## Show this list
	@grep -hE '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}'

## --- Setup -----------------------------------------------------------------

.PHONY: install
install: ## Create the venv (if missing) and install Dendr into it, editable
	@test -x "$(PYTHON)" || python3 -m venv "$(DENDR_VENV)"
	$(PIP) install -e . --quiet

.PHONY: update
update: ## Pull latest, refresh deps, verify models, restart the login daemon
	DENDR_VENV=$(DENDR_VENV) scripts/update.sh

## --- Day to day --------------------------------------------------------------

.PHONY: ingest
ingest: ## Run a single ingest cycle
	$(DENDR) ingest

.PHONY: daemon
daemon: ## Watch Daily/ and auto-ingest on changes (foreground)
	$(DENDR) daemon

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

## --- Login daemon (launchd) ---------------------------------------------------

.PHONY: autostart-install
autostart-install: ## Run the daemon on login via a macOS LaunchAgent
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
