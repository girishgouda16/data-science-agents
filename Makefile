.PHONY: setup run stop dev up ui-build mlflow docker docker-start docker-stop user test test-e2e quick eval lint swarm swarm-status swarm-stop ollama-pull ollama-list migrate migration clean package

# ── Setup ─────────────────────────────────────────────────────────────────────
# Installs the gateway's deps (root requirements.txt) plus each A2A agent's
# own requirements.txt — they're independently deployable, so each pins its
# own dependencies rather than sharing this file. Plus ui/'s npm deps — the
# real UI (ui/src, a React app) needs a build step, unlike the old static
# ui/legacy.html fallback.
setup:
	python -m pip install -r requirements.txt
	python -m pip install -r classification-agent/requirements.txt
	python -m pip install -r regression-agent/requirements.txt
	python -m pip install -r clustering-agent/requirements.txt
	python -m pip install -r anomaly-agent/requirements.txt
	python -m pip install -r visualization-agent/requirements.txt
	python -m pip install -r forecasting-agent/requirements.txt
	python -m pip install -r drift-agent/requirements.txt
	python -m pip install -r serving-agent/requirements.txt
	python -m pip install -r explain-agent/requirements.txt
	python -m pip install -r orchestrator-agent/requirements.txt
	cd ui && npm install
	@# A new .env gets random secrets: the example's devkey and JWT_SECRET are public.
	@[ -f .env ] || sed -e "s/=devkey$$/=$$(python -c 'import secrets; print(secrets.token_urlsafe(32))')/" \
		-e "s/^JWT_SECRET=.*/JWT_SECRET=$$(python -c 'import secrets; print(secrets.token_urlsafe(32))')/" \
		.env.example > .env
	$(MAKE) migrate
	@echo "\n✓  Setup complete. Set your LLM key in .env, then:"
	@echo "   make docker-start && make up"

# ── Database ──────────────────────────────────────────────────────────────────
# Applies pending Alembic migrations. Run once after setup and after every
# pull that touches core/db.py — the app refuses to start against a schema
# that hasn't been migrated (see core/db.py init_db()).
migrate:
	alembic upgrade head

# Autogenerates a new revision from changes to core/db.py's table defs.
# Usage: make migration m="add foo column"
migration:
	alembic revision --autogenerate -m "$(m)"

# ── Run server ────────────────────────────────────────────────────────────────
# gateway/app.py is a thin REST+auth layer with no LLM/ML code of its own —
# its /api/v1/sessions* routes proxy to the orchestrator (:9100), and its
# /api/v1/upload is what the chat UI's attach button actually uses. Start all
# ten agents first (`make swarm` or use `make up`, which starts everything).
#
# Clears a stale process on 9001 first — `uvicorn.run(reload=False)` means a
# leftover background process from a previous run/session is the most common
# reason `make run` fails with "address already in use". (This bites the A2A
# agents too, on their own ports — if one seems stuck, check `lsof -ti:<port>`
# before assuming your restart took effect.)
run: stop migrate
	@echo "API token for curl/testing: $$(python -m scripts.print_token)"
	python -m gateway.app

stop:
	@lsof -ti:9001 | xargs kill -9 2>/dev/null || true
	@lsof -ti:8080 | xargs kill -9 2>/dev/null || true
	@echo "Stopped anything on 9001 (gateway) / 8080 (ui)."

# Gateway + UI only (scripts/dev.sh). Does not start agents or MLflow — use
# `make up` for the full stack, or run `make swarm` in another terminal first.
dev:
	@bash scripts/dev.sh

# ── Agent swarm ───────────────────────────────────────────────────────────────
# Starts all ten A2A agents, then REFUSES to report ready until each one both
# serves its agent card and accepts the shared bearer token from .env. That
# second check is the point: with the *_AGENT_API_KEY vars unset, every agent
# invents its own key at startup and every delegation 401s — which the
# orchestrator turns into a tool-result string the LLM narrates around, so the
# chat shows hand-offs that never happened. Catch it here, not mid-demo.
swarm:
	@bash scripts/swarm.sh

swarm-status:
	@bash scripts/swarm.sh status

swarm-stop:
	@bash scripts/swarm.sh stop

# The agents and services — all ten agents (stale ports cleared, health-checked),
# MLflow on MLFLOW_PORT, the gateway serving the built UI on :9001 — then it
# prints where to go and opens the browser; you use the UI from there.
# No Docker here (make docker-start). UI_DEV=1 for the Vite dev server.
# Ctrl+C stops it. Details: scripts/dev-all.sh.
up:
	@bash scripts/dev-all.sh

# The production UI bundle the gateway serves (make up builds it too).
ui-build:
	cd ui && { [ -d node_modules ] || npm install; } && npm run build

# MLflow UI on MLFLOW_PORT from .env (default 5001), over the registry every agent logs to.
mlflow:
	@bash scripts/mlflow.sh

# ── Docker services (separate from `make up`) ────────────────────────────────
# Keycloak on :8180 (sign-in, realm ml-agents from deploy/keycloak) and
# Langfuse on :3000 (traces), plus the stores they need. docker-start waits
# until Keycloak serves the realm and creates the UI admin (admin/admin, or
# ML_ADMIN_USER / ML_ADMIN_PASSWORD from .env) if it does not exist yet.
# Start these before `make up` and sign-in goes through Keycloak.
docker-start:
	@bash scripts/docker.sh start

docker-stop:
	@bash scripts/docker.sh stop

docker: docker-start

# Add a sign-in account: make user u=alice [admin=1] [p=<password>]
user:
	@bash scripts/keycloak-user.sh "$(u)" "$(admin)" "$(p)"

# ── Tests ─────────────────────────────────────────────────────────────────────
# No live agent and no LLM needed anywhere here — each agent's
# test_skill_gating.py spawns its own MCP subprocess and test_mcp_server/
# calls the tool functions directly.
#
# pytest per agent directory, rather than naming individual files. Naming
# them meant this target silently ran a SUBSET: classification-agent has
# eight test files and this used to invoke two of them, so the readiness-gate
# tests, the split-integrity tests and the positive-class regression tests
# never ran in CI at all. A discovery run cannot go stale when a file is
# added.
#
# Each agent runs from its own directory because each is its own deployable
# with its own requirements.txt — this assumes one env has all of them
# installed (true after `make setup`).
# The test_mcp_server.py files (and a few others) are scripts with a main(),
# not pytest functions — pytest collects nothing from them, so they run as
# scripts below. Without that they never ran under `make test` at all.
SCRIPT_TESTS = classification-agent/scripts/test_gates.py classification-agent/scripts/test_mcp_server.py \
	classification-agent/scripts/test_registry_readiness.py regression-agent/scripts/test_mcp_server.py \
	clustering-agent/scripts/test_mcp_server.py anomaly-agent/scripts/test_mcp_server.py \
	visualization-agent/scripts/test_mcp_server.py forecasting-agent/scripts/test_mcp_server.py \
	drift-agent/scripts/test_mcp_server.py serving-agent/scripts/test_mcp_server.py
test:
	python core/test_gates.py
	python core/test_tracing.py
	python core/test_registry.py
	python -m pytest gateway/test_gateway.py -q
	cd classification-agent/scripts && python -m pytest . -q
	cd regression-agent/scripts && python -m pytest . -q
	cd clustering-agent/scripts && python -m pytest . -q
	cd anomaly-agent/scripts && python -m pytest . -q
	cd visualization-agent/scripts && python -m pytest . -q
	cd forecasting-agent/scripts && python -m pytest . -q
	cd drift-agent/scripts && python -m pytest . -q
	cd serving-agent/scripts && python -m pytest . -q
	cd explain-agent/scripts && python -m pytest . -q
	cd orchestrator-agent && python -m pytest . -q
	@for t in $(SCRIPT_TESTS); do echo "== $$t"; (cd $$(dirname $$t) && python $$(basename $$t)) || exit 1; done
	python -m pytest scripts/test_swarm_e2e.py scripts/test_mlops_all_agents.py scripts/test_multi_user.py \
		scripts/test_tool_boundary.py scripts/test_mcp_servers_serve_every_tool.py \
		scripts/test_pickle_classes_in_sync.py -q
	cd ui && npm run lint && npm test && npm run build

# Fast gate (~1 min): same shape as CI `quick` — no full agent pytest sweep.
quick:
	python -m compileall -q -x 'node_modules|/dist/|mlruns|\.venv' .
	python core/test_gates.py
	cd ui && npm run lint && npm test

# Every agent's tests pass in isolation; the bugs that shipped lived in the
# SEAMS between them (a chart plotting a different class from the report that
# embedded it, a tuned threshold serving dropped on the floor). This drives
# one dataset through the whole chain and asserts only the claims that cross
# an agent boundary. No network, no LLM — the tool functions are called
# directly, so it tests the contracts, not a model's choice of tool.
test-e2e:
	python -m pytest scripts/test_swarm_e2e.py -v

# LLM behaviour evaluation (costs API calls). Optional: eval AGENT=classification
eval:
	@agents="classification regression clustering anomaly forecasting drift"; \
	if [ -n "$(AGENT)" ]; then agents="$(AGENT)"; fi; \
	failed=""; \
	for a in $$agents; do \
	  echo "== $$a-agent eval"; \
	  (cd $$a-agent/scripts && python eval_agent.py) || failed="$$failed $$a"; \
	done; \
	if [ -n "$$failed" ]; then echo "eval failed:$$failed"; exit 1; fi

# ── Ollama ────────────────────────────────────────────────────────────────────
ollama-pull:
	ollama pull qwen2.5:7b
	@echo "Alternative (larger, better reasoning): ollama pull llama3.1:8b"

ollama-list:
	ollama list

# ── Lint ──────────────────────────────────────────────────────────────────────
# Only the UI has a linter configured (oxlint, via ui/package.json). No
# Python linter is set up anywhere in this repo — add one before wiring it
# in here, rather than pointing this at a tool that isn't actually installed.
lint:
	cd ui && npm run lint

# ── Clean ─────────────────────────────────────────────────────────────────────
clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true

# ── Package for deployment ────────────────────────────────────────────────────
# Tarball of the working tree as-is (uncommitted changes included — this
# packages what you're about to run, not what's committed) for copying to a
# server. Excludes .git, secrets (.env), and everything already gitignored
# (venvs, caches, node_modules, datasets, DB/model artifacts, logs) — see
# .gitignore for the authoritative list this mirrors.
PKG_NAME    ?= agentic-ml
PKG_VERSION ?= $(shell date +%Y%m%d%H%M%S)
package:
	@mkdir -p dist
	tar -czf dist/$(PKG_NAME)-$(PKG_VERSION).tar.gz \
		--exclude='./.git' \
		--exclude='./.gitignore' \
		--exclude='./dist' \
		--exclude='__pycache__' \
		--exclude='*.pyc' --exclude='*.pyo' \
		--exclude='./.pytest_cache' --exclude='./.ruff_cache' \
		--exclude='*.egg-info' --exclude='./build' \
		--exclude='.DS_Store' \
		--exclude='./notebooks/.ipynb_checkpoints' \
		--exclude='./.idea' --exclude='./.vscode' --exclude='*.swp' \
		--exclude='./.venv' --exclude='./venv' \
		--exclude='./mlruns' --exclude='*.ckpt' --exclude='*.pt' --exclude='*.pkl' --exclude='*.joblib' \
		--exclude='*.log' --exclude='./logs' \
		--exclude='./data/*.db' --exclude='./data/artifacts' \
		--exclude='./data/runs' --exclude='./data/uploads' \
		--exclude='./data/*.csv' --exclude='./data/*.parquet' --exclude='./data/*.feather' \
		--exclude='./.env' \
		--exclude='./ui/node_modules' --exclude='./ui/dist' --exclude='./ui/.env' \
		-C .. "$$(basename "$$(pwd)")"
	@echo "Package created: dist/$(PKG_NAME)-$(PKG_VERSION).tar.gz"
	@echo "Note: datasets (data/*.csv etc.) and .env are excluded — copy those to the server separately."
