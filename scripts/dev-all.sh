#!/usr/bin/env bash
# `make up` — the agents and services, then you use the UI:
#   agents     :9000–:9900 + orchestrator :9100, via scripts/swarm.sh — clears
#              stale processes on those ports first (else a restart silently keeps
#              serving OLD code), then health-checks every agent; logs in logs/<agent>.log
#   MLflow     :MLFLOW_PORT (.env, default 5001)   log in logs/mlflow.log
#   gateway    :9001   serves the built UI, the API and charts — http://localhost:9001
#   UI_DEV=1   instead runs the Vite dev server on :5173 with hot reload
# When everything answers it prints where to go and opens the browser
# (NO_BROWSER=1 to skip). Ctrl+C stops it all except MLflow (reused next time).
#
# Docker services are NOT started here — `make docker-start` / `make docker-stop`.
# If Keycloak is already running (or OIDC_ISSUER is set in .env), sign-in goes
# through it and agents require a signed user; otherwise dev tokens.
#
# Model choice and every *_AGENT_API_KEY come from .env. Keys missing there
# get ONE random key for this run, shared by every process started here — they
# must MATCH (a per-agent auto-generated key 401s every orchestrator hand-off)
# but must not be guessable: whoever holds an agent key can claim any user.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs

# Python: the active env, else the repo's .venv, else the agentic-ml conda env.
if [ -z "${VIRTUAL_ENV:-}" ] && [ "${CONDA_DEFAULT_ENV:-}" != agentic-ml ]; then
  if [ -f .venv/bin/activate ]; then
    . .venv/bin/activate
  else
    eval "$(conda shell.bash hook)"
    conda activate agentic-ml
  fi
fi

[ -f .env ] && { set -a; . ./.env; set +a; }
RUN_KEY="$(python -c 'import secrets; print(secrets.token_urlsafe(32))')"
for k in CLASSIFICATION REGRESSION CLUSTERING ANOMALY VISUALIZATION FORECASTING DRIFT SERVING EXPLAIN; do
  v="${k}_AGENT_API_KEY"; export "$v=${!v:-$RUN_KEY}"
done
export ORCHESTRATOR_API_KEY="${ORCHESTRATOR_API_KEY:-$RUN_KEY}"
# The gateway signs each user's context and every agent verifies it with this;
# JWT_SECRET also signs dev login tokens (print_token below must match the gateway).
export JWT_SECRET="${JWT_SECRET:-$RUN_KEY}"
export USER_CONTEXT_SECRET="${USER_CONTEXT_SECRET:-$JWT_SECRET}"
export ORCHESTRATOR_AGENT_API_KEY="$ORCHESTRATOR_API_KEY"   # gateway -> orchestrator must match

# Sign-in: Keycloak when it is configured or already running locally.
LOCAL_REALM="http://localhost:${KEYCLOAK_PORT:-8180}/realms/ml-agents"
if [ -z "${OIDC_ISSUER:-}" ] && curl -sf -m 3 "$LOCAL_REALM/.well-known/openid-configuration" >/dev/null 2>&1; then
  export OIDC_ISSUER="$LOCAL_REALM"
fi
[ -n "${OIDC_ISSUER:-}" ] && export AGENTIC_ML_REQUIRE_USER="${AGENTIC_ML_REQUIRE_USER:-1}"

# Stale gateway / UI from an earlier run (swarm.sh clears the agent ports itself).
lsof -ti:${APP_PORT:-9001},5173 -sTCP:LISTEN | xargs kill -9 2>/dev/null || true

PIDS=()
trap 'kill "${PIDS[@]}" 2>/dev/null || true' EXIT

bash scripts/swarm.sh > logs/swarm.log 2>&1 & PIDS+=($!)    # all ten agents, health-checked
bash scripts/mlflow.sh > logs/mlflow.log 2>&1 & PIDS+=($!)   # reuses one already on the port
alembic upgrade head >/dev/null
(cd ui && { [ -d node_modules ] || npm install; })

if [ "${UI_DEV:-0}" = 1 ]; then
  URL="http://localhost:5173"
  python -m gateway.app > logs/gateway.log 2>&1 & PIDS+=($!)
  (cd ui && npm run dev > ../logs/ui.log 2>&1) & PIDS+=($!)
else
  URL="http://localhost:${APP_PORT:-9001}"
  (cd ui && npm run build > ../logs/ui-build.log 2>&1)       # gateway serves ui/dist on its own origin
  python -m gateway.app > logs/gateway.log 2>&1 & PIDS+=($!)
fi

# Ready = the orchestrator answers with our key AND the gateway serves the API.
printf "Starting agents and services"
for _ in $(seq 1 150); do
  curl -sf -m 2 -H "Authorization: Bearer $ORCHESTRATOR_API_KEY" http://localhost:9100/.well-known/agent-card.json >/dev/null 2>&1 \
    && curl -sf -m 2 "http://localhost:${APP_PORT:-9001}/api/v1/auth/config" >/dev/null 2>&1 && break
  printf "."; sleep 2
done
echo
if ! curl -sf -m 2 "http://localhost:${APP_PORT:-9001}/api/v1/auth/config" >/dev/null 2>&1; then
  echo "The gateway did not come up — see logs/gateway.log and logs/swarm.log"; exit 1
fi
grep -E "CORE agents not healthy|optional agents down" logs/swarm.log || true

echo
echo "════════════════════════════════════════════════════════════"
echo "  Data Science Agents is up — open $URL"
if [ -n "${OIDC_ISSUER:-}" ]; then
  echo "  Sign in with your Keycloak account ($OIDC_ISSUER)"
  echo "  No account yet?  make user u=<name>   (admin=1 for ml-admin)"
else
  echo "  Dev sign-in token (Keycloak not running — make docker-start for SSO):"
  echo "  $(python -m scripts.print_token "${USER:-dev-user}")"
fi
echo "  MLflow  http://127.0.0.1:${MLFLOW_PORT:-5001}     logs  logs/"
echo "  Ctrl+C stops everything"
echo "════════════════════════════════════════════════════════════"
if [ "${NO_BROWSER:-0}" != 1 ] && [ -z "${SSH_CONNECTION:-}" ]; then
  (command -v open >/dev/null && open "$URL") || (command -v xdg-open >/dev/null && xdg-open "$URL") || true
fi

wait
