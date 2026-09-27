#!/usr/bin/env bash
# The MLflow UI over the registry every agent logs to. Port from .env
# MLFLOW_PORT (default 5001 — on macOS the AirPlay Receiver answers
# localhost:5000, so a UI there shows a 403 instead of the registry). Store is
# MLFLOW_TRACKING_URI when set, else <repo>/mlruns — the same default
# core/registry.py resolves for every agent. Bound to 127.0.0.1: the UI has no auth.
#   make mlflow            (also started by `make up`)
set -euo pipefail
cd "$(dirname "$0")/.."
env_value() { grep -E "^$1=" .env 2>/dev/null | tail -1 | cut -d= -f2- | tr -d '"' || true; }
PORT="${MLFLOW_PORT:-$(env_value MLFLOW_PORT)}"; PORT="${PORT:-5001}"
# Where the registry lives. On a shared server: MLFLOW_BACKEND_STORE_URI=postgresql://…
# and every agent's MLFLOW_TRACKING_URI=http://127.0.0.1:$PORT — this server is
# then the one writer (it serializes concurrent registrations). Locally the
# agents write the file store directly and this is just the UI over it.
TRACKING="${MLFLOW_TRACKING_URI:-$(env_value MLFLOW_TRACKING_URI)}"
BACKEND="${MLFLOW_BACKEND_STORE_URI:-$(env_value MLFLOW_BACKEND_STORE_URI)}"
if [ -z "$BACKEND" ]; then case "$TRACKING" in ""|http*) BACKEND="file:$(pwd)/mlruns" ;; *) BACKEND="$TRACKING" ;; esac; fi
ARTIFACTS="${MLFLOW_ARTIFACTS_DESTINATION:-$(env_value MLFLOW_ARTIFACTS_DESTINATION)}"; ARTIFACTS="${ARTIFACTS:-$(pwd)/mlruns}"
HOST="${MLFLOW_HOST:-$(env_value MLFLOW_HOST)}"; HOST="${HOST:-127.0.0.1}"   # no auth on the UI: keep it local or behind a proxy
if lsof -nP -iTCP:"$PORT" -sTCP:LISTEN >/dev/null 2>&1; then
  echo "MLflow UI: :$PORT is already in use — if that is MLflow, open http://127.0.0.1:$PORT"
  exit 0
fi
echo "MLflow UI: http://$HOST:$PORT   (registry: $BACKEND)"
exec mlflow server --backend-store-uri "$BACKEND" --artifacts-destination "$ARTIFACTS" --host "$HOST" --port "$PORT"
