#!/usr/bin/env bash
# Starts the gateway (:9001) + UI dev server (:5173). Ctrl+C stops both.
#
# Does NOT start the ten agents — the UI talks to the orchestrator (via the
# gateway) and needs it (and whichever specialist it delegates to) reachable.
# Run `make swarm` first, or use `make up` for the full stack.
#
# The UI (ui/src, a React app) needs `npm install` once — see `make setup` —
# and Vite's dev server for hot reload. For the zero-build fallback
# (ui/legacy.html, plain static file, talks A2A directly instead of through
# the gateway), just `cd ui && python3 -m http.server 8080` instead.
set -euo pipefail
cd "$(dirname "$0")/.."

python -m gateway.app &
GATEWAY_PID=$!
trap 'kill $GATEWAY_PID 2>/dev/null || true' EXIT

cd ui
[ -d node_modules ] || npm install
npm run dev
