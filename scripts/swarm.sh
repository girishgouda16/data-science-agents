#!/usr/bin/env bash
# Start the agent swarm, verify every member actually answers, then hold.
#
# Why this exists: the agents are separate processes on separate ports, and
# the ways a demo dies are all silent. A stale process from an earlier run
# still holds the port, so your restart didn't take and you're talking to old
# code. Or an agent isn't up at all, and the orchestrator turns the connection
# error into a tool result the LLM narrates around — so the chat shows a
# confident hand-off to an agent that was never listening.
#
# So this refuses to hand back a swarm it hasn't proved is talking: every
# agent must serve its A2A card, AND accept the shared bearer token, before
# this script says ready. A 401 here is the same 401 the orchestrator would
# have hit, caught 20 seconds before the demo instead of during it.
#
#   scripts/swarm.sh          start everything, wait for healthy, tail logs
#   scripts/swarm.sh stop     kill everything on the agent ports
#   scripts/swarm.sh status   health-check what's running, change nothing
#
# Ctrl+C stops every agent it started. Logs land in logs/<agent>.log.
set -uo pipefail
cd "$(dirname "$0")/.."
ROOT="$(pwd)"

PYTHON="${SWARM_PYTHON:-$(command -v python)}"
LOG_DIR="$ROOT/logs"

# name:port:path-to-agent.py  — orchestrator last, it depends on the rest.
AGENTS=(
  "classification:9000:classification-agent/scripts/agent.py"
  "visualization:9200:visualization-agent/scripts/agent.py"
  "regression:9300:regression-agent/scripts/agent.py"
  "clustering:9400:clustering-agent/scripts/agent.py"
  "anomaly:9500:anomaly-agent/scripts/agent.py"
  "forecasting:9600:forecasting-agent/scripts/agent.py"
  "drift:9700:drift-agent/scripts/agent.py"
  "serving:9800:serving-agent/scripts/agent.py"
  "explain:9900:explain-agent/scripts/agent.py"
  "orchestrator:9100:orchestrator-agent/agent.py"
)

# Only these are needed for the core demo; the rest are optional extras.
CORE="classification visualization serving explain drift orchestrator"

red()   { printf "\033[31m%s\033[0m\n" "$*"; }
green() { printf "\033[32m%s\033[0m\n" "$*"; }
dim()   { printf "\033[2m%s\033[0m\n" "$*"; }

load_env() {
  [ -f "$ROOT/.env" ] || { red "no .env — copy .env.example and set the agent keys"; exit 1; }
  set -a; . "$ROOT/.env"; set +a
  if [ -z "${CLASSIFICATION_AGENT_API_KEY:-}" ]; then
    red "CLASSIFICATION_AGENT_API_KEY is unset in .env."
    red "Every agent auto-generates its own key when unset, so the orchestrator"
    red "signs with a different secret than each agent expects and EVERY hop 401s."
    red "See the agent-to-agent auth block in .env.example."
    exit 1
  fi
}

key_for() {  # agent name -> its bearer token, defaulting to the orchestrator's
  local var="$(echo "$1" | tr '[:lower:]' '[:upper:]')_AGENT_API_KEY"
  echo "${!var:-${ORCHESTRATOR_API_KEY:-}}"
}

# Health = serves its agent card AND accepts our token. Card-only would pass
# a misconfigured key and leave the real failure for the first delegation.
probe() {
  local port="$1" key="$2"
  curl -sf -m 3 -H "Authorization: Bearer $key" \
    "http://localhost:$port/.well-known/agent-card.json" >/dev/null 2>&1
}

# Python listeners only: cmd_stop kill -9s this pid, and a Docker-published port's host pid is Docker Desktop itself.
port_pid() { lsof -ti:"$1" -sTCP:LISTEN -a -c python 2>/dev/null | head -1; }

cmd_stop() {
  local killed=0
  for entry in "${AGENTS[@]}"; do
    IFS=: read -r name port _ <<< "$entry"
    local pid; pid="$(port_pid "$port")"
    if [ -n "$pid" ]; then kill -9 "$pid" 2>/dev/null && killed=$((killed+1)); fi
  done
  green "stopped $killed agent process(es)"
}

cmd_status() {
  load_env
  local down=0
  for entry in "${AGENTS[@]}"; do
    IFS=: read -r name port _ <<< "$entry"
    local pid; pid="$(port_pid "$port")"
    if [ -z "$pid" ]; then
      printf "  %-15s %-6s %s\n" "$name" "$port" "$(red 'not running')"; down=$((down+1))
    elif probe "$port" "$(key_for "$name")"; then
      printf "  %-15s %-6s %s\n" "$name" "$port" "$(green "healthy (pid $pid)")"
    else
      printf "  %-15s %-6s %s\n" "$name" "$port" "$(red "listening but REJECTED our token (pid $pid)")"
      down=$((down+1))
    fi
  done
  if ! curl -sf -m 2 http://localhost:3000/api/public/health >/dev/null 2>&1; then
    printf "  %-15s %-6s %s\n" "langfuse" "3000" "$(red 'not running — no traces, and agents will spam retries')"
  else
    printf "  %-15s %-6s %s\n" "langfuse" "3000" "$(green 'healthy')"
  fi
  return $down
}

case "${1:-start}" in
  stop)   cmd_stop; exit 0 ;;
  status) cmd_status; exit $? ;;
esac

load_env
mkdir -p "$LOG_DIR"

echo "Clearing stale processes on agent ports…"
cmd_stop >/dev/null

PIDS=()
cleanup() { echo; echo "stopping swarm…"; for p in "${PIDS[@]}"; do kill "$p" 2>/dev/null; done; wait 2>/dev/null; }
trap cleanup EXIT INT TERM

echo "Starting agents (python: $PYTHON)…"
for entry in "${AGENTS[@]}"; do
  IFS=: read -r name port path <<< "$entry"
  if [ ! -f "$ROOT/$path" ]; then
    red "  $name — no agent at $path, skipping"; continue
  fi
  ( cd "$(dirname "$ROOT/$path")" && exec "$PYTHON" "$(basename "$path")" ) \
    > "$LOG_DIR/$name.log" 2>&1 &
  PIDS+=("$!")
  dim "  $name on :$port  (log: logs/$name.log)"
done

echo
echo "Waiting for agents to answer…"
DEADLINE=$(( $(date +%s) + ${SWARM_TIMEOUT:-90} ))
declare -a UNHEALTHY
while :; do
  UNHEALTHY=()
  for entry in "${AGENTS[@]}"; do
    IFS=: read -r name port path <<< "$entry"
    [ -f "$ROOT/$path" ] || continue
    probe "$port" "$(key_for "$name")" || UNHEALTHY+=("$name")
  done
  [ ${#UNHEALTHY[@]} -eq 0 ] && break
  [ "$(date +%s)" -ge "$DEADLINE" ] && break
  sleep 2
done

echo
cmd_status
echo

# A missing optional agent is a warning; a missing CORE agent means the demo
# path itself is broken, and finding that out now is the entire point.
CORE_DOWN=()
for name in $CORE; do
  for bad in "${UNHEALTHY[@]:-}"; do [ "$name" = "$bad" ] && CORE_DOWN+=("$name"); done
done

if [ ${#CORE_DOWN[@]} -gt 0 ]; then
  red "CORE agents not healthy: ${CORE_DOWN[*]}"
  red "The demo path is broken. Check logs/<agent>.log — the usual causes are a"
  red "missing dependency, a port still held by an old process, or a key mismatch."
  echo
  for name in "${CORE_DOWN[@]}"; do dim "--- last 15 lines of logs/$name.log ---"; tail -15 "$LOG_DIR/$name.log" 2>/dev/null; done
  exit 1
fi

[ ${#UNHEALTHY[@]} -gt 0 ] && dim "optional agents down (demo path unaffected): ${UNHEALTHY[*]}"

# Langfuse is where the hand-offs are actually visible — every agent posts
# spans to it via litellm's callback. When it is down, each agent retries the
# export on a backoff and floods its own log with connection errors, which
# buries anything real. More to the point: with no Langfuse there is no trace
# to show, and the trace IS the evidence that the agents coordinated.
if ! curl -sf -m 2 http://localhost:3000/api/public/health >/dev/null 2>&1; then
  echo
  red "Langfuse is not up on :3000 — no traces will be recorded."
  dim "  Every agent will retry span exports and fill its log with connection errors."
  dim "  Start it with:  make docker   (needs Docker — e.g. on the server)"
  echo
fi

green "Swarm ready. Orchestrator: http://localhost:9100"
dim "Talk to it via the gateway (make dev) or the UI. Ctrl+C stops everything."
echo
wait
