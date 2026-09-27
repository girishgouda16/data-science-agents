# Go-live runbook

## 1. Pre-flight (all must pass)

```bash
make test        # every suite: core, gateway, 10 agents, the script-style tool tests,
                 # cross-agent + multi-user + tool-boundary, UI lint + build
# live LLM checks — the agents' JUDGMENT, not just their tools (costs LLM calls)
python classification-agent/scripts/eval_agent.py        # 6 scenarios, expect 19/19
python scripts/live_smoke_agents.py                      # all five training agents, temp stores
```

## 2. The server, once

1. `.env` from `.env.example`. Set real values for `JWT_SECRET`, `USER_CONTEXT_SECRET`,
   every `*_AGENT_API_KEY` (distinct, random), the LLM key, and:
   ```
   OIDC_ISSUER=https://sso.example.com/realms/ml-agents
   AGENTIC_ML_REQUIRE_USER=1
   KEYCLOAK_ADMIN_PASSWORD=…          KEYCLOAK_HOSTNAME=https://sso.example.com
   KEYCLOAK_COMMAND=start --import-realm
   ML_AGENTS_UI_URL=https://ml.example.com
   POSTGRES_PASSWORD=…  (and the other CHANGEME values in docker-compose.yml)
   ```
2. `make docker` — Langfuse (:3000) and Keycloak (:8180, realm `ml-agents` imported from
   `deploy/keycloak/`). Put both behind the TLS reverse proxy with the gateway.
3. In Keycloak (admin console → realm ml-agents): add users, or federate your directory
   (User federation → LDAP/AD, or Identity providers → Azure AD/Okta). Everyone gets `ml-user`;
   give team leads `ml-admin`.
4. Existing exported models (before the tool guard) are refused until recorded as genuine:
   `python -m core.toolguard data/artifacts/*.pkl` — only for files you know `export_model` wrote.
5. `make up`. The gateway serves the UI, API and charts on :9001 — the only port the proxy exposes.
   Agents (:9000–:9900), MLflow and Postgres stay internal.

## 3. Smoke through the UI (the one path no unit test covers)

Signed in as a normal user, for classification and one other agent:
1. Attach a CSV, "train a model for <target>, full pipeline, log to MLflow" — the turn runs in
   the background (status + elapsed time, Stop works); expect questions, then a report whose first
   line is the readiness verdict, charts inline, and a registry version.
2. "Promote version N to champion, I confirm, reason: <why>" → `serving_path` returned.
   A BLOCKED run is only promoted when you say to force it and give a reason.
3. "Score <csv> with the champion" → predictions naming the model version.
4. "Has production drifted?" → drift runs against the predictions log.
5. Models view shows the champion; Runs view shows only your runs (an ml-admin sees all).

## 4. Security model (what is enforced, where)

| Boundary | Enforced by |
|---|---|
| Who the user is | Keycloak token verified by the gateway (issuer, audience, expiry, signature via JWKS) — `core/auth.py` |
| User reaching agents | gateway-signed user context (HS256, `USER_CONTEXT_SECRET`, 4 h); agents reject missing/forged ones when `AGENTIC_ML_REQUIRE_USER=1` — `core/runtime.py` |
| Agent to agent | bearer key per agent; a key alone is not an identity |
| What a tool may touch | `core/toolguard.py`: reads only own uploads / shared datasets / permitted model files, CSV only; writes only `data/artifacts` (export) or own `data/predictions`; a `.pkl` loads only with a matching export record (sha256); dev tools (`dvc_track`, `generate_ci_workflow`) refused |
| Tool process | allow-listed environment (no API keys or secrets), per-call timeout (`AGENT_TOOL_TIMEOUT`) |
| Runs, registry aliases | owner or `ml-admin` role — `core/runtime.access_error`, `core/registry.authorize` |
| Uploads | authenticated, CSV only, per-user folder |

## 5. Execution and durability

- Every agent is a LangGraph graph (`core/agent_host.py`) checkpointed after each step — each tool
  call is a step — to `data/a2a/<agent>-threads.db` (or Postgres via `AGENT_CHECKPOINT_URL`).
  A crash loses at most the tool call in flight; a question asked before a restart is answered after it.
- The gateway never holds a request open: a turn starts non-blocking, a background follower polls
  the orchestrator's task, and a restarted gateway resumes following every turn still running.
  Stop = A2A `tasks/cancel`. Turns longer than `GATEWAY_TURN_TIMEOUT` (4 h) are stopped.

## 6. Settings that matter

| Variable | Default | What it does |
|---|---|---|
| `OIDC_ISSUER` | empty (dev tokens) | Keycloak realm URL; turns on SSO |
| `AGENTIC_ML_REQUIRE_USER` | off | agents refuse requests without a signed user |
| `USER_CONTEXT_SECRET` | `JWT_SECRET` | signs/verifies the user context — same value in every process |
| `AGENTIC_ML_ADMIN_ROLE` / `AGENTIC_ML_ADMINS` | `ml-admin` / empty | admin by Keycloak role / break-glass names |
| `AGENT_CHECKPOINT_URL` | SQLite in `data/a2a/` | LangGraph checkpoints (Postgres for several hosts) |
| `AGENT_TOOL_TIMEOUT` | 1800 | seconds one tool call may run |
| `AGENT_MAX_CONCURRENT_TURNS` | 4 | turns a specialist runs at once; the rest queue |
| `<AGENT>_MAX_TOOL_ROUNDS` / `ORCHESTRATOR_MAX_HOPS` | 30 / 16 | model rounds per turn before pausing resumably |
| `MLFLOW_TRACKING_URI` / `MLFLOW_PORT` | file store / 5001 | the shared registry |
| `AGENTIC_ML_DATA_DIR` | `<repo>/data` | uploads, runs, artifacts, predictions |
| `APP_HOST` / `AGENT_BIND` | `127.0.0.1` | gateway / agent listen address; `0.0.0.0` only when the proxy or other agents are on another host |
| `CHARTS_PUBLIC_URL` | `/charts` | where browsers fetch charts |

## 7. Rollback

- A bad champion: every promotion returns its `rollback` call; `demote_model('champion', confirmed=true)`
  withdraws it.
- Code: check out the previous release tag. If it has an older database schema, run
  `alembic downgrade <its revision>` before `make up`.
- SSO trouble: unset `OIDC_ISSUER` to fall back to dev tokens (admins only — never on an open network).

## 8. Known limits (tell users)

- Runs end `incomplete` until every required step ran; success verdicts can be `INCONCLUSIVE`;
  champion promotion needs a reason; a BLOCKED run needs an explicit, reasoned force.
- Serving is an agent scoring CSVs — no online endpoint; drift runs on request, not on a schedule.
- One host by default (SQLite per agent). Several hosts need Postgres for checkpoints, tasks and MLflow.
- Cancelling stops the orchestrator's turn; a specialist already mid-step finishes that step.
- Runs and uploads created before identity existed have no owner and stay open to everyone.
