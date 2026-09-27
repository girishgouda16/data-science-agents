<h1 align="center">Data Scientist Agents</h1>

<p align="center">
  <em>The senior data scientist who won't ship your model until it earns it.</em>
</p>

<p align="center">
  <img src="https://img.shields.io/github/actions/workflow/status/girishgouda16/data-scientist-agents/ci.yml?branch=main&style=flat-square&label=ci" alt="CI">
  <img src="https://img.shields.io/badge/status-alpha-111111?style=flat-square" alt="Alpha">
  <img src="https://img.shields.io/badge/license-Apache--2.0%20%2B%20BSL-111111?style=flat-square" alt="Apache-2.0 + BSL license">
  <img src="https://img.shields.io/badge/agents-10-111111?style=flat-square" alt="10 agents">
  <img src="https://img.shields.io/badge/python-3.13-111111?style=flat-square" alt="Python 3.13">
  <img src="https://img.shields.io/badge/LLM-any%20via%20litellm-111111?style=flat-square" alt="Any LLM via litellm">
  <img src="https://img.shields.io/badge/domain-telco%20first-111111?style=flat-square" alt="Telco first">
</p>

<p align="center">
  <strong>79 / 80 behaviour checks pass &middot; 25 scenarios &middot; 6 agents &middot; no model ships past a failed gate without a person's reason</strong><br>
  <sub>The real agents, driven by a real LLM (deepseek-v4.1-flash via OpenRouter), on synthetic data with known answers, graded from what each run left behind — not from what the agent said. One run per scenario. Reproduce: <code>make eval</code> or <code>cd &lt;agent&gt;-agent/scripts &amp;&amp; python eval_agent.py</code> (LLM calls).</sub>
</p>

---

You know the one. Twenty years of data science. Before opening the file,
asks what the model is *for*. Splits by customer before anyone says
"random split". Finds the column that was filled in after the outcome.
Refuses to ship a model that loses to "same as last month". Writes down
every assumption, so you can argue with it later.

Data Scientist Agents puts that data scientist behind a chat box: ten
agents that take you from a raw table to a monitored model in production.
Bring a problem. Get a model you can defend.

## Before / after

You ask for a churn model.

A typical AutoML run shuffles the rows, so the same customer lands in both
train and test. A `cancellation_reason` column predicts churn perfectly.
AUC 0.99. Shipped.

With Data Scientist Agents (a condensed, illustrative conversation):

```text
you    Predict churned from subscribers.parquet.
agent  What will a churn score trigger, and how many retention calls can the
       team make a month? What result makes the model worth using?
you    Retention calls, 20,000 a month. Recall of 0.6 or more.
agent  customer_id repeats across 12 months: splitting by customer and by time.
       cancellation_reason is written after the outcome: kept out of the model.
       Built features from each customer's own history — gaps between top-ups,
       usage against their own baseline, calls to competitor numbers.
       The model beats the baseline, every readiness gate passes.
       Report ready (verdict first). Promote version 3 to champion?
```

The questions are the ones only you can answer. Everything a measurement
can settle, it decides — and records in the report.

## Numbers

Each scenario gives the real agent a dataset with a known trap and a
scripted user, then grades the decisions from the run's own records.
Full results and every miss traced: [docs/agent-eval-report.md](docs/agent-eval-report.md).

| Agent | Scenarios | Checks passed | Examples of what is checked |
| --- | --: | --: | --- |
| Classification | 6 | 19 / 19 | splits repeated customers by customer; keeps a label copy out; no SMOTE at 2% positives; invents no success bar |
| Regression | 6 | 13 / 13 | log or Poisson for a skewed target; beats the persistence baseline; prices a 5:1 cost as a 0.83 quantile |
| Clustering | 4 | 15 / 15 | outcome and age describe segments, never define them; Gower k-medoids on mixed data; admits there is no structure in noise |
| Anomaly | 3 | 10 / 10 | alert share from review capacity; keeps a case-status field out; explained review queue without labels |
| Forecasting | 3 | 13 / 14 | weekly season for hourly traffic; judged at the business horizon; every cell in one run |
| Drift | 3 | 9 / 9 | leads with a broken feed; catches a joint shift per-column checks miss; names the week it started |

Measured offline, in CI, on data with known answers:

| | |
| --- | --- |
| Drift, 120 current rows, no real drift | false alarms 25% → **2.5%** (sampling-noise floor), real 0.5 sd shifts still caught 92% |
| Forecasting, hourly cell traffic | season chosen from the data: the naive baseline is an honest 10% WAPE, not a 35% straw man; ETS 7.7% |
| Forecasting, 28 cells in one run | WAPE 9.0% (naive) → **6.9%**, beats each cell's own naive on 86% of cells |
| Anomaly, rare-but-normal category | KNN average precision 1.0 vs IForest 0.12 — flagged automatically, detector picked by evidence |
| Clustering, mixed numeric + categorical | Gower k-medoids ARI 0.75 vs k-means 0.32; stability 0.96 vs 0.59 |

## What it does

| Agent | Port | Brings you |
| --- | --: | --- |
| orchestrator | 9100 | routes each request to the right specialist and relays its questions to you |
| classification | 9000 | churn, fraud (Wangiri, IRSF, SIM swap), any yes/no or multi-class target — calibrated probabilities, threshold tuned to your bar |
| regression | 9300 | ARPU, usage, capacity, durations — a value and a calibrated range; Poisson and quantile objectives |
| clustering | 9400 | stable, described segments; mixed data (Gower k-medoids); how customers move between segments month to month |
| anomaly | 9500 | an alert queue sized to what your team can review, a reason for every alert, with or without labels |
| forecasting | 9600 | one series, a total, or every series (all cells) in one run; backtests at your horizon; P10–P90 bands |
| drift | 9700 | per column (missing values, type changes, PSI above the noise floor), the table as a whole, and when it started |
| visualization | 9200 | charts for any run |
| explain | 9900 | why the model decided this for one row, and what would change it |
| serving | 9800 | scores new data with the champion; every prediction names its model |

Around them: **gateway + UI** on :9001, **MLflow** on :5001, and in Docker
**Keycloak** sign-in on :8180 and **Langfuse** traces on :3000.

## How it works

Every run follows the same senior workflow:

```
1. What is it for?      → the decision, the bar, where labels came from: asked, never invented
2. Where is the data?   → a file (csv, parquet, json, xlsx, archives) or sql: / clickhouse: / store:
3. Behaviour features   → each customer's own history: gaps, trends, own baseline, peers
4. Split honestly       → by customer, by time; rows whose outcome is not known yet held back
5. Screen for leakage   → before training, and after every column change
6. Beat a baseline      → majority class, mean, same-as-last-period, seasonal naive
7. Readiness gates      → blocked = fix it; export and promotion refuse a blocked run
8. Report               → the verdict first, then every assumption and every question asked
9. Registry → champion → serving → drift
```

It **asks** only what the business decides: what the model is for, the
bar, the review capacity, the horizon, which attributes must not drive a
decision, where labels came from. It **decides and records** everything
a measurement settles: imputation, identifier drops, features, the model,
the threshold.

Under the hood each agent is a LangGraph loop (`core/agent_host.py`) that
calls the LLM through litellm and runs its tools as an MCP server. Agents
talk to each other over A2A. Every step is checkpointed, so a restart loses
neither a conversation nor a pending question.

## Install

Needs Python 3.13 and Node 20+, on macOS or Linux. Docker is optional
(Keycloak sign-in and Langfuse traces).

```bash
python3.13 -m venv .venv && source .venv/bin/activate
make setup      # Python + UI deps, .env with random secrets, DB schema (migrate)
```

Prefer conda? `conda env create -f env/env.macos.yaml && conda activate agentic-ml`,
then `make setup`.

In `.env`, set `OPENROUTER_API_KEY` (or another `LLM_PROVIDER` and its
key). That's all you need locally. After a pull that changes the database
schema, run `make migrate`.

No API key? Run a local model with [Ollama](https://ollama.com):
`make ollama-pull`, then set `LLM_PROVIDER=ollama` and
`AGENT_MODEL=qwen2.5:7b` in `.env`. Small local models make weaker
judgment calls than the model the evals above used.

## Run

```bash
make docker-start     # Keycloak + Langfuse; waits until sign-in is ready
make up               # all ten agents, MLflow, gateway + UI; opens http://localhost:9001
```

`make up` is what you want day to day. `make run` starts only the gateway
(API + built UI on :9001); use it when the agents are already up (`make swarm`).
`make dev` is gateway + Vite hot reload without starting agents — start
`make swarm` separately if you need chat delegation.

Sign in as **admin / admin** (created by `make docker-start`, has the
`ml-admin` role). Add people with `make user u=alice` (one-time password,
they choose their own) or `make user u=dana admin=1`.

**Ctrl+C** in the `make up` terminal stops everything. `make docker-stop`
stops the containers. No Docker? `make up` alone still works and prints a
dev sign-in token instead of using Keycloak.

## Use it

1. **New chat** → attach `data/titanic.csv` (📎 or drag and drop) →
   *"Predict Survived. Success = recall ≥ 0.6. Log it to MLflow."*
   Your own CSV works the same way, or ask for a connected source (below).
   Optionally set a **Project**: models register as
   `<agent>-<project>-<target>`.
2. The turn runs in the background ("Agents working · 3m 12s"); **Stop**
   cancels it. Answer questions as they come; *"go with your
   recommendation"* is fine. Or switch on **Autopilot**: agents answer
   their own questions with their recommended option and list each
   decision. They still never force past a failed readiness gate.
3. Read the report: its first line is the readiness verdict. Charts
   appear inline.
4. *"Promote version 1 to champion — I confirm, reason: …"*. A run that
   failed a gate is promoted only if you force it and give a reason; the
   reason is recorded.
5. *"Score new_customers.csv with the champion"*, then *"Has production
   drifted?"*

**Runs** lists your runs and each one's next step. **Models** shows the
registry: versions, champion/challenger, who promoted what and why. Each
user sees only their own chats, uploads and runs; an `ml-admin` sees
everyone's.

Things to try:

| Ask | Agent |
| --- | --- |
| *"Which subscribers will churn next month? Recall ≥ 0.6."* | classification |
| *"Forecast daily traffic for every cell for the next 14 days."* | forecasting |
| *"Find anomalous lines — the team can review 200 a day out of 1M records."* | anomaly |
| *"Segment subscribers for tariff offers, 4–6 segments, and show how they move month to month."* | clustering |
| *"Has this month's feed drifted from training data, and since when?"* | drift |

## Connect your data

Credentials never pass through the model: a source is a **name**, resolved
from the environment. Add to `.env`:

```bash
AGENTIC_ML_SQL_WAREHOUSE=postgresql://user:pass@host:5432/db        # any SQLAlchemy URL
AGENTIC_ML_CLICKHOUSE_EVENTS=https://user:pass@clickhouse:8443/?database=telco
AGENTIC_ML_STORE_LAKE=s3://bucket/prefix                            # s3://, gs://, az://, file://
```

Then ask in chat: *"Load last month's usage from sql:warehouse"*. The agent
runs one read-only `SELECT` (SQL: always rolled back; ClickHouse: read-only
mode), capped at 5M rows, and freezes the result as a parquet snapshot with
its lineage, so the run stays reproducible after the table changes. Push
filters and aggregation into the query.

The UI upload takes CSV today; for parquet, Excel or JSON files, point a
`file://` store at their folder.

## Use the agents from Claude Code

Open this repo in Claude Code: `.mcp.json` starts each agent's MCP server,
so Claude can call the same tools (`eda`, `prepare_dataset`,
`train_model`, `check_readiness`, …) directly, following the playbooks in
each agent's `SKILL.md` and `skills/`. The servers run with the repo's
`.venv` Python, else `python3` on PATH. Using conda? Link it once, with the
env active: `ln -s "$CONDA_PREFIX" .venv`.

## Commands

| Command | What it does |
| --- | --- |
| `make setup` | Python and UI dependencies, `.env` from the example with random secrets, `make migrate` |
| `make migrate` | apply Alembic migrations (also runs at end of `make setup` and start of `make up`) |
| `make docker-start` / `make docker-stop` | Keycloak + Langfuse and their stores (Postgres, ClickHouse, Redis, MinIO) |
| `make up` | all ten agents, MLflow, gateway + UI. `UI_DEV=1 make up` runs the UI with hot reload on :5173 |
| `make dev` | gateway + Vite hot reload on :5173 only — agents not started |
| `make swarm` / `make swarm-status` / `make swarm-stop` | start, check, or stop the ten A2A agents |
| `make run` | gateway on :9001 only (runs `migrate` first); needs agents up for chat |
| `make user u=<name> [admin=1] [p=<password>]` | add a sign-in account |
| `make quick` | fast CI-shaped check: compileall, core gates, UI lint + tests (~1 min) |
| `make test` | full suite + UI lint, tests and build (no LLM calls; matches CI on PRs) |
| `make eval` | all six LLM behaviour evals (`make eval AGENT=classification` for one) |

## On a server

Run `make docker-start` then `make up` behind a TLS reverse proxy that
exposes only :9001 (UI + API) and Keycloak. In `.env` set at least:

```
OIDC_ISSUER=https://sso.example.com/realms/ml-agents
AGENTIC_ML_REQUIRE_USER=1
KEYCLOAK_HOSTNAME=https://sso.example.com
KEYCLOAK_COMMAND=start --import-realm
KEYCLOAK_ADMIN_PASSWORD=…     ML_ADMIN_PASSWORD=…     POSTGRES_PASSWORD=…
JWT_SECRET=…                  # openssl rand -hex 32
*_AGENT_API_KEY=…             # every one: a long random string, not the example's "devkey"
ML_AGENTS_UI_URL=https://ml.example.com
```

Everything listens on 127.0.0.1 by default. Keep the proxy on the same
host, or set `APP_HOST=0.0.0.0` when it runs elsewhere.

Full checklist and security model: [docs/GO_LIVE.md](docs/GO_LIVE.md). Every setting, with comments:
[.env.example](.env.example).

## Troubleshooting

| Symptom | Fix |
| --- | --- |
| "Schema not migrated" / app refuses to start | `make migrate` (or re-run `make setup` / `make up`) |
| Keycloak sign-in page when you expected a dev token (or the reverse) | `make up` uses Keycloak whenever it is running; `make docker-stop` for dev tokens |
| `make docker-start`: "Docker is not running" | start Docker Desktop (or `colima start`) |
| http://localhost:9001 doesn't load | `make up` isn't running or failed: see `logs/gateway.log` |
| an agent is missing in `make swarm-status` | see `logs/<agent>.log` |
| logs full of "Failed to export span batch" | Langfuse keys are set but Langfuse isn't running: `make docker-start`, or clear the `LANGFUSE_*` keys |
| a turn fails with "the model did not answer within 300s, twice" | the LLM provider hung; retry, or raise `LLM_TOTAL_TIMEOUT` |
| no traces in Langfuse | create a project at http://localhost:3000 and put its keys in `LANGFUSE_PUBLIC_KEY` / `LANGFUSE_SECRET_KEY` |
| "has no export record" for an old model | exported before the tool guard: `python -m core.toolguard data/artifacts/*.pkl` (only for files you trust) |

## Layout

```
core/                shared: agent host, auth, tool guard, registry, gates, tracing,
                     data sources, feature tools, validation, eval runner
gateway/             REST API, sessions, uploads; serves the UI and charts
ui/                  React app
<name>-agent/        one per agent: agent.py, SKILL.md, skills/, scripts/mcp_server/ (its tools),
                     scripts/eval_agent.py (LLM evaluation), scripts/test_*.py
deploy/keycloak/     realm imported by Keycloak
data/                uploads, runs, exported models, predictions (runtime, not in git)
```

## FAQ

**Does it replace the data scientist?**
No. It asks what only you know — the decision, the bar, where the labels
came from — and does the rest the way a careful senior would, with every
choice written down for you to challenge.

**Which LLM?**
Any provider litellm supports: set `LLM_PROVIDER` and its key, and
`AGENT_MODEL` (or a per-agent `<NAME>_AGENT_MODEL`). Evaluated with
`deepseek/deepseek-v4.1-flash` via OpenRouter.

**Can it ship a bad model?**
Not silently. A run that failed a readiness gate refuses export and
champion promotion; only a person can force it, with a reason, and the
reason is recorded in the report and the registry.

**Does it scale?**
To one machine: pandas, up to 5M rows per load. Push joins and
aggregation into SQL or ClickHouse; the agents are built to take the
aggregated table.

**Why telco first?**
It is the first domain with deep feature guidance (Wangiri, IRSF, SIM-box,
SIM swap, churn, capacity). Fraud/AML, credit, retail, logistics and
operations playbooks ship too.

**Is it production-ready?**
Alpha. Run `make test` (and `make eval` if you change agent behaviour)
before you deploy, and follow [docs/GO_LIVE.md](docs/GO_LIVE.md).

## Contributing, security, license

Issues and pull requests are welcome; run `make test` before you open one.
A contribution is licensed like the directory it changes. For `core/` and the
`*-agent/` directories, you also grant Girish Kumar Gouda the right to
relicense it under commercial terms.
Report vulnerabilities privately, as described in [SECURITY.md](SECURITY.md).

Copyright 2026 Girish Kumar Gouda. `core/` and the `*-agent/` directories are
source-available under the [Business Source License 1.1](LICENSE.BSL): free to
use, change and self-host, including in production. Offering them to others as
a hosted service needs a commercial license. Each version becomes Apache-2.0
four years after its release. Everything else is [Apache-2.0](LICENSE).
Details: [LICENSING.md](LICENSING.md).
