"""Pre-launch LIVE smoke test: does each training agent's LLM actually follow
the senior-data-scientist workflow end to end, not just its tools?

For regression, clustering, anomaly and forecasting: hand the real agent (its
real LLM, in-process — the same _run_turn the A2A server uses) a small dataset
and "run the full pipeline and log it to MLflow", answer its questions from a
script, then grade the run from its ARTIFACTS:

    business context recorded · leakage screen BEFORE training · a model
    trained · readiness checked · report written · registered in MLflow

Costs real LLM calls (a few minutes per agent) and needs the provider key in
.env, so it is run by hand before a release, not in CI:

    python scripts/live_smoke_agents.py                 # all five, temp stores
    python scripts/live_smoke_agents.py forecasting     # one
    python scripts/live_smoke_agents.py --real          # into the REAL registry + data dir
By default writes only to a temp data dir and a temp MLflow store. --real
uses the configured stores (.env / <repo>/mlruns, <repo>/data), so every
agent's model appears in the MLflow UI (make mlflow) and stays promotable.
"""

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
AGENTS = ["classification", "regression", "clustering", "anomaly", "forecasting"]

# Runs inside the agent's scripts/ dir, so `import agent` is that agent.
_CHILD = r"""
import asyncio, json, re, sys
from pathlib import Path
import numpy as np, pandas as pd
import agent, mcp_server as m

name, workdir = sys.argv[1], Path(sys.argv[2])
existing_runs = {d.name for d in m.RUNS_DIR.iterdir()} if m.RUNS_DIR.exists() else set()
rng = np.random.default_rng(0)
if name == "classification":
    n = 800; tenure, calls = rng.integers(1, 72, n), rng.poisson(2, n)
    churned = (rng.random(n) < 1 / (1 + np.exp(0.06 * tenure - 0.6 * calls))).astype(int)
    df = pd.DataFrame({"customer_id": [f"C{i:05d}" for i in range(n)], "tenure_months": tenure, "support_calls": calls,
                       "monthly_charges": rng.normal(70, 20, n).round(2), "contract": rng.choice(["monthly", "annual"], n),
                       "churned": churned})
    request = "Train a classification model to predict churned."
elif name == "regression":
    n = 400; x1, x2 = rng.normal(0, 1, n), rng.normal(5, 2, n)
    df = pd.DataFrame({"customer_id": [f"C{i:05d}" for i in range(n)], "tenure": x1, "usage": x2,
                       "plan": rng.choice(["a", "b"], n), "monthly_bill": 3 * x1 + x2 + rng.normal(0, 0.5, n)})
    request = "Train a regression model to predict monthly_bill."
elif name == "clustering":
    pts = np.vstack([c + rng.normal(0, 0.8, (120, 2)) for c in ([0, 0], [8, 8], [0, 8])])
    df = pd.DataFrame(pts, columns=["recency", "frequency"]); df.insert(0, "customer_id", [f"C{i:05d}" for i in range(len(df))])
    request = "Segment these customers into groups."
elif name == "anomaly":
    n = 600; df = pd.DataFrame({"calls": rng.poisson(8, n).astype(float), "avg_sec": rng.normal(90, 20, n)})
    df.loc[:14, ["calls", "avg_sec"]] = [[60, 4]] * 15
    request = "Find the anomalous subscribers in this usage data."
else:
    t = np.arange(240); df = pd.DataFrame({"date": pd.date_range("2024-01-01", periods=240, freq="D"),
                                          "sales": 100 + 0.5 * t + 10 * np.sin(t * 2 * np.pi / 7) + rng.normal(0, 1, 240)})
    request = "Forecast daily sales."
path = workdir / f"{name}.csv"; df.to_csv(path, index=False)
answers = [(r"bar|success|target|threshold|metric|good enough", "Use your recommended bar and record it as an assumption."),
           (r"split|group|leak", "Random stratified split is fine."),
           (r"customer_id|identifier|drop", "Drop identifier columns."), (r"date|time", "Use the date column.")]
def reply(q):
    return " ".join(dict.fromkeys(r for p, r in answers if re.search(p, q, re.I))) or "Use your recommended option."
from langgraph.checkpoint.memory import InMemorySaver
host, questions = agent.Host(agent.SPEC, checkpointer=InMemorySaver()), []
text, awaiting = asyncio.run(host.turn("smoke", f"Dataset: {path}\n{request} Run the full pipeline end to end the way "
                                       "a senior data scientist would, generate the report, then log the run to MLflow."))
for _ in range(14):
    if awaiting or text.rstrip().endswith("?"):
        questions.append(text); answer = reply(text)
    elif text.startswith("Paused after"):
        answer = "continue"
    else:
        break
    text, awaiting = asyncio.run(host.turn("smoke", answer))
runs = [d for d in m.RUNS_DIR.iterdir() if d.name not in existing_runs and (d / "meta.json").exists()]
meta = max((json.loads((d / "meta.json").read_text()) | {"_run_id": d.name} for d in runs),
           key=lambda mt: len(mt.get("execution_log") or []), default=None)
tools = [e["tool"] for e in (meta or {}).get("execution_log") or [] if e.get("ok")]
def ran_before(a, b): return a in tools and b in tools and tools.index(a) < tools.index(b)
checks = {
    "business context recorded": bool(meta and (meta.get("business_understanding") or {}).get("business_objective")),
    "leakage screen before training": ran_before("detect_data_leakage", "train_model"),
    "model trained": "train_model" in tools,
    "readiness checked": "check_readiness" in tools,
    "report written": bool(meta) and (m.RUNS_DIR / meta["_run_id"] / "report.md").exists(),
    "registered in MLflow": bool(meta and meta.get("registry")),
}
readiness = m.compute_readiness(meta["_run_id"])["overall_status"] if meta else None
print("RESULT " + json.dumps({"agent": name, "checks": checks, "readiness": readiness, "questions": len(questions),
                               "registry": (meta or {}).get("registry"),
                               "tools": tools, "last_reply": text[:300]}))
"""


def main(names: list[str]) -> int:
    real = "--real" in names
    names = [n for n in names if n != "--real"]
    failed = 0
    for name in names or AGENTS:
        if real:  # the configured stores; the dataset lives in data/smoke so its source path stays valid
            work = ROOT / "data" / "smoke"
            work.mkdir(parents=True, exist_ok=True)
            env = dict(os.environ)
        else:
            work = Path(tempfile.mkdtemp(prefix=f"live-{name}-"))
            env = {
                **os.environ,
                "AGENTIC_ML_DATA_DIR": str(work / "data"),
                "MLFLOW_TRACKING_URI": f"file:{work / 'mlruns'}",
            }
        proc = subprocess.run(
            [sys.executable, "-c", _CHILD, name, str(work)],
            cwd=ROOT / f"{name}-agent" / "scripts",
            env=env,
            capture_output=True,
            text=True,
        )
        line = next(
            (l for l in proc.stdout.splitlines() if l.startswith("RESULT ")), None
        )
        if not line:
            failed += 1
            print(f"{name}: CRASHED\n{proc.stderr[-1500:]}")
            continue
        result = json.loads(line[len("RESULT ") :])
        print(
            f"{name}  (readiness: {result['readiness']}, {result['questions']} question(s), "
            f"registry: {result.get('registry')})",
            flush=True,
        )
        for check, ok in result["checks"].items():
            failed += not ok
            print(f"  {'PASS' if ok else 'FAIL'}  {check}")
        if not all(result["checks"].values()):
            print(
                f"  tools run: {result['tools']}\n  last reply: {result['last_reply']}"
            )
    print(
        "\nall live checks passed" if not failed else f"\n{failed} live check(s) failed"
    )
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
