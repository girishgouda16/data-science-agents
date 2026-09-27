"""Agent-level evals, shared by every agent: does the LLM make the RIGHT
CALLS, not just do the tools work. Each scenario hands the real agent (real
LLM, in-process, the same LangGraph host the A2A server runs) a dataset and a
request, answers its questions from a script, and grades the decisions from
the run's ARTIFACTS — never from what the agent said it did.

Each agent's eval_agent.py sets AGENTIC_ML_DATA_DIR / MLFLOW_TRACKING_URI to
a temp dir BEFORE importing its agent, defines datasets, checks and
scenarios, and calls main(). Costs real LLM calls, so it is not in CI.
"""
import asyncio
import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

MAX_EXCHANGES = 12  # questions answered + "continue"s before a scenario is called stuck
DEFAULT_REPLY = "Use your recommended option."


@dataclass
class Scenario:
    name: str
    target: str
    make_csv: Callable[[Path], Path]
    request: str
    answers: list[tuple[str, str]]  # (regex on the question, scripted reply) — every match is sent
    checks: list[Callable] = field(default_factory=list)
    stateless: bool = False  # no run artifacts (drift): graded from the transcript alone


# ── checks shared by every agent: each returns (name, passed, detail) ─────────

def tools_run(meta) -> list:
    return [e["tool"] for e in meta.get("execution_log") or [] if e.get("ok")]


def ran(*tools):
    def check(meta, questions):
        missing = [t for t in tools if t not in tools_run(meta)]
        return f"runs {', '.join(tools)}", not missing, f"never ran {missing}" if missing else "ran"
    return check


def asked(pattern, what):
    def check(meta, questions):
        hit = next((q for q in questions if re.search(pattern, q, re.I)), None)
        return f"asks {what}", bool(hit), (hit or "never asked")[:120]
    return check


def called(*tools):
    """From the transcript: the tools the agent called (any agent)."""
    def check(meta, questions):
        made = [name for name, _ in meta.get("_calls") or []]
        missing = [t for t in tools if t not in made]
        return f"calls {', '.join(tools)}", not missing, f"never called {missing}" if missing else "called"
    return check


def says(pattern, what):
    def check(meta, questions):
        hit = re.search(pattern, meta.get("_reply") or "", re.I)
        return f"says {what}", bool(hit), (hit.group(0) if hit else (meta.get("_reply") or "")[:120])
    return check


NEGATION = re.compile(r"\b(not|no|never|cannot|can't|doesn't|does not|isn't|is not|without)\b", re.I)


def never_says(pattern, what):
    """The claim, asserted — a sentence that negates it ("this does not mean
    the model is still accurate") is the caveat we want, not the claim."""
    def check(meta, questions):
        sentences = re.split(r"(?<=[.!?\n])\s+", meta.get("_reply") or "")
        claims = [x for x in sentences if re.search(pattern, x, re.I) and not NEGATION.search(x)]
        return f"never says {what}", not claims, claims[0][:120] if claims else "not said"
    return check


def no_bar(meta, questions):
    got = (meta.get("business_understanding") or {}).get("success_target")
    return "invents no bar", not got, f"recorded {got}" if got else "none recorded, as the user said"


def out_of_model(*columns):
    """Dropped, a split key (kept for validation, excluded by the pipeline),
    or a clustering profile column (kept to describe segments, never in the
    distance)."""
    def check(meta, questions):
        keys = {meta.get("group_column"), meta.get("time_column"), *(meta.get("profile_columns") or [])}
        kept = [c for c in columns if c not in (meta.get("dropped_columns") or {}) and c not in keys]
        return f"keeps {list(columns)} out of the model", not kept, f"still a feature: {kept}" if kept else "out"
    return check


# ── the runner ────────────────────────────────────────────────────────────────

def reply(scenario: Scenario, question: str) -> str:
    """Every scripted answer whose topic the question touches — agents bundle
    decisions, and answering only the first leaves the rest unanswered."""
    replies = [r for pattern, r in scenario.answers if re.search(pattern, question, re.I)]
    return " ".join(dict.fromkeys(replies)) or DEFAULT_REPLY


async def run(scenario: Scenario, agent, runs_dir: Path, workdir: Path) -> dict:
    from langgraph.checkpoint.memory import InMemorySaver

    csv = scenario.make_csv(workdir)
    host, thread = agent.Host(agent.SPEC, checkpointer=InMemorySaver()), f"eval-{scenario.name}"
    questions, started = [], time.time()
    before = set(runs_dir.iterdir()) if runs_dir.exists() else set()
    text, awaiting = await host.turn(thread, scenario.request.format(path=csv, target=scenario.target))
    for _ in range(MAX_EXCHANGES):
        if awaiting or text.rstrip().endswith("?"):  # ask_user, or a question asked in prose
            questions.append(text)
            answer = reply(scenario, text)
        elif text.startswith("Paused after"):
            answer = "continue"
        else:
            break
        text, awaiting = await host.turn(thread, answer)
    state = await host.state(thread)
    transcript = {"_reply": text, "_calls": [(c["function"]["name"], c["function"].get("arguments"))
                                             for msg in state.get("messages") or [] for c in msg.get("tool_calls") or []]}
    runs = [d for d in set(runs_dir.iterdir()) - before if (d / "meta.json").exists()] if runs_dir.exists() else []
    metas = sorted((json.loads((d / "meta.json").read_text()) | {"_run_id": d.name} for d in runs),
                   key=lambda meta: len(meta.get("execution_log") or []))
    meta = next((mt for mt in reversed(metas) if mt.get("target") == scenario.target), None)
    meta = transcript if scenario.stateless else meta and meta | transcript
    results = ([check(meta, questions) for check in scenario.checks] if meta else
               [("creates a run", False, f"no run for target {scenario.target}; last reply: {text[:200]}")])
    return {"scenario": scenario.name, "seconds": round(time.time() - started), "questions": questions,
            "reply": text[:3000], "tools_called": [name for name, _ in transcript["_calls"]],
            "run_id": meta and meta.get("_run_id"),
            "checks": [{"check": n, "passed": p, "detail": d} for n, p, d in results]}


def main(names: list[str], scenarios: list[Scenario], agent, runs_dir: Path, workdir: Path, repo: Path) -> int:
    chosen = [s for s in scenarios if not names or s.name in names]
    if not chosen:
        print(f"unknown scenario(s) {names}; choose from {[s.name for s in scenarios]}")
        return 2
    print(f"model: {agent.HOST.model}   workdir: {workdir}\n")
    real_runs = repo / "data" / "runs"
    untouched = set(real_runs.iterdir()) if real_runs.exists() else set()
    report, failed = [], 0
    for scenario in chosen:
        result = asyncio.run(run(scenario, agent, runs_dir, workdir))
        report.append(result)
        print(f"{scenario.name}  ({result['seconds']}s, {len(result['questions'])} question(s))")
        for c in result["checks"]:
            failed += not c["passed"]
            print(f"  {'PASS' if c['passed'] else 'FAIL'}  {c['check']:45s} {c['detail']}")
    leaked = (set(real_runs.iterdir()) if real_runs.exists() else set()) - untouched
    if leaked:  # the MCP server must inherit AGENTIC_ML_DATA_DIR — if not, the eval wrote into real data
        print(f"\nERROR: the agent's MCP server wrote into the real {real_runs}: {sorted(p.name for p in leaked)}")
        failed += 1
    out = workdir / "eval_results.json"
    out.write_text(json.dumps(report, indent=2, default=str))
    total = sum(len(r["checks"]) for r in report)
    print(f"\n{total - failed}/{total} checks passed — details: {out}")
    return 1 if failed else 0
