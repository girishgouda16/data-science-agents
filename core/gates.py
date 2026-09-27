"""The deterministic readiness layer, shared by every ML agent.

Generalised out of classification-agent's gates.py, which is the only agent
that had one. Every other module in an agent answers "what did we find?".
This one answers "is this run actually finished, and is it safe to believe?"
— from artifacts on disk, never from a claim the model wrote.

What lives here is the MECHANISM, which is domain-independent:

  * four statuses, and the rule that `not_run` is not `pass` — absence of
    evidence is never evidence of passing. A gate that legitimately does not
    apply resolves to `not_applicable` WITH a recorded reason; it never
    silently disappears,
  * `readiness()` — roll a domain's gate results up into blocked /
    incomplete / ready,
  * `evaluate_success_criteria()` — compare against a bar recorded BEFORE
    modelling, or report that none exists. It never derives one,
  * `record_execution()` / `execution_order()` — the ledger that makes "that
    tool wasn't available" a checkable claim, and lets a domain assert that
    its screens ran in an order that still describes the current model,
  * `acknowledge()` — the one way past a gate: a recorded human justification.

What does NOT live here is which gates a domain requires or how each is
computed. Those stay in the agent, because "no leakage" means something
different to a forecaster than to a classifier. An agent defines its own
REQUIRED_GATES and builds a {gate_name: gate(...)} dict; this module turns
that into a verdict.
"""

from datetime import datetime, timezone

PASS, FAIL, NOT_RUN, NOT_APPLICABLE = "pass", "fail", "not_run", "not_applicable"

_INTERPRETATION = {
    "blocked": "At least one gate found positive evidence of a problem. Do not ship; fix and re-run.",
    "incomplete": "No failures, but required evidence is missing. Absence of evidence is NOT a pass — run the missing steps.",
    "ready": "Every mechanical gate passed. This covers execution correctness only; methodological and business judgment still need human review.",
}


def gate(status: str, evidence: str) -> dict:
    """One gate's result. `evidence` is what makes it checkable — a number, a
    column name, a tool that did or did not run; never an adjective."""
    if status not in (PASS, FAIL, NOT_RUN, NOT_APPLICABLE):
        raise ValueError(f"unknown gate status {status!r}")
    return {"status": status, "evidence": evidence}


def readiness(run_id: str, checks: dict, required: tuple) -> dict:
    """Roll a domain's gate results up into one verdict.

    A required gate that the domain never produced counts as not_run — a gate
    silently missing from the dict is exactly the failure mode this layer
    exists to stop.
    """
    checks = {
        name: checks.get(
            name, gate(NOT_RUN, "this gate was never computed for this run")
        )
        for name in required
    }
    failed = [k for k, v in checks.items() if v["status"] == FAIL]
    not_run = [k for k, v in checks.items() if v["status"] == NOT_RUN]
    passed = [k for k, v in checks.items() if v["status"] in (PASS, NOT_APPLICABLE)]
    overall = "blocked" if failed else ("incomplete" if not_run else "ready")
    return {
        "run_id": run_id,
        "overall_status": overall,
        "gates_passed": len(passed),
        "gates_total": len(required),
        # Deliberately a fraction of deterministic checks, not a probability.
        # It says "this much of the mechanical contract is satisfied" — it is
        # NOT a claim that the modelling choices or the business framing are
        # right, which no gate here can measure.
        "mechanical_completeness": (
            round(len(passed) / len(required), 3) if required else 0.0
        ),
        "failed_gates": failed,
        "not_run_gates": not_run,
        "checks": checks,
        "interpretation": _INTERPRETATION[overall],
    }


def evaluate_success_criteria(meta: dict, measured, supported_metrics: tuple) -> dict:
    """The bar the run is judged against, or the finding that none was set.

    Never derives a threshold and never rounds one into existence. A report
    that announces "criteria not met: 0.822 vs the 0.85 target" for a target
    nobody set is worse than a report with no bar at all.

    `measured` is the domain's lookup: (meta, metric_name) -> value or None
    when that metric genuinely isn't computable for this run.
    """
    target = (meta.get("business_understanding") or {}).get("success_target") or {}
    metric, threshold = target.get("metric"), target.get("threshold")
    if not metric or threshold is None:
        return gate(
            NOT_RUN,
            "no measurable success threshold was recorded for this run — record_business_context was called "
            "without success_metric/success_threshold, so there is no bar to judge the model against. "
            "A threshold must NOT be invented at report time.",
        )
    if metric not in supported_metrics:
        return gate(
            FAIL,
            f"recorded success metric '{metric}' is not one this agent can measure ({', '.join(supported_metrics)})",
        )
    value = measured(meta, metric)
    if value is None:
        return gate(
            NOT_RUN,
            f"'{metric}' is not computable for this run — nothing to compare against the {threshold} bar",
        )
    direction = target.get("direction", "at_least")
    met = value >= threshold if direction == "at_least" else value <= threshold
    comparison = "at least" if direction == "at_least" else "at most"
    return gate(
        PASS if met else FAIL,
        f"{metric} {round(float(value), 4)} vs the recorded bar of {comparison} {threshold}",
    )


def record_success_target(
    meta: dict,
    metric: str | None,
    threshold: float | None,
    supported_metrics: tuple,
    direction: str = "at_least",
) -> dict | None:
    """Validate a bar at RECORD time, so an unmeasurable metric is rejected
    when it is set rather than silently ignored when the report is written.
    Returns the target to store, or None when no bar was given."""
    if not metric or threshold is None:
        return None
    if metric not in supported_metrics:
        raise ValueError(
            f"success_metric must be one of {', '.join(supported_metrics)} — got '{metric}'"
        )
    if direction not in ("at_least", "at_most"):
        raise ValueError("direction must be 'at_least' or 'at_most'")
    return {"metric": metric, "threshold": float(threshold), "direction": direction}


def record_execution(meta: dict, tool: str, ok: bool = True, detail: str = "") -> dict:
    """Append to the run's execution ledger. This is what makes "that tool was
    not available in this session" a checkable claim rather than a story."""
    log = meta.setdefault("execution_log", [])
    log.append(
        {
            "tool": tool,
            "ok": bool(ok),
            "detail": detail,
            "at": datetime.now(timezone.utc).isoformat(),
        }
    )
    return meta


def execution_order(meta: dict) -> list:
    """Tools that actually succeeded, in order — the input to any ordering check."""
    return [
        entry.get("tool")
        for entry in (meta.get("execution_log") or [])
        if entry.get("ok")
    ]


def last_index(order: list, tool: str):
    """Where `tool` last ran, or None. Ordering gates are written against this:
    a screen that ran BEFORE the step that changed what it screened is stale."""
    return len(order) - 1 - order[::-1].index(tool) if tool in order else None


def check_screen_ordering(
    meta: dict, screen: str, mutators: tuple, model_step: str = "train_model"
) -> dict:
    """Generic form of the ordering failure that no other gate can see: a
    screen runs, the feature set or the fitted model changes underneath it,
    and the stored result keeps looking reasonable while describing something
    that no longer exists."""
    order = execution_order(meta)
    if not order:
        return gate(
            NOT_RUN, "no execution log on this run — ordering cannot be verified"
        )
    screened_at, trained_at = last_index(order, screen), last_index(order, model_step)
    problems = [
        f"{screen} ran before `{mutator}` changed the feature set — it screened columns that are no longer "
        "the ones being modelled; re-run it"
        for mutator in mutators
        if (changed := last_index(order, mutator)) is not None
        and screened_at is not None
        and changed > screened_at
    ]
    if screened_at is None and trained_at is not None:
        problems.append(f"a model was trained without {screen} having run at all")
    if problems:
        return gate(FAIL, "; ".join(problems))
    return gate(
        PASS,
        "every recorded check describes the current feature set and the current fitted model",
    )


def acknowledge(meta: dict, kind: str, key: str, justification: str) -> dict:
    """The only way past a gate: a recorded human justification, not a flag.
    Stored under meta['acknowledged_<kind>'] so the gate can name who let it
    through and why."""
    if not justification or not justification.strip():
        raise ValueError(
            "a justification is required — an acknowledgement with no reason is just a silenced gate"
        )
    meta.setdefault(f"acknowledged_{kind}", {})[key] = {
        "justification": justification.strip(),
        "at": datetime.now(timezone.utc).isoformat(),
    }
    return meta


def install_ledger(mcp, run_dir, load_meta, save_meta):
    """Make every @mcp.tool() on this server append to its run's execution
    ledger. Call once, before the tool modules are imported.

    Wrapping the decorator instead of editing every tool body is deliberate: a
    per-tool call is one more thing to forget on tool #40, and forgetting it
    silently re-opens the hole the ledger closes — a report is only as
    trustworthy as the claim that the work behind it happened.
    """
    import functools
    import inspect
    import json
    import time

    from . import runtime

    register = mcp.tool

    def _resolve_run_id(fn, args, kwargs, result):
        """A tool either takes run_id (most of them) or creates one and returns
        it (prepare_dataset) — or neither (eda/predict work off a path)."""
        try:
            bound = inspect.signature(fn).bind(*args, **kwargs)
            bound.apply_defaults()
            if run_id := bound.arguments.get("run_id"):
                return str(run_id)
        except TypeError:
            pass
        if isinstance(result, str):
            try:
                return json.loads(result).get("run_id")
            except (json.JSONDecodeError, AttributeError):
                pass
        return None

    def _record_step(fn, args, kwargs, result, error, started):
        """Never raises — a broken ledger must not break a working tool call,
        and a tool whose ledger write failed is still a tool that ran."""
        try:
            run_id = _resolve_run_id(fn, args, kwargs, result)
            if not run_id or not (run_dir(run_id) / "meta.json").exists():
                return
            # A tool can return {"error": ...} as a normal value rather than
            # raising — that is still a step that did not do its job.
            if error is None and isinstance(result, str):
                try:
                    error = json.loads(result).get("error")
                except (json.JSONDecodeError, AttributeError):
                    pass
            meta = runtime.stamp_owner(load_meta(run_id))
            meta.setdefault("execution_log", []).append(
                {
                    "tool": fn.__name__,
                    "ok": error is None,
                    "error": error,
                    "at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                    "seconds": round(time.time() - started, 3),
                    "user": runtime.user(),
                }
            )
            save_meta(run_id, meta)
        except Exception:
            pass

    def _denied(fn, args, kwargs) -> str | None:
        """Another user's run is refused before the tool touches it."""
        run_id = _resolve_run_id(fn, args, kwargs, None)
        if not run_id or not (run_dir(run_id) / "meta.json").exists():
            return None
        return runtime.access_error(load_meta(run_id).get("owner"))

    def tool_with_ledger(*d_args, **d_kwargs):
        wrap_register = register(*d_args, **d_kwargs)

        def decorate(fn):
            @functools.wraps(
                fn
            )  # FastMCP builds its JSON schema from the signature — wraps keeps it intact
            def wrapper(*args, **kwargs):
                if denied := _denied(fn, args, kwargs):
                    return json.dumps({"error": denied})
                started = time.time()
                try:
                    result = fn(*args, **kwargs)
                except Exception as exc:
                    _record_step(
                        fn, args, kwargs, None, f"{type(exc).__name__}: {exc}", started
                    )
                    raise
                _record_step(fn, args, kwargs, result, None, started)
                return result

            return wrap_register(wrapper)

        return decorate

    mcp.tool = tool_with_ledger
    return mcp
