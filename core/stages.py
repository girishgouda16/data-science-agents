"""The senior-data-scientist stages every ML agent shares, beyond the gate
engine (core/gates.py) and MLOps (core/mlops.py): recording what a run is FOR
before modelling, the one audited way past a failed gate, a label-leak screen
that works for any column type, and the report a reviewer signs off on.

Each agent keeps what is domain-specific — WHICH gates it requires and how
each is computed — in its own mcp_server/diagnostics.py.
"""

from datetime import datetime, timezone
from pathlib import Path

from . import gates as gate_engine


def record_business_context(
    meta: dict,
    *,
    business_objective: str,
    target_definition: str,
    success_criteria: str,
    domain: str,
    success_metric: str,
    success_threshold,
    direction: str,
    assumptions: str,
    clarifications: str,
    supported_metrics: dict,
) -> dict:
    """Store the business framing on the run. `supported_metrics` maps each
    metric a bar may name to its natural direction ("at_least"/"at_most").
    Raises ValueError for an unmeasurable bar — rejected when it is set, not
    silently ignored when the report is written."""
    if not business_objective.strip() or not target_definition.strip():
        raise ValueError(
            "business_objective and target_definition are required — they define what 'good' means"
        )
    target = None
    if success_metric:
        if success_metric not in supported_metrics:
            raise ValueError(
                f"success_metric must be one of {sorted(supported_metrics)} — got '{success_metric}'"
            )
        target = gate_engine.record_success_target(
            meta,
            success_metric,
            None if success_threshold is None else float(success_threshold),
            tuple(supported_metrics),
            direction or supported_metrics[success_metric],
        )
    meta["business_understanding"] = {
        "business_objective": business_objective,
        "target_definition": target_definition,
        "success_criteria": success_criteria,
        "success_target": target,
        "domain": domain or None,
        "assumptions": (
            [a.strip() for a in assumptions.split("\n") if a.strip()]
            if assumptions
            else []
        ),
        "clarifications": (
            [c.strip() for c in clarifications.split("\n") if c.strip()]
            if clarifications
            else []
        ),
        "recorded_at": datetime.now(timezone.utc).isoformat(),
    }
    return meta["business_understanding"]


def acknowledge_gate(
    meta: dict, gate: str, justification: str, required: tuple
) -> dict:
    """The one way past a FAILED gate: a person's recorded reason. Never a flag."""
    if gate not in required:
        raise ValueError(f"unknown gate '{gate}' — one of {list(required)}")
    if not justification.strip():
        raise ValueError(
            "a justification is required — a gate cannot be waived silently"
        )
    meta.setdefault("acknowledged_gates", {})[gate] = justification
    return meta["acknowledged_gates"]


def apply_acknowledgements(checks: dict, meta: dict) -> dict:
    """A failed gate a person acknowledged passes — with their reason in the
    evidence, so the report shows it was waived and by what argument."""
    for gate, reason in (meta.get("acknowledged_gates") or {}).items():
        if gate in checks and checks[gate]["status"] == gate_engine.FAIL:
            checks[gate] = gate_engine.gate(
                gate_engine.PASS,
                f"ACKNOWLEDGED despite: {checks[gate]['evidence']} — reason on record: {reason}",
            )
    return checks


def single_column_auc(
    df, is_positive, columns: list[str], max_rows: int = 200_000
) -> dict:
    """ROC-AUC of each column ALONE against a binary label: numeric by value
    (either direction), anything else by out-of-fold label rate, so a category
    seen once can't predict itself. >= 0.98 means one column nearly IS the
    label — a copy of it, or a field filled in after the outcome."""
    import numpy as np
    import pandas as pd
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold

    y = np.asarray(is_positive).astype(int)
    if y.min() == y.max():
        return {}
    if len(df) > max_rows:
        keep = np.random.default_rng(42).choice(len(df), max_rows, replace=False)
        df, y = df.iloc[keep], y[keep]
    folds = list(
        StratifiedKFold(n_splits=5, shuffle=True, random_state=42).split(df, y)
    )
    scores = {}
    for col in columns:
        x = df[col]
        if pd.api.types.is_numeric_dtype(x) and not pd.api.types.is_bool_dtype(x):
            auc = roc_auc_score(y, x.fillna(x.median()).to_numpy())
            scores[col] = round(float(max(auc, 1 - auc)), 4)
            continue
        keys, encoded = x.astype(str).to_numpy(), np.empty(len(df))
        for fit_idx, val_idx in folds:
            rates = pd.Series(y[fit_idx]).groupby(keys[fit_idx]).mean()
            encoded[val_idx] = (
                pd.Series(keys[val_idx]).map(rates).fillna(y[fit_idx].mean()).to_numpy()
            )
        scores[col] = round(float(roc_auc_score(y, encoded)), 4)
    return scores


def _table(rows: list[tuple]) -> list[str]:
    return ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]


def render_report(
    agent: str,
    run_id: str,
    meta: dict,
    readiness: dict,
    metrics: dict,
    sections: list[tuple[str, list[str]]] | None = None,
) -> str:
    """The standard report a reviewer signs off on, for any agent: verdict
    first, then what the run was for, the data, the model, how it scored,
    every gate with its evidence, and the execution ledger. Built only from
    the run's artifacts — it cannot make an unfinished run look finished."""
    status = readiness["overall_status"]
    verdict = {
        "ready": "✅ **Proceed to human review** — every mechanical gate passed.",
        "incomplete": "⚠ **Incomplete** — required evidence was never produced: "
        + ", ".join(readiness["not_run_gates"]),
        "blocked": "⛔ **Blocked** — a gate found a problem: "
        + ", ".join(readiness["failed_gates"]),
    }[status]
    bu = meta.get("business_understanding") or {}
    success = readiness["checks"].get("success_criteria", {})
    lines = [
        f"# {agent.title()} report — run `{run_id}`",
        "",
        "## Executive summary",
        "",
        verdict,
        "",
        f"- Model: `{meta.get('model') or meta.get('algorithm')}`"
        + (
            f" with {meta.get('best_params') or meta.get('algorithm_params')}"
            if (meta.get("best_params") or meta.get("algorithm_params"))
            else ""
        ),
        f"- Success bar: {success.get('evidence', 'n/a')}",
        f"- Readiness: {status} ({readiness['gates_passed']}/{readiness['gates_total']} gates)",
        "",
    ]
    lines += ["## Business understanding", ""]
    if bu:
        lines += [
            f"- **Objective:** {bu.get('business_objective')}",
            f"- **Target / unit:** {bu.get('target_definition')}",
            f"- **Success criteria:** {bu.get('success_criteria') or 'not stated'}",
            f"- **Domain:** {bu.get('domain') or 'not declared'}",
        ]
        lines += [f"- Assumption: {a}" for a in bu.get("assumptions") or []]
        lines += [
            f"- Clarified with the user: {c}" for c in bu.get("clarifications") or []
        ]
    else:
        lines += [
            "_Not recorded — record_business_context was never called, so nothing says what this model is for._"
        ]
    lines += [
        "",
        "## Data",
        "",
        f"- Source: `{meta.get('source_path')}`",
        f"- Dropped columns: {meta.get('dropped_columns') or 'none'}",
        f"- Imputation: {meta.get('imputation') or 'none'}",
    ]
    if meta.get("split_type"):
        keys = [k for k in (meta.get("group_column"), meta.get("time_column")) if k]
        lines.append(f"- Split: {meta['split_type']}" + (f" on {', '.join(f'`{k}`' for k in keys)}" if keys else "")
                     + (f"; {meta['entity_overlap_pct']}% of test rows from entities seen in training"
                        if meta.get("entity_overlap_pct") is not None else ""))
    if meta.get("immature_rows_dropped"):
        lines.append(f"- Unsettled targets excluded: {meta['immature_rows_dropped']} rows after "
                     f"{meta.get('immature_after')}")
    source = meta.get("data_source")
    if source:
        lines.append(f"- Loaded from `{source.get('source')}`" + (f", query `{source['query']}`" if source.get("query") else "")
                     + f" — {source.get('rows')} rows at {source.get('fetched_at')}"
                     + (" (truncated at the row cap)" if source.get("truncated_at_max_rows") else ""))
    if meta.get("target_transform"):
        lines.append(f"- Target modelled as {meta['target_transform']}; metrics are in the original units")
    interval = meta.get("prediction_interval")
    if interval:
        lines.append(f"- Prediction interval: ±{interval.get('half_width')} for {interval.get('coverage_target'):.0%} "
                     f"coverage; measured test coverage {interval.get('test_coverage'):.0%}")
    lines.append("")
    lines += [
        "## Metrics",
        "",
        "_Held-out data the model was not fit on unless stated._",
        "",
    ]
    flat = {k: v for k, v in (metrics or {}).items() if not isinstance(v, (dict, list))}
    nested = {
        f"{k}.{kk}": vv
        for k, v in (metrics or {}).items()
        if isinstance(v, dict)
        for kk, vv in v.items()
        if not isinstance(vv, (dict, list))
    }
    lines += _table(
        [("metric", "value"), ("---", "---")] + sorted({**flat, **nested}.items())
    ) + [""]
    for title, body in sections or []:
        lines += [f"## {title}", ""] + body + [""]
    lines += (
        ["## Run readiness", ""]
        + _table(
            [("gate", "status", "evidence"), ("---", "---", "---")]
            + [
                (name, check["status"], check["evidence"].replace("|", "/"))
                for name, check in readiness["checks"].items()
            ]
        )
        + [""]
    )
    log = meta.get("execution_log") or []
    lines += [
        "## Execution log",
        "",
        "_Every tool call recorded on this run, in order._",
        "",
    ]
    lines += [
        f"{i + 1}. `{e['tool']}` — {'ok' if e.get('ok') else 'ERROR: ' + str(e.get('error'))}"
        for i, e in enumerate(log)
    ] or ["_empty_"]
    return "\n".join(lines) + "\n"


def write_report(run_dir: Path, text: str) -> str:
    path = Path(run_dir) / "report.md"
    path.write_text(text)
    return str(path)


def ran_after_training(
    meta: dict, tool: str, training=("train_model", "tune_hyperparams")
) -> bool:
    """Did `tool` succeed AFTER the current model was last fit? Read from the
    execution ledger — a result from an earlier fit describes another model."""
    order = gate_engine.execution_order(meta)
    last_fit = max((i for i, t in enumerate(order) if t in training), default=None)
    ran = gate_engine.last_index(order, tool)
    return ran is not None and last_fit is not None and ran > last_fit


def ran_before_training(meta: dict, tool: str) -> bool:
    order = gate_engine.execution_order(meta)
    ran, fit = (
        gate_engine.last_index(order, tool),
        gate_engine.last_index(order, "train_model"),
    )
    return ran is not None and (fit is None or ran < fit)


def register_tools(
    mcp,
    agent: str,
    *,
    load_meta,
    save_meta,
    run_dir,
    required_gates: tuple,
    supported_metrics: dict,
    compute_readiness,
    report_metrics,
    report_sections=None,
    include=None,
) -> tuple:
    """Add the shared senior-DS tools to an agent's MCP server and return them
    in this order: record_business_context, acknowledge_identifier_column,
    acknowledge_gate, check_readiness, generate_report. `include` limits which
    are registered (an agent that already has its own keeps it); the tuple
    then holds None for those left out."""
    import json

    wanted = set(
        include
        or (
            "record_business_context",
            "acknowledge_identifier_column",
            "acknowledge_gate",
            "check_readiness",
            "generate_report",
        )
    )
    registered = {}

    def tool(fn):
        if fn.__name__ in wanted:
            registered[fn.__name__] = mcp.tool()(fn)
        return fn

    @tool
    def record_business_context(
        run_id: str,
        business_objective: str,
        target_definition: str,
        success_criteria: str = "",
        domain: str = "",
        success_metric: str = "",
        success_threshold: float | None = None,
        direction: str = "",
        assumptions: str = "",
        clarifications: str = "",
    ) -> str:
        """Record what this run is FOR, before modelling: the decision it feeds
        (business_objective), what the target/unit is (target_definition), the
        success bar the user states (success_metric + success_threshold — ask for
        it; NEVER invent one), and every assumption made instead of asking /
        every question asked (newline-separated). The report renders all of it."""
        meta = load_meta(run_id)
        try:
            stored = record_business_context_on(
                meta,
                business_objective,
                target_definition,
                success_criteria,
                domain,
                success_metric,
                success_threshold,
                direction,
                assumptions,
                clarifications,
                supported_metrics,
            )
        except ValueError as exc:
            return json.dumps({"error": str(exc)})
        save_meta(run_id, meta)
        return json.dumps(
            {"run_id": run_id, "business_understanding": stored}, default=str
        )

    @tool
    def acknowledge_identifier_column(
        run_id: str, column: str, justification: str
    ) -> str:
        """Record that a column flagged as identifier-shaped (or as a near-copy of
        the label) is genuinely a feature here, with the person's reason. The
        only way past that gate without dropping the column; an acknowledgement
        nobody gave you is an assumption with a signature on it."""
        if not justification.strip():
            return json.dumps(
                {
                    "error": "a justification is required — this gate cannot be waived silently"
                }
            )
        meta = load_meta(run_id)
        meta.setdefault("acknowledged_identifiers", {})[column] = justification
        save_meta(run_id, meta)
        return json.dumps(
            {"run_id": run_id, "acknowledged": column, "justification": justification}
        )

    @tool
    def acknowledge_gate(run_id: str, gate: str, justification: str) -> str:
        """Record a person's reason for accepting a FAILED gate (e.g. weak
        cluster structure the business still wants to act on). The gate then
        passes, with that reason printed in the report. Ask the user first."""
        meta = load_meta(run_id)
        try:
            acknowledge_gate_on(meta, gate, justification, required_gates)
        except ValueError as exc:
            return json.dumps({"error": str(exc)})
        save_meta(run_id, meta)
        return json.dumps(
            {
                "run_id": run_id,
                "acknowledged_gate": gate,
                "justification": justification,
            }
        )

    @tool
    def check_readiness(run_id: str) -> str:
        """Deterministic readiness gate — computed from this run's artifacts,
        not from anything the agent asserts. blocked = a gate found a problem
        (export refuses); incomplete = required evidence never produced (NOT a
        pass — go run it); ready = every mechanical gate passed. Loop on
        not_run_gates until ready."""
        return json.dumps(compute_readiness(run_id), default=str)

    @tool
    def generate_report(run_id: str) -> str:
        """The standard report a reviewer signs off on — verdict first, business
        framing, data, model, metrics, every gate with its evidence, the
        execution ledger — written to report.md in the run (log_run_to_mlflow
        ships it). Built only from artifacts; hand it to the user as-is."""
        meta = load_meta(run_id)
        text = render_report(
            agent,
            run_id,
            meta,
            compute_readiness(run_id),
            report_metrics(meta),
            report_sections(meta) if report_sections else None,
        )
        write_report(run_dir(run_id), text)
        return text

    return tuple(
        registered.get(name)
        for name in (
            "record_business_context",
            "acknowledge_identifier_column",
            "acknowledge_gate",
            "check_readiness",
            "generate_report",
        )
    )


def record_business_context_on(
    meta,
    business_objective,
    target_definition,
    success_criteria,
    domain,
    success_metric,
    success_threshold,
    direction,
    assumptions,
    clarifications,
    supported_metrics,
):
    return record_business_context(
        meta,
        business_objective=business_objective,
        target_definition=target_definition,
        success_criteria=success_criteria,
        domain=domain,
        success_metric=success_metric,
        success_threshold=success_threshold,
        direction=direction,
        assumptions=assumptions,
        clarifications=clarifications,
        supported_metrics=supported_metrics,
    )


acknowledge_gate_on = acknowledge_gate
