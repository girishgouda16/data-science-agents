"""Feature-engineering tools shared by every supervised agent
(classification, regression): business-meaning features built from the
mechanism behind the label, not mined from correlations. Target-agnostic —
each agent binds its own run storage and its own "signal" measure (AUC for
classification, rank correlation for regression) and registers the tools on
its MCP server:

    feature_tools.bind(load_split=..., run_dir=..., save_meta=..., signal=...)
    globals().update(feature_tools.register(mcp))

Each agent runs as its own process, so the binding is per process.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import runtime, toolguard
from .datasource import read_table
from .validation import _parse_dates as parse_dates

_store: dict = {}


def bind(*, load_split, run_dir, save_meta, signal) -> None:
    """signal(train, target, meta, columns) -> {"scores": {col: float},
    "metric": str, "suspect": float, "weak": float, "groups": Series} —
    the per-feature strength measure and how profile_features groups rows."""
    _store.update(load_split=load_split, run_dir=run_dir, save_meta=save_meta, signal=signal)


def _load_split(run_id):
    return _store["load_split"](run_id)


def _run_dir(run_id):
    return _store["run_dir"](run_id)


def _save_meta(run_id, meta):
    return _store["save_meta"](run_id, meta)


def _eval_safe(col) -> str:
    """pandas eval cannot parse a name that starts with a digit, backticks or
    not — and pre-aggregated telco exports are full of them (`7_lag_to`).
    Expressions reference those as `_7_lag_to`."""
    return f"_{col}" if str(col)[:1].isdigit() else col


def _reasoned_features(meta: dict) -> dict:
    """{name: {"formula": str, "rationale": str}} for every engineered
    feature, normalizing the two older shapes this field has had (a bare
    list of names, then {name: formula}) so reporting can render one shape
    without caring which vintage of run it is reading."""
    existing = meta.get("engineered_features") or {}
    if isinstance(existing, list):
        return {
            name: {
                "formula": "not recorded (added before this field tracked it)",
                "rationale": "",
            }
            for name in existing
        }
    normalized = {}
    for name, value in existing.items():
        normalized[name] = (
            value if isinstance(value, dict) else {"formula": value, "rationale": ""}
        )
    return normalized


def propose_features(run_id: str, top_k: int = 5) -> str:
    """Rank training-fold numeric columns by |correlation| with the target,
    then propose pairwise ratio/product features among the top_k. Read-only,
    generic/correlation-driven — feeds into apply_features after user
    approval. Prefer named domain templates (apply_custom_feature) over
    these whenever the domain is known; these are the fallback, not the
    first move (feature-engineering skill: this is a judgment call, kept
    HITL unlike imputation)."""
    train, _, meta = _load_split(run_id)
    target = meta["target"]
    numeric = train.select_dtypes(include="number").drop(
        columns=[target], errors="ignore"
    )
    if numeric.empty:
        return json.dumps(
            {"candidates": [], "note": "no numeric columns to derive features from"}
        )
    y = (
        pd.factorize(train[target])[0]
        if not pd.api.types.is_numeric_dtype(train[target])
        else train[target]
    )
    corr = (
        numeric.corrwith(pd.Series(y, index=train.index))
        .abs()
        .sort_values(ascending=False)
    )
    top_cols = list(corr.head(top_k).index)

    candidates = {}
    for i, a in enumerate(top_cols):
        for b in top_cols[i + 1 :]:
            candidates[f"{a}_div_{b}"] = {
                "formula": f"{a} / ({b} + 1e-6)",
                "a_corr": round(float(corr[a]), 4),
                "b_corr": round(float(corr[b]), 4),
            }
            candidates[f"{a}_times_{b}"] = {
                "formula": f"{a} * {b}",
                "a_corr": round(float(corr[a]), 4),
                "b_corr": round(float(corr[b]), 4),
            }
    return json.dumps(
        {"run_id": run_id, "top_correlated_columns": top_cols, "candidates": candidates}
    )


def apply_features(run_id: str, features: str) -> str:
    """Add user-approved derived features from propose_features's candidate
    names (comma-separated, or "all"). Computed directly from existing
    columns on both train and test — no fitted statistic, so nothing can
    leak train into test."""
    train, test, meta = _load_split(run_id)
    proposals = json.loads(propose_features(run_id))["candidates"]
    names = (
        list(proposals)
        if features.strip().lower() == "all"
        else [f.strip() for f in features.split(",") if f.strip()]
    )
    unknown = [n for n in names if n not in proposals]
    if unknown:
        return json.dumps(
            {"error": f"unknown feature(s) {unknown}, call propose_features first"}
        )

    added = []
    for df in (train, test):
        for name in names:
            if "_div_" in name:
                a, b = name.split("_div_")
                df[name] = df[a] / (df[b] + 1e-6)
            else:
                a, b = name.split("_times_")
                df[name] = df[a] * df[b]
        added = names
    train.to_csv(_run_dir(run_id) / "train.csv", index=False)
    test.to_csv(_run_dir(run_id) / "test.csv", index=False)
    formulas = {
        name: {
            "formula": proposals[name]["formula"],
            # No mechanism to state: these candidates come from correlation
            # mining, which is the fallback path precisely because it cannot
            # say why a feature should matter.
            "rationale": "correlation-mined candidate (propose_features) — no business mechanism stated",
        }
        for name in added
    }
    meta["engineered_features"] = {**_reasoned_features(meta), **formulas}
    _save_meta(run_id, meta)
    return json.dumps(
        {"run_id": run_id, "added_features": added, "train_shape": train.shape}
    )


def apply_custom_feature(
    run_id: str, name: str, expression: str, rationale: str = ""
) -> str:
    """Add one business-meaning feature via a pandas eval expression over
    EXISTING columns (row-wise only — no groupby/window support here), on
    both train and test. This is the PRIMARY tool for domain templates
    (Wangiri calling-pattern ratios, churn trend signals, credit ratios —
    see feature-engineering/references/domain-features.md); reach for it
    before propose_features/apply_features whenever the domain is known,
    since a business formula beats a correlation-mined one. Example:
    name="short_call_ratio", expression="short_calls / (total_calls + 1e-6)"
    where short_calls and total_calls are columns already present. Features
    needing a groupby are not row-wise: event rows -> per-entity aggregates
    is aggregate_events (before prepare_dataset); per-entity history on
    entity-per-period rows is apply_entity_features. This tool then builds
    ratios between their outputs. A column whose name starts with a digit is
    written with a leading underscore (7_lag_to -> _7_lag_to); a name with
    spaces goes in backticks.

    rationale: why this feature should predict the target — the mechanism in
    a sentence, and, for a fraud or anomaly feature, the legitimate
    population that looks the same and what separates them ("a call centre
    also has high outbound with no inbound; answer rate separates them").
    A formula alone is not reviewable: a stakeholder cannot tell a derived
    signal from a coincidence by reading arithmetic. This is persisted and
    rendered in the report, so the reasoning that produced the feature
    survives the conversation that produced it."""
    train, test, meta = _load_split(run_id)
    try:
        for df in (train, test):
            df[name] = df.rename(columns=_eval_safe).eval(expression)
    except Exception as e:
        return json.dumps({"error": f"expression failed: {e}"})
    train.to_csv(_run_dir(run_id) / "train.csv", index=False)
    test.to_csv(_run_dir(run_id) / "test.csv", index=False)
    meta["engineered_features"] = {
        **_reasoned_features(meta),
        name: {"formula": expression, "rationale": rationale},
    }
    _save_meta(run_id, meta)
    return json.dumps(
        {"run_id": run_id, "added_feature": name, "train_shape": train.shape}
    )


# --- entity × time features ---------------------------------------------------
# Most telecom signal lives in how an entity's events are spread over time —
# the gaps between them, their rhythm, their deviation from the entity's own
# past — and none of that is a row-wise expression. These two tools give the
# agent a small vocabulary it composes instead of a fixed feature list:
# aggregate_events (event rows -> one row per entity, BEFORE the split) and
# apply_entity_features (entity-per-period rows, AFTER the split, prior rows
# only). Ratios between their outputs are then plain apply_custom_feature.

_PSEUDO_COLUMNS = (
    "_gap",
    "_hour",
    "_dow",
    "_day",
)  # from the time column; `_gap` is computed after `where`, so `column` only
_BUILTIN_AGGS = {"sum", "mean", "median", "std", "min", "max", "nunique"}
_NO_COLUMN_AGGS = {"count", "share", "span_days", "recency_days"}
_ZERO_WHEN_EMPTY = {
    "count",
    "share",
    "sum",
}  # an entity with no matching events has 0 of them, not "unknown"


def _entropy(s: pd.Series) -> float:
    p = s.value_counts(normalize=True)
    return float(-(p * np.log2(p)).sum()) if len(p) else np.nan


def _periodicity(s: pd.Series) -> float:
    """Share of values within 5% of their median — ~1.0 for a firmware timer's
    gaps or a meter's constant payload, low for anything a human produces."""
    s = s.dropna()
    return (
        float(((s - s.median()).abs() <= 0.05 * abs(s.median())).mean())
        if len(s)
        else np.nan
    )


_CUSTOM_AGGS = {
    "cv": lambda s: s.std() / (abs(s.mean()) + 1e-9),
    "entropy": _entropy,
    "top_share": lambda s: s.value_counts(normalize=True).max(),
    "periodicity": _periodicity,
}
_ALL_AGGS = _BUILTIN_AGGS | _NO_COLUMN_AGGS | set(_CUSTOM_AGGS) | {"below"}


def _parse_time(series: pd.Series) -> pd.Series:
    """Datetime strings as-is; a numeric column is read as epoch seconds (the
    usual CDR encoding)."""
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_datetime(series, unit="s", errors="coerce")
    return parse_dates(series)


_AT_AGGS = {"count", "share", "sum", "mean", "std", "min", "max", "recency_days"}


def _features_at_decisions(df: pd.DataFrame, evaluable: pd.DataFrame, entity: str, specs: list,
                           at: str, window: str) -> tuple[pd.DataFrame, dict]:
    """One output row per decision row (`at`), each feature computed over the
    same entity's events in the window BEFORE that row — the view a scorer
    has at the moment of the decision. A spec's own "window" overrides the
    call's. `df` is sorted by entity, time."""
    decision = evaluable.eval(at).astype(bool)
    frame = df.assign(_one=1.0)
    features, formulas = {}, {}
    for spec in specs:
        name, agg, col = spec["name"], spec["agg"], spec.get("column")
        span = spec.get("window") or window
        mask = evaluable.eval(spec["where"]).astype(bool) if spec.get("where") else pd.Series(True, index=df.index)
        if agg == "recency_days":  # no window: the last matching event however long ago
            last = df["_t"].where(mask).groupby(df[entity]).shift(1).groupby(df[entity]).ffill()
            v = (df["_t"] - last).dt.total_seconds() / 86400
        elif agg in ("count", "share"):
            hits = _time_window(frame.assign(_hit=mask.astype(float)), entity, "_t", "_hit", span, "sum").fillna(0)
            if agg == "count":
                v = hits
            else:
                everything = _time_window(frame, entity, "_t", "_one", span, "sum").fillna(0)
                v = hits / everything.replace(0, np.nan)
        else:
            v = _time_window(frame.assign(_val=df[col].where(mask)), entity, "_t", "_val", span, agg)
            if agg == "sum":
                v = v.fillna(0)
        features[name] = v[decision]
        scope = "" if agg == "recency_days" else f" in the {span} before"
        formulas[name] = {
            "formula": f"{agg}({col or 'events'}){' where ' + spec['where'] if spec.get('where') else ''}"
                       f"{scope} each row where {at}, per {entity}",
            "rationale": spec.get("rationale", ""),
        }
    out = df[decision].assign(**features)
    return out.drop(columns=[c for c in out.columns if c.startswith("_")]), formulas


def aggregate_events(
    path: str,
    entity_column: str,
    time_column: str,
    features: str,
    period: str = "",
    before: str = "",
    at: str = "",
    window: str = "",
    out_path: str = "",
) -> str:
    """Turn EVENT rows (calls, SMS, data sessions, transactions, care contacts,
    signalling) into model rows. Runs on the raw file BEFORE prepare_dataset —
    it changes what a row is. Writes a new CSV; call prepare_dataset on THAT.
    Three grains:
      default      one row per entity (whole history before `before`)
      period       one row per entity × period ("h", "D", "7D"; fixed lengths)
      at + window  one row per DECISION event — the rows matching `at` (e.g.
                   "event_type == 'sim_change'") — with each feature computed
                   over that entity's events in the `window` ("24h", "7D")
                   strictly before it. Use it when the decision is one event
                   inside a wider stream: a SIM change among journey events, a
                   port request, a login.
                   A spec may carry its own "window" (resets in "24h", SIM
                   changes in "30D") — one call.

    features: JSON list of specs, each
      {"name": str, "agg": str, "column": str (not for count/share/span_days/
       recency_days), "where": pandas-eval filter (optional), "value": number
       (for "below"), "rationale": str (mechanism + innocent twin)}
    aggs: count, share (rows matching `where` / all the entity's rows), sum,
      mean, median, std, min, max, nunique, cv (std/mean), entropy (bits —
      diversity), top_share (share of the most common value — concentration),
      periodicity (share of values within 5% of the median), below (share of
      values < `value`), span_days (first to last event), recency_days (last
      event to the cutoff / period end / decision). With `at`, only count,
      share, sum, mean, std, min, max and recency_days.
    Pseudo-columns, from the time column: `_gap` — seconds since the entity's
      previous event among the rows `where` kept (so gaps between outbound
      calls only is {"column": "_gap", "where": "direction == 'out'"}; not
      with `at`); `_hour`, `_dow` — rhythm (`where: "_hour < 6"`); `_day` —
      the calendar day, so active days is {"column": "_day", "agg": "nunique"}.

    before: only events strictly before this timestamp are used — set it to
      the decision point, or activity that stops BECAUSE the entity was
      caught (a barred SIM goes silent) becomes a feature that encodes the
      label.

    Default and period grains carry through columns constant within every
    entity (the label, plan, tenure); anything else must be aggregated
    explicitly (a per-event fraud flag: {"name": "is_fraud", "column":
    "is_fraud", "agg": "max"}). The `at` grain keeps every column of the
    decision rows, label included. A numeric time column is read as epoch
    seconds."""
    try:
        specs = json.loads(features)
    except json.JSONDecodeError as e:
        return json.dumps({"error": f"features must be a JSON list of specs: {e}"})
    if not isinstance(specs, list) or not specs:
        return json.dumps({"error": "features must be a non-empty JSON list of specs"})
    if at and period:
        return json.dumps({"error": "`at` and `period` are two different grains — pick one"})
    if at and not window and any(not s.get("window") and s.get("agg") != "recency_days" for s in specs):
        return json.dumps({"error": '`at` needs a `window` (e.g. "24h"), on the call or on every spec'})
    df = read_table(path)
    for col in (entity_column, time_column):
        if col not in df.columns:
            return json.dumps(
                {"error": f"column '{col}' not found. Columns: {list(df.columns)}"}
            )
    for spec in specs:
        agg, col = spec.get("agg"), spec.get("column")
        allowed = _AT_AGGS if at else _ALL_AGGS
        if not spec.get("name") or agg not in allowed:
            return json.dumps(
                {"error": f"spec {spec} needs a name and an agg from {sorted(allowed)}"}
            )
        pseudo = () if at else _PSEUDO_COLUMNS
        if agg not in _NO_COLUMN_AGGS and col not in df.columns and col not in pseudo:
            return json.dumps(
                {
                    "error": f"spec '{spec['name']}': column '{col}' not found (pseudo-columns: {pseudo})"
                }
            )
        if agg == "below" and "value" not in spec:
            return json.dumps(
                {"error": f"spec '{spec['name']}': agg 'below' needs a 'value'"}
            )

    df["_t"] = _parse_time(df[time_column])
    if df["_t"].isna().any():
        return json.dumps(
            {
                "error": f"time_column '{time_column}' has {int(df['_t'].isna().sum())} unparseable value(s)"
            }
        )
    rows_in = len(df)
    if before:
        cutoff = _parse_time(
            pd.Series(
                [float(before) if before.replace(".", "", 1).isdigit() else before]
            )
        ).iloc[0]
        df = df[df["_t"] < cutoff]
    df = df.sort_values([entity_column, "_t"], kind="mergesort")
    df["_hour"], df["_dow"], df["_day"] = (
        df["_t"].dt.hour,
        df["_t"].dt.dayofweek,
        df["_t"].dt.floor("D"),
    )
    evaluable = df.rename(columns=_eval_safe)

    if at:
        try:
            out, formulas = _features_at_decisions(
                df, evaluable, entity_column, specs, at, window
            )
        except Exception as e:
            return json.dumps({"error": f"`at` / `where` / window failed: {e}"})
        grain, carried, not_carried = (
            f"decision rows where {at} ({window} lookback)",
            list(df.columns),
            [],
        )
        carried = [c for c in carried if not c.startswith("_")]
    else:
        keys = [entity_column] + (["_period"] if period else [])
        if period:
            df["_period"] = df["_t"].dt.floor(period)
        groups = df.groupby(keys, sort=True)
        total = groups.size()
        out = pd.DataFrame(index=total.index)
        formulas = {}
        for spec in specs:
            name, agg, col = spec["name"], spec["agg"], spec.get("column")
            try:
                sub = (
                    df[evaluable.eval(spec["where"]).astype(bool)]
                    if spec.get("where")
                    else df
                )
            except Exception as e:
                return json.dumps(
                    {"error": f"spec '{name}': where '{spec['where']}' failed: {e}"}
                )
            sub = sub.assign(_gap=sub.groupby(keys)["_t"].diff().dt.total_seconds())
            g = sub.groupby(keys)
            if agg == "count":
                v = g.size()
            elif agg == "share":
                v = g.size() / total
            elif agg == "span_days":
                v = (g["_t"].max() - g["_t"].min()).dt.total_seconds() / 86400
            elif agg == "recency_days":
                last = g["_t"].max()
                if period:
                    ref = pd.Series(
                        last.index.get_level_values("_period")
                        + pd.tseries.frequencies.to_offset(period),
                        index=last.index,
                    )
                else:
                    ref = df["_t"].max() if not before else cutoff
                v = (ref - last).dt.total_seconds() / 86400
            elif agg == "below":
                v = g[col].agg(lambda s, x=spec["value"]: (s.dropna() < x).mean())
            elif agg in _CUSTOM_AGGS:
                v = g[col].agg(_CUSTOM_AGGS[agg])
            else:
                v = g[col].agg(agg)
            v = v.reindex(out.index)
            out[name] = v.fillna(0) if agg in _ZERO_WHEN_EMPTY else v
            detail = f"{agg}({col or 'rows'}{' < ' + str(spec['value']) if agg == 'below' else ''})"
            formulas[name] = {
                "formula": f"{detail}{' where ' + spec['where'] if spec.get('where') else ''} per {' x '.join(keys)}",
                "rationale": spec.get("rationale", ""),
            }
        candidates = [
            c
            for c in df.columns
            if c not in (entity_column, time_column)
            and not c.startswith("_")
            and c not in out.columns
        ]
        # ponytail: one nunique pass over every column; fine to a few million rows
        varying = (
            groups[candidates].nunique(dropna=False).max() > 1
            if candidates
            else pd.Series(dtype=bool)
        )
        carried = [c for c in candidates if not varying[c]]
        not_carried = [c for c in candidates if c not in carried]
        out = (
            out.join(groups[carried].first())
            .reset_index()
            .rename(columns={"_period": time_column})
        )
        grain = " x ".join(
            [entity_column] + ([f"{time_column} ({period})"] if period else [])
        )

    src = Path(path)
    suffix = f"_at_{window}" if at else f"_{period}" if period else ""
    out_file = (
        Path(out_path)
        if out_path
        else src.with_name(f"{src.stem}_by_{entity_column}{suffix}.csv")
    )
    if (
        runtime.user() and not out_path
    ):  # the source may be a shared dataset; the default lands in the user's own folder
        out_file = (
            toolguard.data_dir()
            / "uploads"
            / toolguard.user_folder(runtime.user())
            / out_file.name
        )
    try:  # the guard skips an empty out_path, so the default has to be checked here
        out_file = toolguard.check_output(str(out_file), "aggregate_events")
    except toolguard.Refused as refused:
        return json.dumps({"error": f"refused: {refused}"})
    out.to_csv(out_file, index=False)
    # prepare_dataset reads this back into meta["engineered_features"], so the
    # report can show what each aggregate is and why it was built.
    Path(f"{out_file}.features.json").write_text(json.dumps(formulas, indent=2))
    time_split = bool(at or period)
    return json.dumps(
        {
            "out_path": str(out_file),
            "grain": grain,
            "rows_in": rows_in,
            "rows_before_cutoff": len(df),
            "rows_out": len(out),
            "features": list(formulas),
            "summary": json.loads(
                out[list(formulas)]
                .describe()
                .T[["mean", "50%", "max"]]
                .round(4)
                .to_json(orient="index")
            ),
            "carried_columns": carried,
            "not_carried": not_carried,
            "note": "not_carried columns vary within an entity — aggregate any you need (the label especially) with an "
            "explicit spec. Next: prepare_dataset on out_path"
            + (
                f" with group_column='{entity_column}', time_column='{time_column}'"
                if time_split
                else ""
            )
            + ".",
        },
        default=str,
    )


_ROLL_OPS = {
    "roll_mean": "mean",
    "roll_std": "std",
    "roll_min": "min",
    "roll_max": "max",
    "roll_sum": "sum",
    "roll_count": "count",
}
_ENTITY_OPS = {
    "lag",
    "diff",
    "pct_change",
    "vs_roll_mean",
    "gap_prev",
    "since",
    "prior_count",
    "streak",
    *_ROLL_OPS,
}


def _time_window(
    frame: pd.DataFrame, keys, time_col: str, col: str, window: str, how: str
) -> pd.Series:
    """`how` over each row's entity rows in [t - window, t) — strictly earlier
    in time, so the row itself and anything at the same instant are out.
    `frame` must already be sorted by keys then time: rolling(on=) returns
    rows in that order, indexed by time, so the result is aligned by position."""
    res = (
        frame.groupby(keys, sort=False, dropna=False)
        .rolling(window, on=time_col, closed="left")[col]
        .agg(how)
    )
    if len(res) != len(frame):
        raise ValueError(
            "time-window result does not line up with the rows — is the frame sorted by entity, time?"
        )
    return pd.Series(res.to_numpy(), index=frame.index)


def apply_entity_features(run_id: str, features: str) -> str:
    """Backward-looking per-entity features on rows where EVERY row is a
    decision — caller-day, subscriber-month, or one row per transaction or
    call — using the run's group_column and time_column from prepare_dataset.
    Every value comes from the entity's EARLIER rows only. Train and test are
    ordered together, so a test row's history includes the entity's training
    rows — exactly the history a production scorer would have.

    Call it right after prepare_dataset, BEFORE data-cleaning: cleaning drops
    the entity id and decomposes the timestamp, after which there is nothing
    to order or group by.

    features: JSON list of specs {"name", "op", "column", "n" | "window",
      "rationale"}. A window is either "n": the previous n ROWS, or "window":
      a time span ("1h", "24h", "7D", "30D") covering [t - window, t). Use
      the time span whenever rows are not exactly one per fixed period
      (transactions, calls, or a file with rows only on active days).
      lag           value n rows back
      diff          value - lag n
      pct_change    (value - lag n) / |lag n|
      roll_mean / roll_std / roll_min / roll_max / roll_sum
                    over the window (current row excluded)
      roll_count    earlier rows inside a time window — velocity ("window" only)
      vs_roll_mean  value / roll_mean over the window — deviation from the
                    entity's OWN baseline
      gap_prev      time since the entity's previous row (days, or the time
                    column's own units if it is an integer period index)
      since         time since the entity's last earlier row where
                    `column` > 0 (last dropped call, top-up, ticket)
      prior_count   how many earlier rows of this entity had the same value
                    of `column` — 0 = first time (new payee, new country,
                    new device on this line)
      streak        consecutive earlier rows with `column` > 0, up to the
                    previous row; a streak that just broke is streak > 0
                    with the current value 0 (then apply_custom_feature)
    A spec may use a column created earlier in the same call (gap_prev, then
    roll_std of it = irregularity of the entity's activity). The target
    column is refused: a lagged label is only a feature if the label is known
    at scoring time, which in fraud and churn it rarely is."""
    train, test, meta = _load_split(run_id)
    group, time_col, target = (
        meta.get("group_column"),
        meta.get("time_column"),
        meta.get("target"),  # none for clustering
    )
    if not group or not time_col:
        return json.dumps(
            {
                "error": "needs both group_column and time_column — set them at prepare_dataset "
                f"(this run has group_column={group!r}, time_column={time_col!r})"
            }
        )
    gone = [c for c in (group, time_col) if c not in train.columns]
    if gone:
        return json.dumps(
            {
                "error": f"{gone} no longer in the training fold — data-cleaning dropped or decomposed "
                "them. Entity features must be built before data-cleaning; re-run prepare_dataset "
                "and call this first."
            }
        )
    try:
        specs = json.loads(features)
    except json.JSONDecodeError as e:
        return json.dumps({"error": f"features must be a JSON list of specs: {e}"})
    if not isinstance(specs, list) or not specs:
        return json.dumps({"error": "features must be a non-empty JSON list of specs"})

    both = pd.concat(
        [
            train.assign(_fold=0, _pos=range(len(train))),
            test.assign(_fold=1, _pos=range(len(test))),
        ],
        ignore_index=True,
    )
    is_datetime = not pd.api.types.is_numeric_dtype(both[time_col])
    both["_t"] = parse_dates(both[time_col]) if is_datetime else both[time_col]
    both = both.sort_values([group, "_t", "_fold", "_pos"], kind="mergesort")
    g = both.groupby(group, sort=False)

    def elapsed(delta: pd.Series) -> pd.Series:
        return delta.dt.total_seconds() / 86400 if is_datetime else delta

    formulas = {}
    for spec in specs:
        name, op, col, window = (
            spec.get("name"),
            spec.get("op"),
            spec.get("column"),
            spec.get("window"),
        )
        n = int(spec.get("n", 1))
        if not name or op not in _ENTITY_OPS:
            return json.dumps(
                {
                    "error": f"spec {spec} needs a name and an op from {sorted(_ENTITY_OPS)}"
                }
            )
        if op != "gap_prev":
            if col == target:
                return json.dumps(
                    {
                        "error": f"spec '{name}': '{col}' is the target — a lagged label is only usable "
                        "if it is known at scoring time; build it upstream if it is"
                    }
                )
            if col not in both.columns:
                return json.dumps({"error": f"spec '{name}': column '{col}' not found"})
        if window and not is_datetime:
            return json.dumps(
                {
                    "error": f"spec '{name}': a time window needs a date/time column; "
                    f"'{time_col}' is numeric — use \"n\" rows instead"
                }
            )
        if op == "roll_count" and not window:
            return json.dumps(
                {
                    "error": f"spec '{name}': roll_count needs a time \"window\" (over n rows it is just n)"
                }
            )
        if op == "gap_prev":
            v = elapsed(both["_t"] - g["_t"].shift(1))
        elif op == "since":
            last = (
                both["_t"]
                .where(both[col] > 0)
                .groupby(both[group])
                .shift(1)
                .groupby(both[group])
                .ffill()
            )
            v = elapsed(both["_t"] - last)
        elif op == "prior_count":
            v = both.groupby([group, col], sort=False, dropna=False).cumcount()
        elif op == "streak":
            positive = both[col] > 0
            run = (
                positive.astype(int)
                .groupby([both[group], (~positive).groupby(both[group]).cumsum()])
                .cumsum()
            )
            v = run.groupby(both[group]).shift(1)
        elif op in ("lag", "diff", "pct_change"):
            prior = g[col].shift(n)
            v = (
                prior
                if op == "lag"
                else (
                    both[col] - prior
                    if op == "diff"
                    else (both[col] - prior) / (prior.abs() + 1e-6)
                )
            )
        else:
            how = _ROLL_OPS.get(op, "mean")
            if window:
                source = both.assign(_one=1.0) if op == "roll_count" else both
                stat = _time_window(
                    source,
                    group,
                    "_t",
                    "_one" if op == "roll_count" else col,
                    window,
                    how,
                )
                if op in ("roll_count", "roll_sum"):
                    stat = stat.fillna(
                        0
                    )  # no earlier rows in the window = none, not unknown
            else:
                stat = (
                    g[col]
                    .shift(1)
                    .groupby(both[group], sort=False)
                    .rolling(n, min_periods=1)
                    .agg(how)
                    .droplevel(0)
                )
            v = both[col] / (stat + 1e-6) if op == "vs_roll_mean" else stat
        both[name] = v
        sized = op in ("lag", "diff", "pct_change", "vs_roll_mean", *_ROLL_OPS)
        span = f", window={window}" if window else f", n={n}" if sized else ""
        formulas[name] = {
            "formula": f"{op}({col or time_col}{span}) per {group} ordered by {time_col}, earlier rows only",
            "rationale": spec.get("rationale", ""),
        }

    both = both.sort_values(["_fold", "_pos"])
    helpers = ["_fold", "_pos", "_t"]
    train = both[both["_fold"] == 0].drop(columns=helpers)
    test = both[both["_fold"] == 1].drop(columns=helpers)
    train.to_csv(_run_dir(run_id) / "train.csv", index=False)
    test.to_csv(_run_dir(run_id) / "test.csv", index=False)
    meta["engineered_features"] = {**_reasoned_features(meta), **formulas}
    _save_meta(run_id, meta)
    return json.dumps(
        {
            "run_id": run_id,
            "added_features": list(formulas),
            # NaN = no prior row to look at (an entity's first period) — data-cleaning imputes it
            "missing_pct_train": {
                k: round(float(train[k].isna().mean()) * 100, 2) for k in formulas
            },
            "train_shape": train.shape,
        }
    )


MIN_PEER_ROWS = (
    20  # a peer group thinner than this falls back to the whole training fold
)


def apply_peer_features(run_id: str, features: str) -> str:
    """Compare each row with its PEERS — rows of the same kind: the same cell,
    plan, device model, merchant category, APN. Answers a different question
    from the entity's own baseline: not "is this unusual for THEM" but "is
    this unusual for THEIR KIND" — a subscriber dropping calls on a cell where
    everyone drops calls has a network problem, not a personal one. Peer
    statistics are fitted on the TRAINING fold only and applied to both, so
    the test fold never shapes them.

    features: JSON list of specs {"name", "column", "by", "stat", "rationale"}
      stat "ratio"   value / peer median
      stat "zscore"  (value - peer mean) / peer std
    Peer groups with fewer than MIN_PEER_ROWS training rows, and groups never
    seen in training, use the whole training fold's statistic. `by` may not
    be the run's group_column — an entity's own history must look backwards
    in time; that is apply_entity_features. Call before data-cleaning, which
    may drop or encode the `by` column."""
    train, test, meta = _load_split(run_id)
    target, group = meta.get("target"), meta.get("group_column")
    try:
        specs = json.loads(features)
    except json.JSONDecodeError as e:
        return json.dumps({"error": f"features must be a JSON list of specs: {e}"})
    if not isinstance(specs, list) or not specs:
        return json.dumps({"error": "features must be a non-empty JSON list of specs"})
    formulas, unseen = {}, {}
    for spec in specs:
        name, col, by, stat = (
            spec.get("name"),
            spec.get("column"),
            spec.get("by"),
            spec.get("stat"),
        )
        if not name or stat not in ("ratio", "zscore"):
            return json.dumps(
                {"error": f"spec {spec} needs a name and a stat of 'ratio' or 'zscore'"}
            )
        if target in (col, by):
            return json.dumps(
                {
                    "error": f"spec '{name}': the target cannot be a peer column or a peer key"
                }
            )
        if by == group:
            return json.dumps(
                {
                    "error": f"spec '{name}': '{by}' is the run's entity — its own history must look "
                    "backwards in time; use apply_entity_features (vs_roll_mean)"
                }
            )
        for c in (col, by):
            if c not in train.columns:
                return json.dumps({"error": f"spec '{name}': column '{c}' not found"})
        if not pd.api.types.is_numeric_dtype(train[col]):
            return json.dumps({"error": f"spec '{name}': '{col}' is not numeric"})
        peers = train.groupby(by)[col].agg(["median", "mean", "std", "count"])
        overall = train[col].agg(["median", "mean", "std"])
        peers.loc[peers["count"] < MIN_PEER_ROWS, ["median", "mean", "std"]] = (
            overall.to_numpy()
        )
        for df in (train, test):
            fitted = {
                s: df[by].map(peers[s]).fillna(overall[s])
                for s in ("median", "mean", "std")
            }
            df[name] = (
                df[col] / (fitted["median"].abs() + 1e-6)
                if stat == "ratio"
                else (df[col] - fitted["mean"]) / (fitted["std"] + 1e-6)
            )
        unseen[name] = round(float((~test[by].isin(peers.index)).mean()) * 100, 2)
        formulas[name] = {
            "formula": f"{col} {'/ peer median' if stat == 'ratio' else 'z-score vs peers'} by {by} "
            f"(fitted on the training fold)",
            "rationale": spec.get("rationale", ""),
        }
    train.to_csv(_run_dir(run_id) / "train.csv", index=False)
    test.to_csv(_run_dir(run_id) / "test.csv", index=False)
    meta["engineered_features"] = {**_reasoned_features(meta), **formulas}
    _save_meta(run_id, meta)
    return json.dumps(
        {
            "run_id": run_id,
            "added_features": list(formulas),
            # a high share means the peer key churns faster than the training window
            "test_rows_with_unseen_peer_group_pct": unseen,
            "train_shape": train.shape,
        }
    )


def profile_features(run_id: str, columns: str = "") -> str:
    """Read-only check between BUILDING features and TRAINING: for each
    feature (default: every engineered feature on the run), its missing
    share, distinct values, its median per class (classification) or per
    target quartile (regression), and the strength it reaches ALONE on the
    training fold — ROC-AUC for classification, |Spearman| for regression
    (categoricals scored out-of-fold). Flags:
      leakage_suspect  near-perfect alone — a copy of the label, data written
                       after the outcome, or a label produced by a rule over
                       the same inputs, until proven otherwise. A mechanism
                       feature CAN be this strong on investigated labels;
                       label_provenance decides which
      constant         one distinct value — carries nothing
      mostly_missing   over half missing — check the column, `where` or window
      weak_alone       little signal alone — NOT a reason to drop by itself:
                       discriminators are weak alone and strong in combination
    Use it to revise the mechanism before paying for a model."""
    train, _, meta = _load_split(run_id)
    target = meta["target"]
    names = [c.strip() for c in columns.split(",") if c.strip()] or [
        c for c in _reasoned_features(meta) if c in train.columns
    ]
    if not names:
        return json.dumps({"error": "no engineered features on this run — pass `columns` to profile others"})
    missing = [c for c in names if c not in train.columns]
    if missing:
        return json.dumps({"error": f"not in the training fold: {missing}"})
    signal = _store["signal"](train, target, meta, names)
    scores, groups = signal["scores"], signal["groups"]
    report = {}
    for c in names:
        x = train[c]
        entry = {"missing_pct": round(float(x.isna().mean()) * 100, 2), "distinct": int(x.nunique())}
        if pd.api.types.is_numeric_dtype(x) and not pd.api.types.is_bool_dtype(x):
            entry["median_by_group"] = {str(k): round(float(v), 4)
                                        for k, v in x.groupby(groups, observed=True).median().items()}
        else:
            entry["top_value_by_group"] = {str(k): (str(v.mode().iloc[0]) if v.notna().any() else None)
                                           for k, v in x.groupby(groups, observed=True)}
        s = scores.get(c)
        if s is not None:
            entry[f"{signal['metric']}_alone"] = s
        entry["flag"] = ("constant" if entry["distinct"] <= 1 else
                         "mostly_missing" if entry["missing_pct"] > 50 else
                         "leakage_suspect" if s is not None and s >= signal["suspect"] else
                         "weak_alone" if s is not None and s < signal["weak"] else "ok")
        report[c] = entry
    return json.dumps({"run_id": run_id, "features": report, "signal_metric": signal["metric"],
                       "note": "training fold only"})


TOOLS = (propose_features, apply_features, apply_custom_feature, aggregate_events, apply_entity_features, apply_peer_features, profile_features)


def register(mcp, only: tuple | None = None) -> dict:
    """Register the tools on `mcp` (its own guard / ledger wrappers apply) and
    return the registered callables by name, for the agent to export. `only`
    limits the set — an agent without a target (clustering) takes the ones
    that do not need one."""
    return {fn.__name__: mcp.tool()(fn) for fn in TOOLS if only is None or fn.__name__ in only}


reasoned_features = _reasoned_features
