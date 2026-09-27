"""The split decisions that silently invalidate a run if they are wrong.

Everything here is about ORDER and PROVENANCE rather than accuracy: a
forward-chained split really putting the past in train and the future in
test, frequency encoding learning its map from the training fold only, and
SMOTE refusing to quietly undo either of them. A model trained through a
broken version of any of these still reports a good score — that is exactly
why they need a test instead of a glance at the metrics.

Run: python test_split_integrity.py
"""

import os as _os
import tempfile as _tempfile

# Tests never write into the real data/ (runs, artifacts, predictions) that
# the orchestrator reports as "recent runs"; CI may point this elsewhere.
_os.environ.setdefault(
    "AGENTIC_ML_DATA_DIR", _tempfile.mkdtemp(prefix="agentic-ml-test-")
)
# ...nor register test models into the real MLflow registry (the UI on :5000).
_os.environ.setdefault(
    "MLFLOW_TRACKING_URI", "file:" + _tempfile.mkdtemp(prefix="agentic-ml-test-mlruns-")
)

import json

import joblib
import numpy as np
import pandas as pd

import mcp_server as m


def _entity_day_frame(n_entities=40, n_days=20, seed=0):
    """Telecom-shaped: one row per entity per day, positives spread across
    the whole window so a temporal split is legitimate."""
    rng = np.random.default_rng(seed)
    rows = []
    for day in range(n_days):
        for e in range(n_entities):
            rows.append(
                {
                    "caller": f"C{e:03d}",
                    "day": day,
                    "total_calls": int(rng.integers(1, 100)),
                    # "p123" not "+123": pandas parses a leading-plus digit string
                    # as an int64 on read_csv, which would make this column numeric
                    # and quietly skip every categorical path under test. That is a
                    # real trap for telecom prefixes, not just a fixture detail.
                    "prefix": f"p{rng.integers(1, 400):03d}",
                    "label": int((e + day) % 7 == 0),
                }
            )
    return pd.DataFrame(rows)


def _write(tmp_path, df, name="d.csv"):
    p = tmp_path / name
    df.to_csv(p, index=False)
    return str(p)


def test_temporal_split_puts_every_training_row_before_every_test_row(tmp_path):
    path = _write(tmp_path, _entity_day_frame())
    res = json.loads(m.prepare_dataset(path, "label", time_column="day"))
    assert "error" not in res, res
    assert res["split_type"] == "temporal"
    train, test, _ = m._load_split(res["run_id"])
    # The whole point: no test row may be older than the newest training row.
    assert train["day"].max() <= test["day"].min(), (
        f"train reaches day {train['day'].max()} but test starts at day {test['day'].min()} — "
        "a trailing-window feature would be computed from rows the model trained on"
    )
    assert set(train["label"]) == {0, 1} and set(test["label"]) == {0, 1}


def test_temporal_split_reports_entity_overlap_instead_of_hiding_it(tmp_path):
    path = _write(tmp_path, _entity_day_frame())
    res = json.loads(
        m.prepare_dataset(path, "label", time_column="day", group_column="caller")
    )
    assert "error" not in res, res
    # Every caller appears on every day here, so the overlap is total — the
    # contract is that this is REPORTED, not silently purged away.
    assert res["entity_overlap_pct"] == 100.0, res
    train, test, _ = m._load_split(res["run_id"])
    assert len(train) and len(test), "reporting overlap must not cost us the folds"


def test_temporal_split_refuses_a_single_class_fold(tmp_path):
    df = _entity_day_frame()
    # All positives in the first half: the future fold has nothing to detect.
    df.loc[df["day"] >= 10, "label"] = 0
    df.loc[df["day"] < 10, "label"] = (df.loc[df["day"] < 10, "day"] % 2 == 0).astype(
        int
    )
    res = json.loads(
        m.prepare_dataset(_write(tmp_path, df), "label", time_column="day")
    )
    assert "error" in res and "one class" in res["error"], res


def test_temporal_split_refuses_an_unorderable_column(tmp_path):
    df = _entity_day_frame()
    df["day"] = df["day"].astype(str)
    df.loc[0, "day"] = "not-a-date"
    res = json.loads(
        m.prepare_dataset(_write(tmp_path, df), "label", time_column="day")
    )
    assert "error" in res and "orderable" in res["error"], res


def test_frequency_encoding_learns_its_map_from_the_training_fold_only(tmp_path):
    """The leakage discipline: a value that exists only in test must encode
    to 0.0, because the model was never told it existed."""
    df = _entity_day_frame()
    path = _write(tmp_path, df)
    res = json.loads(m.prepare_dataset(path, "label", time_column="day"))
    run_id = res["run_id"]
    train_before, test_before, _ = m._load_split(run_id)
    train_freq = train_before["prefix"].value_counts(normalize=True)

    out = json.loads(m.apply_frequency_encoding(run_id, "prefix"))
    assert "error" not in out, out
    train, test, meta = m._load_split(run_id)
    assert "prefix" not in train.columns and "prefix_freq" in train.columns

    unseen_mask = ~test_before["prefix"].isin(train_freq.index)
    if unseen_mask.any():
        assert (test.loc[unseen_mask.values, "prefix_freq"] == 0.0).all(), (
            "a value never seen in training must encode to 0.0, not to its test-fold frequency"
        )
    seen_mask = ~unseen_mask
    if seen_mask.any():
        expected = test_before.loc[seen_mask, "prefix"].map(train_freq).to_numpy()
        assert np.allclose(
            test.loc[seen_mask.values, "prefix_freq"].to_numpy(), expected
        ), "test rows must be encoded with the TRAINING fold's frequencies"
    assert meta["frequency_encoded"]["prefix"]["new_column"] == "prefix_freq"


def test_frequency_encoding_refuses_the_group_column(tmp_path):
    path = _write(tmp_path, _entity_day_frame())
    res = json.loads(m.prepare_dataset(path, "label", group_column="caller"))
    out = json.loads(m.apply_frequency_encoding(res["run_id"], "caller"))
    assert "error" in out and "fingerprint" in out["error"], out


def test_wide_categorical_is_offered_encoding_but_not_proposed_for_dropping(tmp_path):
    """The band between "real feature" and "identifier": a destination prefix
    one-hots into hundreds of columns, but dropping it is the wrong answer."""
    df = _entity_day_frame(n_entities=60, n_days=30)
    path = _write(tmp_path, df)
    run_id = json.loads(m.prepare_dataset(path, "label", time_column="day"))["run_id"]
    out = json.loads(m.propose_drop_columns(run_id))
    assert "prefix" in out["frequency_encodable_instead"], out
    assert "prefix" not in out["proposals"], (
        "a wide categorical is a feature, not a column to drop"
    )


def test_telecom_identity_columns_are_dropped_without_asking(tmp_path):
    """IMSI/IMEI/MSISDN name a standardised identity with exactly one meaning
    — a name match on those is strong, not a question worth a round-trip."""
    df = _entity_day_frame()
    df["imsi"] = [
        "I" + str(i % 37) for i in range(len(df))
    ]  # repeats: no cardinality signal
    df["order_no"] = [
        "O" + str(i % 41) for i in range(len(df))
    ]  # genuinely ambiguous name
    run_id = json.loads(m.prepare_dataset(_write(tmp_path, df), "label"))["run_id"]
    from mcp_server.data_cleaning import identifier_columns

    train, _, _ = m._load_split(run_id)
    signals = identifier_columns(train, "label")
    assert signals["imsi"]["strength"] == "strong", signals["imsi"]
    assert signals["imsi"]["rule"] == "unambiguous_id_name"
    assert signals["order_no"]["strength"] == "weak", (
        "an ambiguous name must still be gated, not auto-dropped"
    )


def test_smote_is_not_recommended_on_a_non_random_split(tmp_path):
    path = _write(tmp_path, _entity_day_frame())
    res = json.loads(m.prepare_dataset(path, "label", time_column="day"))
    proposal = json.loads(m.propose_smote(res["run_id"]))
    assert proposal["recommend_smote"] is False
    assert proposal["split_type"] == "temporal"
    # And if applied anyway, it must say so rather than comply silently.
    applied = json.loads(m.apply_smote(res["run_id"]))
    assert "warning" in applied and "temporal" in applied["warning"], applied


def _trained_run(tmp_path, seed=3, n=4000):
    """A small separable binary problem — enough signal for a real model and
    a real threshold, small enough to train in a test."""
    rng = np.random.default_rng(seed)
    rows = [
        {
            "x1": rng.normal(3 if (f := int(rng.random() < 0.08)) else 0, 1),
            "x2": rng.normal(),
            "label": f,
        }
        for _ in range(n)
    ]
    path = _write(tmp_path, pd.DataFrame(rows))
    run_id = json.loads(m.prepare_dataset(path, "label"))["run_id"]
    m.train_model(run_id, "random_forest")
    return run_id


def test_operating_point_is_persisted_exported_and_applied(tmp_path):
    """The reviewed model and the shipped model must be the same model."""
    run_id = _trained_run(tmp_path)
    tuned = json.loads(m.tune_threshold(run_id, target_recall=0.9))
    threshold = tuned["chosen"]["threshold"]

    _, _, meta = m._load_split(run_id)
    assert meta["operating_point"]["threshold"] == threshold, (
        "threshold must survive the call that chose it"
    )

    pkl = str(tmp_path / "model.pkl")
    json.loads(m.export_model(run_id, pkl, force=True))
    import joblib

    assert joblib.load(pkl)["operating_point"]["threshold"] == threshold, (
        "threshold must travel with the model"
    )

    scored = json.loads(
        m.predict(pkl, _write(tmp_path, pd.read_csv(tmp_path / "d.csv"), "new.csv"))
    )
    assert scored["decision_threshold"] == threshold, (
        "predict must apply the tuned threshold, not 0.5"
    )

    out = pd.read_csv(scored["out_path"])
    expected = (out["positive_proba"] >= threshold).astype(int)
    assert (out["prediction"] == expected).all(), (
        "predictions must be the thresholded probabilities"
    )


def test_refitting_the_pipeline_invalidates_the_threshold(tmp_path):
    run_id = _trained_run(tmp_path)
    m.tune_threshold(run_id, target_recall=0.9)
    m.train_model(
        run_id, "logistic_regression"
    )  # refit: the old threshold no longer describes this model
    _, _, meta = m._load_split(run_id)
    assert not meta.get("operating_point"), (
        "a refit must clear a threshold tuned against the previous fit"
    )


def test_alert_budget_mode_respects_the_budget(tmp_path):
    run_id = _trained_run(tmp_path)
    res = json.loads(m.tune_threshold(run_id, max_alert_rate=0.05))
    assert "error" not in res, res
    # The budget is met where the threshold was CHOSEN (out-of-fold training
    # predictions). On the test fold it is measured, not guaranteed — a
    # cutoff can overshoot a budget on unseen rows, and saying so is the point.
    assert res["selected_on"]["alert_rate"] <= 0.05, res["selected_on"]
    assert res["chosen"]["alert_rate"] <= 0.05 * 1.5, (
        "an overshoot this large means the choice didn't generalise"
    )
    assert "max_alert_rate" in res["chosen_by"]


def test_tune_threshold_refuses_two_modes_at_once(tmp_path):
    run_id = _trained_run(tmp_path)
    res = json.loads(m.tune_threshold(run_id, target_recall=0.9, max_alert_rate=0.05))
    assert "error" in res and "at most one" in res["error"]


def test_label_rule_screen_catches_a_rule_generated_target(tmp_path):
    """The exact Wangiri failure mode: the label IS a threshold on a column."""
    rng = np.random.default_rng(7)
    ratio = rng.random(4000)
    df = pd.DataFrame(
        {
            "per_to": ratio,
            "noise": rng.normal(size=4000),
            "label": (ratio < 0.15).astype(int),
        }
    )  # the label is literally the rule
    run_id = json.loads(m.prepare_dataset(_write(tmp_path, df), "label"))["run_id"]
    m.train_model(run_id, "random_forest")
    res = json.loads(m.check_label_rule(run_id))
    assert res["label_rule_suspicion"] is True, res
    assert res["best_single_column_rule"]["column"] == "per_to", res

    readiness = json.loads(m.check_readiness(run_id))
    assert readiness["checks"]["label_rule_screened"]["status"] == "fail"


def test_label_rule_screen_clears_when_the_signal_is_spread_across_features(tmp_path):
    """No single column should reproduce the label when the label genuinely
    depends on an interaction — that is the shape of a model earning its score."""
    rng = np.random.default_rng(5)
    n = 6000
    a, b, c = rng.normal(size=n), rng.normal(size=n), rng.normal(size=n)
    logit = 1.6 * a * b + 1.2 * c - 2.5  # interaction: no one column carries it
    y = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
    df = pd.DataFrame({"a": a, "b": b, "c": c, "label": y})
    run_id = json.loads(m.prepare_dataset(_write(tmp_path, df), "label"))["run_id"]
    m.train_model(run_id, "random_forest")
    res = json.loads(m.check_label_rule(run_id))
    assert res["label_rule_suspicion"] is False, res
    assert (
        json.loads(m.check_readiness(run_id))["checks"]["label_rule_screened"]["status"]
        == "pass"
    )


def test_a_single_driver_domain_can_be_acknowledged_but_not_silently(tmp_path):
    """One dominant column is ambiguous — a rule-written label, or a real
    physical driver. The gate demands an answer rather than guessing."""
    run_id = _trained_run(tmp_path, seed=11)  # x1 genuinely carries the signal
    m.check_label_rule(run_id)
    gate = lambda: json.loads(m.check_readiness(run_id))["checks"][
        "label_rule_screened"
    ]["status"]
    assert gate() == "fail", (
        "a dominant single column must stop the run until someone answers for it"
    )
    assert "error" in json.loads(m.acknowledge_label_rule(run_id, "   ")), (
        "cannot be waived with an empty reason"
    )
    m.acknowledge_label_rule(
        run_id, "labels are analyst-investigated; x1 is the physical driver"
    )
    assert gate() == "pass"


def test_feature_engineering_gate_needs_features_or_a_stated_reason(tmp_path):
    run_id = _trained_run(tmp_path)
    gate = lambda: json.loads(m.check_readiness(run_id))["checks"][
        "feature_engineering_considered"
    ]["status"]
    assert gate() == "not_run", "an untouched run must not silently pass the step"
    m.declare_feature_engineering_not_applicable(
        run_id, "file is pre-aggregated; every archetype needs a column it lacks"
    )
    assert gate() == "not_applicable"


def test_engineered_feature_rationale_survives_into_meta(tmp_path):
    path = _write(tmp_path, _entity_day_frame())
    res = json.loads(m.prepare_dataset(path, "label", time_column="day"))
    run_id = res["run_id"]
    out = json.loads(
        m.apply_custom_feature(
            run_id,
            "calls_per_day_ratio",
            "total_calls / (total_calls.mean() + 1e-6)",
            rationale="burst vs own baseline; a new genuine SIM looks the same, ramp shape separates them",
        )
    )
    assert "error" not in out, out
    _, _, meta = m._load_split(run_id)
    entry = meta["engineered_features"]["calls_per_day_ratio"]
    assert entry["formula"].startswith("total_calls /")
    assert "ramp shape" in entry["rationale"]


def test_custom_feature_reaches_digit_leading_columns(tmp_path):
    """Pre-aggregated telco exports name windows `7_lag_to`; pandas eval
    cannot parse that even in backticks."""
    df = _entity_day_frame().assign(**{"7_lag_to": lambda d: d["total_calls"] * 7})
    run_id = json.loads(m.prepare_dataset(_write(tmp_path, df), "label"))["run_id"]
    out = json.loads(
        m.apply_custom_feature(
            run_id, "burst_7d", "total_calls / (_7_lag_to / 7 + 1e-6)"
        )
    )
    assert "error" not in out, out
    train, _, _ = m._load_split(run_id)
    assert (train["burst_7d"] - 1).abs().max() < 1e-6


def test_aggregate_events_gap_features_respect_filter_and_cutoff(tmp_path):
    """Firmware on a 300s timer vs a bursty human: the gap vocabulary must
    separate them, `where` must recompute gaps within the filtered rows, and
    nothing after `before` may reach a feature."""
    rows = []
    for e in range(3):  # machines: an event every 300s exactly
        rows += [
            {"sim": f"M{e}", "ts": 1_000_000 + 300 * i, "dir": "out", "m2m": 1}
            for i in range(20)
        ]
    for e in range(
        3
    ):  # humans: irregular, mixed direction, plus one event after the cutoff
        ts = 1_000_000 + np.cumsum([5, 900, 20, 4000, 60, 30, 7000, 15, 300, 2500])
        rows += [
            {"sim": f"H{e}", "ts": int(t), "dir": "out" if i % 2 else "in", "m2m": 0}
            for i, t in enumerate(ts)
        ]
        rows.append({"sim": f"H{e}", "ts": 3_000_000, "dir": "out", "m2m": 0})
    path = _write(tmp_path, pd.DataFrame(rows), "events.csv")
    specs = [
        {
            "name": "gap_cv",
            "column": "_gap",
            "agg": "cv",
            "rationale": "timer vs human",
        },
        {"name": "gap_max", "column": "_gap", "agg": "max"},
        {"name": "gap_periodicity", "column": "_gap", "agg": "periodicity"},
        {
            "name": "out_gap_min",
            "column": "_gap",
            "agg": "min",
            "where": "dir == 'out'",
        },
        {"name": "out_share", "agg": "share", "where": "dir == 'out'"},
    ]
    res = json.loads(
        m.aggregate_events(path, "sim", "ts", json.dumps(specs), before="2000000")
    )
    assert "error" not in res, res
    out = pd.read_csv(res["out_path"]).set_index("sim")
    assert "m2m" in res["carried_columns"]
    assert out.loc["M0", "gap_cv"] < 1e-6 and out.loc["M0", "gap_periodicity"] == 1.0
    assert out.loc["H0", "gap_cv"] > 1
    assert out.loc["H0", "gap_max"] == 7000, (
        "the post-cutoff event must not create a gap"
    )
    assert out.loc["H0", "out_gap_min"] == 60 + 30, (
        "gaps between OUTBOUND events only (all-event min is 15)"
    )
    assert out.loc["H0", "out_share"] == 0.5

    run = json.loads(m.prepare_dataset(res["out_path"], "m2m", test_size=0.34))
    _, _, meta = m._load_split(run["run_id"])
    assert meta["engineered_features"]["gap_cv"]["rationale"] == "timer vs human", (
        "definitions reach the run"
    )


def test_aggregate_events_never_writes_beside_a_shared_dataset(tmp_path):
    """With a user bound, an empty out_path must not drop a file next to a
    shared CSV in data/ — it goes to the user's own uploads folder. (No
    pytest fixtures: CI also runs this file as a plain script.)"""
    shared = m.core.toolguard.data_dir() / "datasets" / "shared_events.csv"
    shared.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"sim": ["A", "A", "B"], "ts": [1, 5, 9], "y": [0, 0, 1]}).to_csv(
        shared, index=False
    )
    spec = json.dumps([{"name": "n", "agg": "count"}])
    previous = _os.environ.get("AGENTIC_ML_USER")
    _os.environ["AGENTIC_ML_USER"] = "ds-alice"
    try:
        res = json.loads(m.aggregate_events(str(shared), "sim", "ts", spec))
        refused = json.loads(
            m.aggregate_events(str(shared), "sim", "ts", spec, out_path=str(shared.parent / "mine.csv"))
        )
    finally:
        if previous is None:
            _os.environ.pop("AGENTIC_ML_USER", None)
        else:
            _os.environ["AGENTIC_ML_USER"] = previous
    assert "error" not in res, res
    assert (
        "/uploads/ds-alice/" in res["out_path"]
        and not (shared.parent / "shared_events_by_sim.csv").exists()
    )
    assert "refused" in refused["error"]


def test_entity_features_look_only_backwards_across_the_split(tmp_path):
    raw = _entity_day_frame()
    res = json.loads(
        m.prepare_dataset(
            _write(tmp_path, raw), "label", group_column="caller", time_column="day"
        )
    )
    run_id = res["run_id"]
    specs = [
        {"name": "calls_lag1", "op": "lag", "column": "total_calls"},
        {"name": "calls_vs_3d", "op": "vs_roll_mean", "column": "total_calls", "n": 3},
        {"name": "days_since_prev", "op": "gap_prev"},
    ]
    out = json.loads(m.apply_entity_features(run_id, json.dumps(specs)))
    assert "error" not in out, out
    train, test, _ = m._load_split(run_id)
    both = pd.concat([train, test]).set_index(["caller", "day"]).sort_index()
    calls = raw.set_index(["caller", "day"])["total_calls"]
    # the first test day's lag comes from the last TRAINING day — history crosses the split, the future never does
    first_test_day = test["day"].min()
    row = both.loc[("C005", first_test_day)]
    assert row["calls_lag1"] == calls[("C005", first_test_day - 1)]
    baseline = calls.loc["C005"].loc[first_test_day - 3 : first_test_day - 1].mean()
    assert (
        abs(row["calls_vs_3d"] - calls[("C005", first_test_day)] / (baseline + 1e-6))
        < 1e-6
    )
    assert both.loc[("C005", 0), ["calls_lag1", "days_since_prev"]].isna().all(), (
        "day 0 has no history"
    )
    assert (both.loc["C005"].loc[1:, "days_since_prev"] == 1).all()

    refused = json.loads(
        m.apply_entity_features(
            run_id, json.dumps([{"name": "x", "op": "lag", "column": "label"}])
        )
    )
    assert "target" in refused["error"]
    # The entity id is a split key: cleaning keeps it, so history features
    # still work after cleaning, and the model never sees it.
    kept = json.loads(m.apply_drop_columns(run_id, '["caller"]'))
    assert kept["kept_split_keys"] == ["caller"] and kept["dropped_columns"] == []
    late = json.loads(m.apply_entity_features(run_id, json.dumps([{**specs[0], "name": "calls_lag1_again"}])))
    assert "error" not in late, late


def _transactions(n_accounts=30, seed=4):
    """One row per transaction: time-stamped, irregular, label per row."""
    rng = np.random.default_rng(seed)
    rows = []
    for a in range(n_accounts):
        t = pd.Timestamp("2026-01-01")
        for i in range(25):
            t += pd.Timedelta(minutes=float(rng.exponential(300)))
            rows.append(
                {
                    "acct": f"A{a:02d}",
                    "ts": t,
                    "amount": float(rng.integers(1, 100)),
                    "payee": f"P{rng.integers(3)}",
                    "flag": int(rng.random() < 0.5),
                    "label": int(rng.random() < 0.3),
                }
            )
    return pd.DataFrame(rows)


def test_time_windows_prior_count_and_streak_use_only_earlier_rows(tmp_path):
    raw = _transactions()
    run_id = json.loads(
        m.prepare_dataset(
            _write(tmp_path, raw), "label", group_column="acct", time_column="ts"
        )
    )["run_id"]
    specs = [
        {"name": "n_1h", "op": "roll_count", "window": "1h"},
        {"name": "amt_24h", "op": "roll_sum", "column": "amount", "window": "24h"},
        {"name": "payee_seen", "op": "prior_count", "column": "payee"},
        {"name": "flag_streak", "op": "streak", "column": "flag"},
    ]
    specs[0]["column"] = "amount"
    out = json.loads(m.apply_entity_features(run_id, json.dumps(specs)))
    assert "error" not in out, out
    train, test, _ = m._load_split(run_id)
    got = (
        pd.concat([train, test])
        .assign(ts=lambda d: pd.to_datetime(d["ts"]))
        .sort_values(["acct", "ts"])
    )
    for _, acct in got.groupby("acct"):
        for i in range(len(acct)):
            row, earlier = acct.iloc[i], acct.iloc[:i]
            assert (
                row["n_1h"] == (earlier["ts"] >= row["ts"] - pd.Timedelta("1h")).sum()
            )
            assert (
                row["amt_24h"]
                == earlier.loc[
                    earlier["ts"] >= row["ts"] - pd.Timedelta("24h"), "amount"
                ].sum()
            )
            assert row["payee_seen"] == (earlier["payee"] == row["payee"]).sum()
            run = 0
            for f in earlier["flag"][::-1]:
                if f <= 0:
                    break
                run += 1
            assert (i == 0 and np.isnan(row["flag_streak"])) or row[
                "flag_streak"
            ] == run
    refused = json.loads(
        m.apply_entity_features(
            run_id, json.dumps([{"name": "x", "op": "roll_count", "column": "amount"}])
        )
    )
    assert "window" in refused["error"], "roll_count over n rows is meaningless"


def test_aggregate_at_decisions_sees_only_the_window_before_each_decision(tmp_path):
    t0 = pd.Timestamp("2026-03-01 08:00")
    ev = lambda h, kind, label=0: {
        "line": "L1",
        "ts": t0 + pd.Timedelta(hours=h),
        "kind": kind,
        "takeover": label,
    }  # noqa: E731
    rows = [
        ev(0, "reset"),
        ev(1, "care"),
        ev(2, "sim_change", 1),
        ev(2.5, "otp"),
        ev(30, "sim_change"),
    ]
    rows += [
        {"line": f"L{i}", "ts": t0, "kind": "sim_change", "takeover": i % 2}
        for i in range(2, 8)
    ]
    path = _write(tmp_path, pd.DataFrame(rows), "journey.csv")
    specs = [
        {"name": "events_24h", "agg": "count"},
        {"name": "resets_24h", "agg": "count", "where": "kind == 'reset'"},
        {"name": "days_since_reset", "agg": "recency_days", "where": "kind == 'reset'"},
        {"name": "changes_30D", "agg": "count", "where": "kind == 'sim_change'", "window": "30D"},
    ]
    res = json.loads(
        m.aggregate_events(
            path,
            "line",
            "ts",
            json.dumps(specs),
            at="kind == 'sim_change'",
            window="24h",
        )
    )
    assert "error" not in res, res
    out = pd.read_csv(res["out_path"])
    first, second = out[out["line"] == "L1"].sort_values("ts").to_dict("records")
    assert (first["events_24h"], first["resets_24h"], first["takeover"]) == (2, 1, 1), (
        "reset + care, not the OTP after"
    )
    assert (second["events_24h"], second["resets_24h"]) == (0, 0), (
        "nothing in the 24h before hour 30"
    )
    assert second["days_since_reset"] == 30 / 24, "recency looks back beyond the window"
    assert abs(first["days_since_reset"] - 2 / 24) < 1e-9
    assert (first["changes_30D"], second["changes_30D"]) == (0, 1), "a spec's own window overrides the call's"
    assert len(out) == 2 + 6, "one row per decision event: L1's two, one each for L2-L7"


def test_immature_after_drops_unsettled_rows_before_the_split(tmp_path):
    res = json.loads(
        m.prepare_dataset(
            _write(tmp_path, _entity_day_frame()),
            "label",
            time_column="day",
            immature_after="16",
        )
    )
    assert res["immature_rows_dropped"] == 3 * 40, res
    train, test, meta = m._load_split(res["run_id"])
    assert (
        max(train["day"].max(), test["day"].max()) == 16
        and meta["immature_after"] == "16"
    )


def test_peer_features_are_fitted_on_the_training_fold_only(tmp_path):
    df = _entity_day_frame().assign(
        cell=lambda d: "c" + (d["caller"].str[-1].astype(int) % 2).astype(str)
    )
    run_id = json.loads(
        m.prepare_dataset(
            _write(tmp_path, df), "label", group_column="caller", time_column="day"
        )
    )["run_id"]
    train, test, _ = m._load_split(run_id)
    test.loc[test["cell"] == "c0", "total_calls"] = (
        10_000  # a test-only extreme must not move the peer median
    )
    test.to_csv(m._run_dir(run_id) / "test.csv", index=False)
    spec = [
        {
            "name": "calls_vs_cell",
            "column": "total_calls",
            "by": "cell",
            "stat": "ratio",
        }
    ]
    assert "error" not in json.loads(m.apply_peer_features(run_id, json.dumps(spec)))
    train2, test2, _ = m._load_split(run_id)
    median_c0 = train.loc[train["cell"] == "c0", "total_calls"].median()
    row = test2[test2["cell"] == "c0"].iloc[0]
    assert abs(row["calls_vs_cell"] - 10_000 / median_c0) < 1e-3
    own = [{"name": "x", "column": "total_calls", "by": "caller", "stat": "ratio"}]
    assert (
        "apply_entity_features"
        in json.loads(m.apply_peer_features(run_id, json.dumps(own)))["error"]
    )


def test_profile_features_flags_a_label_copy_and_a_dead_feature(tmp_path):
    run_id = json.loads(
        m.prepare_dataset(_write(tmp_path, _entity_day_frame()), "label")
    )["run_id"]
    m.apply_custom_feature(run_id, "label_echo", "label * 3 + 1")
    m.apply_custom_feature(run_id, "flat", "total_calls * 0")
    report = json.loads(m.profile_features(run_id))["features"]
    assert report["label_echo"]["flag"] == "leakage_suspect", report["label_echo"]
    assert report["flat"]["flag"] == "constant"



def test_every_cv_respects_the_split_and_the_model_never_sees_the_keys(tmp_path):
    """Temporal folds validate strictly later rows (by the time column, not
    file order); grouped folds never share an entity; the split keys stay in
    the data but never reach the model, and no gate asks to drop them."""
    frame = _entity_day_frame()
    X, y = frame.drop(columns=["label"]), frame["label"]
    shuffled = X.sample(frac=1, random_state=1)
    folds, scheme = m.core._validation_folds(shuffled, y.loc[shuffled.index], {"time_column": "day"})
    assert "forward-chained" in scheme and len(folds) >= 3, scheme
    for fit, val in folds:
        assert shuffled["day"].iloc[fit].max() <= shuffled["day"].iloc[val].min()
    folds, scheme = m.core._validation_folds(X, y, {"group_column": "caller"})
    assert "grouped" in scheme
    for fit, val in folds:
        assert not set(X["caller"].iloc[fit]) & set(X["caller"].iloc[val])

    run_id = json.loads(m.prepare_dataset(_write(tmp_path, frame), "label", group_column="caller", time_column="day"))["run_id"]
    m.detect_data_leakage(run_id)
    trained = json.loads(m.train_model(run_id, "random_forest"))
    assert "forward-chained" in trained["cv_scheme"], trained["cv_scheme"]
    pipeline = joblib.load(m._run_dir(run_id) / "pipeline.pkl")
    features = set(pipeline.named_steps["encode"].get_feature_names_out())
    assert not any(f == "day" or f.startswith("caller") for f in features), features
    gates = json.loads(m.check_readiness(run_id))["checks"]
    assert gates["no_identifier_in_model"]["status"] == "pass", gates["no_identifier_in_model"]
    assert "caller" not in json.loads(m.propose_drop_columns(run_id))["proposals"]


def _device_frame(n=600, seed=0):
    rng = np.random.default_rng(seed)
    cls = rng.choice(["human", "static_iot", "mobile_iot"], n, p=[0.6, 0.25, 0.15])
    return pd.DataFrame({
        "gap_cv": np.where(cls == "human", 1.5, 0.05) + rng.normal(0, 0.6, n),
        "bytes": rng.gamma(2, 50, n),
        "code_copy": pd.Series(cls).map({"human": 2, "static_iot": 0, "mobile_iot": 1}),  # label as shuffled codes
        "name_copy": cls,                                                                  # label as text
        "device_class": cls,
    })


def test_multiclass_runs_screen_label_copies_and_every_model_speaks_the_labels(tmp_path):
    run_id = json.loads(m.prepare_dataset(_write(tmp_path, _device_frame()), "device_class"))["run_id"]
    leak = json.loads(m.detect_data_leakage(run_id))
    assert {"code_copy", "name_copy"} <= set(leak["near_perfect_predictors"]), leak["single_column_auc_top"]
    m.apply_drop_columns(run_id, '["code_copy", "name_copy"]')
    for model in ("xgboost", "hist_gradient_boosting"):
        out = json.loads(m.train_model(run_id, model))
        assert "error" not in out, out
        pipeline = joblib.load(m._run_dir(run_id) / "pipeline.pkl")
        _, test, _ = m._load_split(run_id)
        assert set(pipeline.predict(test.drop(columns=["device_class"]))) <= {"human", "static_iot", "mobile_iot"}
        explained = json.loads(m.explain_model(run_id, method="shap"))
        assert "error" not in explained, explained


def test_xgboost_weights_the_minority_whichever_label_sorts_second(tmp_path):
    y = pd.Series(["legit"] * 90 + ["fraud"] * 10)  # sorts fraud=0, legit=1: code 1 is the MAJORITY
    assert m.core._make_model("xgboost", y).get_params()["scale_pos_weight"] < 1
    y = pd.Series(["no"] * 90 + ["yes"] * 10)       # code 1 is the minority
    assert m.core._make_model("xgboost", y).get_params()["scale_pos_weight"] > 1


def test_logistic_regression_scales_and_survives_missing_values(tmp_path):
    frame = _entity_day_frame().assign(big=lambda d: d["total_calls"] * 1e6)
    frame.loc[::7, "total_calls"] = np.nan  # e.g. a first lag
    run_id = json.loads(m.prepare_dataset(_write(tmp_path, frame), "label"))["run_id"]
    out = json.loads(m.train_model(run_id, "logistic_regression"))
    assert "error" not in out, out
    encode = joblib.load(m._run_dir(run_id) / "pipeline.pkl").named_steps["encode"]
    assert "num" in encode.named_transformers_

if __name__ == "__main__":
    import tempfile
    from pathlib import Path

    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            with tempfile.TemporaryDirectory() as d:
                fn(Path(d))
            print(f"ok {name}")
    print("all split-integrity checks passed")
