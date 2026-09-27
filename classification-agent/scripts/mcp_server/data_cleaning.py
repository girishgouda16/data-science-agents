"""Every propose_* -> ask_user -> apply_* data-changing step: imputation,
column-dropping, datetime decomposition, SMOTE. Every value a propose_*
computes comes from the training fold only, and every apply_* writes both
train.csv and test.csv so the two stay the same shape — see the
`data-cleaning` skill for the HITL cycle these follow."""

import json

import pandas as pd

from .core import (
    _load_meta,
    _load_split,
    _run_dir,
    _save_meta,
    _to_native,
    mcp,
    parse_dates,
    split_keys,
)


def _reasoned_drops(meta: dict) -> dict:
    """dropped_columns used to be a bare list (no reason kept) — normalize
    old runs' meta.json on read so reporting can always render {col: reason}."""
    existing = meta.get("dropped_columns") or {}
    if isinstance(existing, list):
        return {
            col: "reason not recorded (dropped before this field tracked why)"
            for col in existing
        }
    return existing


@mcp.tool()
def propose_imputation(run_id: str) -> str:
    """Suggest a per-column imputation strategy (median/mode/none), computed
    from the training fold only — does not modify anything. Present to the
    user before apply_imputation."""
    train, _, meta = _load_split(run_id)
    proposals = {}
    for col in train.columns:
        if col == meta["target"]:
            continue
        missing_pct = round(train[col].isna().mean() * 100, 2)
        if missing_pct == 0:
            continue
        if missing_pct > 50:
            strategy = "drop_column"
        elif pd.api.types.is_numeric_dtype(train[col]):
            strategy = "median"
        else:
            strategy = "mode"
        proposals[col] = {"missing_pct": missing_pct, "suggested_strategy": strategy}
    return json.dumps({"proposals": proposals})


@mcp.tool()
def apply_imputation(run_id: str, strategies: str) -> str:
    """Apply user-approved per-column strategies. strategies: JSON string
    {col: "median"|"mode"|"drop_column"|<literal value>}. Fill values are
    computed from the training fold only, then applied to both train and
    test — the test fold never influences what value gets filled in."""
    train, test, meta = _load_split(run_id)
    plan = json.loads(strategies)
    dropped, values, missing_pcts = [], {}, {}
    for col, strategy in plan.items():
        if strategy == "drop_column":
            dropped.append(col)
            missing_pcts[col] = round(train[col].isna().mean() * 100, 2)
            continue
        if strategy == "median":
            value = float(train[col].median())
        elif strategy == "mode":
            value = _to_native(train[col].mode().iloc[0])
        else:
            value = strategy
        values[col] = value
        train[col] = train[col].fillna(value)
        test[col] = test[col].fillna(value)

    if dropped:
        train = train.drop(columns=dropped)
        test = test.drop(columns=[c for c in dropped if c in test.columns])
        reasons = {
            col: f"{missing_pcts[col]}% missing — above the 50% imputation-reliability threshold"
            for col in dropped
        }
        meta["dropped_columns"] = {**_reasoned_drops(meta), **reasons}

    meta["imputation"] = {**meta["imputation"], **values}
    train.to_csv(_run_dir(run_id) / "train.csv", index=False)
    test.to_csv(_run_dir(run_id) / "test.csv", index=False)
    _save_meta(run_id, meta)
    return json.dumps(
        {
            "run_id": run_id,
            "dropped_columns": dropped,
            "imputed_columns": list(values),
            "train_shape": train.shape,
        }
    )


# The identifier heuristics live here, in ONE place, because three callers
# need the identical answer: propose_drop_columns (what to drop),
# diagnostics.detect_data_leakage (what to flag), and eda.propose_group_column
# (what to group-split on). They used to be re-implemented per caller and
# drifted — detect_data_leakage's copy required uniqueness == 100% where
# propose_drop_columns wanted >= 98%, so a column could be "safe to keep"
# and "an ID column" in the same run. Change a threshold here, every caller
# moves together.
#
# The three cardinality signals cover three different corners of the
# (uniqueness ratio x absolute distinct count) space. Each was added after a
# real dataset slipped through the others:
#
#   ratio >= 0.98, any dtype .... a classic per-row id (PassengerId, row_id).
#   non-numeric, count > 1000 ... low ratio, huge absolute count. A real telco
#                                 Wangiri file: the hashed caller-ID repeats
#                                 across the 1/3/7-day lag rows, so its ratio
#                                 was only ~0.65 — far under the near-unique
#                                 bar — but 2.28M distinct values is not a
#                                 categorical feature by any reading.
#   non-numeric, ratio >= 0.5 ... high ratio, MODERATE absolute count. Titanic's
#                                 `Ticket`: 0.76 ratio but only ~544 distinct,
#                                 so it cleared the near-unique bar AND the
#                                 absolute bar and reached the model, where
#                                 one-hot turned it into hundreds of
#                                 memorization columns that dominated SHAP.
#                                 Same shape as order/invoice/claim numbers.
#
# A legitimate categorical FEATURE (region, plan, channel) is low on both
# axes, so it trips none of them. Numeric columns are exempt from the two
# non-numeric rules because a continuous numeric feature is naturally
# high-ratio — `Fare` is not an identifier.
HIGH_CARDINALITY_CATEGORICAL_THRESHOLD = 1000
NEAR_UNIQUE_RATIO = 0.98
IDENTIFIER_RATIO_THRESHOLD = 0.5
# Below this many distinct values one-hot encoding stays small enough that a
# high ratio is harmless (a 12-distinct column in 20 rows is ratio 0.6 but
# encodes to 12 columns, not 500) — keeps the ratio rule off small fixtures
# and genuinely small categoricals.
IDENTIFIER_RATIO_MIN_DISTINCT = 20

# The minority share below which SMOTE is worth escalating to, given that
# class_weight="balanced" is already applied to every model in this agent.
# Single source of truth: the data-cleaning skill quotes this number, it does
# not carry its own copy — the two drifted apart once (skill said 2%, the tool
# recommended at 10%, so a 5% churn set got opposite answers from the same run).
SMOTE_MIN_RATIO = 0.02

# Identifier rules whose columns are still worth keeping AS A FREQUENCY. Both
# describe a column with too many distinct values to one-hot — which is a
# statement about the ENCODER, not about the column being meaningless. A
# destination prefix, a cell id and an APN all trip high_cardinality, and how
# often a value occurs is exactly the signal ("this call went somewhere almost
# nobody calls"). near_unique is excluded on purpose: a value seen once has a
# frequency of 1/n for every row, which is noise with extra steps.
FREQUENCY_ENCODABLE_RULES = ("high_cardinality", "identifier_ratio")

# A categorical wide enough that one-hot encoding it is already a bad deal,
# but not wide enough to trip any identifier rule — the band between "a real
# categorical feature" and "an identifier". A destination prefix (a few
# hundred distinct), a cell id, a merchant id and a device TAC all sit here:
# nothing flags them, so they reach the encoder and become hundreds of sparse
# columns that can only memorize. They are NOT proposed for dropping, because
# they are genuine features — they are offered frequency encoding instead.
WIDE_CATEGORICAL_MIN_DISTINCT = 50

# ponytail: substring match on the column NAME, independent of cardinality —
# catches an identifier the cardinality checks above can miss entirely (most
# importantly a NUMERIC id that repeats, e.g. an integer customer_id, which
# the two non-numeric rules skip by design and the near-unique rule misses
# because it repeats). Every real dataset names its identifiers differently
# ("FROM" here, "msisdn"/"customer_id"/"account_number" elsewhere) — this is
# a short list of common tokens, not exhaustive. A name match alone is a WEAK
# signal (a real feature can contain "order"), so it proposes and warns but
# never hard-blocks a readiness gate on its own — see _identifier_signal's
# strength field. Extend as new datasets surface names it misses.
# A name match is normally a WEAK signal, because a real feature can contain
# "order" or "claim" and a human should confirm before it is dropped. These
# tokens are the exception: each one names a globally-standardised subscriber,
# SIM or device identity with exactly one meaning, so there is no domain in
# which a column called IMSI is a feature. Asking about them wastes a
# round-trip on a question with one answer. Note this covers the RAW
# identifier only — a value DERIVED from one can be a genuine feature (the
# TAC, the first 8 digits of an IMEI, resolves to the device model; a country
# code taken from an MSISDN is a destination feature). Derive those before
# prepare_dataset and give them their own names; the raw column still goes.
UNAMBIGUOUS_ID_NAMES = ("msisdn", "imsi", "imei", "iccid", "uuid", "guid", "ssn")

ID_NAME_PATTERNS = (
    "_id",
    "id_",
    "uuid",
    "guid",
    "msisdn",
    "imei",
    "imsi",
    "phone",
    "caller",
    "callee",
    "account_no",
    "acct_no",
    "customer_no",
    "ssn",
    "email",
    "hash",
    "ticket",
    "invoice",
    "order_no",
    "claim",
    "reference",
    "serial",
    "policy_no",
)


def _looks_like_id_by_name(col: str) -> bool:
    name = col.lower()
    return name == "id" or any(p in name for p in ID_NAME_PATTERNS)


def _is_integer_like(series) -> bool:
    """True for an int column, or a float column holding only whole numbers
    (ids survive a CSV round-trip as floats whenever the column has a NaN)."""
    if pd.api.types.is_integer_dtype(series):
        return True
    values = series.dropna()
    return bool(len(values)) and bool((values % 1 == 0).all())


def _identifier_signal(series, n_rows: int, col: str) -> dict | None:
    """The single source of truth for "is this column an identifier rather
    than a feature". Returns None for a normal feature, else {"reason",
    "strength", "distinct", "ratio"}.

    strength="strong" — a cardinality signal, safe to gate on: a column this
    shape cannot carry generalizable signal through a one-hot encoder, it can
    only memorize.
    strength="weak" — name-pattern match only, with unremarkable cardinality.
    Worth surfacing to a human, never worth failing a run over by itself.
    """
    distinct = int(series.nunique(dropna=True))
    if distinct <= 1 or n_rows == 0:
        return None
    ratio = distinct / n_rows
    numeric = pd.api.types.is_numeric_dtype(series)
    common = {"distinct": distinct, "ratio": round(ratio, 4)}

    if ratio >= NEAR_UNIQUE_RATIO and (not numeric or _is_integer_like(series)):
        # The numeric guard matters: a continuous measurement (a price, a score,
        # a rate) is naturally unique per row, and without it EVERY float feature
        # reads as an identifier. Real ids are whole numbers — 3.7 is a
        # measurement, not a key — so integer-likeness is what separates
        # `PassengerId` from `Fare`. This was a latent false positive while it
        # only produced a droppable suggestion; as a blocking gate it would have
        # failed any run with a unique-valued float feature.
        return {
            "reason": f"near-unique ({distinct} distinct in {n_rows} rows, ratio {ratio:.2f}) — an ID column, not a feature",
            "strength": "strong",
            "rule": "near_unique",
            **common,
        }
    if not numeric and distinct > HIGH_CARDINALITY_CATEGORICAL_THRESHOLD:
        return {
            "reason": f"high-cardinality categorical ({distinct} distinct values) — an identifier that repeats "
            "across rows (e.g. one row per entity per time window), not a real feature",
            "strength": "strong",
            "rule": "high_cardinality",
            **common,
        }
    if (
        not numeric
        and ratio >= IDENTIFIER_RATIO_THRESHOLD
        and distinct >= IDENTIFIER_RATIO_MIN_DISTINCT
    ):
        return {
            "reason": f"identifier-shaped categorical ({distinct} distinct in {n_rows} rows, ratio {ratio:.2f}) — "
            "too many distinct values to generalize; one-hot encoding this memorizes rows "
            "(ticket/order/invoice/claim numbers have exactly this shape)",
            "strength": "strong",
            "rule": "identifier_ratio",
            **common,
        }
    name = col.lower()
    if any(tok in name for tok in UNAMBIGUOUS_ID_NAMES):
        return {
            "reason": f"'{col}' names a standardised subscriber/SIM/device identity ({distinct} distinct values) "
            "— an identifier of the record, never a feature. Drop it. If you want the information "
            "inside it (device model from an IMEI's TAC, destination country from an MSISDN), derive "
            "that as its own column before prepare_dataset.",
            "strength": "strong",
            "rule": "unambiguous_id_name",
            **common,
        }
    if _looks_like_id_by_name(col):
        return {
            "reason": f"column name matches a common identifier pattern ({distinct} distinct values seen) — "
            "verify this is really an identifier for your domain before dropping, name matches "
            "alone can be wrong",
            "strength": "weak",
            "rule": "name_pattern",
            **common,
        }
    return None


def identifier_columns(df, target: str | None = None) -> dict:
    """{column: signal} for every identifier-shaped column in df. The shared
    entry point for propose_drop_columns / detect_data_leakage /
    propose_group_column."""
    return {
        col: signal
        for col in df.columns
        if col != target
        and (signal := _identifier_signal(df[col], len(df), col)) is not None
    }


@mcp.tool()
def propose_drop_columns(run_id: str) -> str:
    """Suggest columns to drop (near-constant, near-unique/ID-like, a
    high-cardinality categorical that's really an identifier with repeat
    rows, or a column whose NAME matches a common identifier pattern
    regardless of its cardinality), computed from the training fold — does
    not modify anything. Present to the user before apply_drop_columns —
    ask them to confirm each flagged column really is an identifier in
    their domain rather than assuming; a name match especially is a weak
    signal on its own (e.g. a genuine categorical feature could coincidentally
    contain "id")."""
    train, _, meta = _load_split(run_id)
    proposals = {}
    for col in train.columns:
        if col == meta["target"]:
            continue
        if train[col].nunique(dropna=True) <= 1:
            proposals[col] = "constant column, no signal"
    encodable = {}
    identifiers = identifier_columns(train, meta["target"])
    for col, signal in identifiers.items():
        proposals.setdefault(col, signal["reason"])
        if signal["rule"] in FREQUENCY_ENCODABLE_RULES and col != meta.get(
            "group_column"
        ):
            encodable[col] = signal["distinct"]
    for col in train.columns:
        if (
            col == meta["target"]
            or col in identifiers
            or col == meta.get("group_column")
        ):
            continue
        if pd.api.types.is_numeric_dtype(train[col]):
            continue
        distinct = int(train[col].nunique(dropna=True))
        if distinct >= WIDE_CATEGORICAL_MIN_DISTINCT:
            # Deliberately not added to `proposals`: this is a real feature
            # that the encoder handles badly, not a column to throw away.
            encodable[col] = distinct
    # The entity id and time column are split keys: _build_pipeline already
    # keeps them out of the model, and grouped/temporal validation needs them.
    keys = split_keys(meta)
    for key in keys:
        proposals.pop(key, None)
        encodable.pop(key, None)
    return json.dumps(
        {
            "proposals": proposals,
            "split_keys_kept": keys,
            # Dropping is not the only answer to high cardinality — it is the only
            # answer available to a one-hot encoder. See apply_frequency_encoding.
            "frequency_encodable_instead": encodable,
            "note": (
                (
                    "columns under frequency_encodable_instead one-hot into more columns than they are worth, but may "
                    "carry signal as a frequency (a rare destination prefix, an unusual cell, a seldom-seen APN) — "
                    "consider apply_frequency_encoding. Those that ALSO appear in proposals are identifier-shaped and "
                    "are a genuine keep-or-drop choice; those that appear only here are real features that the one-hot "
                    "encoder handles badly, and dropping them is the wrong response"
                )
                if encodable
                else ""
            ),
        }
    )


@mcp.tool()
def apply_drop_columns(run_id: str, columns: str) -> str:
    """Drop user-approved columns from both train and test. columns: JSON
    list of column names."""
    train, test, meta = _load_split(run_id)
    # Every other tool here returns {"error": ...} on bad input; this one used
    # to raise JSONDecodeError straight out of the MCP call, which surfaces as
    # a crashed tool rather than something the agent can read and correct.
    # A model passing `msisdn` or `msisdn,plan` instead of `["msisdn"]` is a
    # routine slip, not an exceptional one — accept both and say so.
    try:
        parsed = json.loads(columns)
    except (json.JSONDecodeError, TypeError):
        parsed = [c.strip() for c in str(columns).split(",") if c.strip()]
    if isinstance(parsed, str):
        parsed = [parsed]
    if not isinstance(parsed, list):
        return json.dumps(
            {"error": f"columns must be a JSON list of column names, got {columns!r}"}
        )
    unknown = [c for c in parsed if c not in train.columns]
    kept_keys = [c for c in parsed if c in split_keys(meta)]
    cols = [c for c in parsed if c in train.columns and c not in kept_keys]
    if not cols and kept_keys:
        return json.dumps({
            "run_id": run_id,
            "dropped_columns": [],
            "kept_split_keys": kept_keys,
            "note": "split keys stay in the data for grouped/temporal validation; the model pipeline already "
                    "excludes them, so there is nothing to drop",
        })
    if not cols:
        return json.dumps(
            {
                "error": f"none of {parsed} are columns of this run's training fold",
                "available": list(train.columns),
            }
        )
    reasons_available = json.loads(propose_drop_columns(run_id))["proposals"]
    train = train.drop(columns=cols)
    test = test.drop(columns=[c for c in cols if c in test.columns])
    new_reasons = {
        c: reasons_available.get(c, "dropped by explicit user request") for c in cols
    }
    meta["dropped_columns"] = {**_reasoned_drops(meta), **new_reasons}
    train.to_csv(_run_dir(run_id) / "train.csv", index=False)
    test.to_csv(_run_dir(run_id) / "test.csv", index=False)
    _save_meta(run_id, meta)
    return json.dumps(
        {"run_id": run_id, "dropped_columns": cols, "train_shape": train.shape,
         **({"kept_split_keys": kept_keys} if kept_keys else {})}
    )


@mcp.tool()
def propose_datetime_features(run_id: str) -> str:
    """Detects training-fold columns that are datetimes (by dtype, or by
    >=95% of non-null values successfully date-parsing) and proposes
    decomposing each into year/month/day/dayofweek/hour numeric columns — a
    raw timestamp/date string carries no signal to a one-hot/numeric
    pipeline as-is, but its calendar structure often does (e.g. fraud
    clustering at odd hours). Read-only — pairs with
    apply_datetime_features via the same propose -> ask_user -> apply cycle
    as imputation/column-dropping."""
    train, _, meta = _load_split(run_id)
    skip = {meta["target"], *(meta.get("dropped_columns") or [])}
    candidates = []
    for col in train.columns:
        if col in skip:
            continue
        if pd.api.types.is_datetime64_any_dtype(train[col]):
            candidates.append(col)
        elif (
            train[col].dtype == object
            and parse_dates(train[col]).notna().mean() >= 0.95
        ):
            candidates.append(col)
    return json.dumps({"run_id": run_id, "datetime_columns": candidates})


@mcp.tool()
def apply_datetime_features(run_id: str, columns: str) -> str:
    """Replaces each user-approved column (comma-separated) with
    <col>_year/_month/_day/_dayofweek/_hour numeric columns, on both train
    and test — no parameters are fit from data here (just calendar math), so
    there's nothing that could leak train into test the way a fitted
    imputation value could.

    The run's own time_column is decomposed but KEPT (it orders temporal
    validation; the pipeline excludes it from the model), and it gets no
    `_year`: on a forward-in-time split the test period's year is one the
    model never saw — a trend a tree cannot extrapolate, not a season."""
    train, test, meta = _load_split(run_id)
    cols = [c.strip() for c in columns.split(",") if c.strip()]
    time_key = meta.get("time_column")
    for df in (train, test):
        for col in cols:
            if col not in df.columns:
                continue
            parsed = parse_dates(df[col])
            if col != time_key:
                df[f"{col}_year"] = parsed.dt.year
            df[f"{col}_month"] = parsed.dt.month
            df[f"{col}_day"] = parsed.dt.day
            df[f"{col}_dayofweek"] = parsed.dt.dayofweek
            df[f"{col}_hour"] = parsed.dt.hour
            if col != time_key:
                df.drop(columns=[col], inplace=True)
    train.to_csv(_run_dir(run_id) / "train.csv", index=False)
    test.to_csv(_run_dir(run_id) / "test.csv", index=False)
    meta["datetime_features"] = sorted(
        set(meta.get("datetime_features") or []) | set(cols)
    )
    _save_meta(run_id, meta)
    return json.dumps(
        {"run_id": run_id, "decomposed_columns": cols, "train_shape": train.shape}
    )


@mcp.tool()
def propose_smote(run_id: str) -> str:
    """Check whether SMOTE oversampling would help, based on the training
    fold — does not change anything.

    Recommended only below SMOTE_MIN_RATIO minority share, with enough
    minority rows (>=10) for k-NN synthesis, AND only on a plain random
    split. class_weight="balanced" is already on for every model in this
    agent and costs nothing, so SMOTE is the escalation for extreme
    imbalance, not the default response to any imbalance.

    On a grouped or temporal split it is NOT recommended, and the reason is
    structural rather than statistical: SMOTE interpolates between a
    minority row and its nearest minority neighbours, and on entity-per-
    period data those neighbours belong to OTHER entities and OTHER periods.
    The synthetic rows are blends of subscribers who never existed, or of a
    January caller with a March one. That is not leakage into the test fold
    — the resampler lives inside the pipeline and only ever sees training
    data — it is a training set that no longer represents the population the
    split was carefully built to model."""
    train, _, meta = _load_split(run_id)
    counts = train[meta["target"]].value_counts()
    ratio = float(counts.min() / counts.max())
    split_type = meta.get("split_type") or (
        "grouped" if meta.get("group_column") else "random"
    )
    structurally_sound = split_type == "random"
    recommend = bool(
        ratio < SMOTE_MIN_RATIO and counts.min() >= 10 and structurally_sound
    )
    return json.dumps(
        {
            "counts": counts.to_dict(),
            "imbalance_ratio": round(ratio, 4),
            "recommend_smote": recommend,
            "split_type": split_type,
            "reasoning": (
                f"split_type='{split_type}' — synthesising minority rows by interpolating between different "
                "entities/periods would fabricate rows the split was built to keep apart; use class_weight="
                "'balanced' (already on) and tune_threshold instead"
                if not structurally_sound
                else (
                    "minority class is small enough that oversampling may help beyond class_weight alone"
                    if recommend
                    else f"minority share {ratio:.4f} is above the {SMOTE_MIN_RATIO} escalation bar — class_weight="
                    "'balanced' is already on and is usually enough on its own"
                )
            ),
        }
    )


@mcp.tool()
def apply_smote(run_id: str, sampling_strategy: float = 1.0) -> str:
    """Turn on SMOTE for this run (user-approved step). Doesn't touch any
    file — SMOTE becomes a step inside the model pipeline that only ever
    fires during training (train_model/tune_hyperparams); it's structurally
    impossible for it to touch the held-out test fold or a future real
    prediction.

    sampling_strategy: ratio of minority:majority after resampling.
      1.0 = full balance 1:1 (default, slowest — generates most synthetic samples)
      0.1 = bring minority up to 10% of majority (much faster, usually enough)
      0.2 = bring minority up to 20% of majority
    For creditcard fraud (0.17% positive rate), 0.1 is a good starting point.
    """
    meta = _load_meta(run_id)
    meta["use_smote"] = True
    meta["smote_sampling_strategy"] = sampling_strategy
    _save_meta(run_id, meta)
    split_type = meta.get("split_type") or (
        "grouped" if meta.get("group_column") else "random"
    )
    result = {
        "run_id": run_id,
        "use_smote": True,
        "sampling_strategy": sampling_strategy,
    }
    if split_type != "random":
        # Applied anyway — the user may have a reason — but never silently:
        # this is the one combination where the resampler undoes the split.
        result["warning"] = (
            f"this run uses a {split_type} split; SMOTE will interpolate between rows from different "
            "entities/periods, producing synthetic rows that the split exists to prevent. Surface this to the "
            "user and prefer class_weight='balanced' plus tune_threshold unless they have overridden it knowingly."
        )
    return json.dumps(result)


@mcp.tool()
def apply_frequency_encoding(run_id: str, columns: str) -> str:
    """Replace each high-cardinality categorical column with how OFTEN its
    value occurs, turning a column that one-hot encoding could only memorize
    into a single numeric feature that generalizes.

    columns: comma-separated column names, from propose_drop_columns's
    frequency_encodable_instead list.

    The third option between keeping and dropping. propose_drop_columns
    flags a column like a destination prefix, a cell id or an APN as
    high-cardinality, and that flag is correct about one-hot encoding —
    hundreds of sparse columns can only memorize rows. It is not correct
    that the column is meaningless: how rare a value is often IS the signal
    (a call to a prefix almost nobody dials, a session on a seldom-seen
    cell). This maps each value to its share of TRAINING-fold rows, so the
    frequency is a fitted statistic learned from train alone and applied to
    test — the same leakage discipline as imputation. A value that appears
    only in test maps to 0.0, which is the honest encoding of "never seen
    while training", and the count of those is reported so an agent can see
    when a column is too volatile to be worth encoding at all.

    Refused for the run's group_column: encoding an entity id by its own row
    count hands the model a per-entity fingerprint, which is precisely what
    splitting by that column was meant to prevent."""
    train, test, meta = _load_split(run_id)
    cols = [c.strip() for c in columns.split(",") if c.strip()]
    missing = [c for c in cols if c not in train.columns]
    if missing:
        return json.dumps({"error": f"column(s) {missing} not in the training fold"})
    if meta.get("group_column") in cols:
        return json.dumps(
            {
                "error": f"'{meta['group_column']}' is this run's group_column — frequency-encoding it would give the "
                "model a per-entity fingerprint, defeating the grouped split. Drop it instead.",
            }
        )

    encoded = {}
    for col in cols:
        freq = train[col].value_counts(normalize=True)
        unseen = float((~test[col].isin(freq.index)).mean()) * 100 if len(test) else 0.0
        for df in (train, test):
            df[f"{col}_freq"] = df[col].map(freq).fillna(0.0)
            df.drop(columns=[col], inplace=True)
        encoded[col] = {
            "distinct_values_in_train": int(freq.shape[0]),
            "new_column": f"{col}_freq",
            "test_rows_with_unseen_value_pct": round(unseen, 2),
        }
    train.to_csv(_run_dir(run_id) / "train.csv", index=False)
    test.to_csv(_run_dir(run_id) / "test.csv", index=False)
    meta["frequency_encoded"] = {**(meta.get("frequency_encoded") or {}), **encoded}
    _save_meta(run_id, meta)
    return json.dumps(
        {"run_id": run_id, "frequency_encoded": encoded, "train_shape": train.shape}
    )
