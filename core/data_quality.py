"""Is this column an identifier, or a feature? Shared by every agent.

Copied from classification-agent, which is where these rules were tuned
against real failures — an identifier column (`Ticket`, 0.79 uniqueness)
reached a model, one-hot expanded into hundreds of columns and dominated
SHAP. Nothing about that failure is specific to classification: a regressor,
a clusterer and an anomaly detector all memorize the same way. So every
OTHER agent screens with the rules from here.

Deliberately a copy, not an import: classification-agent owns the original in
its own mcp_server/data_cleaning.py and is left alone. If a rule changes
there, change it here too — `core/test_data_quality.py` pins the behaviour
both copies have to agree on.

`identifier_columns(df, target)` is the entry point. Each hit carries a
`strength`: "strong" is a cardinality signal safe to gate a run on, "weak" is
a name-pattern match worth showing a human but never worth failing a run over
by itself.
"""

import pandas as pd

# One place, because every caller needs the identical answer — they used to
# be re-implemented per caller and drifted, so a column could be "safe to
# keep" and "an ID column" in the same run. Change a threshold here and
# every caller, in every agent, moves together.
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

# Identifier rules whose columns are still worth keeping AS A FREQUENCY. Both
# describe a column with too many distinct values to one-hot — which is a
# statement about the ENCODER, not about the column being meaningless. A
# destination prefix, a cell id and an APN all trip high_cardinality, and how
# often a value occurs is exactly the signal ("this call went somewhere almost
# nobody calls"). near_unique is excluded on purpose: a value seen once has a
# frequency of 1/n for every row, which is noise with extra steps.
FREQUENCY_ENCODABLE_RULES = ("high_cardinality", "identifier_ratio")

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
