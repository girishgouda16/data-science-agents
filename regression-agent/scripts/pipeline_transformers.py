"""Custom sklearn-compatible transformers used inside the pipelines
mcp_server/ builds and export_model saves. Deliberately its own module,
not defined inside mcp_server/ — a class pickled from a script's __main__
scope can't be unpickled by any other process (it looks for the class under
"__main__", which means something different in every process). Anything
that loads an exported model — including from outside this repo — needs
this file importable alongside it, e.g. on sys.path next to the .pkl.
"""

from sklearn.base import BaseEstimator, TransformerMixin


class ColumnDropper(BaseEstimator, TransformerMixin):
    """Drops the given columns if present — a no-op for any that are already
    missing, so replaying this on new raw data at inference time doesn't
    break if the caller already trimmed them."""

    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X.drop(columns=[c for c in self.columns if c in X.columns])


class ColumnFiller(BaseEstimator, TransformerMixin):
    """Fills NaN in the given columns with the given (train-fitted) values —
    the same fill values decided when the run was prepared, never
    recomputed from whatever data flows through later. Recomputing from
    whatever's being transformed is exactly the leakage bug the run-based
    pipeline design in mcp_server/ exists to remove."""

    def __init__(self, values):
        self.values = values

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        for col, value in self.values.items():
            if col in X.columns:
                X[col] = X[col].fillna(value)
        return X


class FeatureRecipes(BaseEstimator, TransformerMixin):
    """First pipeline step: rebuilds every engineered feature from raw
    columns, in the order the run applied them, so an exported model scores
    rows shaped like the ORIGINAL file. A recipe whose outputs are already
    present (the run's own train/test files) is skipped, so training and
    evaluation see exactly the values the run computed.

    kinds: expr (pandas eval over columns), datetime (calendar parts),
    frequency / peer (statistics fitted on the training fold, stored in the
    recipe), history (needs the entity's earlier rows — cannot be rebuilt
    from one row, so it must arrive as a column)."""

    def __init__(self, recipes):
        self.recipes = recipes

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        import pandas as _pd

        X = X.copy()
        for r in self.recipes:
            if all(o in X.columns for o in r["outputs"]):
                continue
            kind = r["kind"]
            if kind == "expr":
                safe = {c: f"_{c}" for c in X.columns if str(c)[:1].isdigit()}
                X[r["outputs"][0]] = X.rename(columns=safe).eval(r["expression"])
            elif kind == "datetime":
                raw = X[r["column"]]
                parsed = _pd.to_datetime(raw, errors="coerce")
                if parsed.isna().sum() > raw.isna().sum():
                    parsed = _pd.to_datetime(raw, errors="coerce", format="mixed")
                for out in r["outputs"]:
                    part = out[len(r["column"]) + 1:]
                    X[out] = getattr(parsed.dt, part)
                if r.get("drop_source"):
                    X = X.drop(columns=[r["column"]])
            elif kind == "frequency":
                X[r["outputs"][0]] = X[r["column"]].map(dict(r["map"])).fillna(0.0)
                X = X.drop(columns=[r["column"]])
            elif kind == "peer":
                stats = {k: dict(v) for k, v in r["stats"].items()}
                fitted = {s: X[r["by"]].map(stats[s]).fillna(r["overall"][s]) for s in ("median", "mean", "std")}
                col = X[r["column"]]
                X[r["outputs"][0]] = (
                    col / (fitted["median"].abs() + 1e-6)
                    if r["stat"] == "ratio"
                    else (col - fitted["mean"]) / (fitted["std"] + 1e-6)
                )
            else:
                missing = [o for o in r["outputs"] if o not in X.columns]
                raise ValueError(
                    f"columns {missing} are entity-history features ({kind}): they need the entity's earlier "
                    "rows, which one scoring row does not carry — compute them upstream (same definition as "
                    "the run's report) and pass them in as columns"
                )
        return X
