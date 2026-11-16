"""Custom sklearn-compatible transformers used inside the pipelines
mcp_server.py builds and export_model saves. Deliberately its own module,
not defined inside mcp_server.py — a class pickled from a script's __main__
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
    pipeline design in mcp_server.py exists to remove."""

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


try:
    import numpy as _np
    from xgboost import XGBClassifier as _XGBClassifier

    class XGBLabelClassifier(_XGBClassifier):
        """XGBoost that takes and returns the run's own labels ("human",
        "static_iot"), not just 0..K-1 — plain XGBClassifier refuses string
        classes. Codes follow sorted label order, the same order
        predict_proba's columns and `classes_` use, so every positive-class
        lookup elsewhere keeps working unchanged."""

        def fit(self, X, y, **kwargs):
            labels, codes = _np.unique(_np.asarray(y), return_inverse=True)
            self._fitting = True  # XGBoost validates y against classes_ mid-fit: it must see codes there
            try:
                super().fit(X, codes, **kwargs)
            finally:
                self._fitting = False
            self.labels_ = labels
            return self

        @property
        def classes_(self):
            if getattr(self, "_fitting", False) or not hasattr(self, "labels_"):
                return super().classes_
            return self.labels_

        def predict(self, X, **kwargs):
            return self.labels_[_np.asarray(super().predict(X, **kwargs)).astype(int)]
except ImportError:  # xgboost is optional where only exported non-xgboost models are loaded
    pass


class RunFolds:
    """CV splitter for classification's calibrate_model: the run's own folds
    (forward-chained, grouped or stratified — core/validation.py), recomputed
    on whatever rows it is handed. Passed as CalibratedClassifierCV(cv=...)
    so the calibrator never learns from an entity's own rows or from the
    future, and a clone refit on a subsample still gets valid folds. It ends
    up inside the pickled model, so loaders need the class; split() only runs
    during fit, never at predict time."""

    def __init__(self, meta, n_splits=5):
        self.meta = {k: meta.get(k) for k in ("time_column", "group_column")}
        self.n_splits = n_splits

    def split(self, X, y=None, groups=None):
        from core.validation import validation_folds

        return iter(validation_folds(X, y, self.meta, self.n_splits)[0])

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits if X is None else len(list(self.split(X, y)))


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
