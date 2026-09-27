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
