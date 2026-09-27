"""Byte-for-byte copy of classification-agent/scripts/pipeline_transformers.py
— this agent's plot_confusion_matrix/plot_roc_curve unpickle that agent's
pipeline.pkl directly (see mcp_server/'s RUNS_DIR), and pickle resolves a
class by (module name, class name), not by structural equality — this
process needs a module literally named `pipeline_transformers` on its own
sys.path to satisfy that, the same way MODELS/_make_model are duplicated
below rather than imported across the two agents. If you change one copy
(e.g. ColumnFiller's fill logic), change both — an unpickle here doesn't
re-run classification-agent's code, it just needs the class shape to match.
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


import numpy as _np  # noqa: E402
from scipy.spatial.distance import cdist as _cdist  # noqa: E402
from sklearn.base import ClusterMixin as _ClusterMixin  # noqa: E402


class KMedoids(BaseEstimator, _ClusterMixin):
    """k-medoids under Manhattan distance, CLARA-style: fitted on a sample of
    at most max_rows (k-medoids++ seeding, alternating medoid updates, best
    of n_init), then every row goes to its nearest medoid. Medoids are real
    rows, so new rows are assigned the same way. On the clustering agent's
    Gower space (min-max numerics, one-hot categoricals weighted 0.5)
    Manhattan distance IS Gower distance times the number of variables:
    this is Gower k-medoids, the standard method for mixed data."""

    def __init__(self, n_clusters=4, max_rows=2000, n_init=3, max_iter=30, random_state=42):
        self.n_clusters = n_clusters
        self.max_rows = max_rows
        self.n_init = n_init
        self.max_iter = max_iter
        self.random_state = random_state

    def fit(self, X, y=None):
        X = _np.asarray(X, dtype=float)
        rng = _np.random.default_rng(self.random_state)
        S = X[rng.choice(len(X), size=min(len(X), self.max_rows), replace=False)]
        best_cost, best = _np.inf, None
        for _ in range(self.n_init):
            centers = S[[rng.integers(len(S))]]
            while len(centers) < self.n_clusters:  # k-medoids++: far rows are likelier seeds
                d = _cdist(S, centers, "cityblock").min(axis=1)
                pick = rng.choice(len(S), p=d / d.sum()) if d.sum() else rng.integers(len(S))
                centers = _np.vstack([centers, S[pick]])
            for _ in range(self.max_iter):
                labels = _cdist(S, centers, "cityblock").argmin(axis=1)
                moved = _np.array([self._medoid(S[labels == c]) if (labels == c).any() else centers[c]
                                   for c in range(self.n_clusters)])
                if _np.array_equal(moved, centers):
                    break
                centers = moved
            cost = _cdist(S, centers, "cityblock").min(axis=1).sum()
            if cost < best_cost:
                best_cost, best = cost, centers
        self.cluster_centers_ = best
        self.labels_ = self.predict(X)
        return self

    @staticmethod
    def _medoid(members):
        return members[_cdist(members, members, "cityblock").sum(axis=1).argmin()]

    def predict(self, X):
        return _cdist(_np.asarray(X, dtype=float), self.cluster_centers_, "cityblock").argmin(axis=1)
