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
