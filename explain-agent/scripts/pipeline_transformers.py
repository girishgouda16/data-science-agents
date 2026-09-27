"""Classes a serving-agent predict() call needs importable under exactly
these names to unpickle another agent's exported .pkl — pickle records a
class as `<module>.<ClassName>`, and every upstream agent's mcp_server/
built its bundle with these classes living in a module literally named
`pipeline_transformers`, so that's what has to be on sys.path here too (see
each of those agents' own pipeline_transformers.py docstring).

This file is the union of the two shapes serving-agent has to load:
  - ColumnDropper/ColumnFiller — used inside the sklearn-Pipeline bundles
    exported by classification/regression/clustering/anomaly-agent. Their
    pipeline_transformers.py are byte-identical to each other, so one copy
    here covers all four.
  - XGBLabelClassifier (classification's xgboost with string labels) and
    KMedoids (clustering's Gower k-medoids) — copied from those agents'
    files; scripts/test_pickle_classes_in_sync.py fails CI if one is missing.
  - ForecastModel (+ its module-level helpers) — the bespoke, non-Pipeline
    artifact exported by forecasting-agent (see its own
    pipeline_transformers.py docstring for why it isn't an sklearn
    Pipeline). Copied verbatim, not imported across agents — this agent
    stays independently deployable, same convention as every agent's
    run_persistence.py.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
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
    recomputed from whatever data flows through later."""

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


def _calendar_features(dates: pd.Series) -> pd.DataFrame:
    """Cheap, always-computed calendar features. XGBoost ignores whichever
    ones don't carry signal for a given frequency (hour on daily data,
    dayofweek on monthly) — cheaper than branching per-frequency, and
    harmless. Hour of day is the strongest single feature of hourly telecom
    traffic; the weekend flag separates the two weekly regimes."""
    dt = pd.to_datetime(dates)
    return pd.DataFrame(
        {
            "cal_hour": dt.dt.hour,
            "cal_dayofweek": dt.dt.dayofweek,
            "cal_is_weekend": (dt.dt.dayofweek >= 5).astype(int),
            "cal_day": dt.dt.day,
            "cal_weekofyear": dt.dt.isocalendar().week.astype(int).to_numpy(),
            "cal_month": dt.dt.month,
            "cal_quarter": dt.dt.quarter,
        },
        index=dates.index,
    )


def _lag_rolling_features(
    series: pd.Series, lags: list[int], rolling_windows: list[int]
) -> pd.DataFrame:
    """lag_N = value N steps back; roll_N_mean = trailing N-step mean
    (shifted by 1 so it never includes the current/target step itself —
    that would be leaking the label into its own feature)."""
    out = {}
    for lag in lags:
        out[f"lag_{lag}"] = series.shift(lag)
    for window in rolling_windows:
        out[f"roll_{window}_mean"] = series.shift(1).rolling(window).mean()
    return pd.DataFrame(out, index=series.index)


def build_features(
    dates: pd.Series,
    target: pd.Series,
    exog: pd.DataFrame | None,
    lags: list[int],
    rolling_windows: list[int],
) -> pd.DataFrame:
    """Full feature frame for the xgboost model type — calendar + lag/rolling
    features on `target`, plus any exogenous columns as-is. Rows at the
    start with insufficient history come back with NaN lag/roll columns;
    callers drop those before fitting (see mcp_server/'s train_model)."""
    parts = [
        _calendar_features(dates),
        _lag_rolling_features(target, lags, rolling_windows),
    ]
    if exog is not None and not exog.empty:
        parts.append(exog.reset_index(drop=True).set_axis(dates.index))
    return pd.concat(parts, axis=1)


class ForecastModel:
    """model_type: "naive_seasonal" | "exponential_smoothing" | "xgboost".

    history: the full training-fold target series (values + dates), kept so
    forecast() can extrapolate forward from wherever training ended without
    needing the caller to resupply it.
    """

    def __init__(
        self,
        model_type: str,
        date_column: str,
        target: str,
        exog_columns: list[str],
        seasonal_periods: int,
        lags: list[int],
        rolling_windows: list[int],
        freq: str,
        fitted,
        encoder,
        history_dates: pd.Series,
        history_target: pd.Series,
        history_exog: pd.DataFrame | None,
    ):
        self.model_type = model_type
        self.date_column = date_column
        self.target = target
        self.exog_columns = exog_columns
        self.seasonal_periods = seasonal_periods
        self.lags = lags
        self.rolling_windows = rolling_windows
        self.freq = freq
        self.fitted = fitted  # statsmodels results obj, xgboost model, or the naive_seasonal cycle array
        self.encoder = encoder  # fitted OneHotEncoder for categorical exog, or None
        self.history_dates = history_dates
        self.history_target = history_target
        self.history_exog = history_exog

    def _encode_exog(self, exog: pd.DataFrame) -> pd.DataFrame:
        if self.encoder is None or not self.exog_columns:
            return exog
        cat_cols = [c for c in self.exog_columns if exog[c].dtype == object]
        if not cat_cols:
            return exog
        encoded = self.encoder.transform(exog[cat_cols])
        encoded_df = pd.DataFrame(
            encoded,
            columns=self.encoder.get_feature_names_out(cat_cols),
            index=exog.index,
        )
        return pd.concat([exog.drop(columns=cat_cols), encoded_df], axis=1)

    def future_dates(self, horizon: int) -> pd.DatetimeIndex:
        last = pd.to_datetime(self.history_dates.iloc[-1])
        return pd.date_range(last, periods=horizon + 1, freq=self.freq)[1:]

    def forecast(
        self, horizon: int, future_exog: pd.DataFrame | None = None
    ) -> pd.DataFrame:
        """Returns a DataFrame with `date_column` and `forecast` columns,
        `horizon` rows. future_exog (if the model has exogenous columns): a
        DataFrame with `horizon` rows for those columns, in order — if
        omitted, the last known value of each is carried forward flat
        (a documented approximation, not a real forecast of the regressors
        themselves)."""
        dates = self.future_dates(horizon)

        if self.exog_columns and future_exog is None and self.model_type == "xgboost":
            last_row = self.history_exog.iloc[[-1]]
            future_exog = pd.concat([last_row] * horizon, ignore_index=True)

        if self.model_type == "naive_seasonal":
            cycle = self.fitted  # last `seasonal_periods` training values, in order
            values = [cycle[i % len(cycle)] for i in range(horizon)]
            return self._finish(dates, values)

        if self.model_type == "exponential_smoothing":
            return self._finish(dates, np.asarray(self.fitted.forecast(horizon)))

        # xgboost — recursive: each step's prediction extends the series the
        # NEXT step's lag/rolling features are computed from, since there's
        # no real future target to look up.
        series = self.history_target.reset_index(drop=True)
        preds = []
        for i in range(horizon):
            feat_row = _calendar_features(pd.Series([dates[i]]))
            for lag in self.lags:
                feat_row[f"lag_{lag}"] = (
                    series.iloc[-lag] if len(series) >= lag else np.nan
                )
            for window in self.rolling_windows:
                feat_row[f"roll_{window}_mean"] = (
                    series.iloc[-window:].mean() if len(series) >= window else np.nan
                )
            if future_exog is not None:
                row_exog = future_exog.iloc[[i]].reset_index(drop=True)
                feat_row = pd.concat([feat_row, self._encode_exog(row_exog)], axis=1)
            pred = float(
                self.fitted.predict(feat_row[self.fitted.feature_names_in_])[0]
            )
            preds.append(pred)
            series = pd.concat([series, pd.Series([pred])], ignore_index=True)

        return self._finish(dates, preds)

    def _finish(self, dates, values) -> pd.DataFrame:
        """The forecast frame, clipped at 0 for a series that never went
        below it (traffic, counts, revenue), with the prediction interval
        when one was calibrated: per-step quantiles of the signed backtest
        errors (actual - forecast) at the same step ahead, so the band widens
        with the horizon and can be asymmetric. Steps beyond the calibrated
        horizon widen by sqrt(step / horizon)."""
        out = pd.DataFrame({self.date_column: dates, "forecast": np.asarray(values, dtype=float)})
        nonneg = bool((self.history_target.dropna() >= 0).all())
        interval = getattr(self, "interval", None)
        if interval:
            lower, upper = np.asarray(interval["lower"], dtype=float), np.asarray(interval["upper"], dtype=float)
            steps = np.arange(len(out))
            grow = np.sqrt(np.maximum(steps + 1, len(lower)) / len(lower))
            idx = np.minimum(steps, len(lower) - 1)
            out["lower"] = out["forecast"] + lower[idx] * grow
            out["upper"] = out["forecast"] + upper[idx] * grow
        if nonneg:
            out[[c for c in ("forecast", "lower", "upper") if c in out]] = out[
                [c for c in ("forecast", "lower", "upper") if c in out]].clip(lower=0)
        return out


class PanelForecastModel(ForecastModel):
    """Many series in one model — every cell, region or product of a long
    table. model_type:
      "naive_seasonal"         each series repeats its own last season
      "exponential_smoothing"  one Holt-Winters fit per series
      "xgboost"                ONE global model trained on every series at
                               once, each series scaled by its own training
                               mean so a quiet cell and a busy one share what
                               they have in common (the daily and weekly
                               shape, calendar effects), with the series'
                               code and size as features for what differs.
    forecast() returns one row per series per future step; the prediction
    interval is calibrated on backtest errors in each series' own scale.
    A subclass of ForecastModel so everything that loads a forecasting
    artifact (serving, explain) accepts it unchanged."""

    def __init__(self, model_type, date_column, target, series_column, exog_columns, seasonal_periods, lags,
                 rolling_windows, freq, fitted, scales, codes, history):
        self.model_type = model_type
        self.date_column = date_column
        self.target = target
        self.series_column = series_column
        self.exog_columns = exog_columns
        self.seasonal_periods = seasonal_periods
        self.lags = lags
        self.rolling_windows = rolling_windows
        self.freq = freq
        self.fitted = fitted  # global regressor, or {series: ETS results | naive cycle}
        self.encoder = None
        self.scales = scales  # {series: training mean of |target|}
        self.codes = codes  # {series: integer code}
        self.history = history  # long frame: series, date, target, exog — sorted
        self.history_dates = pd.Series(sorted(pd.to_datetime(history[date_column]).unique()))
        self.history_target = history[target]
        self.history_exog = None

    def forecast(self, horizon: int, future_exog: pd.DataFrame | None = None) -> pd.DataFrame:
        """future_exog (xgboost with exogenous columns): rows with the series
        column and the exogenous columns, `horizon` rows per series in time
        order; without it each series' last known value is carried flat."""
        dates = self.future_dates(horizon)
        series = list(self.scales)
        values = {}
        if self.model_type == "naive_seasonal":
            for s in series:
                cycle = self.fitted[s]
                values[s] = [cycle[i % len(cycle)] for i in range(horizon)]
        elif self.model_type == "exponential_smoothing":
            for s in series:
                f = self.fitted[s]  # a Holt-Winters fit, or a naive cycle where the series was too short for one
                values[s] = (list(np.asarray(f.forecast(horizon))) if hasattr(f, "forecast")
                             else [f[i % len(f)] for i in range(horizon)])
        else:
            hist = {s: list(g[self.target].to_numpy(dtype=float) / self.scales[s])
                    for s, g in self.history.groupby(self.series_column, sort=False)}
            last = self.history.groupby(self.series_column, sort=False).tail(1).set_index(self.series_column)
            future = ({s: g.reset_index(drop=True) for s, g in future_exog.groupby(self.series_column, sort=False)}
                      if future_exog is not None else {})
            preds = {s: [] for s in series}
            names = list(self.fitted.feature_names_in_)
            for i in range(horizon):
                rows = []
                for s in series:
                    h = hist[s]
                    row = {f"lag_{lag}": (h[-lag] if len(h) >= lag else np.nan) for lag in self.lags}
                    row.update({f"roll_{w}_mean": (float(np.mean(h[-w:])) if len(h) >= w else np.nan)
                                for w in self.rolling_windows})
                    row.update(series_code=self.codes[s], series_log_scale=float(np.log1p(self.scales[s])))
                    for col in self.exog_columns:
                        f = future.get(s)
                        row[col] = f[col].iloc[i] if f is not None and i < len(f) else last.at[s, col]
                    rows.append(row)
                feats = pd.concat([_calendar_features(pd.Series([dates[i]] * len(series))), pd.DataFrame(rows)], axis=1)
                step = self.fitted.predict(feats[names])
                for s, p in zip(series, step):
                    hist[s].append(float(p))
                    preds[s].append(float(p) * self.scales[s])
            values = preds
        frames = []
        interval = getattr(self, "interval", None)
        nonneg = bool((self.history[self.target].dropna() >= 0).all())
        for s in series:
            out = pd.DataFrame({self.series_column: s, self.date_column: dates,
                                "forecast": np.asarray(values[s], dtype=float)})
            if interval:
                lower, upper = np.asarray(interval["lower"], dtype=float), np.asarray(interval["upper"], dtype=float)
                steps = np.arange(horizon)
                grow = np.sqrt(np.maximum(steps + 1, len(lower)) / len(lower))
                idx = np.minimum(steps, len(lower) - 1)
                out["lower"] = out["forecast"] + lower[idx] * grow * self.scales[s]
                out["upper"] = out["forecast"] + upper[idx] * grow * self.scales[s]
            frames.append(out)
        out = pd.concat(frames, ignore_index=True)
        if nonneg:
            cols = [c for c in ("forecast", "lower", "upper") if c in out]
            out[cols] = out[cols].clip(lower=0)
        return out


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
