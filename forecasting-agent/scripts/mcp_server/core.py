"""MCP server exposing time-series forecasting tools. prepare_dataset sorts
by the date column and splits train/test CHRONOLOGICALLY (last test_size
fraction as holdout) — a random/shuffled split is the classic time-series
leakage bug (it lets the model train on rows that happen after the ones
it's evaluated on), so there is deliberately no `shuffle` option anywhere in
this file. Every CV here is walk-forward (sklearn's TimeSeriesSplit) for the
same reason: plain K-fold shuffles.

Everything after prepare_dataset is keyed by run_id, same convention as
classification/regression/clustering/anomaly-agent. Mirrors their
propose/apply human-in-the-loop pattern for imputation and column-dropping;
the domain-specific step (train_model et al.) is genuinely different code,
not a relabeled copy — see pipeline_transformers.py's docstring for why the
fitted artifact isn't an sklearn Pipeline here.

Run: python -m mcp_server.server  (cwd=scripts/; agent.py does this)

Shared kernel of the package: imports, constants, run storage, every
helper and the `mcp` instance. Tools live in one module per skill."""

import json

import os

import shutil

import sys

import time

import uuid

from pathlib import Path

import joblib

import numpy as np

import optuna

import pandas as pd

from pipeline_transformers import ForecastModel, PanelForecastModel, _calendar_features, build_features

from run_persistence import save_training_run

from mcp.server.fastmcp import FastMCP

from scipy import stats

from sklearn.metrics import mean_absolute_error, mean_squared_error

from sklearn.model_selection import TimeSeriesSplit

from sklearn.preprocessing import OneHotEncoder

from xgboost import XGBRegressor

sys.path.insert(
    0, str(Path(__file__).parent.resolve().parents[2])
)  # repo root, for core.*

from core import mlops as core_mlops  # noqa: E402

mcp = FastMCP("forecasting-agent")
from core import toolguard  # noqa: E402

toolguard.install(
    mcp
)  # every tool's paths/models checked before it runs (core/toolguard.py)

optuna.logging.set_verbosity(optuna.logging.WARNING)

# AGENTIC_ML_DATA_DIR relocates data/ for every agent at once (the test suites
# set it so they never write into the real data/runs).
RUNS_DIR = (
    Path(
        os.environ.get("AGENTIC_ML_DATA_DIR")
        or Path(__file__).parent.parent.parent.parent / "data"
    )
    / "runs"
)

RUNS_DIR.mkdir(parents=True, exist_ok=True)

# ponytail: flat TTL swept opportunistically, not a scheduler — see
# regression-agent/scripts/mcp_server/'s identical note.
RUN_RETENTION_DAYS = float(os.environ.get("RUN_RETENTION_DAYS", 7))

MODELS = ["naive_seasonal", "exponential_smoothing", "xgboost"]

# Maps a pandas offset alias to how many periods make one seasonal cycle.
# Anything not listed here gets seasonal_periods=1 (no assumed seasonality)
# rather than erroring — an unusual frequency shouldn't block forecasting,
# it just means ETS/naive fall back to a non-seasonal model.
FREQ_SEASONAL_PERIODS = {
    "H": 24,
    "h": 24,
    "ME": 12,
    "T": 60,
    "min": 60,
    "D": 7,
    "B": 5,
    "W": 52,
    "M": 12,
    "MS": 12,
    "Q": 4,
    "QS": 4,
    "Y": 1,
    "A": 1,
    "AS": 1,
}

PARAM_GRIDS = {
    "naive_seasonal": {},  # nothing to tune — tune_hyperparams just re-scores it
    "exponential_smoothing": {
        "trend": ["add", None],
        "seasonal": ["add", "mul"],
        "damped_trend": [True, False],
    },
    "xgboost": {
        "n_estimators": [100, 200, 400],
        "max_depth": [3, 5, 7],
        "learning_rate": [0.01, 0.1, 0.3],
    },
}

# ponytail: relative (not absolute) gap threshold — RMSE/MAE are in the
# target's own units, so a flat number like regression's R^2-based 0.15
# doesn't transfer across datasets. 0.5 = test RMSE 50% worse than backtest.
OVERFIT_GAP_THRESHOLD = 0.5

# P10-P90 by default: the band capacity planning and stock decisions read.
INTERVAL_LEVEL = 0.8

# Candidate lags and rolling windows per frequency — several seasonalities
# where they exist (hourly traffic repeats daily AND weekly). prepare_dataset
# keeps those shorter than a third of the training history.
FREQ_LAGS = {"H": [1, 2, 24, 48, 168], "h": [1, 2, 24, 48, 168], "D": [1, 2, 7, 14, 28], "B": [1, 5, 10],
             "W": [1, 4, 52], "M": [1, 3, 12], "MS": [1, 3, 12], "ME": [1, 3, 12], "Q": [1, 4], "QS": [1, 4]}
FREQ_WINDOWS = {"H": [24, 168], "h": [24, 168], "D": [7, 28], "B": [5, 20], "W": [4, 13],
                "M": [3, 12], "MS": [3, 12], "ME": [3, 12], "Q": [4], "QS": [4]}
# Seasonal lags whose autocorrelation prepare_dataset reports, so the choice
# of seasonal period is made on evidence (hourly: daily or weekly cycle).
SEASONAL_CANDIDATES = {"H": [24, 168], "h": [24, 168], "D": [7, 365], "B": [5], "W": [52], "M": [12], "MS": [12],
                       "ME": [12], "Q": [4], "QS": [4]}


def _run_dir(run_id: str) -> Path:
    return RUNS_DIR / run_id


def _cleanup_old_runs(max_age_days: float = RUN_RETENTION_DAYS) -> None:
    cutoff = time.time() - max_age_days * 86400
    for run_dir in RUNS_DIR.iterdir():
        if not run_dir.is_dir():
            continue
        meta_path = run_dir / "meta.json"
        mtime = (
            meta_path.stat().st_mtime if meta_path.exists() else run_dir.stat().st_mtime
        )
        if mtime >= cutoff:
            continue
        # A run registered in MLflow is pinned: promoting its version exports
        # from this directory, so sweeping it strands a registry version.
        try:
            if json.loads(meta_path.read_text()).get("registry"):
                continue
        except (OSError, ValueError):
            pass
        shutil.rmtree(run_dir, ignore_errors=True)


def _load_meta(run_id: str) -> dict:
    path = _run_dir(run_id) / "meta.json"
    if not path.exists():
        raise FileNotFoundError(
            f"Unknown run_id '{run_id}' — call prepare_dataset first."
        )
    return json.loads(path.read_text())


def _save_meta(run_id: str, meta: dict) -> None:
    (_run_dir(run_id) / "meta.json").write_text(json.dumps(meta, indent=2, default=str))


def _load_split(run_id: str) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    meta = _load_meta(run_id)
    train = pd.read_csv(
        _run_dir(run_id) / "train.csv", parse_dates=[meta["date_column"]]
    )
    test = pd.read_csv(_run_dir(run_id) / "test.csv", parse_dates=[meta["date_column"]])
    return train, test, meta


def _to_native(value):
    return value.item() if hasattr(value, "item") else value


def is_panel(meta: dict) -> bool:
    return bool(meta.get("panel"))


def _exog_columns(meta: dict) -> list[str]:
    skip = {meta["date_column"], meta["target"], meta.get("series_column"), *(meta.get("dropped_columns") or [])}
    return [c for c in meta.get("all_columns", []) if c not in skip]


def _fit_ets(y: pd.Series, seasonal_periods: int, params: dict):
    """Holt-Winters. Damped or not is an empirical question: a damped trend
    is safer over long horizons, an undamped one right for a steady ramp —
    fit both, keep the lower AIC (tune_hyperparams can still force it)."""
    from statsmodels.tsa.holtwinters import ExponentialSmoothing

    sp = seasonal_periods if seasonal_periods > 1 else None
    trend = params.get("trend", "add")
    kwargs = dict(trend=trend, seasonal=(params.get("seasonal", "add") if sp else None), seasonal_periods=sp,
                  initialization_method="estimated")
    series = y.reset_index(drop=True)
    if trend and "damped_trend" not in params:
        return min((ExponentialSmoothing(series, damped_trend=d, **kwargs).fit() for d in (True, False)),
                   key=lambda res: res.aic)
    if trend:
        kwargs["damped_trend"] = params["damped_trend"]
    return ExponentialSmoothing(series, **kwargs).fit()


def _panel_features(df: pd.DataFrame, meta: dict, scales: dict, codes: dict) -> tuple:
    """Training features for the global model, every series at once: its
    own lags and trailing means in its own scale (target / series mean),
    calendar, the series' code and log size, numeric drivers."""
    s_col, d_col = meta["series_column"], meta["date_column"]
    scale = df[s_col].map(scales)
    y = df[meta["target"]].astype(float) / scale
    g = y.groupby(df[s_col])
    parts = {f"lag_{lag}": g.shift(lag) for lag in meta["lags"]}
    parts.update({f"roll_{w}_mean": g.transform(lambda v, w=w: v.shift(1).rolling(w).mean())
                  for w in meta["rolling_windows"]})
    X = pd.concat([_calendar_features(df[d_col]), pd.DataFrame(parts, index=df.index)], axis=1)
    X["series_code"] = df[s_col].map(codes)
    X["series_log_scale"] = np.log1p(scale)
    for col in _exog_columns(meta):
        X[col] = df[col]
    return X, y


def _fit_panel_model(df: pd.DataFrame, meta: dict, model_type: str, params: dict) -> PanelForecastModel:
    s_col, d_col, target = meta["series_column"], meta["date_column"], meta["target"]
    sp = meta["seasonal_periods"]
    df = df.sort_values([s_col, d_col]).reset_index(drop=True)
    groups = dict(tuple(df.groupby(s_col, sort=True)))
    scales = {s: float(g[target].abs().mean()) or 1.0 for s, g in groups.items()}
    codes = {s: i for i, s in enumerate(groups)}
    if model_type == "naive_seasonal":
        fitted = {s: (list(g[target].iloc[-sp:]) if sp > 1 else [float(g[target].iloc[-1])]) for s, g in groups.items()}
    elif model_type == "exponential_smoothing":
        fitted = {}
        for s, g in groups.items():
            try:
                fitted[s] = _fit_ets(g[target], sp, params)
            except Exception:  # too short for a seasonal fit: its own last season stands in
                fitted[s] = list(g[target].iloc[-sp:]) if sp > 1 else [float(g[target].iloc[-1])]
    else:
        X, y = _panel_features(df, meta, scales, codes)
        keep = X.notna().all(axis=1) & y.notna()
        fitted = XGBRegressor(random_state=42, n_estimators=params.get("n_estimators", 300),
                              max_depth=params.get("max_depth", 6), learning_rate=params.get("learning_rate", 0.1))
        fitted.fit(X[keep], y[keep])
    history = df[[s_col, d_col, target, *_exog_columns(meta)]]
    return PanelForecastModel(model_type, d_col, target, s_col, _exog_columns(meta), sp, meta["lags"],
                              meta["rolling_windows"], meta["freq"], fitted, scales, codes, history)


def _fit_forecast_model(
    df: pd.DataFrame, meta: dict, model_type: str, params: dict
) -> ForecastModel:
    if is_panel(meta):
        return _fit_panel_model(df, meta, model_type, params)
    date_col, target_col = meta["date_column"], meta["target"]
    exog_cols = _exog_columns(meta)
    dates, y = df[date_col], df[target_col]
    exog = df[exog_cols] if exog_cols else None
    seasonal_periods, lags, rolling_windows = (
        meta["seasonal_periods"],
        meta["lags"],
        meta["rolling_windows"],
    )

    if model_type == "naive_seasonal":
        fitted = (
            list(y.iloc[-seasonal_periods:])
            if seasonal_periods > 1
            else [float(y.iloc[-1])]
        )
        encoder = None
    elif model_type == "exponential_smoothing":
        fitted = _fit_ets(y, seasonal_periods, params)
        encoder = None
    else:  # xgboost
        feats = build_features(dates, y, exog, lags, rolling_windows)
        combined = pd.concat(
            [feats, y.rename("__y__").reset_index(drop=True).set_axis(feats.index)],
            axis=1,
        ).dropna()
        X, y_train = combined.drop(columns="__y__"), combined["__y__"]
        cat_cols = [c for c in exog_cols if df[c].dtype == object] if exog_cols else []
        encoder = None
        if cat_cols:
            encoder = OneHotEncoder(
                handle_unknown="ignore", sparse_output=False, drop="if_binary"
            )
            encoded = encoder.fit_transform(X[cat_cols])
            encoded_df = pd.DataFrame(
                encoded, columns=encoder.get_feature_names_out(cat_cols), index=X.index
            )
            X = pd.concat([X.drop(columns=cat_cols), encoded_df], axis=1)
        fitted = XGBRegressor(
            random_state=42,
            n_estimators=params.get("n_estimators", 200),
            max_depth=params.get("max_depth", 5),
            learning_rate=params.get("learning_rate", 0.1),
        )
        fitted.fit(X, y_train)

    return ForecastModel(
        model_type=model_type,
        date_column=date_col,
        target=target_col,
        exog_columns=exog_cols,
        seasonal_periods=seasonal_periods,
        lags=lags,
        rolling_windows=rolling_windows,
        freq=meta["freq"],
        fitted=fitted,
        encoder=encoder,
        history_dates=dates.reset_index(drop=True),
        history_target=y.reset_index(drop=True),
        history_exog=exog.reset_index(drop=True) if exog is not None else None,
    )


def _mase_scale(train_df: pd.DataFrame, meta: dict) -> float | None:
    """In-sample MAE of the seasonal naive forecast on the training fold —
    MASE's denominator. MASE < 1 = better than repeating last season."""
    if is_panel(meta):
        return None  # per series: _score_on_test
    y = train_df[meta["target"]].to_numpy(dtype=float)
    m = meta["seasonal_periods"] if len(y) > meta["seasonal_periods"] * 2 else 1
    scale = float(np.nanmean(np.abs(y[m:] - y[:-m]))) if len(y) > m else None
    return scale or None


def _rmse_mae_mape(y_true: np.ndarray, y_pred: np.ndarray, scale: float | None = None) -> dict:
    """Error in the target's units (RMSE, MAE), scale-free (MASE against the
    seasonal naive, WAPE = total absolute error / total volume — the
    business number for traffic and demand), bias (+ = over-forecast), and
    MAPE only when no actual is (near) zero: one near-zero period makes it
    explode, and WAPE says the same thing without that failure."""
    y_true, y_pred = np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)
    err = y_pred - y_true
    rmse = float(np.sqrt(np.mean(err**2)))
    mae = float(np.mean(np.abs(err)))
    volume = float(np.sum(np.abs(y_true)))
    near_zero = np.abs(y_true) <= 1e-9 + 0.01 * np.mean(np.abs(y_true))
    mape = float(np.mean(np.abs(err / y_true)) * 100) if not near_zero.any() else None
    return {
        "rmse": round(rmse, 4),
        "mae": round(mae, 4),
        "mase": round(mae / scale, 4) if scale else None,
        "wape": round(float(np.sum(np.abs(err)) / volume * 100), 4) if volume else None,
        "bias_pct": round(float(np.sum(err) / volume * 100), 4) if volume else None,
        "mape": round(mape, 4) if mape is not None else None,
    }


def _backtest(train_df: pd.DataFrame, meta: dict, model_type: str, params: dict, n_splits: int = 5) -> dict:
    """Rolling-origin backtest AT THE BUSINESS HORIZON: the last n_splits
    non-overlapping windows of `horizon` steps in the training fold, each
    forecast from an origin that has seen only what came before it, with the
    SAME recursive multi-step logic production uses — never on true
    intervening values. Returns the RMSE across origins and every origin's
    signed errors by step ahead (what the prediction interval is calibrated
    on; in each series' own scale for a multi-series run). An origin with too
    little history is skipped. Multi-series runs use at most 3 origins."""
    d_col = meta["date_column"]
    timeline = sorted(pd.to_datetime(train_df[d_col]).unique()) if is_panel(meta) else None
    n = len(timeline) if is_panel(meta) else len(train_df)
    h = int(meta.get("horizon") or max(1, n // (n_splits + 1)))
    min_needed = max(meta["seasonal_periods"] * 2, max(meta["lags"], default=1) + 5, 10)
    exog_cols = _exog_columns(meta)
    scores, residuals = [], []
    for back in range(min(n_splits, 3) if is_panel(meta) else n_splits, 0, -1):
        cutoff = n - back * h
        if cutoff < min_needed:
            continue
        try:
            if is_panel(meta):
                stamps = pd.to_datetime(train_df[d_col])
                fit_df = train_df[stamps < timeline[cutoff]]
                val = train_df[stamps.isin(timeline[cutoff:cutoff + h])]
                fm = _fit_forecast_model(fit_df, meta, model_type, params)
                errs, err_units = _panel_errors(fm, val, meta)
                residuals.extend(errs)
                scores.append(float(np.sqrt(np.mean(np.square(err_units)))))
                continue
            val = train_df.iloc[cutoff:cutoff + h]
            fm = _fit_forecast_model(train_df.iloc[:cutoff], meta, model_type, params)
            fc = fm.forecast(len(val), val[exog_cols].reset_index(drop=True) if exog_cols else None)
        except Exception:
            continue
        err = val[meta["target"]].to_numpy(dtype=float) - fc["forecast"].to_numpy()
        residuals.append(err.tolist())
        scores.append(float(np.sqrt(np.mean(err**2))))
    if not scores:
        return {"rmse_mean": None, "rmse_std": None, "origins": 0, "horizon": h, "residuals": []}
    return {"rmse_mean": round(float(np.mean(scores)), 4), "rmse_std": round(float(np.std(scores)), 4),
            "origins": len(scores), "horizon": h, "residuals": residuals}


def _panel_forecast_for(fm, frame: pd.DataFrame, meta: dict) -> pd.DataFrame:
    """The multi-series model's forecast for the periods in `frame`, joined
    to its actuals (series and date)."""
    s_col, d_col = meta["series_column"], meta["date_column"]
    steps = int(pd.to_datetime(frame[d_col]).nunique())
    exog_cols = _exog_columns(meta)
    future = frame.sort_values([s_col, d_col])[[s_col, *exog_cols]] if exog_cols else None
    fc = fm.forecast(steps, future)
    actual = frame[[s_col, d_col, meta["target"]]].assign(**{d_col: pd.to_datetime(frame[d_col])})
    fc[d_col] = pd.to_datetime(fc[d_col])
    return actual.merge(fc, on=[s_col, d_col], how="inner")


def _panel_errors(fm, frame: pd.DataFrame, meta: dict) -> tuple:
    """Per series: signed errors by step, in the series' own scale (for the
    interval), and all errors in target units (for RMSE)."""
    joined = _panel_forecast_for(fm, frame, meta)
    joined["err"] = joined[meta["target"]].astype(float) - joined["forecast"]
    scaled = [(g["err"] / fm.scales[s]).tolist()
              for s, g in joined.sort_values(meta["date_column"]).groupby(meta["series_column"])]
    return scaled, joined["err"].to_numpy()


def _overfit_gate(
    backtest_rmse: float | None, test_rmse: float
) -> tuple[float | None, bool]:
    if not backtest_rmse:
        return None, False
    gap = round((test_rmse - backtest_rmse) / backtest_rmse, 4)
    return gap, gap > OVERFIT_GAP_THRESHOLD


def interval_from_residuals(residuals: list, horizon: int, level: float = INTERVAL_LEVEL) -> dict | None:
    """Empirical prediction interval per step ahead: quantiles of the signed
    backtest errors at that step, pooled with the two steps either side (a
    handful of origins is too few alone), then forced to widen with the
    horizon. None without enough errors to estimate a band."""
    pooled_all = [e for r in residuals for e in r]
    if len(pooled_all) < 10:
        return None
    lo_q, hi_q = (1 - level) / 2, 1 - (1 - level) / 2
    lower, upper = [], []
    for step in range(horizon):
        pool = [r[j] for r in residuals for j in range(max(0, step - 2), min(len(r), step + 3))]
        pool = pool if len(pool) >= 10 else pooled_all
        lower.append(float(np.quantile(pool, lo_q)))
        upper.append(float(np.quantile(pool, hi_q)))
    return {"level": level, "lower": np.minimum.accumulate(lower).round(6).tolist(),
            "upper": np.maximum.accumulate(upper).round(6).tolist(), "origins": len(residuals)}


def _score_on_test(fm, train: pd.DataFrame, test: pd.DataFrame, meta: dict) -> tuple:
    """Forecast the held-out period exactly as predict would and score it.
    Returns (metrics, forecast frame with actuals). Multi-series: errors are
    pooled (RMSE, WAPE, bias across every series and step), MASE is the
    MEDIAN series' MASE against its own seasonal naive, and the share of
    series where the model beats that naive is reported with the worst five
    series by WAPE."""
    target = meta["target"]
    if not is_panel(meta):
        exog_cols = _exog_columns(meta)
        forecast = fm.forecast(len(test), test[exog_cols].reset_index(drop=True) if exog_cols else None)
        actual = test[target].to_numpy(dtype=float)
        metrics = _rmse_mae_mape(actual, forecast["forecast"].to_numpy(), _mase_scale(train, meta))
        forecast = forecast.assign(actual=actual)
    else:
        s_col = meta["series_column"]
        forecast = _panel_forecast_for(fm, test, meta).rename(columns={target: "actual"})
        metrics = _rmse_mae_mape(forecast["actual"].to_numpy(), forecast["forecast"].to_numpy())
        per_series = []
        for s, g in forecast.groupby(s_col):
            hist = train.loc[train[s_col] == s, target].to_numpy(dtype=float)
            m = meta["seasonal_periods"] if len(hist) > 2 * meta["seasonal_periods"] else 1
            scale = float(np.nanmean(np.abs(hist[m:] - hist[:-m]))) if len(hist) > m else np.nan
            err = np.abs(g["forecast"] - g["actual"])
            per_series.append({"series": _to_native(s), "mase": float(err.mean() / scale) if scale else np.nan,
                               "wape": float(err.sum() / max(g["actual"].abs().sum(), 1e-9) * 100)})
        table = pd.DataFrame(per_series)
        metrics["mase"] = round(float(table["mase"].median()), 4)
        metrics["series"] = int(len(table))
        metrics["series_beating_naive_share"] = round(float((table["mase"] < 1).mean()), 4)
        metrics["worst_series_by_wape"] = table.sort_values("wape", ascending=False).head(5).round(4).to_dict("records")
    if "lower" in forecast:
        inside = (forecast["actual"] >= forecast["lower"]) & (forecast["actual"] <= forecast["upper"])
        metrics["interval_level"] = fm.interval["level"]
        metrics["interval_coverage"] = round(float(inside.mean()), 4)
    return metrics, forecast


def _fit_and_score(train: pd.DataFrame, test: pd.DataFrame, meta: dict, model: str, params: dict,
                   n_splits: int = 5) -> tuple:
    """Backtest, fit on the whole training fold, attach the calibrated
    interval, forecast the held-out period exactly as predict would.
    Returns (ForecastModel, metrics, forecast frame)."""
    bt = _backtest(train, meta, model, params, n_splits=n_splits)
    fm = _fit_forecast_model(train, meta, model, params)
    fm.interval = interval_from_residuals(bt["residuals"], bt["horizon"])
    test_eval, forecast = _score_on_test(fm, train, test, meta)
    gap, overfitting_warning = _overfit_gate(bt["rmse_mean"], test_eval["rmse"])
    metrics = {
        "backtest_rmse_mean": bt["rmse_mean"],
        "backtest_rmse_std": bt["rmse_std"],
        "backtest_origins": bt["origins"],
        "horizon": bt["horizon"],
        "cv_to_test_gap": gap,
        "overfitting_warning": overfitting_warning,
        **test_eval,
    }
    return fm, metrics, forecast


def _current_metrics(meta: dict) -> dict:
    """The current model's metrics — tuned if tune_hyperparams ran."""
    return meta.get("tuned_metrics") or meta.get("baseline_metrics") or {}


# Every tool call on a run is recorded in its execution ledger (core/gates.py),
# so the readiness gates judge what ACTUALLY ran, in what order. Installed here,
# before any tool module decorates its tools.
from core import gates as _gate_engine  # noqa: E402

_gate_engine.install_ledger(mcp, _run_dir, _load_meta, _save_meta)
