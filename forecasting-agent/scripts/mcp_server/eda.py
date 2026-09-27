"""forecasting agent — `eda` tools. Helpers and shared imports live in core.py."""

from .core import *  # noqa: F401,F403 — shared imports, constants, run storage
from .core import _cleanup_old_runs, _load_split, _run_dir, _save_meta  # noqa: F401
from core import data_tools
from core.datasource import read_table


@mcp.tool()
def eda(path: str) -> str:
    """Shape, dtypes, missingness, and summary stats for a data file in any
    supported format. Read-only — call this on the raw file before
    prepare_dataset."""
    df = read_table(path)
    return json.dumps(
        {
            "shape": df.shape,
            "dtypes": df.dtypes.astype(str).to_dict(),
            "missing_pct": (df.isna().mean() * 100).round(2).to_dict(),
            "describe": json.loads(df.describe(include="all").to_json()),
        }
    )


def _regular_freq(dates: pd.Series) -> str | None:
    """pandas' infer_freq, or — when a few periods are missing, which makes
    it give up — the step that covers 80%+ of consecutive differences, for
    steps up to a week (hourly, daily, weekly data with gaps)."""
    freq = pd.infer_freq(dates)
    if freq:
        return freq
    diffs = dates.diff().dropna()
    if diffs.empty:
        return None
    step = diffs.mode().iloc[0]
    if (diffs == step).mean() >= 0.8 and step <= pd.Timedelta(days=7):
        return pd.tseries.frequencies.to_offset(step).freqstr
    return None


def _season_and_lags(total: pd.Series, freq: str, seasonal_periods: int, usable: int) -> tuple:
    """(seasonal_periods, basis, autocorrelation per candidate, lags,
    rolling windows) — the season with the strongest autocorrelation among
    those with 3+ full cycles of history, unless one was given."""
    base = freq.split("-")[0].lstrip("0123456789") or freq
    y = total.astype(float)
    seasonality = {int(lag): round(float(y.autocorr(lag)), 3) for lag in SEASONAL_CANDIDATES.get(base, [])
                   if len(y.dropna()) > 2 * lag}
    fitting = {lag: r for lag, r in seasonality.items() if len(y.dropna()) >= 3 * lag}
    if seasonal_periods:
        basis = "given"
    elif fitting:
        seasonal_periods, basis = max(fitting, key=fitting.get), "strongest autocorrelation"
    else:
        seasonal_periods, basis = FREQ_SEASONAL_PERIODS.get(base, 1), "frequency default"
    seasonal_periods = int(seasonal_periods)
    lags = sorted({lag for lag in FREQ_LAGS.get(base, [1, 2, 3]) + [seasonal_periods] if 1 <= lag <= usable} or {1})
    windows = sorted({w for w in FREQ_WINDOWS.get(base, [3]) + ([seasonal_periods] if seasonal_periods > 1 else [])
                      if 2 <= w <= usable} or {min(3, max(usable, 2))})
    return seasonal_periods, basis, seasonality, lags, windows


def _prepare_panel(df, path, date_column, target, series_column, test_size, freq, horizon, aggregate,
                   seasonal_periods) -> str:
    """prepare_dataset for every series of a long table in one run."""
    stamps = pd.to_datetime(df[date_column], errors="coerce")
    if stamps.isna().any():
        return json.dumps({"error": f"date_column '{date_column}' has {int(stamps.isna().sum())} value(s) that "
                                    "don't parse as dates"})
    df = df.assign(**{date_column: stamps})
    dropped_text = [c for c in df.columns if c not in (date_column, target, series_column)
                    and not pd.api.types.is_numeric_dtype(df[c])]
    df = df.drop(columns=dropped_text)
    keys = [series_column, date_column]
    duplicates = int(df.duplicated(keys).sum())
    if duplicates and not aggregate:
        return json.dumps({"error": f"{duplicates} rows repeat a {series_column} and {date_column} — pass aggregate "
                                    "(sum for volumes, mean for levels) to combine them within each series"})
    if duplicates:
        rules = {c: (aggregate if c == target else "mean") for c in df.columns if c not in keys}
        df = df.groupby(keys, as_index=False).agg(rules)
    timeline = pd.Series(sorted(df[date_column].unique()))
    inferred_freq = freq or _regular_freq(timeline)
    if not inferred_freq:
        return json.dumps({"error": "could not infer a regular frequency — pass freq explicitly (e.g. 'h', 'D')"})
    grid = pd.date_range(timeline.iloc[0], timeline.iloc[-1], freq=inferred_freq)
    if not timeline.isin(grid).all():
        return json.dumps({"error": f"timestamps do not sit on a regular {inferred_freq} grid — pass freq"})
    n_test = int(horizon) if horizon else max(1, round(len(grid) * test_size))
    test_start = grid[-n_test]
    frames, gaps, stale, short = [], 0, [], []
    for s, g in df.groupby(series_column, sort=True):
        g = g.set_index(date_column).sort_index()
        if g.index[-1] < test_start:
            stale.append(s)
            continue
        own = grid[grid >= g.index[0]]
        gaps += len(own) - len(g)
        frames.append(g.reindex(own).rename_axis(date_column).reset_index().assign(**{series_column: s}))
    if not frames:
        return json.dumps({"error": "no series reports into the held-out period"})
    panel = pd.concat(frames, ignore_index=True)
    total = panel.groupby(date_column)[target].sum(min_count=1)
    sp, basis, seasonality, lags, windows = _season_and_lags(total, inferred_freq, seasonal_periods,
                                                            (len(grid) - n_test) // 3)
    min_needed = max(sp * 2, max(lags) + 5, 10) + n_test
    lengths = panel.groupby(series_column).size()
    short = sorted(lengths[lengths < min_needed].index.tolist(), key=str)
    panel = panel[~panel[series_column].isin(short)]
    if panel.empty:
        return json.dumps({"error": f"every series is shorter than the {min_needed} periods a backtest at this "
                                    "horizon needs"})
    panel = panel.sort_values([series_column, date_column]).reset_index(drop=True)
    in_test = panel[date_column] >= test_start
    train, test = panel[~in_test].reset_index(drop=True), panel[in_test].reset_index(drop=True)

    run_id = str(uuid.uuid4())
    run_dir = _run_dir(run_id)
    run_dir.mkdir(parents=True)
    train.to_csv(run_dir / "train.csv", index=False)
    test.to_csv(run_dir / "test.csv", index=False)
    meta = {
        "date_column": date_column, "target": target, "source_path": path, "all_columns": list(panel.columns),
        "dropped_columns": [], "imputation": {}, "freq": inferred_freq, "seasonal_periods": sp,
        "seasonality": seasonality, "seasonal_periods_basis": basis, "horizon": n_test,
        "aggregate": aggregate or None, "duplicates_aggregated": duplicates, "panel": True,
        "series_column": series_column, "n_series": int(panel[series_column].nunique()),
        "series": f"each of {panel[series_column].nunique()} {series_column} values",
        "series_left_out": {"stopped_before_held_out": [str(s) for s in stale][:50], "too_short": [str(s) for s in short][:50]},
        "gaps_inserted": gaps, "lags": lags, "rolling_windows": windows, "model": None, "best_params": None,
        "baseline_metrics": None, "tuned_metrics": None,
    }
    lineage = data_tools.lineage_for(path)
    if lineage:
        meta["data_source"] = lineage
    _save_meta(run_id, meta)
    result = {
        "run_id": run_id, "series": meta["n_series"], "train_shape": train.shape, "test_shape": test.shape,
        "freq": inferred_freq, "horizon": n_test, "seasonal_periods": sp, "seasonal_periods_basis": basis,
        "seasonality_autocorrelation": seasonality, "lags": lags,
        "date_range": [str(grid[0]), str(grid[-1])],
    }
    if stale or short:
        result["series_left_out"] = {"stopped_before_held_out": len(stale), "too_short": len(short)}
    if duplicates:
        result["duplicates_aggregated"] = f"{duplicates} rows combined by {aggregate} within each series"
    if dropped_text:
        result["text_columns_left_out"] = dropped_text
    if gaps:
        result["gaps_inserted"] = (f"{gaps} missing period(s) inserted across series — the target is empty there; "
                                   "fill it with apply_imputation (per series) before training")
    return json.dumps(result, default=str)


@mcp.tool()
def prepare_dataset(
    path: str,
    date_column: str,
    target: str,
    test_size: float = 0.2,
    freq: str = "",
    horizon: int = 0,
    aggregate: str = "",
    series_column: str = "",
    series_value: str = "",
    seasonal_periods: int = 0,
    each_series: bool = False,
) -> str:
    """Sorts by date_column and splits CHRONOLOGICALLY — the held-out period
    is the END of the series, never a random sample.
    horizon: how many steps ahead the business needs (28 days, 12 weeks, 168
      hours). The held-out period is exactly that long and every backtest
      origin forecasts that far — a model judged one step ahead says little
      about week four. 0 = test_size of the rows.
    aggregate: several rows per timestamp (per cell, per product, per
      channel) are summed / averaged ("sum"|"mean"|"max"|"min"|"last") into
      one series — the user's decision: summing traffic and averaging a
      price are both right for their column. Numeric exogenous columns are
      averaged, others take the last value.
    series_column + series_value: forecast ONE series of a long table (one
      cell, one region); series_column + aggregate: the total across series;
      series_column + each_series=true: EVERY series, one run — per-series
      naive and Holt-Winters, or one global xgboost model learning the shared
      shape from all of them (the right tool for hundreds of cells). Series
      that stopped reporting before the held-out period, or are too short to
      backtest, are left out and listed; aggregate then combines duplicate
      rows WITHIN a series and period.
    freq: pandas offset alias ("h", "D", "W", "MS"...); inferred when empty,
      also when a few periods are missing. Missing periods are INSERTED so
      the grid is regular (lags must mean "one day ago", not "one row ago");
      the target is empty there until apply_imputation fills it.
    seasonal_periods: override the default season (hourly 24, daily 7...);
      the result reports the autocorrelation at each candidate season (hourly
      data: 24 and 168) so the choice rests on evidence.
    Returns a run_id; every other tool takes run_id, not a file path."""
    _cleanup_old_runs()
    df = read_table(path)
    if df.empty:
        return json.dumps({"error": f"'{path}' has no rows"})
    for col, what in ((target, "target"), (date_column, "date")):
        if col not in df.columns:
            return json.dumps({"error": f"{what} column '{col}' not found. Columns: {list(df.columns)}"})
    if not pd.api.types.is_numeric_dtype(df[target]):
        return json.dumps({"error": f"target column '{target}' is not numeric — forecasting needs a continuous target"})
    if aggregate and aggregate not in ("sum", "mean", "max", "min", "last"):
        return json.dumps({"error": "aggregate must be sum, mean, max, min or last"})

    series = None
    if series_column:
        if series_column not in df.columns:
            return json.dumps({"error": f"series column '{series_column}' not found"})
        if series_value:
            df = df[df[series_column].astype(str) == str(series_value)]
            if df.empty:
                return json.dumps({"error": f"no rows with {series_column} = {series_value}"})
            series = f"{series_column} = {series_value}"
        elif each_series:
            return _prepare_panel(df, path, date_column, target, series_column, test_size, freq, horizon,
                                  aggregate, seasonal_periods)
        elif aggregate:
            series = f"{aggregate} across {df[series_column].nunique()} {series_column} values"
        else:
            return json.dumps({"error": f"{series_column} holds {df[series_column].nunique()} series — pass "
                                        "series_value for one of them, aggregate for their total, or "
                                        "each_series=true to forecast every one of them"})
        df = df.drop(columns=[series_column])

    parsed_dates = pd.to_datetime(df[date_column], errors="coerce")
    if parsed_dates.isna().any():
        return json.dumps({"error": f"date_column '{date_column}' has {int(parsed_dates.isna().sum())} value(s) "
                                    "that don't parse as dates"})
    df = df.assign(**{date_column: parsed_dates}).sort_values(date_column).reset_index(drop=True)

    duplicates = int(df[date_column].duplicated().sum())
    if duplicates and not aggregate:
        return json.dumps({"error": f"date_column '{date_column}' has {duplicates} duplicate timestamps — several "
                                    "series or events per period. Ask the user how to combine them and pass "
                                    "aggregate (sum for volumes like traffic or calls, mean for levels like price), "
                                    "or series_column + series_value for one series"})
    if duplicates:
        rules = {c: (aggregate if c == target else "mean" if pd.api.types.is_numeric_dtype(df[c]) else "last")
                 for c in df.columns if c != date_column}
        df = df.groupby(date_column, as_index=False).agg(rules)

    inferred_freq = freq or _regular_freq(df[date_column])
    if not inferred_freq:
        return json.dumps({"error": "could not infer a regular frequency from the timestamps — pass freq "
                                    "explicitly (e.g. 'h', 'D', 'W', 'MS')"})
    grid = pd.date_range(df[date_column].iloc[0], df[date_column].iloc[-1], freq=inferred_freq)
    gaps_inserted = 0
    if df[date_column].isin(grid).all() and len(grid) > len(df):
        gaps_inserted = len(grid) - len(df)
        df = df.set_index(date_column).reindex(grid).rename_axis(date_column).reset_index()

    base = inferred_freq.split("-")[0].lstrip("0123456789") or inferred_freq
    y = df[target].astype(float)
    seasonality = {int(lag): round(float(y.autocorr(lag)), 3) for lag in SEASONAL_CANDIDATES.get(base, [])
                   if len(y.dropna()) > 2 * lag}
    # The season with the strongest autocorrelation, among those with 3+ full
    # cycles of history: hourly traffic usually repeats WEEKLY (weekends differ),
    # and a daily-season baseline is then a straw man any model beats.
    fitting = {lag: r for lag, r in seasonality.items() if len(y.dropna()) >= 3 * lag}
    if seasonal_periods:
        basis = "given"
    elif fitting:
        seasonal_periods, basis = max(fitting, key=fitting.get), "strongest autocorrelation"
    else:
        seasonal_periods, basis = FREQ_SEASONAL_PERIODS.get(base, 1), "frequency default"
    seasonal_periods = int(seasonal_periods)

    n_test = int(horizon) if horizon else max(1, round(len(df) * test_size))
    if not horizon and seasonal_periods > 1:
        n_test = max(n_test, min(seasonal_periods, len(df) // 3))
    if len(df) - n_test < max(seasonal_periods * 2, 10):
        return json.dumps({"error": f"only {len(df)} period(s) — need at least {max(seasonal_periods * 2, 10) + n_test} "
                                    f"for a seasonal_periods={seasonal_periods} series with a {n_test}-step held-out "
                                    "period"})
    train, test = df.iloc[:-n_test].reset_index(drop=True), df.iloc[-n_test:].reset_index(drop=True)
    usable = len(train) // 3
    lags = sorted({lag for lag in FREQ_LAGS.get(base, [1, 2, 3]) + [seasonal_periods] if 1 <= lag <= usable} or {1})
    rolling_windows = sorted({w for w in FREQ_WINDOWS.get(base, [3]) + ([seasonal_periods] if seasonal_periods > 1 else [])
                              if 2 <= w <= usable} or {min(3, max(usable, 2))})

    run_id = str(uuid.uuid4())
    run_dir = _run_dir(run_id)
    run_dir.mkdir(parents=True)
    train.to_csv(run_dir / "train.csv", index=False)
    test.to_csv(run_dir / "test.csv", index=False)
    meta = {
        "date_column": date_column,
        "target": target,
        "source_path": path,
        "all_columns": list(df.columns),
        "dropped_columns": [],
        "imputation": {},
        "freq": inferred_freq,
        "seasonal_periods": seasonal_periods,
        "seasonality": seasonality,
        "seasonal_periods_basis": basis,
        "horizon": n_test,
        "aggregate": aggregate or None,
        "duplicates_aggregated": duplicates,
        "series": series,
        "gaps_inserted": gaps_inserted,
        "lags": lags,
        "rolling_windows": rolling_windows,
        "model": None,
        "best_params": None,
        "baseline_metrics": None,
        "tuned_metrics": None,
    }
    lineage = data_tools.lineage_for(path)  # load_dataset's source record
    if lineage:
        meta["data_source"] = lineage
    _save_meta(run_id, meta)
    result = {
        "run_id": run_id,
        "train_shape": train.shape,
        "test_shape": test.shape,
        "freq": inferred_freq,
        "horizon": n_test,
        "seasonal_periods": seasonal_periods,
        "seasonal_periods_basis": basis,
        "seasonality_autocorrelation": seasonality,
        "lags": lags,
        "date_range": [str(df[date_column].min()), str(df[date_column].max())],
        "target_summary": train[target].describe().round(4).to_dict(),
    }
    if duplicates:
        result["duplicates_aggregated"] = f"{duplicates} rows combined by {aggregate}"
    if series:
        result["series"] = series
    if gaps_inserted:
        result["gaps_inserted"] = (f"{gaps_inserted} missing period(s) inserted — the target is empty there; "
                                   "fill it with apply_imputation before training")
    return json.dumps(result, default=str)


@mcp.tool()
def detect_outliers(run_id: str) -> str:
    """Z-score outliers (|z| > 3) in the target, training fold only. A
    time-series target outlier (e.g. a demand spike) is frequently the most
    operationally important row, not noise — see the `forecasting` skill's
    domain notes before assuming it should be smoothed away."""
    train, _, meta = _load_split(run_id)
    y = train[meta["target"]].astype(float)
    if meta.get("panel"):  # against each series' own level: a busy cell is not an outlier
        g = y.groupby(train[meta["series_column"]])
        z = ((y - g.transform("mean")) / g.transform("std").replace(0, np.nan)).to_numpy()
    else:
        z = stats.zscore(y, nan_policy="omit")
    return json.dumps({"target_outlier_count": int(np.nansum(np.abs(z) > 3))})


_intake = data_tools.register(mcp)
list_data_sources = _intake["list_data_sources"]
load_dataset = _intake["load_dataset"]
