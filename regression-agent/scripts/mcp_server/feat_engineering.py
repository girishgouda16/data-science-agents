"""regression agent — `feature-engineering` tools. The tools are shared with
the classification agent (core/feature_tools.py); this module binds them to
this agent's run storage and to regression's per-feature signal — the
|Spearman| correlation a feature reaches alone against the target
(categoricals target-mean encoded out-of-fold, so a value seen once cannot
predict itself)."""
import numpy as np
import pandas as pd
from core import feature_tools
from scipy import stats
from sklearn.model_selection import KFold

from .core import _load_split, _run_dir, _save_meta, mcp

SIGNAL_SAMPLE_ROWS = 200_000


def _regression_signal(train, target, meta, columns) -> dict:
    df = train if len(train) <= SIGNAL_SAMPLE_ROWS else train.sample(SIGNAL_SAMPLE_ROWS, random_state=42)
    y = df[target].to_numpy(dtype=float)
    folds = list(KFold(n_splits=5, shuffle=True, random_state=42).split(df))
    scores = {}
    for col in columns:
        x = df[col]
        if pd.api.types.is_numeric_dtype(x) and not pd.api.types.is_bool_dtype(x):
            values = x.to_numpy(dtype=float)
        else:
            keys, values = x.astype(str).to_numpy(), np.empty(len(df))
            for fit_idx, val_idx in folds:
                means = pd.Series(y[fit_idx]).groupby(keys[fit_idx]).mean()
                values[val_idx] = pd.Series(keys[val_idx]).map(means).fillna(y[fit_idx].mean()).to_numpy()
        rho = stats.spearmanr(values, y, nan_policy="omit").statistic
        scores[col] = round(abs(float(rho)), 4) if rho is not None and np.isfinite(rho) else None
    quartiles = pd.qcut(train[target].rank(method="first"), 4, labels=["q1_lowest", "q2", "q3", "q4_highest"])
    return {"scores": scores, "metric": "spearman", "suspect": 0.98, "weak": 0.05, "groups": quartiles}


feature_tools.bind(load_split=_load_split, run_dir=_run_dir, save_meta=_save_meta, signal=_regression_signal)
_tools = feature_tools.register(mcp)
propose_features = _tools["propose_features"]
apply_features = _tools["apply_features"]
apply_custom_feature = _tools["apply_custom_feature"]
aggregate_events = _tools["aggregate_events"]
apply_entity_features = _tools["apply_entity_features"]
apply_peer_features = _tools["apply_peer_features"]
profile_features = _tools["profile_features"]
_reasoned_features = feature_tools.reasoned_features
