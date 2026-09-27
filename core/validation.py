"""Validation that respects how a run was split — shared by every agent that
trains a supervised model, so "cross-validated" means the same thing in
classification and regression.

A run is split one of three ways (prepare_dataset): forward in time on its
time_column, by entity on its group_column, or randomly. Every CV score,
every search, every threshold or interval chosen on held-out predictions
must use folds that obey the same rule, or the score is computed on folds
the split itself would have forbidden: the future predicting the past, or an
entity recognised from its own rows.
"""
import numpy as np
import pandas as pd

CV_SAMPLE_MAX_ROWS = 200_000
VALIDATION_FOLDS = 5


def split_keys(meta: dict) -> list[str]:
    """The run's entity id and time column — what the split and every
    cross-validation respect, never what the model learns from. They stay in
    train/test for exactly that reason, and each agent's pipeline drops them
    structurally, so no cleaning step has to (and none should)."""
    return [c for c in (meta.get("group_column"), meta.get("time_column")) if c]


def _parse_dates(series: pd.Series) -> pd.Series:
    parsed = pd.to_datetime(series, errors="coerce")
    if parsed.isna().sum() > series.isna().sum():
        parsed = pd.to_datetime(series, errors="coerce", format="mixed")
    return parsed


def cv_sample(X: pd.DataFrame, y: pd.Series, meta: dict | None = None, stratify: bool = True,
              max_rows: int = CV_SAMPLE_MAX_ROWS) -> tuple[pd.DataFrame, pd.Series, int]:
    """Sample the training fold down to `max_rows` for cross_val_score / a
    hyperparameter search — NEVER for the final fit. Keeps the split's
    structure: the NEWEST rows in time order on a temporal run, whole
    entities on a grouped run, stratified (classification) or uniform
    otherwise. Returns the row count actually used."""
    if len(X) <= max_rows:
        return X, y, len(X)
    meta = meta or {}
    if meta.get("time_column"):
        return X.tail(max_rows), y.tail(max_rows), max_rows
    group = meta.get("group_column")
    if group and group in X.columns:
        sizes = X[group].value_counts()
        order = sizes.sample(frac=1, random_state=42)
        keep = order.index[order.cumsum().to_numpy() <= max_rows]
        mask = X[group].isin(keep)
        return X[mask], y[mask], int(mask.sum())
    from sklearn.model_selection import train_test_split

    X_s, _, y_s, _ = train_test_split(X, y, train_size=max_rows, stratify=y if stratify else None, random_state=42)
    return X_s, y_s, len(X_s)


def validation_folds(X: pd.DataFrame, y: pd.Series, meta: dict, n_splits: int = VALIDATION_FOLDS,
                     seed: int = 42, stratify: bool = True) -> tuple[list, str]:
    """(fit_idx, val_idx) positional pairs over the training fold that respect
    the run's split:

      temporal  expanding window: fold k fits on the oldest k blocks and
                validates on the next, ordered by the time column — never the
                past predicted from the future
      grouped   (Stratified)GroupKFold on the entity column — no entity on
                both sides
      random    (Stratified)KFold

    With `stratify` (classification) a temporal fold is kept only when its
    validation block holds two or more classes, all seen in its fitting rows."""
    from sklearn.model_selection import GroupKFold, KFold, StratifiedGroupKFold, StratifiedKFold

    n, note = len(X), ""
    time_col, group = meta.get("time_column"), meta.get("group_column")
    if time_col:
        if time_col in X.columns:
            stamp = X[time_col] if pd.api.types.is_numeric_dtype(X[time_col]) else _parse_dates(X[time_col])
            order = np.argsort(stamp.to_numpy(), kind="stable")
        else:
            order = np.arange(n)  # train.csv is written in time order
        edges = np.linspace(0, n, n_splits + 2, dtype=int)
        folds = []
        for k in range(1, n_splits + 1):
            fit, val = order[: edges[k]], order[edges[k]: edges[k + 1]]
            if not len(fit) or not len(val):
                continue
            if stratify:
                seen, held = set(y.iloc[fit]), set(y.iloc[val])
                if len(held) < 2 or not held <= seen:
                    continue
            folds.append((fit, val))
        if folds:
            return folds, f"{len(folds)}-fold out-of-fold, forward-chained (expanding window on `{time_col}`)"
        note = " (no forward fold had every class on both sides — fell back)"
    if group:
        if group in X.columns:
            groups = X[group].astype(str)
            if stratify:
                cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
                splits = cv.split(X, y, groups=groups)
            else:
                splits = GroupKFold(n_splits=n_splits).split(X, y, groups=groups)
            return list(splits), f"{n_splits}-fold out-of-fold, grouped by `{group}`{note}"
        note += f" (grouped CV unavailable: `{group}` is no longer in the training fold — scores may be optimistic)"
    cv = (StratifiedKFold if stratify else KFold)(n_splits=n_splits, shuffle=True, random_state=seed)
    kind = "stratified" if stratify else "shuffled"
    return list(cv.split(X, y)), f"{n_splits}-fold out-of-fold, {kind}{note}"
