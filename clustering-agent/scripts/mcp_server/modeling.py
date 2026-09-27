"""clustering agent — `modeling` tools. Helpers and shared imports live in core.py."""

from .core import *  # noqa: F401,F403 — shared imports, constants, run storage
from .diagnostics import compute_readiness  # noqa: F401
from .core import (
    _algorithm_params,
    _assign_nearest_centroids,
    _build_pipeline,
    _build_preprocessor,
    _current_metrics,
    _label_counts,
    _load_meta,
    _load_split,
    _make_clusterer,
    _nearest_centroids,
    _round_or_none,
    _run_dir,
    _safe_cluster_metrics,
    _save_meta,
    _to_native,
    _validate_feature_matrix,
)  # noqa: F401
from core import runtime, toolguard
from core.datasource import read_table
from core.validation import _parse_dates
from sklearn.tree import DecisionTreeClassifier, export_text


@mcp.tool()
def propose_k(run_id: str, k_range: str = "2,3,4,5,6,7,8", algorithm: str = "kmeans") -> str:
    """Try one algorithm (kmeans default; kmedoids for mixed numeric +
    categorical data; minibatch_kmeans, gmm, hierarchical) across a k range
    on the fit fold and report, per k:
    silhouette (sampled), Davies-Bouldin, inertia (kmeans' elbow curve), and
    STABILITY — the adjusted Rand index between the full fit and refits on
    80% subsamples. A k whose segments do not come back when the data is
    resampled is an artefact of this sample, however good its silhouette.
    recommended_k is the best-silhouette k among the stable ones
    (stability >= STABLE_ARI); if none is stable, the best silhouette, with
    a warning. Read-only — no model is saved."""
    fit, _, meta = _load_split(run_id)
    ks = [int(k.strip()) for k in k_range.split(",") if k.strip()]
    if not ks:
        return json.dumps({"error": "k_range must contain at least one integer"})
    if algorithm not in ALGORITHMS - DENSITY_BASED:
        return json.dumps({"error": f"propose_k compares k for {sorted(ALGORITHMS - DENSITY_BASED)}; "
                                    "density-based algorithms find k themselves"})
    if algorithm == "hierarchical" and len(fit) > HIERARCHICAL_MAX_ROWS:
        return json.dumps({"error": f"hierarchical needs memory quadratic in rows — {len(fit)} fit rows is "
                                    f"above {HIERARCHICAL_MAX_ROWS}"})
    try:
        X_fit = _build_preprocessor(meta, fit, gower=algorithm in GOWER).fit_transform(fit)
        _validate_feature_matrix(X_fit)
    except Exception as exc:
        return json.dumps({"error": str(exc)})

    evaluations = []
    for k in ks:
        if k < 2 or k >= len(fit):
            evaluations.append({"k": k, "error": f"k must be between 2 and {len(fit) - 1}"})
            continue
        model = _make_clusterer(algorithm, n_clusters=k).fit(X_fit)
        labels = fit_labels(model, X_fit)
        metrics = _safe_cluster_metrics(X_fit, labels, metric=distance_metric(algorithm))
        metrics.update({"k": k, "inertia": _round_or_none(getattr(model, "inertia_", None)),
                        **bootstrap_stability(X_fit, algorithm, {"n_clusters": k}, labels)})
        evaluations.append(metrics)

    scored = [row for row in evaluations if row.get("silhouette_score") is not None]
    by_quality = sorted(scored, key=lambda row: (-row["silhouette_score"], row["davies_bouldin_score"]))
    stable = [row for row in by_quality if (row.get("stability_ari_mean") or 0) >= STABLE_ARI]
    recommended = (stable or by_quality or [{}])[0].get("k")
    result = {
        "run_id": run_id,
        "algorithm": algorithm,
        "evaluations": evaluations,
        "elbow_curve": [{"k": row["k"], "inertia": row.get("inertia")} for row in evaluations if "inertia" in row],
        "recommended_k": recommended,
        "ranked_recommendations": by_quality,
        "stable_ks": [row["k"] for row in stable],
    }
    if by_quality and not stable:
        result["warning"] = (f"no k in {ks} is stable (stability ARI >= {STABLE_ARI}) — the data may have no "
                             "reproducible segments at this granularity; say so rather than presenting one")
    elif recommended is not None and recommended == max(ks):
        result["warning"] = (f"recommended_k={recommended} is at the top edge of the searched k_range {ks}; "
                             "widen k_range to confirm this is a true optimum, not a search-boundary artifact.")
    return json.dumps(result)


@mcp.tool()
def train_model(
    run_id: str,
    algorithm: str = "kmeans",
    n_clusters: int = 4,
    eps: float = 0.5,
    min_samples: int = 5,
    linkage: str = "ward",
    min_cluster_size: int = 15,
    covariance_type: str = "full",
) -> str:
    """Fits one pipeline (drop profile/dropped columns -> impute -> log the
    skewed numerics -> standardise numerics / one-hot categoricals ->
    clusterer) on the fit fold, saves it as pipeline.pkl, and reports:
      fit          separation (silhouette sampled, noise excluded), sizes
      stability    ARI of refits on 80% subsamples vs the full fit — the
                   check that the segments are real; overfitting_warning is
                   set when it falls below STABLE_ARI
      holdout_ari  the model's assignment of holdout rows vs a clustering
                   fitted on the holdout itself (not for density-based)
    algorithm: kmeans (default; spherical, same-size-ish segments),
    minibatch_kmeans (the same at millions of rows), kmedoids (Gower
    distance: mixed numeric + categorical data, sampled CLARA-style so it
    scales; medoids are real rows), gmm (elliptical,
    overlapping segments, soft membership; n_clusters + covariance_type),
    hierarchical (up to 20k rows), hdbscan (density; finds k itself, labels
    noise; min_cluster_size), dbscan (density with a fixed eps)."""
    if algorithm not in ALGORITHMS:
        return json.dumps({"error": f"unknown algorithm '{algorithm}', use one of {sorted(ALGORITHMS)}"})
    if algorithm not in DENSITY_BASED and n_clusters < 2:
        return json.dumps({"error": "n_clusters must be >= 2 for kmeans/minibatch_kmeans/gmm/hierarchical"})

    fit, holdout, meta = _load_split(run_id)
    if algorithm == "hierarchical" and len(fit) > HIERARCHICAL_MAX_ROWS:
        return json.dumps({"error": f"hierarchical clustering needs memory quadratic in rows — {len(fit)} fit "
                                    f"rows is above {HIERARCHICAL_MAX_ROWS}; use kmeans / minibatch_kmeans / gmm"})
    params = _algorithm_params(algorithm, n_clusters=n_clusters, eps=eps, min_samples=min_samples,
                               linkage=linkage, min_cluster_size=min_cluster_size, covariance_type=covariance_type)
    try:
        pipeline = _build_pipeline(meta, algorithm, fit, **params)
        pipeline.fit(fit)
        X_fit = pipeline.named_steps["preprocess"].transform(fit)
        X_holdout = pipeline.named_steps["preprocess"].transform(holdout)
        _validate_feature_matrix(X_fit)
        labels = fit_labels(pipeline.named_steps["clusterer"], X_fit)
    except Exception as exc:
        return json.dumps({"error": str(exc)})

    fit_metrics = _safe_cluster_metrics(X_fit, labels, metric=distance_metric(algorithm))
    stability = bootstrap_stability(X_fit, algorithm, params, labels)
    unstable = stability.get("stability_ari_mean") is not None and stability["stability_ari_mean"] < STABLE_ARI
    centroids = {k: v for k, v in _nearest_centroids(X_fit, labels).items() if k != "-1"}  # noise has no centre

    holdout_assignment_counts, holdout_ari = None, None
    if algorithm not in DENSITY_BASED:
        assigned = (pipeline.named_steps["clusterer"].predict(X_holdout) if algorithm in PREDICTS
                    else _assign_nearest_centroids(X_holdout, centroids))
        holdout_assignment_counts = _label_counts(assigned)
        if len(holdout) >= 10:
            refit = _make_clusterer(algorithm, **params).fit(X_holdout)
            holdout_ari = round(float(adjusted_rand_score(np.asarray(assigned).astype(str),
                                                          fit_labels(refit, X_holdout).astype(str))), 4)

    sizes = np.asarray([v for k, v in fit_metrics["cluster_size_distribution"].items() if k != "-1"])
    size_warnings = []
    if len(sizes):
        if (sizes / len(fit) < 0.01).any():
            size_warnings.append("a segment holds under 1% of rows — too small to act on; merge or explain it")
        if sizes.max() / len(fit) > 0.7:
            size_warnings.append("one segment holds over 70% of rows — the split may just peel off outliers")

    model_warnings = []
    distance_cats = [c for c in fit.columns
                     if c not in set(meta.get("dropped_columns") or []) | set(meta.get("profile_columns") or [])
                     and not pd.api.types.is_numeric_dtype(fit[c])]
    if algorithm == "gmm" and distance_cats:
        model_warnings.append(
            f"gmm treats the one-hot dummies of {distance_cats} as continuous: inside a component their variance "
            "collapses and the likelihood rewards isolating a rare category as its own segment — use kmeans for "
            "mixed data, or keep gmm to numeric features (move these to set_profile_columns)")
    np.save(_run_dir(run_id) / "fit_labels.npy", labels)
    joblib.dump(pipeline, str(_run_dir(run_id) / "pipeline.pkl"))
    training_metrics = {
        "fit": fit_metrics,
        "stability": stability,
        "holdout_ari": holdout_ari,
        "overfitting_warning": bool(unstable),
        "holdout_assignment_counts": holdout_assignment_counts,
        "log_scaled_columns": log_scaled_columns(pipeline),
        "profile_columns_excluded": meta.get("profile_columns") or [],
        "size_warnings": size_warnings,
        "model_warnings": model_warnings,
    }
    meta["algorithm"] = algorithm
    meta["algorithm_params"] = params
    meta["training_metrics"] = training_metrics
    meta["cluster_centroids"] = centroids
    meta.pop("explain", None)  # the previous model's profiles describe segments that no longer exist
    _save_meta(run_id, meta)

    save_training_run(
        run_id,
        target_column="__unsupervised__",
        best_model=algorithm,
        metric="silhouette",
        best_score=fit_metrics["silhouette_score"] if fit_metrics["silhouette_score"] is not None else -1.0,
        test_auc=None,
        cv_to_test_gap=None,
        overfitting_warning=bool(unstable),
        all_model_scores=training_metrics,
        best_params=params,
        label_classes=sorted(centroids),
        artifact_path=str(_run_dir(run_id) / "pipeline.pkl"),
    )
    return json.dumps({"run_id": run_id, "algorithm": algorithm, "params": params, **training_metrics},
                      default=_to_native)


@mcp.tool()
def explain_model(run_id: str) -> str:
    """What the segments ARE, on the fit fold, in original units:
      profiles        size, share, numeric medians, most common categories
      distinguishing  per segment, the features that set it apart — numeric
                      (segment mean - overall mean) / overall std, and
                      categories over-represented (share in segment / overall
                      share); name segments from these, not from the means
      outcomes        every profile column (churn, ARPU, a protected
                      attribute) per segment, with eta^2 — the share of its
                      variance the segmentation explains. Segments that do
                      not differ on the outcome the business cares about are
                      not useful, however well separated
      rules           a depth-3 decision tree reproducing the segments from
                      the features, and how accurately — high accuracy means
                      the segments can be described (and operationalised) as
                      a few simple rules"""
    pipeline_path = _run_dir(run_id) / "pipeline.pkl"
    labels_path = _run_dir(run_id) / "fit_labels.npy"
    if not pipeline_path.exists():
        return json.dumps({"error": "no fitted model for this run yet — call train_model first"})
    fit, _, meta = _load_split(run_id)
    if labels_path.exists():
        labels = np.load(labels_path, allow_pickle=False)
    else:  # runs trained before labels were stored
        labels = getattr(joblib.load(pipeline_path).named_steps["clusterer"], "labels_", None)
        if labels is None:
            return json.dumps({"error": "current clusterer does not expose fitted labels — retrain"})
    labels = np.asarray(labels)
    excluded = set(meta.get("dropped_columns") or [])
    outcome_cols = [c for c in (meta.get("profile_columns") or []) if c in fit.columns]
    feature_cols = [c for c in fit.columns if c not in excluded and c not in outcome_cols]
    numeric = [c for c in feature_cols if pd.api.types.is_numeric_dtype(fit[c])]
    categorical = [c for c in feature_cols if c not in numeric]
    overall_mean, overall_std = fit[numeric].mean(), fit[numeric].std().replace(0, np.nan)

    profiles = []
    for label in sorted(np.unique(labels).tolist()):
        group = fit[labels == label]
        distinct = []
        for col in numeric:
            z = (group[col].mean() - overall_mean[col]) / overall_std[col]
            if np.isfinite(z):
                distinct.append({"feature": col, "effect": f"{z:+.2f} sd", "_score": abs(z)})
        for col in categorical:
            share_in, share_all = group[col].value_counts(normalize=True), fit[col].value_counts(normalize=True)
            for value, share in share_in.head(3).items():
                lift = share / share_all.get(value, np.nan)
                if np.isfinite(lift) and lift >= 1.5:
                    distinct.append({"feature": f"{col}={value}", "effect": f"x{lift:.1f} over-represented",
                                     "_score": float(np.log(lift))})
        distinct.sort(key=lambda d: -d["_score"])
        profiles.append({
            "cluster": _to_native(label),
            "noise": bool(label == -1),
            "size": int(len(group)),
            "share": round(len(group) / len(fit), 4),
            "distinguishing": [{k: v for k, v in d.items() if k != "_score"} for d in distinct[:5]],
            "numeric_medians": {c: _round_or_none(group[c].median()) for c in numeric},
            "categorical_modes": {c: (_to_native(group[c].mode(dropna=True).iloc[0])
                                      if group[c].notna().any() else None) for c in categorical},
        })

    outcomes = {}
    for col in outcome_cols:
        y = fit[col]
        if pd.api.types.is_numeric_dtype(y) and y.notna().any():
            by, counts = y.groupby(labels).mean(), y.groupby(labels).count()
            total = ((y - y.mean()) ** 2).sum()
            between = (counts * (by - y.mean()) ** 2).sum()  # eta^2 = between-segment / total sum of squares
            outcomes[col] = {"mean_by_cluster": {str(k): _round_or_none(v) for k, v in by.items()},
                             "eta_squared": _round_or_none(between / total) if total else None}
        else:
            outcomes[col] = {"top_by_cluster": {str(g): _to_native(y[labels == g].mode(dropna=True).iloc[0])
                                                if y[labels == g].notna().any() else None
                                                for g in np.unique(labels)}}

    rules = None
    kept = labels != -1
    if feature_cols and kept.sum() >= 10 and len(np.unique(labels[kept])) >= 2:
        X_rules = pd.get_dummies(fit.loc[kept, feature_cols], dummy_na=False).fillna(0)
        tree = DecisionTreeClassifier(max_depth=3, random_state=42).fit(X_rules, labels[kept])
        rules = {"accuracy": round(float(tree.score(X_rules, labels[kept])), 4),
                 "text": export_text(tree, feature_names=list(X_rules.columns), max_depth=3)[:2500]}

    result = {"run_id": run_id, "algorithm": meta.get("algorithm"), "cluster_profiles": profiles,
              "cluster_sizes": _label_counts(labels), "outcomes": outcomes, "rules": rules}
    meta["explain"] = {"profiled": True, "outcomes": outcomes,
                       "rules_accuracy": rules and rules["accuracy"]}
    _save_meta(run_id, meta)
    return json.dumps(result, default=_to_native)


@mcp.tool()
def export_model(run_id: str, out_path: str = "", force: bool = False) -> str:
    """Exports the run's current fitted pipeline as one bundle. Refuses to
    export when fit-vs-holdout silhouette gap is large unless force=True."""
    pipeline_path = _run_dir(run_id) / "pipeline.pkl"
    if not pipeline_path.exists():
        return json.dumps(
            {"error": "no fitted model for this run yet — call train_model first"}
        )

    meta = _load_meta(run_id)
    current_metrics = _current_metrics(meta)
    if current_metrics.get("overfitting_warning") and not force:
        return json.dumps(
            {
                "error": "overfitting_warning is set for this run's current model — the segments are not "
                f"stable (stability ARI {(current_metrics.get('stability') or {}).get('stability_ari_mean')} < "
                f"{STABLE_ARI}): refits on resampled data do not recover them. Ask the user before exporting; "
                "retry with force=true if they confirm.",
            }
        )
    # A run whose own evidence found a problem does not ship (core/gates.py).
    readiness = compute_readiness(run_id)
    if readiness["overall_status"] == "blocked" and not force:
        return json.dumps(
            {
                "error": "readiness gate BLOCKED this run — exporting it would ship a model with known defects.",
                "failed_gates": {
                    g: readiness["checks"][g]["evidence"]
                    for g in readiness["failed_gates"]
                },
                "remedy": "fix the failed gate(s) and re-run, or retry with force=true if the user explicitly accepts it.",
            }
        )
    if not out_path:
        (RUNS_DIR.parent / "artifacts").mkdir(parents=True, exist_ok=True)
        suggested = str(
            RUNS_DIR.parent / "artifacts" / f"{meta['algorithm']}_{run_id}.pkl"
        )  # data/artifacts: where serving looks
        return json.dumps(
            {
                "prompt": "Where should the .pkl be saved?",
                "suggested_out_path": suggested,
            }
        )

    pipeline = joblib.load(pipeline_path)
    joblib.dump(
        {
            "pipeline": pipeline,
            "algorithm": meta["algorithm"],
            "n_clusters": (meta.get("algorithm_params") or {}).get("n_clusters"),
            "cluster_centroids": meta.get("cluster_centroids"),
            "algorithm_params": meta.get("algorithm_params"),
            # Which model this is — serving stamps every assignment with it.
            "run_id": run_id,
            "readiness_at_export": readiness["overall_status"],
            "registry": meta.get("registry"),
        },
        out_path,
    )
    fit, _, _ = _load_split(
        run_id
    )  # the drift baseline travels with the model: the run dir is swept
    profile = core_mlops.write_monitoring_profile(
        out_path, run_id, fit, None, meta.get("dropped_columns"), None
    )
    return json.dumps(
        {
            "out_path": out_path,
            "algorithm": meta["algorithm"],
            "n_clusters": (meta.get("algorithm_params") or {}).get("n_clusters"),
            "monitoring_profile": profile,
        }
    )


@mcp.tool()
def predict(pkl_path: str, data_path: str) -> str:
    """Loads a model exported by export_model and assigns each new row to a
    cluster. DBSCAN exports are intentionally analysis-only."""
    if not Path(pkl_path).exists():
        return json.dumps(
            {"error": f"no exported model at '{pkl_path}' — call export_model first"}
        )
    data_file = Path(data_path)
    if not data_file.exists():
        return json.dumps({"error": f"no data file at '{data_path}'"})

    bundle = joblib.load(pkl_path)
    algorithm = bundle["algorithm"]
    if algorithm in DENSITY_BASED:
        # ponytail: density clusters have no robust out-of-sample predict; reject new-data scoring instead of inventing one.
        return json.dumps(
            {
                "error": f"{algorithm.upper()} exports do not support predict() on brand-new data; use explain_model/analysis on the original run instead"
            }
        )

    X = read_table(data_file)
    predictions = assign_segments(bundle["pipeline"], algorithm, bundle["cluster_centroids"], X)
    out = X.copy()
    out["cluster"] = predictions
    out_file = data_file.with_name(f"{data_file.name.split('.')[0] or 'data'}_clusters.csv")
    if runtime.user():  # beside the input when allowed, else the user's predictions folder
        try:
            toolguard.check_output(str(out_file), "predict")
        except toolguard.Refused:
            out_file = toolguard.data_dir() / "predictions" / toolguard.user_folder(runtime.user()) / out_file.name
            toolguard.check_output(str(out_file), "predict")
    out_path = str(out_file)
    out.to_csv(out_path, index=False)
    return json.dumps(
        {
            "algorithm": algorithm,
            "n_rows": len(out),
            "out_path": out_path,
            "cluster_counts": _label_counts(predictions),
            "preview": out.head(5).to_dict(orient="records"),
        },
        default=_to_native,
    )


@mcp.tool()
def segment_migration(run_id: str, entity_column: str, period_column: str, data_path: str = "") -> str:
    """How customers MOVE between segments over time. The run's current model
    — fitted once, never refitted per period, so a segment means the same
    thing in every period — assigns every row of a panel (one row per entity
    per period: the run's own data, or data_path in any format), then each
    entity's consecutive periods are compared:
      stay_rate             share of period-to-period steps that stay put
      stay_rate_by_segment  low = a transient state (onboarding, a promo),
                            high = a durable identity
      matrix / counts       from -> to, row shares and raw counts
      top_moves             the largest flows between different segments —
                            high-value -> at-risk is the one to act on
      by_period             share of entities that moved at each step; a
                            jump flags a tariff change, an outage or drift
    The entity and period columns must stay out of the distance: keep them
    with set_profile_columns before train_model (apply_drop_columns removes
    them from the run's data — then pass the raw file as data_path)."""
    pipeline_path = _run_dir(run_id) / "pipeline.pkl"
    if not pipeline_path.exists():
        return json.dumps({"error": "no fitted model for this run yet — call train_model first"})
    meta = _load_meta(run_id)
    algorithm = meta.get("algorithm")
    if algorithm in DENSITY_BASED:
        return json.dumps({"error": f"{algorithm} segments cannot place rows from other periods — track "
                                    "migration with a kmeans / kmedoids / gmm / hierarchical model"})
    if data_path:
        data = read_table(data_path)
    else:
        fit, holdout, _ = _load_split(run_id)
        data = pd.concat([fit, holdout], ignore_index=True)
    missing = [c for c in (entity_column, period_column) if c not in data.columns]
    if missing:
        return json.dumps({"error": f"columns not in the data: {missing} — dropped columns are gone from the "
                                    "run's data; pass the raw file as data_path, or keep them with "
                                    "set_profile_columns and retrain"})
    pipeline = joblib.load(pipeline_path)
    encode = pipeline.named_steps["preprocess"].named_steps["encode"]
    in_distance = [c for c in (entity_column, period_column)
                   if any(c in list(cols) for name, _, cols in encode.transformers_ if name != "remainder")]
    if in_distance:
        return json.dumps({"error": f"{in_distance} are part of the distance, so segments are partly defined by "
                                    "who or when — set_profile_columns (keeps them out of the distance), retrain, "
                                    "then retry"})

    panel = pd.DataFrame({"entity": data[entity_column].to_numpy(), "period": data[period_column].to_numpy(),
                          "segment": assign_segments(pipeline, algorithm, meta.get("cluster_centroids"),
                                                     data).astype(str)})
    order = panel["period"] if pd.api.types.is_numeric_dtype(panel["period"]) else _parse_dates(panel["period"])
    panel["_order"] = order if order.notna().all() else panel["period"].astype(str)
    duplicate_rows = int(panel.duplicated(["entity", "_order"]).sum())
    panel = panel.drop_duplicates(["entity", "_order"], keep="last").sort_values(["entity", "_order"])
    step = panel.groupby("entity")
    panel["to"], panel["to_period"], panel["_to_order"] = (
        step["segment"].shift(-1), step["period"].shift(-1), step["_order"].shift(-1))
    moves = panel.dropna(subset=["to"])
    if moves.empty:
        return json.dumps({"error": f"no {entity_column} appears in two periods — there is nothing to track"})

    counts = pd.crosstab(moves["segment"], moves["to"])
    shares = counts.div(counts.sum(axis=1), axis=0)
    stay = (moves["segment"] == moves["to"])
    flows = sorted(((f, t, int(counts.loc[f, t])) for f in counts.index for t in counts.columns
                    if f != t and counts.loc[f, t] > 0), key=lambda m: -m[2])
    by_period = [{"from_period": _to_native(g["period"].iloc[0]), "to_period": _to_native(g["to_period"].iloc[0]),
                  "entities": int(len(g)), "moved_share": round(float((g["segment"] != g["to"]).mean()), 4)}
                 for _, g in moves.groupby(["_order", "_to_order"], sort=True)]
    result = {
        "run_id": run_id, "algorithm": algorithm,
        "entities": int(panel["entity"].nunique()), "transitions": int(len(moves)),
        "stay_rate": round(float(stay.mean()), 4),
        "stay_rate_by_segment": {k: round(float(v), 4) for k, v in stay.groupby(moves["segment"]).mean().items()},
        "matrix": {f: {t: round(float(v), 4) for t, v in row.items()} for f, row in shares.iterrows()},
        "counts": {f: {t: int(v) for t, v in row.items()} for f, row in counts.iterrows()},
        "top_moves": [{"from": f, "to": t, "count": c, "share_of_from": round(float(shares.loc[f, t]), 4)}
                      for f, t, c in flows[:5]],
        "by_period": by_period[-24:],
    }
    warnings = []
    if duplicate_rows:
        warnings.append(f"{duplicate_rows} rows repeat an entity and period — the last one was kept")
    if (meta.get("training_metrics") or {}).get("overfitting_warning"):
        warnings.append("the segments are not stable (stability ARI below the bar), so moves between them may "
                        "be noise — report migration only for a stable segmentation")
    if warnings:
        result["warnings"] = warnings
    meta["migration"] = {k: result[k] for k in ("stay_rate", "transitions", "top_moves")} | {
        "entity_column": entity_column, "period_column": period_column}
    _save_meta(run_id, meta)
    return json.dumps(result, default=_to_native)


@mcp.tool()
def compare_runs(run_ids: str) -> str:
    """Side-by-side comparison of already-trained runs (comma-separated
    run_ids) — reads each run's saved meta.json, no retraining."""
    ids = [r.strip() for r in run_ids.split(",") if r.strip()]
    rows = []
    for run_id in ids:
        try:
            meta = _load_meta(run_id)
        except FileNotFoundError as exc:
            rows.append({"run_id": run_id, "error": str(exc)})
            continue
        metrics = _current_metrics(meta)
        fit_metrics = metrics.get("fit") or {}
        rows.append(
            {
                "run_id": run_id,
                "algorithm": meta.get("algorithm"),
                "silhouette_score": fit_metrics.get("silhouette_score"),
                "davies_bouldin_score": fit_metrics.get("davies_bouldin_score"),
                "calinski_harabasz_score": fit_metrics.get("calinski_harabasz_score"),
                "n_clusters_found": fit_metrics.get("n_clusters_found"),
                "stability_ari": (metrics.get("stability") or {}).get("stability_ari_mean"),
            }
        )
    ranked = sorted(
        [row for row in rows if row.get("silhouette_score") is not None],
        key=lambda row: (-row["silhouette_score"], row["davies_bouldin_score"]),
    )
    return json.dumps(
        {"runs": rows, "ranked_by_silhouette": [row["run_id"] for row in ranked]}
    )
