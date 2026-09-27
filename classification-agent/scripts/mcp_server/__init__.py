"""Tabular-classification MCP tools, one module per skill — as granular as
the skill split itself: business (business-understanding gate), eda (shape/
split/balance/outliers), data_cleaning (impute/drop/datetime/SMOTE),
modeling (train/tune/explain/threshold/export/predict), diagnostics
(leakage/baseline/error-analysis/segments/stability/fairness/reflection),
feat_engineering (business + correlation features), feat_selection
(importance-driven column removal), reporting (the standard report), mlops
(mlflow/dvc/CI). `core` is the shared kernel underneath all of them (the
`mcp` FastMCP instance + run storage / pipeline-building helpers) and owns
no skill of its own. `gates` is the deterministic layer across all of them:
what is true about a run computed from its artifacts, never from what the
agent says about it — the readiness gate reporting and export both consult. Importing this package (or server) registers every
module's tools onto `mcp`.
"""

from .core import *  # noqa: F401,F403
from .core import (
    mcp,
    _cv_sample,
    _load_meta,
    _run_dir,
    _load_split,
    _save_meta,
    _to_native,
)  # noqa: F401
from . import (  # noqa: F401
    business,
    data_cleaning,
    diagnostics,
    eda,
    feat_engineering,
    feat_selection,
    gates,
    mlops,
    modeling,
    reporting,
)
from .business import record_business_context  # noqa: F401
from .gates import (  # noqa: F401
    acknowledge_identifier_column,
    acknowledge_label_rule,
    check_readiness,
    compute_readiness,
    declare_fairness_not_applicable,
)
from .eda import (  # noqa: F401
    check_imbalance,
    detect_outliers,
    inspect_target,
    prepare_dataset,
    propose_group_column,
    propose_time_column,
    list_data_sources,
    load_dataset,
)
from .eda import (
    eda as eda_tool,
)  # noqa: F401 — avoid shadowing the `eda` submodule name
from .data_cleaning import (  # noqa: F401
    apply_datetime_features,
    apply_frequency_encoding,
    apply_drop_columns,
    apply_imputation,
    apply_smote,
    propose_datetime_features,
    propose_drop_columns,
    propose_imputation,
    propose_smote,
)
from .diagnostics import (  # noqa: F401
    check_calibration,
    analyze_segments,
    assess_fairness,
    backtest_on_new_data,
    check_label_rule,
    check_model_stability,
    declare_feature_engineering_not_applicable,
    detect_data_leakage,
    error_analysis,
    evaluate_model,
    record_reflection,
    train_baseline,
)
from .feat_engineering import (  # noqa: F401
    aggregate_events,
    apply_custom_feature,
    apply_entity_features,
    apply_peer_features,
    apply_features,
    profile_features,
    propose_features,
)
from .feat_selection import propose_feature_selection  # noqa: F401
from .mlops import (  # noqa: F401
    compare_model_versions,
    demote_model,
    dvc_track,
    generate_ci_workflow,
    list_model_versions,
    log_run_to_mlflow,
    promote_model,
)
from .modeling import (  # noqa: F401
    calibrate_model,
    compare_models,
    compare_runs,
    explain_model,
    export_model,
    predict,
    run_standard_diagnostics,
    train_model,
    tune_hyperparams,
    tune_threshold,
)
from .reporting import generate_report  # noqa: F401

# `eda` above is the submodule; callers doing `m.eda(path)` (the tool, as it
# was on the old monolithic mcp_server.py) need the function, not the
# module — rebind the package attribute to the tool after both are imported.
eda = eda_tool
del eda_tool
