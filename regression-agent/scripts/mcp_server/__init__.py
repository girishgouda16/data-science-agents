"""regression agent MCP tools, one module per skill. Importing the package registers
every tool on `mcp`; every tool and helper is re-exported, so `mcp_server.<name>`
keeps working for tests and callers. Run: python -m mcp_server.server"""

from . import core  # noqa: F401
from .core import *  # noqa: F401,F403
from .core import (
    _build_pipeline,
    _cleanup_old_runs,
    _current_metrics,
    _evaluate,
    _explainability_gates,
    _load_meta,
    _load_split,
    _make_model,
    _overfit_gate,
    _run_dir,
    _save_meta,
    _to_native,
)  # noqa: F401
from . import eda as _eda_module  # noqa: F401
from . import data_cleaning as _data_cleaning_module  # noqa: F401
from . import modeling as _modeling_module  # noqa: F401
from . import diagnostics as _diagnostics_module  # noqa: F401
from . import mlops as _mlops_module  # noqa: F401
from . import feat_engineering as _feat_engineering_module  # noqa: F401
from .eda import (  # noqa: F401
    eda, prepare_dataset, detect_outliers, propose_group_column, propose_time_column,
    list_data_sources, load_dataset,
)
from .feat_engineering import (  # noqa: F401
    aggregate_events, apply_custom_feature, apply_entity_features, apply_features,
    apply_peer_features, profile_features, propose_features,
)
from .data_cleaning import (
    propose_imputation,
    apply_imputation,
    propose_drop_columns,
    apply_drop_columns,
    propose_datetime_features,
    apply_datetime_features,
    apply_target_transform,
)  # noqa: F401
from .modeling import (
    set_objective,
    compare_models,
    train_model,
    tune_hyperparams,
    explain_model,
    compare_runs,
    export_model,
    predict,
)  # noqa: F401
from .diagnostics import (
    record_business_context,
    detect_data_leakage,
    acknowledge_identifier_column,
    train_baseline,
    check_residuals,
    check_model_stability,
    calibrate_intervals,
    analyze_errors,
    check_readiness,
)  # noqa: F401
from .mlops import (
    log_run_to_mlflow,
    list_model_versions,
    compare_model_versions,
    promote_model,
    demote_model,
)  # noqa: F401
from .diagnostics import generate_report  # noqa: F401
