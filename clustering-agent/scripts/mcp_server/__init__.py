"""clustering agent MCP tools, one module per skill. Importing the package registers
every tool on `mcp`; every tool and helper is re-exported, so `mcp_server.<name>`
keeps working for tests and callers. Run: python -m mcp_server.server"""

from . import core  # noqa: F401
from .core import *  # noqa: F401,F403
from .core import (
    _algorithm_params,
    _assign_nearest_centroids,
    _build_pipeline,
    _build_preprocessor,
    _cleanup_old_runs,
    _cluster_count,
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
from . import eda as _eda_module  # noqa: F401
from . import data_cleaning as _data_cleaning_module  # noqa: F401
from . import modeling as _modeling_module  # noqa: F401
from . import diagnostics as _diagnostics_module  # noqa: F401
from . import mlops as _mlops_module  # noqa: F401
from . import feat_engineering as _feat_engineering_module  # noqa: F401
from .feat_engineering import aggregate_events, apply_custom_feature, apply_peer_features  # noqa: F401
from .eda import eda, list_data_sources, load_dataset, prepare_dataset  # noqa: F401
from .data_cleaning import (
    propose_imputation,
    apply_imputation,
    propose_drop_columns,
    apply_drop_columns,
    set_profile_columns,
)  # noqa: F401
from .modeling import (
    propose_k,
    train_model,
    explain_model,
    export_model,
    predict,
    segment_migration,
    compare_runs,
)  # noqa: F401
from .mlops import (
    log_run_to_mlflow,
    list_model_versions,
    compare_model_versions,
    promote_model,
    demote_model,
)  # noqa: F401
from .diagnostics import (
    compute_readiness,
    detect_data_leakage,
    record_business_context,
    acknowledge_identifier_column,
    acknowledge_gate,
    check_readiness,
    generate_report,
)  # noqa: F401
