"""forecasting agent MCP tools, one module per skill. Importing the package registers
every tool on `mcp`; every tool and helper is re-exported, so `mcp_server.<name>`
keeps working for tests and callers. Run: python -m mcp_server.server"""

from . import core  # noqa: F401
from .core import *  # noqa: F401,F403
from .core import (
    _backtest,
    _cleanup_old_runs,
    _exog_columns,
    _fit_forecast_model,
    _load_meta,
    _load_split,
    _overfit_gate,
    _rmse_mae_mape,
    _run_dir,
    _save_meta,
    _to_native,
)  # noqa: F401
from . import eda as _eda_module  # noqa: F401
from . import data_cleaning as _data_cleaning_module  # noqa: F401
from . import modeling as _modeling_module  # noqa: F401
from . import diagnostics as _diagnostics_module  # noqa: F401
from . import mlops as _mlops_module  # noqa: F401
from .eda import eda, prepare_dataset, detect_outliers, list_data_sources, load_dataset  # noqa: F401
from .data_cleaning import (
    propose_imputation,
    apply_imputation,
    propose_drop_columns,
    apply_drop_columns,
)  # noqa: F401
from .modeling import (
    compare_models,
    train_model,
    tune_hyperparams,
    explain_model,
    compare_runs,
    export_model,
    predict,
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
    train_baseline,
)  # noqa: F401
