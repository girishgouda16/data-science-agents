"""visualization agent MCP tools, one module per skill. Importing the package registers
every tool on `mcp`; every tool and helper is re-exported, so `mcp_server.<name>`
keeps working for tests and callers. Run: python -m mcp_server.server"""

from . import core  # noqa: F401
from .core import *  # noqa: F401,F403
from .core import (
    _classifier_mismatch,
    _fitted,
    _load_run,
    _looks_negative,
    _positive_index,
    _positive_label,
    _run_meta,
    _save,
)  # noqa: F401
from . import exploratory as _exploratory_module  # noqa: F401
from . import model_eval as _model_eval_module  # noqa: F401
from .exploratory import (
    plot_histogram,
    plot_boxplot,
    plot_scatter,
    plot_correlation_heatmap,
    plot_class_distribution,
    plot_missingness,
)  # noqa: F401
from .model_eval import (
    plot_feature_importance,
    plot_confusion_matrix,
    plot_roc_curve,
    plot_pr_curve,
    plot_shap_beeswarm,
    plot_calibration_curve,
    plot_regression_diagnostics,
    plot_forecast,
    plot_clusters,
    plot_anomaly_scores,
)  # noqa: F401
