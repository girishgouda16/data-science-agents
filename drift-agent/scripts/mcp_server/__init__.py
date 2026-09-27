"""drift agent MCP tools, one module per skill. Importing the package registers
every tool on `mcp`; every tool and helper is re-exported, so `mcp_server.<name>`
keeps working for tests and callers. Run: python -m mcp_server.server"""

from . import core  # noqa: F401
from .core import *  # noqa: F401,F403
from .core import (
    _categorical_drift,
    _column_drift,
    _json_default,
    _numeric_drift,
    _overall_verdict,
    _psi,
    _read_csv_or_error,
    _read_profile,
    _severity,
    _structural_columns,
    domain_classifier,
    psi_noise_floor,
)  # noqa: F401
from . import eda as _eda_module  # noqa: F401
from . import drift as _drift_module  # noqa: F401
from .eda import eda, list_data_sources, load_dataset  # noqa: F401
from .drift import compare_distributions, detect_drift, detect_multivariate_drift, drift_over_time  # noqa: F401
