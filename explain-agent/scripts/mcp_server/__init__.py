"""explain agent MCP tools, one module per skill. Importing the package registers
every tool on `mcp`; every tool and helper is re-exported, so `mcp_server.<name>`
keeps working for tests and callers. Run: python -m mcp_server.server"""

from . import core  # noqa: F401
from .core import *  # noqa: F401,F403
from .core import (
    _DATA_DIR,
    _decision,
    _encode,
    _explainer_for,
    _final_step_name,
    _load_artifact,
    _model_name,
    _narrate,
    _native,
    _positive_index,
    _prepare,
    _reference_frame,
    _score,
    _shap_matrix,
    _stability,
    _task_of,
    _unwrap_calibrated,
)  # noqa: F401
from . import decision as _decision_module  # noqa: F401
from . import counterfactual as _counterfactual_module  # noqa: F401
from . import model_explanation as _model_explanation_module  # noqa: F401
from .decision import inspect_explainability, explain_prediction  # noqa: F401
from .counterfactual import explain_counterfactual  # noqa: F401
from .model_explanation import explain_model  # noqa: F401
