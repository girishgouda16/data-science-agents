"""serving agent MCP tools, one module per skill. Importing the package registers
every tool on `mcp`; every tool and helper is re-exported, so `mcp_server.<name>`
keeps working for tests and callers. Run: python -m mcp_server.server"""

from . import core  # noqa: F401
from .core import *  # noqa: F401,F403
from .core import (
    _assign_nearest_centroids,
    _json_default,
    _load_artifact,
    _positive_index,
    _predict_forecast,
    _predict_sklearn,
)  # noqa: F401
from . import serving as _serving_module  # noqa: F401
from .serving import list_exported_models, inspect_model, predict  # noqa: F401
