"""clustering agent — `feature-engineering` tools, shared with the supervised
agents (core/feature_tools.py). Only the ones that need no target: segments
are built from behaviour — per-subscriber aggregates of their events
(aggregate_events: gap statistics, rhythm, shares, diversity), ratios
between them (apply_custom_feature) and comparison with peers
(apply_peer_features)."""
from core import feature_tools

from .core import _load_split, _run_dir, _save_meta, mcp


def _no_signal(train, target, meta, columns):
    raise ValueError("clustering has no target to measure a feature's signal against")


feature_tools.bind(load_split=_load_split, run_dir=_run_dir, save_meta=_save_meta, signal=_no_signal)
_tools = feature_tools.register(mcp, only=("aggregate_events", "apply_custom_feature", "apply_peer_features"))
aggregate_events = _tools["aggregate_events"]
apply_custom_feature = _tools["apply_custom_feature"]
apply_peer_features = _tools["apply_peer_features"]
_reasoned_features = feature_tools.reasoned_features
