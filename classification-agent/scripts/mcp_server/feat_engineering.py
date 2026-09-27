"""Business-meaning-first feature engineering. The tools themselves are shared
with the regression agent (core/feature_tools.py); this module binds them to
this agent's run storage and to classification's per-feature signal — the
ROC-AUC a feature reaches alone — and registers them on this server."""
from core import feature_tools

from .core import _load_split, _run_dir, _save_meta, mcp, positive_label
from .diagnostics import NEAR_PERFECT_AUC, _single_column_auc


def _classification_signal(train, target, meta, columns) -> dict:
    return {
        "scores": _single_column_auc(train, target, positive_label(meta, train[target]), columns),
        "metric": "auc",
        "suspect": NEAR_PERFECT_AUC,
        "weak": 0.55,
        "groups": train[target],
    }


feature_tools.bind(load_split=_load_split, run_dir=_run_dir, save_meta=_save_meta, signal=_classification_signal)
_tools = feature_tools.register(mcp)
propose_features = _tools["propose_features"]
apply_features = _tools["apply_features"]
apply_custom_feature = _tools["apply_custom_feature"]
aggregate_events = _tools["aggregate_events"]
apply_entity_features = _tools["apply_entity_features"]
apply_peer_features = _tools["apply_peer_features"]
profile_features = _tools["profile_features"]
_reasoned_features = feature_tools.reasoned_features
