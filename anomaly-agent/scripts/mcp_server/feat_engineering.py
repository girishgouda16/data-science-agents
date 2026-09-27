"""anomaly agent — `feature-engineering` tools, shared with every agent
(core/feature_tools.py). Telco anomalies live in behaviour, not in single
rows: per-subscriber aggregates of their events (aggregate_events: gap
regularity, bursts, destination diversity), deviation from the entity's own
history (apply_entity_features: vs_roll_mean, roll_count velocity, first-time
values) and from its peers (apply_peer_features). With a label, a feature's
signal is its held-out AUC against it; without one, profile_features is
unavailable (there is nothing to measure against)."""
import numpy as np
from core import feature_tools
from sklearn.metrics import roc_auc_score

from .core import _label_to_binary, _load_split, _run_dir, _save_meta, mcp


def _anomaly_signal(train, target, meta, columns) -> dict:
    if not meta.get("evaluation_available"):
        raise ValueError("no label column — a feature's signal cannot be measured; judge features by the "
                         "review queue explain_model produces")
    y = _label_to_binary(train[meta["label_column"]], meta).to_numpy()
    scores = {}
    for col in columns:
        x = train[col]
        if x.dtype == object or x.nunique(dropna=True) < 2:
            continue
        values = x.astype(float).fillna(x.astype(float).median()).to_numpy()
        auc = roc_auc_score(y, values)
        scores[col] = round(float(max(auc, 1 - auc)), 4)
    groups = np.where(y == 1, "anomaly", "normal")
    return {"scores": scores, "metric": "auc", "suspect": 0.98, "weak": 0.55, "groups": groups}


feature_tools.bind(load_split=_load_split, run_dir=_run_dir, save_meta=_save_meta, signal=_anomaly_signal)
_tools = feature_tools.register(mcp, only=("aggregate_events", "apply_custom_feature", "apply_entity_features",
                                           "apply_peer_features", "profile_features"))
aggregate_events = _tools["aggregate_events"]
apply_custom_feature = _tools["apply_custom_feature"]
apply_entity_features = _tools["apply_entity_features"]
apply_peer_features = _tools["apply_peer_features"]
profile_features = _tools["profile_features"]
_reasoned_features = feature_tools.reasoned_features
