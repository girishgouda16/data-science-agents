"""Self-check for the shared gate engine — the rules every agent's gate set
inherits, independent of any domain:

  * a required gate nobody computed is not_run, never a pass,
  * one failure blocks; missing evidence only makes a run incomplete,
  * a success bar is compared or reported absent, never derived,
  * a screen that ran before the feature set changed is stale.

Run: python test_gates.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core import gates

REQUIRED = ("a", "b", "c")

# A gate the domain forgot to compute cannot pass by being absent — this is the
# whole point of the layer.
forgotten = gates.readiness("r1", {"a": gates.gate(gates.PASS, "ran")}, REQUIRED)
assert forgotten["overall_status"] == "incomplete"
assert set(forgotten["not_run_gates"]) == {"b", "c"}
assert forgotten["gates_passed"] == 1 and forgotten["gates_total"] == 3

# One failure blocks, even when everything else passed.
blocked = gates.readiness(
    "r1",
    {
        "a": gates.gate(gates.PASS, "ran"),
        "b": gates.gate(gates.FAIL, "identifier in the model"),
        "c": gates.gate(gates.NOT_APPLICABLE, "no protected attribute in this dataset"),
    },
    REQUIRED,
)
assert blocked["overall_status"] == "blocked" and blocked["failed_gates"] == ["b"]

# not_applicable WITH a reason counts as satisfied; it never disappears.
ready = gates.readiness(
    "r1",
    {
        "a": gates.gate(gates.PASS, "ran"),
        "b": gates.gate(gates.PASS, "ran"),
        "c": gates.gate(gates.NOT_APPLICABLE, "no protected attribute in this dataset"),
    },
    REQUIRED,
)
assert ready["overall_status"] == "ready" and ready["mechanical_completeness"] == 1.0
assert "human review" in ready["interpretation"]

try:
    gates.gate("probably_fine", "…")
    raise AssertionError("an unknown status must be refused")
except ValueError:
    pass

# --- success criteria: compared, or reported absent. Never invented. --------
measured = lambda meta, metric: {"r2": 0.71}.get(metric)  # noqa: E731
SUPPORTED = ("r2", "rmse")

absent = gates.evaluate_success_criteria({}, measured, SUPPORTED)
assert (
    absent["status"] == gates.NOT_RUN and "must NOT be invented" in absent["evidence"]
)

met = gates.evaluate_success_criteria(
    {
        "business_understanding": {
            "success_target": {
                "metric": "r2",
                "threshold": 0.7,
                "direction": "at_least",
            }
        }
    },
    measured,
    SUPPORTED,
)
assert met["status"] == gates.PASS and "0.71" in met["evidence"]

missed = gates.evaluate_success_criteria(
    {
        "business_understanding": {
            "success_target": {
                "metric": "r2",
                "threshold": 0.8,
                "direction": "at_least",
            }
        }
    },
    measured,
    SUPPORTED,
)
assert missed["status"] == gates.FAIL

# "lower is better" metrics are judged in the right direction.
lower_is_better = gates.evaluate_success_criteria(
    {
        "business_understanding": {
            "success_target": {"metric": "rmse", "threshold": 5, "direction": "at_most"}
        }
    },
    lambda meta, metric: 4.2,
    SUPPORTED,
)
assert lower_is_better["status"] == gates.PASS

# A bar on a metric this agent cannot measure is rejected at RECORD time.
try:
    gates.record_success_target({}, "auc", 0.9, SUPPORTED)
    raise AssertionError(
        "an unmeasurable success metric must be refused when it is set"
    )
except ValueError:
    pass
assert gates.record_success_target({}, None, None, SUPPORTED) is None

# --- ledger + ordering ------------------------------------------------------
meta = {}
for tool in ("detect_data_leakage", "train_model"):
    gates.record_execution(meta, tool)
gates.record_execution(meta, "train_model", ok=False, detail="boom")
assert gates.execution_order(meta) == [
    "detect_data_leakage",
    "train_model",
], "a failed step never counts as done"

fine = gates.check_screen_ordering(meta, "detect_data_leakage", ("apply_drop_columns",))
assert fine["status"] == gates.PASS

gates.record_execution(meta, "apply_drop_columns")
stale = gates.check_screen_ordering(
    meta, "detect_data_leakage", ("apply_drop_columns",)
)
assert stale["status"] == gates.FAIL and "re-run it" in stale["evidence"]

never = gates.check_screen_ordering(
    {"execution_log": [{"tool": "train_model", "ok": True}]}, "detect_data_leakage", ()
)
assert (
    never["status"] == gates.FAIL and "without detect_data_leakage" in never["evidence"]
)

empty = gates.check_screen_ordering({}, "detect_data_leakage", ())
assert empty["status"] == gates.NOT_RUN

# --- acknowledgements need a reason, not a flag -----------------------------
acked = gates.acknowledge(
    {}, "identifiers", "customer_id", "one row per customer; it IS the unit of analysis"
)
assert acked["acknowledged_identifiers"]["customer_id"]["justification"].startswith(
    "one row"
)
try:
    gates.acknowledge({}, "identifiers", "customer_id", "   ")
    raise AssertionError("an acknowledgement with no reason is just a silenced gate")
except ValueError:
    pass

print("gate engine OK")
