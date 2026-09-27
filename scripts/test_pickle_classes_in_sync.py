"""A pickled model records its classes as `pipeline_transformers.<Class>`, so
every agent that unpickles another agent's export needs a copy of each class
the exporting agent can put in a bundle. A class added upstream and not
copied here breaks serving/explain/visualization only at predict time, on
the first model that uses it — this catches it at CI time instead."""
import ast
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
EXPORTERS = ["classification", "regression", "clustering", "anomaly", "forecasting"]
LOADERS = {  # loader -> the exporters whose pickles it opens
    "serving": EXPORTERS,
    "explain": EXPORTERS,
    "visualization": ["classification", "regression", "clustering", "anomaly"],  # forecasts are read as data
}


def _tree(agent: str) -> ast.Module:
    return ast.parse((REPO / f"{agent}-agent/scripts/pipeline_transformers.py").read_text())


def _classes(agent: str) -> set:
    return {node.name for node in ast.walk(_tree(agent)) if isinstance(node, ast.ClassDef)}


def _logic(agent: str) -> dict:
    """Every class and function by name, docstrings stripped — what must be
    identical in a copy for an unpickled model to behave the same."""
    out = {}
    top = [n for stmt in _tree(agent).body for n in (stmt.body if isinstance(stmt, ast.Try) else [stmt])]
    for node in top:  # module level, and inside `try:` (the optional-xgboost class)
        if isinstance(node, (ast.ClassDef, ast.FunctionDef)):
            for n in ast.walk(node):
                body = getattr(n, "body", None)
                if isinstance(body, list) and body and isinstance(body[0], ast.Expr) \
                        and isinstance(getattr(body[0], "value", None), ast.Constant) \
                        and isinstance(body[0].value.value, str):
                    n.body = body[1:] or [ast.Pass()]
            out[node.name] = ast.dump(node)
    return out


@pytest.mark.parametrize("loader", sorted(LOADERS))
def test_loader_defines_every_class_its_exporters_pickle(loader):
    missing = {agent: sorted(_classes(agent) - _classes(loader)) for agent in LOADERS[loader]}
    assert not any(missing.values()), f"{loader}-agent/scripts/pipeline_transformers.py lacks {missing}"


@pytest.mark.parametrize("loader", sorted(LOADERS))
def test_copied_classes_behave_the_same(loader):
    """A copy that drifted (a forecast step changed upstream, not here) loads
    fine and then forecasts differently — the worst kind of failure."""
    mine = _logic(loader)
    drifted = {agent: sorted(name for name, code in _logic(agent).items() if name in mine and mine[name] != code)
               for agent in LOADERS[loader]}
    assert not any(drifted.values()), f"{loader}-agent/scripts/pipeline_transformers.py differs in {drifted}"
