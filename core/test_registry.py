"""Self-check for core/registry.py — the inspect-then-promote loop.

What has to hold for the loop to work at all:

  1. successive runs register as VERSIONS of one name, not as separate models
     (otherwise nothing accumulates and nothing is comparable),
  2. list_versions reports the same version numbers and aliases the MLflow UI
     shows — that number is the handle the user quotes back,
  3. compare defaults to newest-vs-previous and reports signed deltas,
  4. promote moves an alias and reports what it displaced, so a promotion is
     reversible,
  5. the failure paths return {"error": ...} instead of raising: unknown
     version, unknown alias, unknown model.

Runs against a throwaway file store in a temp dir (MLFLOW_TRACKING_URI), so it
touches neither the repo's mlruns/ nor any server.

Run: python test_registry.py
"""

import shutil
import sys
import tempfile
from pathlib import Path

_STORE = Path(tempfile.mkdtemp(prefix="registry-selfcheck-"))
import os

os.environ["MLFLOW_TRACKING_URI"] = f"file:{_STORE / 'mlruns'}"

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core import registry

import mlflow
from sklearn.dummy import DummyClassifier


def _log_run(name: str, auc: float) -> int:
    """Train the smallest possible model, log it, register it. Returns the new
    registry version."""
    mlflow.set_tracking_uri(registry.tracking_uri())
    mlflow.set_experiment("selfcheck")
    with mlflow.start_run() as run:
        mlflow.log_metric("roc_auc", auc)
        mlflow.log_param("model", "dummy")
        model = DummyClassifier(strategy="prior").fit([[0], [1]], [0, 1])
        mlflow.sklearn.log_model(model, "model")
        run_id = run.info.run_id
    result = registry.register(run_id, name)
    assert "version" in result, result
    return result["version"]


def main() -> None:
    name = registry.model_name("classification", "is_fraud")
    assert name == "classification-is_fraud", name
    # A target with characters that are illegal in a registry name must not
    # produce an unusable name.
    assert " " not in registry.model_name("classification", "churn flag/v2")

    # 1. two experiments stack as versions of ONE model
    v1 = _log_run(name, auc=0.80)
    v2 = _log_run(name, auc=0.91)
    assert (v1, v2) == (1, 2), (v1, v2)

    # 2. listing reports both, newest first, with metrics
    listing = registry.list_versions(name)
    assert [row["version"] for row in listing["versions"]] == [2, 1], listing
    assert listing["versions"][0]["metrics"]["roc_auc"] == 0.91

    # 3. compare defaults to newest vs previous, signed delta
    cmp = registry.compare(name)
    assert cmp["candidate"]["version"] == 2 and cmp["incumbent"]["version"] == 1, cmp
    assert abs(cmp["metrics"]["roc_auc"]["delta"] - 0.11) < 1e-9, cmp["metrics"]
    # explicit order is honoured, and flips the sign
    assert registry.compare(name, 1, 2)["metrics"]["roc_auc"]["delta"] < 0

    # 4. promotion moves the alias and reports what it displaced
    first = registry.promote(name, 1, registry.CHAMPION)
    assert first["previous_version"] is None and first["rollback"] is None, first
    second = registry.promote(name, 2, registry.CHAMPION)
    assert second["previous_version"] == 1, second
    assert second["model_uri"] == f"models:/{name}@champion"
    assert "promote_model" in second["rollback"], second

    # the alias now resolves to v2, and v1 no longer carries it
    by_version = {
        row["version"]: row for row in registry.list_versions(name)["versions"]
    }
    assert by_version[2]["aliases"] == ["champion"], by_version
    assert by_version[1]["aliases"] == [], by_version

    # champion and challenger are independent
    registry.promote(name, 1, registry.CHALLENGER)
    by_version = {
        row["version"]: row for row in registry.list_versions(name)["versions"]
    }
    assert by_version[1]["aliases"] == ["challenger"], by_version
    assert by_version[2]["aliases"] == ["champion"], by_version

    # the audit trail lands on the version itself
    registry.promote(
        name,
        2,
        registry.CHAMPION,
        tags={"champion.reason": "higher recall", "champion.forced": "false"},
    )
    tags = {r["version"]: r for r in registry.list_versions(name)["versions"]}[2][
        "version_tags"
    ]
    assert (
        tags["champion.reason"] == "higher recall"
        and tags["champion.forced"] == "false"
    ), tags

    # demote clears an alias outright — "no champion" is a state promotion cannot express
    assert registry.demote(name, registry.CHAMPION)["was_pointing_at"] == 2
    assert (
        "champion.demoted_at"
        in {r["version"]: r for r in registry.list_versions(name)["versions"]}[2][
            "version_tags"
        ]
    )
    by_version = {r["version"]: r for r in registry.list_versions(name)["versions"]}
    assert by_version[2]["aliases"] == [], by_version
    assert registry.demote(name, registry.CHAMPION)["was_pointing_at"] is None, (
        "clearing twice must not raise"
    )
    assert "error" in registry.demote(name, "nonsense")

    # listings carry what each experiment WAS, not just how it scored
    top = registry.list_versions(name)["versions"][0]
    assert top["model"] == "dummy" and top["params"]["model"] == "dummy", top

    # 5. failure paths report, never raise
    assert "error" in registry.promote(name, 99, registry.CHAMPION), (
        "unknown version must error"
    )
    assert "error" in registry.promote(name, 1, "champoin"), (
        "typo'd alias must be rejected, not created"
    )
    assert "error" in registry.list_versions("no-such-model")
    assert "error" in registry.compare("no-such-model")
    # a model with a single version has nothing to compare against
    solo = registry.model_name("classification", "solo")
    _log_run(solo, auc=0.5)
    assert "error" in registry.compare(solo)

    # 6. the champion link is the one path serving loads: it follows each
    #    promotion atomically, and demotion removes it (the versioned files stay)
    registry.ARTIFACTS_DIR = _STORE / "artifacts"
    registry.ARTIFACTS_DIR.mkdir()
    for v in (1, 2):
        (registry.ARTIFACTS_DIR / f"{name}-v{v}.pkl").write_text(f"model v{v}")
        (registry.ARTIFACTS_DIR / f"{name}-v{v}.monitoring.json").write_text(
            f"profile v{v}"
        )
    served = Path(
        registry.link_champion(name, str(registry.ARTIFACTS_DIR / f"{name}-v1.pkl"))
    )
    assert served == registry.champion_path(name) and served.read_text() == "model v1"
    registry.link_champion(name, str(registry.ARTIFACTS_DIR / f"{name}-v2.pkl"))
    assert (
        served.read_text() == "model v2"
        and served.with_suffix(".monitoring.json").read_text() == "profile v2"
    )
    assert len(registry.unlink_champion(name)) == 2 and not served.exists()
    assert (registry.ARTIFACTS_DIR / f"{name}-v2.pkl").exists(), (
        "demotion must keep the versioned export"
    )
    assert registry.unlink_champion(name) == [], "unlinking twice must not raise"

    # 7. a delta gets an interval only when both versions were scored on the
    #    SAME test rows — then it is paired, and says if it is distinguishable
    import numpy as np
    import pandas as pd

    rng = np.random.default_rng(0)
    y = (rng.random(200) < 0.4).astype(int)

    def _log_scored(model_name: str, labels, noise: float) -> None:
        """A version plus the per-row test scores log_run_to_mlflow ships."""
        from sklearn.metrics import roc_auc_score

        scores = pd.DataFrame(
            {"y_true": labels, "score": labels * 0.5 + rng.random(len(labels)) * noise}
        )
        _log_run(model_name, auc=roc_auc_score(scores["y_true"], scores["score"]))
        run_id = registry.list_versions(model_name)["versions"][0]["run_id"]
        scores.to_csv(_STORE / "test_scores.csv", index=False)
        mlflow.tracking.MlflowClient().log_artifact(
            run_id, str(_STORE / "test_scores.csv")
        )

    paired = registry.model_name("classification", "paired")
    _log_scored(paired, y, noise=1.0)
    _log_scored(paired, y, noise=0.3)  # clearly better ranking on the same rows
    cmp = registry.compare(paired)
    assert cmp["comparability"].startswith("same test rows"), cmp["comparability"]
    assert (
        cmp["metrics"]["roc_auc"]["distinguishable"] is True
        and cmp["metrics"]["roc_auc"]["delta_ci"][0] > 0
    ), cmp["metrics"]

    unpaired = registry.model_name("classification", "unpaired")
    _log_scored(unpaired, y, noise=1.0)
    _log_scored(
        unpaired, 1 - y, noise=1.0
    )  # different rows -> no row-by-row difference exists
    cmp = registry.compare(unpaired)
    assert "delta_ci" not in cmp["metrics"]["roc_auc"] and cmp[
        "comparability"
    ].startswith("no per-row"), cmp

    # An experiment deleted in the MLflow UI must not stop logging (it made set_experiment refuse), and a
    # user-named experiment is created on first use.
    from core import mlops

    client = mlflow.MlflowClient()
    client.delete_experiment(client.create_experiment("classification-agent"))
    assert "restored" in (mlops.use_experiment("classification-agent") or "")
    assert (
        client.get_experiment_by_name("classification-agent").lifecycle_stage
        == "active"
    )
    assert mlops.use_experiment("titanic") is None
    with mlflow.start_run() as run:
        pass
    assert client.get_experiment(run.info.experiment_id).name == "titanic"

    print(
        f"OK — {name} v1/v2 registered, compared, promoted; aliases, champion link, paired delta intervals, "
        "failure paths and experiment restore/naming hold"
    )


try:
    main()
finally:
    shutil.rmtree(_STORE, ignore_errors=True)
