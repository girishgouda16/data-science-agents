"""Track a run in MLflow, version a data/artifact file with DVC, and emit a
CI workflow. None of these gate — they record/version/scaffold around a
model, they don't touch it. See the `mlops` skill."""

import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # repo root, for core.*
from core import mlops as core_mlops, registry, runtime  # noqa: E402


def _resolve_name(target: str, name: str) -> tuple[str | None, str | None]:
    """(registry name, error). Names carry the project, so a bare target can
    match one model per project — refuse to guess between them."""
    if name:
        return name, None
    if not target:
        return (
            None,
            "pass either name (as log_run_to_mlflow / list_model_versions report it) or target",
        )
    names = registry.find_names("classification", target)
    if len(names) == 1:
        return names[0], None
    if not names:
        return (
            None,
            f"no registered classification model for target '{target}' — log a run to MLflow first",
        )
    return (
        None,
        f"target '{target}' is registered under several projects: {names} — pass the one you mean as name=",
    )


from .core import RUNS_DIR, _load_meta, _run_dir, _save_meta, mcp
from .gates import compute_readiness
from .modeling import export_model
from .reporting import generate_report


@mcp.tool()
def log_run_to_mlflow(run_id: str, experiment: str = "") -> str:
    """Log this run's model, params, and metrics to a local MLflow file
    store (./mlruns, no server needed). No gate — read-only side effect.
    experiment: only when the user names one (created if new); by default
    it is the registered model name, classification-<project>-<target>.
    Returns an error string (not a raised exception) if mlflow isn't
    installed."""
    try:
        import mlflow
    except ImportError:
        return json.dumps(
            {"error": "mlflow not installed — `pip install mlflow` to enable this tool"}
        )
    meta = _load_meta(run_id)
    metrics = meta.get("tuned_metrics") or meta.get("baseline_metrics") or {}
    auc = metrics.get("auc") or {}
    name = core_mlops.model_name(
        "classification", meta
    )  # classification-<project>-<target>
    experiment = (
        experiment.strip() or name
    )  # one experiment per registered model: its runs are comparable
    note = core_mlops.use_experiment(experiment)
    with mlflow.start_run(run_name=run_id) as run:
        mlflow.log_param("model", meta.get("model"))
        mlflow.log_param("target", meta.get("target"))
        if meta.get("best_params"):
            mlflow.log_params(meta["best_params"])
        mlflow.log_metric(
            "roc_auc", auc.get("roc_auc", auc.get("roc_auc_macro", 0)) or 0
        )
        mlflow.log_metric("pr_auc", auc.get("pr_auc", auc.get("pr_auc_macro", 0)) or 0)
        # Threshold-dependent metrics too, so comparing versions doesn't need
        # the (swept-after-7-days) run directories to replay them. The
        # classification_report is pipeline.predict(), i.e. the 0.5 cutoff —
        # named so, because the model ships at the operating point below.
        report = metrics.get("classification_report") or {}
        if "accuracy" in report:
            mlflow.log_metric("accuracy_at_0.5", report["accuracy"])
        if "weighted avg" in report:
            mlflow.log_metric("f1_weighted_at_0.5", report["weighted avg"]["f1-score"])
        if str(meta.get("positive_label")) in report:
            mlflow.log_metric(
                "f1_positive_at_0.5",
                report[str(meta.get("positive_label"))]["f1-score"],
            )
        op = meta.get("operating_point") or {}
        if op.get("threshold") is not None and op.get("pipeline_version") == meta.get(
            "pipeline_version"
        ):
            # What the served model actually does: export ships this threshold.
            for key in ("precision", "recall", "f1", "alert_rate"):
                if op.get(key) is not None:
                    mlflow.log_metric(f"{key}_at_threshold", op[key])
        # The choice between versions should rest on CV (training folds), not
        # on the test rows every version was reported on — so log it.
        if metrics.get("cv_metric"):
            mlflow.log_param("cv_metric", metrics["cv_metric"])
            mlflow.log_metric(
                "cv_score",
                metrics.get("cv_score_best", metrics.get("cv_score_mean")) or 0,
            )
        for metric_name in ("roc_auc", "pr_auc"):
            if metric_name in (metrics.get("ci") or {}):
                lo, hi = metrics["ci"][metric_name]
                mlflow.log_metric(f"{metric_name}_ci_low", lo)
                mlflow.log_metric(f"{metric_name}_ci_high", hi)
        # Lineage: the exact data bytes, the exact test rows, and whether the
        # code that ran is the commit MLflow records (a dirty tree is not).
        mlflow.set_tag("data_sha256", meta.get("source_sha256") or "unrecorded")
        mlflow.set_tag("test_fingerprint", meta.get("test_sha256") or "unrecorded")
        mlflow.set_tag("git_dirty", _git_dirty())
        # The bar this version was judged against. Recorded even when absent,
        # so compare's changed_params shows two versions judged against
        # different bars (or none) instead of implying they are comparable.
        bar = (meta.get("business_understanding") or {}).get("success_target")
        mlflow.log_param(
            "success_bar",
            f"{bar['metric']} {bar['direction']} {bar['threshold']}" if bar else "none",
        )
        # The agent's own run_id, as a tag: promote_model needs it to find the
        # readiness verdict behind a registry version. run_name isn't queryable
        # the same way, so the join gets its own tag.
        mlflow.set_tag("agent_run_id", run_id)
        mlflow.set_tags(
            {
                "project": core_mlops.project_of(meta) or "",
                "owner": meta.get("owner") or runtime.user() or "",
            }
        )
        mlflow.set_tag("readiness", compute_readiness(run_id)["overall_status"])
        # The registered artifact is THIS pipeline. Recording its version lets the
        # promotion gate tell "same model, more evidence since" (trust the live
        # gate) from "refit since logging" (live gate describes another model) —
        # the same idiom gates.py uses to catch a threshold outliving its fit.
        mlflow.set_tag("pipeline_version", str(meta.get("pipeline_version", 0)))
        # What the model decides at. Without these the registry record can't
        # say — v3's tuned threshold was unrecoverable from it.
        # ponytail: recorded, not applied — mlflow.sklearn's predict() still
        # thresholds at 0.5. Nothing serves from models:/ URIs (promoting to
        # champion exports the bundle, which applies it, and serving loads
        # that file); wrap in a pyfunc if anything ever loads models:/...@champion.
        if op.get("threshold") is not None and op.get("pipeline_version") == meta.get(
            "pipeline_version"
        ):
            mlflow.log_param("threshold", op["threshold"])
        if meta.get("positive_label") is not None:
            mlflow.log_param("positive_label", meta["positive_label"])
        pipeline_path = _run_dir(run_id) / "pipeline.pkl"
        report_path = _run_dir(run_id) / "report.md"
        if report_path.exists():
            mlflow.log_artifact(str(report_path))
        for chart in _current_charts(run_id, meta):
            mlflow.log_artifact(str(chart), "charts")
        with tempfile.TemporaryDirectory() as tmp:
            scores = _test_scores(run_id)
            if scores is not None:
                # Per-row test scores: what lets compare put an interval on
                # the DIFFERENCE between two versions scored on the same rows.
                scores.to_csv(Path(tmp) / "test_scores.csv", index=False)
                mlflow.log_artifact(str(Path(tmp) / "test_scores.csv"))
            if meta.get("tuning_trials"):
                (Path(tmp) / "tuning_trials.json").write_text(
                    json.dumps(meta["tuning_trials"], default=str)
                )
                mlflow.log_artifact(str(Path(tmp) / "tuning_trials.json"))

        # Logged as an MLflow *model* (not just a pickled artifact) because
        # only a logged model can be registered, and registration is what
        # gives the version numbers the promotion flow refers to.
        registered = None
        if pipeline_path.exists():
            mlflow.log_artifact(str(pipeline_path))
            try:
                import joblib

                mlflow.sklearn.log_model(joblib.load(pipeline_path), "model")
            except Exception as exc:
                registered = {
                    "error": f"could not log sklearn model ({type(exc).__name__}: {exc})"
                }
        mlflow_run_id = run.info.run_id

    # Registration happens after the run closes: registering a model inside its
    # own still-open run races the artifact upload on the file store.
    if registered is None:
        registered = (
            registry.register(mlflow_run_id, name)
            if pipeline_path.exists()
            else {
                "error": "no pipeline.pkl for this run — train_model first; nothing was registered"
            }
        )
    if "version" in registered:
        # Pins the run dir against _cleanup_old_runs: promoting this version to
        # champion exports from it, and a swept run is a version nobody can serve.
        meta = _load_meta(run_id)
        meta["registry"] = {"name": name, "version": registered["version"]}
        _save_meta(run_id, meta)

    return json.dumps(
        {
            "run_id": run_id,
            "mlflow_run_id": mlflow_run_id,
            "experiment": experiment,
            "note": note,
            "tracking_uri": mlflow.get_tracking_uri(),
            "registered": registered,
            "next": (
                f"Tell the user it is registered as '{name}' version {registered['version']} — that version "
                f"number is what they quote back to promote it after checking the MLflow UI."
                if "version" in registered
                else None
            ),
        }
    )


@mcp.tool()
def dvc_track(path: str) -> str:
    """Shell out to `dvc add <path>` to version a data file/artifact.
    Requires the dvc CLI on PATH and an initialized DVC repo (`dvc init`
    once, at the repo root) — returns an error string naming the missing
    precondition rather than trying to init DVC silently."""
    if shutil.which("dvc") is None:
        return json.dumps(
            {
                "error": "dvc CLI not found on PATH — `pip install dvc` (or the appropriate dvc[...] extra) first"
            }
        )
    if not Path(path).exists():
        return json.dumps({"error": f"no file at '{path}'"})
    result = subprocess.run(["dvc", "add", path], capture_output=True, text=True)
    if result.returncode != 0:
        hint = (
            "run `dvc init` at the repo root first"
            if "not a dvc repository" in result.stderr.lower()
            else result.stderr.strip()
        )
        return json.dumps({"error": hint})
    return json.dumps(
        {"path": path, "dvc_file": f"{path}.dvc", "stdout": result.stdout.strip()}
    )


@mcp.tool()
def generate_ci_workflow(out_path: str = ".github/workflows/ml-ci.yml") -> str:
    """Write a GitHub Actions workflow (lint + pytest + a smoke train run)
    for this agent. Overwrites out_path if it already exists — this is
    scaffolding, not something to hand-edit and expect preserved. Tell the
    user before overwriting a differing existing file."""
    workflow = """name: classification-agent CI

on:
  push:
    paths: ["classification-agent/**"]
  pull_request:
    paths: ["classification-agent/**"]

jobs:
  test:
    runs-on: ubuntu-latest
    defaults:
      run:
        working-directory: classification-agent
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
      - run: pip install -r requirements.txt
      - run: pytest scripts/ -v
"""
    dest = Path(out_path)
    existed = dest.exists()
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(workflow)
    return json.dumps({"out_path": out_path, "overwrote_existing": existed})


# ── model registry: inspect in the UI, then promote ─────────────────────────
# The loop these three serve: the agent registers a run (log_run_to_mlflow),
# the user reads the versions and their metrics in the MLflow UI, then tells
# the agent which version to promote. The agent never picks champion on its
# own — see promote_model.


@mcp.tool()
def list_model_versions(target: str = "", name: str = "") -> str:
    """Every registered version of a classification model, newest first, with
    each version's metrics and current aliases (champion/challenger).

    Pass the run's `target` column (the usual case — the registry name is
    derived from it) or an explicit registry `name`. Call this before
    promoting anything: the version numbers it returns are the same ones
    shown in the MLflow UI, so the user can point at one unambiguously."""
    name, error = _resolve_name(target, name)
    if error:
        return json.dumps({"error": error})
    listing = registry.list_versions(name)
    for row in listing.get("versions", []):
        # `readiness` is the snapshot taken at logging; this is the verdict
        # promote_model will actually enforce. Report this one.
        row["readiness_effective"] = _readiness(
            row["readiness"], row["agent_run_id"], row["pipeline_version"]
        )[0]
    return json.dumps(listing)


@mcp.tool()
def compare_model_versions(
    target: str = "", name: str = "", version_a: int = 0, version_b: int = 0
) -> str:
    """Metric deltas between two registered versions — by default the two
    newest, i.e. 'how does the latest experiment compare to the previous one'.

    Reads the MLflow registry, not the local run directories, so the
    comparison still works after `_cleanup_old_runs` has swept the older run's
    working files. Returns per-metric deltas and no verdict: which metric
    decides a promotion is the user's call, not the agent's."""
    name, error = _resolve_name(target, name)
    if error:
        return json.dumps({"error": error})
    return json.dumps(
        registry.compare(
            name,
            version_a or None,
            version_b or None,
        )
    )


@mcp.tool()
def promote_model(
    version: int,
    alias: str = "challenger",
    target: str = "",
    name: str = "",
    confirmed: bool = False,
    force: bool = False,
    reason: str = "",
    approved_by: str = "",
) -> str:
    """Point `champion` or `challenger` at a registered version.

    `challenger` promotes freely — it serves no traffic and exists to be
    evaluated. `champion` changes what serves, so it requires
    confirmed=true: ask the user in plain words which version they want as
    champion and what it displaces, and only pass confirmed=true once they
    have answered. Do not infer confirmation from the request that started
    the conversation.

    Champion also needs `reason` — the user's why, in their words (ask for
    it in the same question as the confirmation) — and takes `approved_by`
    (who confirmed, if known). Both are stored as tags on the version, with
    whether it was forced, so the registry answers "who put this in
    production, and why" without the chat log.

    Champion requires the evaluation charts (confusion matrix, ROC, PR) for
    the exact fit behind the version — nobody signs off a model unseen. On
    success the charts and a freshly generated report are logged to the
    version's MLflow run.

    Promoting to champion also EXPORTS that version to
    data/artifacts/<name>-v<N>.pkl and points data/artifacts/<name>-champion.pkl
    (the path serving loads) at it. It refuses — and leaves the alias
    untouched — if the run's pipeline is no longer the registered one (refit
    since logging) or its run directory is gone. Report `serving_path` back.

    Refuses either alias if the run behind the version was BLOCKED by the
    readiness gate, unless force=true — the same rule export_model applies,
    for the same reason: a model too defective to export is too defective to
    serve. Returns the displaced version and the exact call that rolls the
    promotion back."""
    name, error = _resolve_name(target, name)
    if error:
        return json.dumps({"error": error})
    if denied := registry.authorize(
        name, version, alias
    ):  # before anything is exported
        return json.dumps(denied)

    if alias == registry.CHAMPION and confirmed and not reason.strip():
        return json.dumps(
            {
                "error": "champion promotion needs a `reason` — it is recorded on the version as the audit trail.",
                "remedy": "ask the user why this version should serve (their words, one line), then retry with reason=...",
            }
        )
    if alias == registry.CHAMPION and not confirmed:
        return json.dumps(
            {
                "error": "champion promotion needs explicit user confirmation — it changes which model serves traffic.",
                "remedy": f"show the user list_model_versions/compare_model_versions for '{name}', ask whether to make "
                f"version {version} champion (naming what it displaces), then retry with confirmed=true.",
            }
        )

    blocked = _blocked_reason(name, version)
    if blocked and not force:
        return json.dumps(
            {
                "error": f"readiness gate BLOCKED the run behind version {version} — promoting it would "
                "serve a model with known defects.",
                "detail": blocked,
                "remedy": "promote a version whose run passed, or retry with force=true if the user explicitly "
                "accepts promoting a blocked model.",
            }
        )
    # Export BEFORE moving the alias: a champion that can't be exported is a
    # label on a model nothing serves, which is exactly the gap this closes.
    exported = (
        _export_version(name, version, force) if alias == registry.CHAMPION else None
    )
    if exported and "error" in exported:
        return json.dumps(
            {
                **exported,
                "error": f"version {version} was NOT promoted — it could not be exported "
                f"for serving: {exported['error']}",
            }
        )
    # The audit trail lives on the version, not in a chat log.
    tags = {
        "promoted_at": datetime.now(timezone.utc).isoformat(),
        "approved_by": (runtime.user() or approved_by or "user (confirmed in chat)")
        + (" (autopilot)" if runtime.autopilot() else ""),
        "reason": reason or "",
        "forced": blocked or "false",
    }
    result = registry.promote(
        name, version, alias, tags={f"{alias}.{k}": v for k, v in tags.items()}
    )
    if exported and "error" not in result:
        result["serving_path"] = registry.link_champion(name, exported["out_path"])
        result["export"] = exported
        result["logged_to_mlflow"] = _log_review_evidence(
            exported["mlflow_run_id"], exported["agent_run_id"]
        )
    if blocked:
        result["warning"] = f"promoted a BLOCKED run on force=true — {blocked}"
    return json.dumps(result)


def _sha256(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _export_version(name: str, version: int, force: bool) -> dict:
    """export_model for a REGISTERED version, to data/artifacts/<name>-v<N>.pkl.

    export_model reads the run directory, which can be refit after logging.
    The registry holds a byte copy of pipeline.pkl from logging time, so the
    two are compared: a mismatch means exporting would ship a different model
    under this version's number, and is refused."""
    client, error = registry._client()
    if error:
        return error
    import mlflow

    try:
        mlflow_run_id = client.get_model_version(name, str(version)).run_id
        agent_run_id = client.get_run(mlflow_run_id).data.tags.get("agent_run_id")
        with tempfile.TemporaryDirectory() as tmp:
            registered_hash = _sha256(
                mlflow.artifacts.download_artifacts(
                    run_id=mlflow_run_id, artifact_path="pipeline.pkl", dst_path=tmp
                )
            )
    except Exception as exc:
        return {
            "error": f"could not read version {version}'s registered pipeline ({type(exc).__name__}: {exc})"
        }
    local = _run_dir(agent_run_id) / "pipeline.pkl" if agent_run_id else None
    if local is None or not local.exists():
        return {
            "error": f"the run directory behind version {version} (agent run {agent_run_id}) is gone — "
            "nothing to export from. Re-train and register a new version."
        }
    if _sha256(local) != registered_hash:
        return {
            "error": f"agent run {agent_run_id} was refit after version {version} was registered — its "
            "pipeline.pkl is no longer the registered model. Re-log the run to register the "
            "current fit as a new version, and promote that."
        }
    charts = (
        None
        if force
        else compute_readiness(agent_run_id)["checks"]["evaluation_charts"]
    )
    if charts and charts["status"] != "pass":
        return {
            "error": f"no evaluation charts for the fit behind version {version}: {charts['evidence']}",
            "remedy": f"have the visualization agent render plot_confusion_matrix, plot_roc_curve and "
            f"plot_pr_curve for run_id {agent_run_id}, show them to the user, then retry.",
        }
    registry.ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    exported = json.loads(
        export_model(
            agent_run_id,
            str(registry.ARTIFACTS_DIR / f"{name}-v{version}.pkl"),
            force=force,
        )
    )
    return {**exported, "agent_run_id": agent_run_id, "mlflow_run_id": mlflow_run_id}


def _git_dirty() -> str:
    """'false', 'true (N files)', or 'unknown' — MLflow records the commit,
    which says nothing about uncommitted edits that actually ran."""
    try:
        out = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=Path(__file__).resolve().parents[3],
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    if out.returncode != 0:
        return "unknown"
    changed = [line for line in out.stdout.splitlines() if line.strip()]
    return f"true ({len(changed)} files)" if changed else "false"


def _test_scores(run_id: str):
    """y_true (1 = positive class) and P(positive) per test row for the
    CURRENT pipeline, or None (multiclass / no predict_proba / no model)."""
    import joblib
    import pandas as pd

    from .core import _load_split, positive_index, positive_label

    path = _run_dir(run_id) / "pipeline.pkl"
    if not path.exists():
        return None
    _, test, meta = _load_split(run_id)
    y = test[meta["target"]]
    pipeline = joblib.load(path)
    if y.nunique() != 2 or not hasattr(pipeline, "predict_proba"):
        return None
    proba = pipeline.predict_proba(test.drop(columns=[meta["target"]]))[
        :, positive_index(meta, y=y)
    ]
    return pd.DataFrame(
        {"y_true": (y == positive_label(meta, y)).astype(int), "score": proba}
    )


def _current_charts(run_id: str, meta: dict) -> list[Path]:
    """The visualization agent's charts of this run's CURRENT fit."""
    return sorted(
        (_run_dir(run_id) / "charts").glob(f"pv{meta.get('pipeline_version', 0)}_*.png")
    )


def _log_review_evidence(mlflow_run_id: str, run_id: str) -> list[str] | dict:
    """At sign-off, attach what the reviewer saw to the version's MLflow run:
    a report regenerated now (so it embeds the charts drawn after logging)
    and the charts themselves, at the same relative paths the report links."""
    client, error = registry._client()
    if error:
        return error
    generate_report(run_id)
    logged = []
    try:
        client.log_artifact(mlflow_run_id, str(_run_dir(run_id) / "report.md"))
        logged.append("report.md")
        for chart in _current_charts(run_id, _load_meta(run_id)):
            client.log_artifact(mlflow_run_id, str(chart), "charts")
            logged.append(f"charts/{chart.name}")
    except Exception as exc:
        return {
            "error": f"promoted, but logging review evidence failed ({type(exc).__name__}: {exc})",
            "logged": logged,
        }
    return logged


def _effective_readiness(
    snapshot: str | None,
    live: str | None,
    logged_pv: str | None,
    current_pv: str | None,
) -> str | None:
    """Which readiness verdict applies to a REGISTERED version.

    The logged tag is a snapshot: evidence added after logging (a baseline, a
    reflection) only shows up in the live gate. But the live gate describes
    whatever the run directory holds NOW, so trust it only while that is
    provably still the registered pipeline. When that can't be shown — refit
    since, logged before pipeline_version was recorded, run swept — take the
    worse of the two rather than guess which model a verdict belongs to."""
    if live is not None and logged_pv is not None and logged_pv == current_pv:
        return live
    return "blocked" if "blocked" in (snapshot, live) else snapshot


def _blocked_reason(name: str, version: int) -> str | None:
    """Why a registry version must not be promoted, or None. Reads the gate
    through _effective_readiness, so a version logged `ready` whose model has
    since FAILED a gate is refused — the snapshot alone would have let it
    through."""
    client, error = registry._client()
    if error:
        return None
    try:
        versions = [
            v
            for v in client.search_model_versions(f"name = '{name}'")
            if int(v.version) == version
        ]
        if not versions:
            return None
        tags = client.get_run(versions[0].run_id).data.tags
    except Exception:
        return None
    run_id = tags.get("agent_run_id")
    status, live = _readiness(
        tags.get("readiness"), run_id, tags.get("pipeline_version")
    )
    if status != "blocked":
        return None
    gates = (live or {}).get("failed_gates") or []
    return (
        f"run {run_id} failed: {gates}"
        if gates
        else "the run was blocked at logging time"
    )


def _readiness(
    snapshot: str | None, run_id: str | None, logged_pv: str | None
) -> tuple[str | None, dict | None]:
    """(effective verdict, live gate result) for a registered version — the
    one rule both promote_model and list_model_versions report from."""
    live, current_pv = None, None
    if run_id:
        try:
            current_pv = str(_load_meta(run_id).get("pipeline_version", 0))
            live = compute_readiness(run_id)
        except Exception:
            pass  # run directory swept — only the snapshot is left
    return (
        _effective_readiness(
            snapshot, live and live["overall_status"], logged_pv, current_pv
        ),
        live,
    )


@mcp.tool()
def demote_model(
    alias: str, target: str = "", name: str = "", confirmed: bool = False
) -> str:
    """Clear `champion` or `challenger` so nothing holds it.

    Use when a model is withdrawn from service or put back under review —
    "there is no champion right now" is a statement promoting some other
    version cannot make. Clearing `champion` means nothing serves, so it
    needs confirmed=true, the same bar promoting TO champion has.

    Clearing champion also removes data/artifacts/<name>-champion.pkl, so
    serving stops loading it; the versioned export stays for rollback.

    Returns the version it was pointing at and the call that restores it."""
    name, error = _resolve_name(target, name)
    if error:
        return json.dumps({"error": error})
    if alias == registry.CHAMPION and not confirmed:
        return json.dumps(
            {
                "error": "clearing champion needs explicit user confirmation — afterwards nothing serves.",
                "remedy": "confirm with the user that they want no champion, then retry with confirmed=true.",
            }
        )
    result = registry.demote(name, alias)
    if alias == registry.CHAMPION and "error" not in result:
        result["removed_serving_links"] = registry.unlink_champion(name)
    return json.dumps(result)
