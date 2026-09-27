"""MLflow tracking, registry promotion and export-for-serving for the
regression, clustering, anomaly and forecasting agents — the EDA-to-MLOps
tail every agent needs and none of them had.

One call per agent, `register_tools(...)`, adds five MCP tools to that agent:
log_run_to_mlflow, list_model_versions, compare_model_versions,
promote_model, demote_model. The loop is the classification agent's: log a
run (registers a version), compare versions, promote on the user's word.
Promoting to `champion` exports that version and points
data/artifacts/<name>-champion.pkl — the one path serving loads — at it.

ponytail: the classification agent predates this module and keeps its own
richer copy (charts gate, paired test scores); fold it onto this after a
production cycle rather than the day before one.
"""

import hashlib
import json
import re
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from . import registry, runtime

# Uploads arrive as "<uuid>_<name>.csv"; the name is what versions stack under.
_UPLOAD_PREFIX = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}_"
)
_UNSUPERVISED = ("clustering", "anomaly")


def project_of(meta: dict) -> str | None:
    """The project a run's versions stack under. A shared, team one only when a
    person named it for the chat (runtime.project, stamped on the run) — the
    model can't pick one: it invented "billing" for two strangers and stacked
    their unrelated models. Otherwise a personal one, <owner>.<dataset>, so two
    people who both upload churn.csv never share a champion."""
    if meta.get("project"):
        return meta["project"]
    dataset = _UPLOAD_PREFIX.sub("", Path(meta.get("source_path") or "").stem) or None
    return f"{meta['owner']}.{dataset}" if meta.get("owner") and dataset else dataset


def model_name(agent: str, meta: dict) -> str:
    """Registry name successive experiments stack under:
    <agent>-<project>-<target> for a supervised agent, <agent>-<project> for
    clustering/anomaly (no target to key on)."""
    target = None if agent in _UNSUPERVISED else meta.get("target")
    return registry.model_name(agent, target, project_of(meta))


def sha256(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def git_dirty() -> str:
    """'false', 'true (N files)' or 'unknown' — the commit MLflow records says
    nothing about uncommitted edits that actually ran."""
    try:
        out = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=Path(__file__).resolve().parents[1],
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


def _scalars(metrics: dict) -> dict:
    """Numeric metrics, one level of nesting flattened ("auc.roc_auc")."""
    flat = {}
    for key, value in (metrics or {}).items():
        if isinstance(value, dict):
            flat.update({f"{key}.{k}": v for k, v in value.items()})
        else:
            flat[key] = value
    return {
        k: float(v)
        for k, v in flat.items()
        if isinstance(v, (int, float)) and not isinstance(v, bool) and v == v
    }  # v == v drops NaN


try:
    import mlflow.pyfunc

    class _JoblibModel(mlflow.pyfunc.PythonModel):
        """Registrable MLflow model around the agent's own joblib artifact.
        Nothing serves from models:/ URIs here (serving loads the exported
        champion file); this exists because only a logged model can be
        registered, and registration is what numbers the versions."""

        def load_context(self, context):
            import joblib

            self.model = joblib.load(context.artifacts["model"])

        def predict(self, context, model_input, params=None):
            return self.model.predict(model_input)

except ImportError:  # the tools report it; importing an agent must not crash
    _JoblibModel = None


def use_experiment(name: str) -> str | None:
    """Make `name` the active MLflow experiment. One deleted in the MLflow UI
    is restored — MLflow refuses to log to a deleted name, and a new one
    cannot take it. Returns a note for the user when it restored one."""
    import mlflow

    mlflow.set_tracking_uri(registry.tracking_uri())
    client = mlflow.MlflowClient()
    existing = client.get_experiment_by_name(name)
    note = None
    if existing and existing.lifecycle_stage == "deleted":
        client.restore_experiment(existing.experiment_id)
        note = f"MLflow experiment '{name}' had been deleted — restored it (with its earlier runs) to log this one"
    mlflow.set_experiment(name)
    return note


def log_run(
    agent: str,
    run_id: str,
    run_dir: Path,
    meta: dict,
    model_file: str,
    metrics: dict,
    readiness: str | None = None,
    experiment: str = "",
) -> dict:
    """Log a run's model, params, metrics and lineage to MLflow and register
    it as the next version of model_name(agent, meta)."""
    try:
        import mlflow
    except ImportError:
        return {
            "error": "mlflow not installed — `pip install mlflow` to enable this tool"
        }
    model_path = run_dir / model_file
    if not model_path.exists():
        return {
            "error": "no fitted model for this run yet — call train_model first; nothing was logged"
        }
    name = model_name(agent, meta)
    source = Path(meta.get("source_path") or "")
    experiment = (
        experiment.strip() or name
    )  # one experiment per registered model: its runs are comparable
    note = use_experiment(experiment)
    with mlflow.start_run(run_name=run_id) as run:
        mlflow.log_param("model", meta.get("model") or meta.get("algorithm"))
        mlflow.log_param(
            "target", meta.get("target") or meta.get("label_column") or "none"
        )
        params = meta.get("best_params") or meta.get("algorithm_params") or {}
        if params:
            mlflow.log_params({k: str(v)[:500] for k, v in params.items()})
        for key, value in _scalars(metrics).items():
            mlflow.log_metric(re.sub(r"[^A-Za-z0-9_.\-/ ]", "_", key), value)
        mlflow.set_tags(
            {
                "agent": agent,
                "agent_run_id": run_id,  # the join promote_model uses to find the run behind a version
                "project": project_of(meta) or "",
                "owner": meta.get("owner") or runtime.user() or "",
                "readiness": readiness or "not gated",
                "model_sha256": sha256(model_path),
                "data_sha256": sha256(source) if source.is_file() else "unrecorded",
                "git_dirty": git_dirty(),
            }
        )
        mlflow.log_artifact(str(model_path))
        if (run_dir / "report.md").exists():
            mlflow.log_artifact(str(run_dir / "report.md"))
        registrable = None
        if _JoblibModel is not None:
            try:
                mlflow.pyfunc.log_model(
                    "model",
                    python_model=_JoblibModel(),
                    artifacts={"model": str(model_path)},
                    pip_requirements=["joblib"],
                )
            except Exception as exc:
                registrable = {
                    "error": f"could not log a registrable model ({type(exc).__name__}: {exc})"
                }
        mlflow_run_id = run.info.run_id
    # After the run closes: registering inside it races the artifact upload.
    registered = registrable or registry.register(mlflow_run_id, name)
    return {
        "run_id": run_id,
        "mlflow_run_id": mlflow_run_id,
        "name": name,
        "registered": registered,
        "experiment": experiment,
        "note": note,
        "tracking_uri": registry.tracking_uri(),
        "next": (
            f"Tell the user it is registered as '{name}' version {registered['version']} — the number they "
            "quote back to promote it."
            if "version" in registered
            else None
        ),
    }


def _version_run(name: str, version: int) -> tuple[str | None, str | None, dict | None]:
    """(mlflow_run_id, agent_run_id, error)."""
    client, error = registry._client()
    if error:
        return None, None, error
    try:
        mlflow_run_id = client.get_model_version(name, str(version)).run_id
        return (
            mlflow_run_id,
            client.get_run(mlflow_run_id).data.tags.get("agent_run_id"),
            None,
        )
    except Exception as exc:
        return (
            None,
            None,
            {
                "error": f"'{name}' has no readable version {version} ({type(exc).__name__}: {exc})"
            },
        )


def _export_version(
    name: str, version: int, run_dir, model_file: str, export_model, force: bool
) -> dict:
    """export_model for a REGISTERED version, to data/artifacts/<name>-v<N>.pkl
    — refused if the run directory no longer holds the model that was
    registered (refit since, or swept)."""
    import mlflow

    mlflow_run_id, agent_run_id, error = _version_run(name, version)
    if error:
        return error
    local = run_dir(agent_run_id) / model_file if agent_run_id else None
    if local is None or not local.exists():
        return {
            "error": f"the run directory behind version {version} (agent run {agent_run_id}) is gone — "
            "re-train and register a new version"
        }
    try:
        with tempfile.TemporaryDirectory() as tmp:
            registered_hash = sha256(
                mlflow.artifacts.download_artifacts(
                    run_id=mlflow_run_id, artifact_path=model_file, dst_path=tmp
                )
            )
    except Exception as exc:
        return {
            "error": f"could not read version {version}'s registered model ({type(exc).__name__}: {exc})"
        }
    if sha256(local) != registered_hash:
        return {
            "error": f"agent run {agent_run_id} was refit after version {version} was registered — exporting "
            "it would serve a different model under this version's number. Re-log and promote the new version."
        }
    registry.ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    exported = json.loads(
        export_model(
            agent_run_id,
            str(registry.ARTIFACTS_DIR / f"{name}-v{version}.pkl"),
            force=force,
        )
    )
    return {**exported, "agent_run_id": agent_run_id}


def promote(
    name: str,
    version: int,
    alias: str,
    *,
    run_dir,
    model_file: str,
    export_model,
    readiness=None,
    confirmed: bool = False,
    force: bool = False,
    reason: str = "",
    approved_by: str = "",
) -> dict:
    """The champion/challenger rules every agent shares: champion needs the
    user's confirmation AND reason; a readiness-BLOCKED run (where the agent
    has gates) needs force; champion is exported and linked for serving
    BEFORE the alias moves, so a label never points at nothing."""
    if alias not in registry.ALIASES:
        return {
            "error": f"unknown alias '{alias}' — use one of {list(registry.ALIASES)}"
        }
    if alias == registry.CHAMPION and not confirmed:
        return {
            "error": "champion promotion needs explicit user confirmation — it changes which model serves.",
            "remedy": f"show the user list_model_versions/compare_model_versions for '{name}', ask whether to "
            f"make version {version} champion (naming what it displaces), then retry with confirmed=true.",
        }
    if alias == registry.CHAMPION and not reason.strip():
        return {
            "error": "champion promotion needs a `reason` — it is recorded on the version as the audit trail.",
            "remedy": "ask the user why this version should serve (their words, one line), then retry.",
        }
    if denied := registry.authorize(
        name, version, alias
    ):  # before anything is exported
        return denied
    _, agent_run_id, error = _version_run(name, version)
    if error:
        return error
    blocked = None
    if readiness and agent_run_id and (run_dir(agent_run_id) / "meta.json").exists():
        status = readiness(agent_run_id)
        blocked = (
            f"readiness gate BLOCKED agent run {agent_run_id}"
            if status == "blocked"
            else None
        )
    if blocked and not force:
        return {
            "error": f"{blocked} — promoting it would serve a model with known defects.",
            "remedy": "promote a version whose run passed, or retry with force=true if the user explicitly accepts it.",
        }
    exported = None
    if alias == registry.CHAMPION:
        exported = _export_version(
            name, version, run_dir, model_file, export_model, force
        )
        if "error" in exported:
            return {
                **exported,
                "error": f"version {version} was NOT promoted — it could not be exported for "
                f"serving: {exported['error']}",
            }
    tags = {
        "promoted_at": datetime.now(timezone.utc).isoformat(),
        "approved_by": (runtime.user() or approved_by or "user (confirmed in chat)")
        + (" (autopilot)" if runtime.autopilot() else ""),
        "reason": reason,
        "forced": blocked or "false",
    }
    result = registry.promote(
        name, version, alias, tags={f"{alias}.{k}": v for k, v in tags.items()}
    )
    if exported and "error" not in result:
        result["serving_path"] = registry.link_champion(name, exported["out_path"])
        result["export"] = exported
    if blocked:
        result["warning"] = f"promoted a BLOCKED run on force=true — {blocked}"
    return result


def demote(name: str, alias: str, confirmed: bool = False) -> dict:
    if alias == registry.CHAMPION and not confirmed:
        return {
            "error": "clearing champion needs explicit user confirmation — afterwards nothing serves.",
            "remedy": "confirm with the user that they want no champion, then retry with confirmed=true.",
        }
    result = registry.demote(name, alias)
    if alias == registry.CHAMPION and "error" not in result:
        result["removed_serving_links"] = registry.unlink_champion(name)
    return result


def write_monitoring_profile(
    out_path: str,
    run_id: str,
    train,
    target: str | None,
    dropped,
    importances,
    extra: dict | None = None,
) -> str:
    """The drift baseline beside an exported model: the training fold (which
    outlives the swept run directory), the raw columns the model reads, and
    the ones its explainer ranked — mapped back from encoded names (Sex_male
    -> Sex), or they match no raw column and silently stop counting."""
    reference_path = Path(out_path).with_suffix(".reference.csv")
    train.to_csv(reference_path, index=False)
    features = [c for c in train.columns if c != target and c not in set(dropped or [])]
    key_columns = []
    for feature, _ in importances or []:
        raw = max(
            (c for c in features if feature == c or str(feature).startswith(f"{c}_")),
            key=len,
            default=None,
        )
        if raw and raw not in key_columns:
            key_columns.append(raw)
    profile_path = Path(out_path).with_suffix(".monitoring.json")
    profile_path.write_text(
        json.dumps(
            {
                "model_path": str(out_path),
                "run_id": run_id,
                "exported_at": datetime.now(timezone.utc).isoformat(),
                "reference_path": str(reference_path),
                "reference_rows": len(train),
                "target": target,
                "feature_columns": features,
                "key_columns": key_columns[:10],
                "key_columns_source": (
                    "explain_model"
                    if key_columns
                    else "not available — explain_model was never run"
                ),
                "predictions_log": str(
                    registry.ARTIFACTS_DIR.parent / "predictions" / f"{run_id}.csv"
                ),
                **(extra or {}),
            },
            indent=2,
            default=str,
        )
    )
    return str(profile_path)


def register_tools(
    mcp,
    agent: str,
    *,
    load_meta,
    save_meta,
    run_dir,
    model_file: str,
    export_model,
    metrics_of,
    readiness=None,
) -> tuple:
    """Add log_run_to_mlflow / list_model_versions / compare_model_versions /
    promote_model / demote_model to an agent's MCP server, and return them in
    that order so the agent binds them as module attributes like its other
    tools (tests and callers reach them the same way)."""

    def _name(name: str, run_id: str) -> str | dict:
        if name:
            return name
        if not run_id:
            return {
                "error": "pass either name (as list_model_versions shows it) or a run_id of this agent"
            }
        return model_name(agent, load_meta(run_id))

    @mcp.tool()
    def log_run_to_mlflow(run_id: str, experiment: str = "") -> str:
        """Log this run's model, params, metrics and lineage (data hash, git
        dirty flag, model hash) to MLflow and register it as the next VERSION
        of this dataset's/target's model. Tell the user the version number —
        it is what they quote back to promote. Pins the run against cleanup.
        experiment: only when the user names one (created if new); by
        default it is the registered model name, <agent>-<project>-<target>."""
        meta = load_meta(run_id)
        result = log_run(
            agent,
            run_id,
            run_dir(run_id),
            meta,
            model_file,
            metrics_of(meta),
            readiness(run_id) if readiness else None,
            experiment,
        )
        if "version" in (result.get("registered") or {}):
            meta = load_meta(run_id)
            meta["registry"] = {
                "name": result["name"],
                "version": result["registered"]["version"],
            }
            save_meta(run_id, meta)
        return json.dumps(result, default=str)

    @mcp.tool()
    def list_model_versions(name: str = "", run_id: str = "") -> str:
        """Every registered version of a model, newest first, with metrics,
        params, aliases (champion/challenger) and the promotion audit tags.
        Pass the registry `name`, or a `run_id` to use that run's model name."""
        resolved = _name(name, run_id)
        return json.dumps(
            (
                resolved
                if isinstance(resolved, dict)
                else registry.list_versions(resolved)
            ),
            default=str,
        )

    @mcp.tool()
    def compare_model_versions(
        name: str = "", run_id: str = "", version_a: int = 0, version_b: int = 0
    ) -> str:
        """Signed metric deltas between two versions (default: newest vs
        previous) and what changed between them. No verdict: which metric
        justifies a promotion is the user's call."""
        resolved = _name(name, run_id)
        if isinstance(resolved, dict):
            return json.dumps(resolved)
        return json.dumps(
            registry.compare(resolved, version_a or None, version_b or None),
            default=str,
        )

    @mcp.tool()
    def promote_model(
        version: int,
        alias: str = "challenger",
        name: str = "",
        run_id: str = "",
        confirmed: bool = False,
        force: bool = False,
        reason: str = "",
        approved_by: str = "",
    ) -> str:
        """Point `champion` or `challenger` at a registered version.
        challenger is free. champion needs confirmed=true AND the user's
        `reason` (ask in plain words, naming what it displaces) — and exports
        the version to data/artifacts/<name>-v<N>.pkl, pointing
        data/artifacts/<name>-champion.pkl (what serving loads) at it; relay
        `serving_path` and the `rollback` call."""
        resolved = _name(name, run_id)
        if isinstance(resolved, dict):
            return json.dumps(resolved)
        return json.dumps(
            promote(
                resolved,
                version,
                alias,
                run_dir=run_dir,
                model_file=model_file,
                export_model=export_model,
                readiness=readiness,
                confirmed=confirmed,
                force=force,
                reason=reason,
                approved_by=approved_by,
            ),
            default=str,
        )

    @mcp.tool()
    def demote_model(
        alias: str, name: str = "", run_id: str = "", confirmed: bool = False
    ) -> str:
        """Clear `champion` or `challenger`. Clearing champion (confirmed=true
        required) also removes the champion link, so nothing serves until a
        new champion is promoted; the versioned export stays for rollback."""
        resolved = _name(name, run_id)
        if isinstance(resolved, dict):
            return json.dumps(resolved)
        return json.dumps(demote(resolved, alias, confirmed), default=str)

    return (
        log_run_to_mlflow,
        list_model_versions,
        compare_model_versions,
        promote_model,
        demote_model,
    )
