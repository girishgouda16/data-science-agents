"""MLflow model registry: register a run's model, then promote a version to
`champion` / `challenger` once a human has looked at it in the MLflow UI.

Split from any one agent because the registry is cross-cutting — classification,
regression, clustering, anomaly and forecasting all produce models that get
registered and promoted the same way. Only the thin `@mcp.tool()` wrappers live
per agent; everything that talks to MLflow lives here, next to `tracing`/`gates`
for the same reason those do.

Two deliberate choices:

  * **Aliases, not stages.** MLflow's stage API (`transition_model_version_stage`)
    is deprecated. `set_registered_model_alias` is the supported mechanism and,
    unlike stages, an alias is visible and editable in the UI — so the version a
    human inspects and the version an agent promotes are the same object, which
    is the whole point of the inspect-then-promote loop.

  * **A stable registered-model name per (agent, project, target).** Versions
    only accumulate — and only become comparable — if successive experiments
    land under one name: `classification-retention-churned` gets v1, v2, v3 as
    the agent re-runs. The project keeps two teams that both predict
    `churned` from stacking unrelated models (different features) under one
    champion — where promoting one would swap the other's serving model.

  * **Owned versions.** Each version records the user who registered it; an
    alias only moves away from, or onto, a version the caller owns (or an
    admin in AGENTIC_ML_ADMINS). See core/runtime.py for where the user
    comes from.

Every function returns a plain dict (the agent layer JSON-encodes it) and turns
a missing/misconfigured MLflow into an `{"error": ...}` rather than an
exception, matching how the agents' other infra tools degrade.
"""

import contextlib
import fcntl
import logging
import os
import re
from datetime import datetime, timezone
from pathlib import Path

from . import runtime

logger = logging.getLogger(__name__)

# The two aliases this repo promotes to. `champion` is the version meant to
# serve traffic; `challenger` is the candidate being evaluated against it.
# Anything else is rejected rather than silently created, because a typo'd
# alias ("champoin") would look like a successful promotion and serve nothing.
#
# The alias alone serves nothing: serving/drift/explain load .pkl FILES, never
# models:/ URIs. A champion only reaches production through champion_path()
# below, which the agent layer points at the exported version on promotion.
CHAMPION = "champion"
CHALLENGER = "challenger"
ALIASES = (CHAMPION, CHALLENGER)

# Where every agent's export lands and where serving looks by default.
ARTIFACTS_DIR = (
    Path(
        os.environ.get("AGENTIC_ML_DATA_DIR")
        or Path(__file__).resolve().parents[1] / "data"
    )
    / "artifacts"
)
# The champion's model and its drift profile — export_model writes both side by side.
_CHAMPION_FILES = (".pkl", ".monitoring.json")

# Registered-model names are used in URLs and as MLflow identifiers; keep them
# to a conservative charset instead of trusting a target column name.
_UNSAFE = re.compile(r"[^A-Za-z0-9_.-]+")


def tracking_uri() -> str:
    """The shared file store at the repo root.

    Anchored to this file, never Path.cwd(): agents are launched from their own
    directories, and a cwd-relative store would give each one a private mlruns/
    — every agent would register into a registry none of the others could see.
    """
    if os.getenv("MLFLOW_TRACKING_URI"):
        return os.environ["MLFLOW_TRACKING_URI"]
    return f"file:{Path(__file__).resolve().parents[1] / 'mlruns'}"


def model_name(agent: str, target: str | None, project: str | None = None) -> str:
    """The stable registry name for an (agent, project, target) — the thing
    successive experiments stack up as versions of."""
    parts = [agent] + [_UNSAFE.sub("_", p).strip("_-") for p in (project, target) if p]
    return "-".join(p for p in parts if p) or agent


def find_names(agent: str, target: str) -> list[str]:
    """Registered names of `agent` ending in this target — one per project,
    plus a pre-project `<agent>-<target>`. For callers that only know the target."""
    client, error = _client()
    if error or not target:
        return []
    suffix = "-" + _UNSAFE.sub("_", target).strip("_-")
    try:
        models = client.search_registered_models(
            filter_string=f"name LIKE '{agent}-%'", max_results=1000
        )
    except Exception:
        return []
    return sorted(m.name for m in models if m.name.endswith(suffix))


@contextlib.contextmanager
def _write_lock():
    """Serialize registry writes across agent processes. The file store numbers
    a new version by listing what exists, so two agents registering the same
    name at once could both take version N. A tracking server (http/database
    URI) serializes its own writes; this covers the default file store.
    ponytail: host-local flock — several hosts need a tracking server anyway."""
    uri = tracking_uri()
    if not uri.startswith("file:"):
        yield
        return
    store = Path(uri[len("file:") :])
    store.mkdir(parents=True, exist_ok=True)
    with open(store / ".registry.lock", "w") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def _owner(row: dict | None) -> str | None:
    return (row or {}).get("version_tags", {}).get("owner")


def authorize(name: str, version: int | None, alias: str) -> dict | None:
    """{"error": ...} unless the caller may point `alias` at `version` (None =
    just clear it): they must own the version, and the version the alias
    leaves. Moving a colleague's champion is a team lead's call (admin)."""
    listing = list_versions(name, limit=1000)
    if "error" in listing:
        return None  # nothing registered to protect; the write itself reports it
    rows = {row["version"]: row for row in listing["versions"]}
    holder = next((row for row in listing["versions"] if alias in row["aliases"]), None)
    for row, what in (
        (rows.get(version), f"'{name}' version {version}"),
        (
            holder,
            f"the version '{name}' @{alias} points at now (v{(holder or {}).get('version')})",
        ),
    ):
        if row is not None and (denied := runtime.access_error(_owner(row), what)):
            return {"error": denied}
    return None


def _client():
    """(client, error). Import is deferred so an agent without mlflow installed
    still starts — the tool reports it, nothing crashes at import time."""
    try:
        import mlflow
        from mlflow.tracking import MlflowClient
    except ImportError:
        return None, {
            "error": "mlflow not installed — `pip install mlflow` to enable the model registry"
        }
    mlflow.set_tracking_uri(tracking_uri())
    return MlflowClient(tracking_uri=tracking_uri()), None


def _aliases_of(version) -> list[str]:
    """Which aliases point at this version. `aliases` appears on ModelVersion
    only in newer MLflow builds, so this falls back rather than raising."""
    return list(getattr(version, "aliases", None) or [])


def list_models() -> dict:
    """Every registered model with its newest version and where its aliases
    point — the team's registry at a glance (the UI's Models view)."""
    client, error = _client()
    if error:
        return error
    try:
        models = client.search_registered_models()
    except Exception as exc:
        return {"error": f"could not read the registry ({type(exc).__name__}: {exc})"}
    rows = []
    for m in models:
        latest = max((int(v.version) for v in (m.latest_versions or [])), default=None)
        rows.append(
            {
                "name": m.name,
                "latest_version": latest,
                "aliases": {
                    alias: int(version)
                    for alias, version in (getattr(m, "aliases", None) or {}).items()
                },
                "updated": getattr(m, "last_updated_timestamp", None),
            }
        )
    return {
        "models": sorted(rows, key=lambda r: r["updated"] or 0, reverse=True),
        "tracking_uri": tracking_uri(),
    }


def list_versions(name: str, limit: int = 20) -> dict:
    """Every registered version of `name`, newest first, with its metrics and
    current aliases — what the user reads before deciding what to promote."""
    client, error = _client()
    if error:
        return error
    try:
        versions = client.search_model_versions(f"name = '{name}'")
    except Exception as exc:
        return {
            "error": f"could not read registry for '{name}' ({type(exc).__name__}: {exc})"
        }
    if not versions:
        return {
            "error": f"no registered model named '{name}' — has a run been registered yet?",
            "registered_models": [m.name for m in client.search_registered_models()][
                :20
            ],
        }

    rows = []
    for version in sorted(versions, key=lambda v: int(v.version), reverse=True)[:limit]:
        # params/tags as well as metrics: "explain both experiments" needs to
        # say WHAT each one was (algorithm, hyperparameters, whether its
        # readiness gate passed), not just how it scored.
        metrics, params, tags = {}, {}, {}
        try:
            data = client.get_run(version.run_id).data
            metrics, params, tags = data.metrics, data.params, data.tags
        except Exception:
            # A registered version whose run was deleted still exists and is
            # still promotable; it just has no metrics to show.
            pass
        rows.append(
            {
                "version": int(version.version),
                "run_id": version.run_id,
                "aliases": _aliases_of(version),
                "metrics": metrics,
                "params": params,
                "model": params.get("model"),
                "target": params.get("target"),
                "readiness": tags.get("readiness"),
                "agent_run_id": tags.get("agent_run_id"),
                "pipeline_version": tags.get("pipeline_version"),
                "test_fingerprint": tags.get("test_fingerprint"),
                # Promotion audit trail (who, why, forced) — set by promote().
                "version_tags": dict(getattr(version, "tags", None) or {}),
                "created": getattr(version, "creation_timestamp", None),
            }
        )
    return {"name": name, "versions": rows, "tracking_uri": tracking_uri()}


def register(run_id: str, name: str, model_uri: str | None = None) -> dict:
    """Register an MLflow run's logged model under `name`, returning the new
    version number. That number is the handle the user reads in the UI and
    quotes back to promote, so it is what callers must surface."""
    client, error = _client()
    if error:
        return error
    import mlflow

    try:
        with _write_lock():
            version = mlflow.register_model(model_uri or f"runs:/{run_id}/model", name)
            if runtime.user():
                client.set_model_version_tag(
                    name, version.version, "owner", runtime.user()
                )
    except Exception as exc:
        return {"error": f"registration failed ({type(exc).__name__}: {exc})"}
    return {
        "name": name,
        "version": int(version.version),
        "run_id": run_id,
        "tracking_uri": tracking_uri(),
    }


def compare(name: str, v1: int | None = None, v2: int | None = None) -> dict:
    """Metric deltas between two versions — by default the two newest, which is
    the "latest experiment vs the previous one" question.

    Reads the registry rather than the agents' local run directories on purpose:
    those are swept after RUN_RETENTION_DAYS, so a local comparison silently
    loses history the registry still has.
    """
    listing = list_versions(name)
    if "error" in listing:
        return listing
    by_version = {row["version"]: row for row in listing["versions"]}
    if v1 is None or v2 is None:
        newest = sorted(by_version, reverse=True)[:2]
        if len(newest) < 2:
            return {
                "error": f"'{name}' has only {len(newest)} version — nothing to compare it against"
            }
        v1, v2 = newest[0], newest[1]  # v1 = candidate (newer), v2 = incumbent
    missing = [v for v in (v1, v2) if v not in by_version]
    if missing:
        return {
            "error": f"no such version(s) of '{name}': {missing}",
            "available": sorted(by_version, reverse=True),
        }

    a, b = by_version[v1], by_version[v2]
    # What actually CHANGED between the two experiments — the thing a user
    # asking "why is v2 different" wants, and the thing a metric delta alone
    # never says.
    changed_params = {
        key: {"candidate": a["params"].get(key), "incumbent": b["params"].get(key)}
        for key in set(a["params"]) | set(b["params"])
        if a["params"].get(key) != b["params"].get(key)
    }
    deltas = {
        metric: {
            "candidate": a["metrics"][metric],
            "incumbent": b["metrics"].get(metric),
            "delta": round(a["metrics"][metric] - b["metrics"][metric], 6),
        }
        for metric in a["metrics"]
        if metric in b["metrics"]
    }
    # A delta is only a finding if it is bigger than the noise of the rows it
    # was measured on — v1->v2's +0.0145 ROC-AUC on 179 rows is not.
    fingerprints = {a.get("test_fingerprint"), b.get("test_fingerprint")} - {
        None,
        "unrecorded",
    }
    paired = (
        _paired_delta_ci(a["run_id"], b["run_id"]) if len(fingerprints) <= 1 else None
    )
    for metric, interval in (paired or {}).items():
        if metric in deltas:
            deltas[metric].update(interval)
    return {
        "name": name,
        "comparability": (
            "same test rows — delta_ci is a paired 95% bootstrap interval; a delta whose interval spans 0 is "
            "not a demonstrated difference"
            if paired
            else "no per-row scores for the same test rows on both versions — deltas are bare point differences with "
            "no interval; do not promote on a small one. Compare cv_score where both have it."
        ),
        "candidate": {
            "version": v1,
            "aliases": a["aliases"],
            "run_id": a["run_id"],
            "model": a["model"],
            "readiness": a["readiness"],
            "params": a["params"],
        },
        "incumbent": {
            "version": v2,
            "aliases": b["aliases"],
            "run_id": b["run_id"],
            "model": b["model"],
            "readiness": b["readiness"],
            "params": b["params"],
        },
        "metrics": deltas,
        "changed_params": changed_params,
        # Deliberately no verdict: "better" depends on which metric the user
        # cares about, and a single-number winner is how a model gets promoted
        # on an improvement that did not matter.
        "note": "deltas are candidate minus incumbent; higher is better for auc/precision/recall/f1, "
        "lower is better for error metrics (mae/rmse/mape).",
    }


def _paired_delta_ci(run_a: str, run_b: str, n_boot: int = 1000) -> dict | None:
    """95% paired-bootstrap interval on (a - b) for ROC-AUC and PR-AUC, when
    both MLflow runs logged per-row `test_scores.csv` for the SAME test rows
    (identical y_true sequence). None otherwise — versions scored on
    different rows can't be differenced row by row."""
    import tempfile

    try:
        import mlflow
        import numpy as np
        import pandas as pd
        from sklearn.metrics import average_precision_score, roc_auc_score

        with tempfile.TemporaryDirectory() as tmp:
            a, b = (
                pd.read_csv(
                    mlflow.artifacts.download_artifacts(
                        run_id=run,
                        artifact_path="test_scores.csv",
                        dst_path=f"{tmp}/{i}",
                    )
                )
                for i, run in enumerate((run_a, run_b))
            )
    except Exception:
        return None
    if len(a) != len(b) or not (a["y_true"].to_numpy() == b["y_true"].to_numpy()).all():
        return None
    y, sa, sb = a["y_true"].to_numpy(), a["score"].to_numpy(), b["score"].to_numpy()
    rng, out = np.random.default_rng(42), {}
    for metric, fn in (("roc_auc", roc_auc_score), ("pr_auc", average_precision_score)):
        diffs = []
        for _ in range(n_boot):
            idx = rng.integers(0, len(y), len(y))
            if y[idx].min() != y[idx].max():
                diffs.append(fn(y[idx], sa[idx]) - fn(y[idx], sb[idx]))
        lo, hi = (round(float(v), 4) for v in np.percentile(diffs, [2.5, 97.5]))
        out[metric] = {"delta_ci": [lo, hi], "distinguishable": bool(lo > 0 or hi < 0)}
    return out


def promote(
    name: str, version: int, alias: str = CHALLENGER, tags: dict | None = None
) -> dict:
    """Point `alias` at `version`, then write `tags` onto that version (the
    caller's audit trail: who approved, why, whether it was forced).

    Champion is not gated here — the gating is the caller's, because only the
    agent layer knows whether the run behind this version passed its readiness
    checks and whether the user confirmed. This function does the registry
    write and reports what it displaced.
    """
    if alias not in ALIASES:
        return {"error": f"unknown alias '{alias}' — use one of {list(ALIASES)}"}
    client, error = _client()
    if error:
        return error

    listing = list_versions(name)
    if "error" in listing:
        return listing
    available = {row["version"] for row in listing["versions"]}
    if version not in available:
        return {
            "error": f"'{name}' has no version {version}",
            "available": sorted(available, reverse=True),
        }
    if denied := authorize(name, version, alias):
        return denied

    # What the alias pointed at before, so a promotion is reversible without
    # digging through MLflow's history to find what was displaced.
    previous = next(
        (row["version"] for row in listing["versions"] if alias in row["aliases"]), None
    )
    try:
        with _write_lock():
            client.set_registered_model_alias(name, alias, str(version))
            for key, value in (tags or {}).items():
                client.set_model_version_tag(name, str(version), key, str(value))
    except Exception as exc:
        return {"error": f"could not set alias '{alias}' ({type(exc).__name__}: {exc})"}
    return {
        "name": name,
        "alias": alias,
        "version": version,
        "previous_version": previous,
        "model_uri": f"models:/{name}@{alias}",
        "rollback": (
            None
            if previous is None
            else f"promote_model('{name}', {previous}, '{alias}')"
        ),
    }


def demote(name: str, alias: str) -> dict:
    """Clear an alias entirely, so nothing is champion/challenger any more.

    Needed because promotion alone cannot express "this is no longer the one":
    pointing champion at some other version is not the same statement as
    withdrawing it, and a model under review should have no champion rather
    than a stale one.
    """
    if alias not in ALIASES:
        return {"error": f"unknown alias '{alias}' — use one of {list(ALIASES)}"}
    client, error = _client()
    if error:
        return error
    listing = list_versions(name)
    if "error" in listing:
        return listing
    previous = next(
        (row["version"] for row in listing["versions"] if alias in row["aliases"]), None
    )
    if previous is None:
        return {
            "name": name,
            "alias": alias,
            "was_pointing_at": None,
            "note": f"'{alias}' was not set on '{name}' — nothing to clear",
        }
    if denied := authorize(name, None, alias):
        return denied
    try:
        with _write_lock():
            client.delete_registered_model_alias(name, alias)
        # The version's own record says when it stopped holding the alias.
        client.set_model_version_tag(
            name,
            str(previous),
            f"{alias}.demoted_at",
            datetime.now(timezone.utc).isoformat(),
        )
    except Exception as exc:
        return {
            "error": f"could not clear alias '{alias}' ({type(exc).__name__}: {exc})"
        }
    return {
        "name": name,
        "alias": alias,
        "was_pointing_at": previous,
        "restore": f"promote_model('{name}', {previous}, '{alias}')",
    }


def delete_model(name: str) -> dict:
    """Delete a registered model and all its versions — only if the caller owns
    every version (or is an ml-admin), and never while it has a champion:
    serving loads that one. Its MLflow runs and experiment stay."""
    listing = list_versions(name, limit=1000)
    if "error" in listing:
        return {"error": listing["error"], "status": 404}
    if any(CHAMPION in row["aliases"] for row in listing["versions"]):
        return {
            "error": f"'{name}' has a champion that serving loads — demote it in a chat first",
            "status": 409,
        }
    for row in listing["versions"]:
        if denied := runtime.access_error(_owner(row), f"'{name}' v{row['version']}"):
            return {"error": denied, "status": 403}
    client, error = _client()
    if error:
        return error
    try:
        with _write_lock():
            client.delete_registered_model(name)
    except Exception as exc:
        return {"error": f"could not delete '{name}' ({type(exc).__name__}: {exc})"}
    return {"deleted": name}


def champion_path(name: str) -> Path:
    """The one stable path serving loads for `name`'s champion — so promoting
    a version changes what serves without anyone re-pointing serving."""
    return ARTIFACTS_DIR / f"{name}-champion.pkl"


def link_champion(name: str, exported_pkl: str) -> str:
    """Point champion_path(name) (and its .monitoring.json) at an exported
    version's files. Each symlink is swapped with os.replace, so a reader never
    sees a missing champion mid-promotion. Relative links, so moving the repo
    (or unpacking it elsewhere) keeps them valid."""
    source = Path(exported_pkl)
    for suffix in _CHAMPION_FILES:
        link = champion_path(name).with_suffix(suffix)
        if not source.with_suffix(suffix).exists():
            # e.g. a forecast model exports no drift profile — and the previous
            # champion's must not keep answering for this one.
            if link.is_symlink():
                link.unlink()
            continue
        tmp = link.with_name(link.name + ".tmp")
        tmp.unlink(missing_ok=True)
        tmp.symlink_to(os.path.relpath(source.with_suffix(suffix), link.parent))
        os.replace(tmp, link)
    return str(champion_path(name))


def unlink_champion(name: str) -> list[str]:
    """Remove the champion links, so a withdrawn champion stops serving. The
    versioned files they pointed at stay — that is what rollback re-links."""
    removed = []
    for suffix in _CHAMPION_FILES:
        link = champion_path(name).with_suffix(suffix)
        if link.is_symlink():
            link.unlink()
            removed.append(str(link))
    return removed
