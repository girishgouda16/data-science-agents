"""What ML runs exist on disk and how far each got — read from the run
folders every training agent writes, not from conversation history. Used by
the orchestrator's inspect_runs tool (routing) and the gateway's Runs view.

Only the asking user's runs are listed (admins see all): with many data
scientists on one box, "the newest run" is otherwise a colleague's.
"""

import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path

from . import registry, runtime, toolguard


def data_dir() -> Path:
    return Path(
        os.environ.get("AGENTIC_ML_DATA_DIR")
        or Path(__file__).resolve().parents[1] / "data"
    )


def run_phase(meta: dict) -> tuple[str, str]:
    """Where a run has got to, and the single next step that follows from
    that. Deliberately coarse: this decides ROUTING, not method."""
    if meta.get("exported_path"):
        if meta.get("export_readiness") == "blocked":
            return "exported_blocked", (
                "exported past a BLOCKED readiness gate (force=true) — do not serve it unless the user "
                "explicitly accepts that; say which gates failed"
            )
        return (
            "exported",
            "serve it (serving-routing) or set up monitoring (drift-routing)",
        )
    if meta.get("model"):
        tuned = meta.get("best_params") is not None
        if meta.get("operating_point"):
            return "threshold_tuned", "generate the report, then export"
        return ("tuned" if tuned else "trained"), (
            "check calibration and tune a decision threshold, explain the model, then report"
        )
    if meta.get("target"):
        return "prepared", "train a model"
    return "unknown", "ask the owning agent what state this run is in"


def list_runs(run_id: str = "", limit: int = 5) -> dict:
    """Grounded answer to 'what have we actually got?'."""
    runs_dir, artifacts_dir = data_dir() / "runs", data_dir() / "artifacts"
    if not runs_dir.exists():
        return {
            "runs": [],
            "note": "no runs directory yet — nothing has been trained on this machine",
        }
    dirs = [d for d in runs_dir.iterdir() if d.is_dir() and (d / "meta.json").exists()]
    if run_id:
        dirs = [d for d in dirs if d.name == run_id]
        if not dirs:
            return {
                "error": f"no run '{run_id}' on disk — it may have been swept (runs are disposable "
                "working state with a retention window) or it never existed"
            }
    dirs.sort(key=lambda d: (d / "meta.json").stat().st_mtime, reverse=True)

    runs = []
    user = runtime.user()
    for d in dirs:
        if len(runs) >= limit:
            break
        try:
            meta = json.loads((d / "meta.json").read_text())
        except (json.JSONDecodeError, OSError):
            continue
        # Only the asking user's runs: with many data scientists on one box,
        # "the newest run" is otherwise a colleague's, and routing continues it.
        if user and not runtime.is_admin() and meta.get("owner") != user:
            continue
        phase, next_step = run_phase(meta)
        bu = meta.get("business_understanding") or {}
        runs.append(
            {
                "run_id": d.name,
                "phase": phase,
                "suggested_next": next_step,
                "target": meta.get("target"),
                "positive_label": meta.get("positive_label"),
                "model": meta.get("model"),
                "domain": bu.get("domain"),
                "source_path": meta.get("source_path"),
                "has_report": (d / "report.md").exists(),
                "owner": meta.get("owner"),
                "registry": meta.get("registry"),
                "last_touched": datetime.fromtimestamp(
                    (d / "meta.json").stat().st_mtime, tz=timezone.utc
                ).isoformat(),
                # Which agent owns it, so a follow-up is routed to the one that
                # can actually act on this run_id rather than to whichever agent
                # was spoken to most recently.
                "owning_agent": (
                    "classification"
                    if "target" in meta and "algorithm" not in meta
                    else "another ML agent"
                ),
            }
        )
    profiles = (
        sorted(artifacts_dir.glob("*.monitoring.json"))
        if artifacts_dir.exists()
        else []
    )
    return {
        "runs": runs,
        "monitoring_profiles": [str(p) for p in profiles[:limit]],
        "note": "phase/next-step are read from each run's meta.json on disk, not from conversation history — "
        "trust these over anything an agent said earlier about where a run got to.",
    }


def delete_run(run_id: str) -> dict:
    """Remove a run's folder — the caller's own, or any for an ml-admin. A run
    behind a registered version stays: promoting that version exports from it."""
    folder = data_dir() / "runs" / run_id
    if not toolguard._RUN_ID.match(run_id) or not (folder / "meta.json").is_file():
        return {"error": f"no run '{run_id}'", "status": 404}
    meta = json.loads((folder / "meta.json").read_text())
    if denied := runtime.access_error(meta.get("owner")):
        return {"error": denied, "status": 403}
    reg = meta.get("registry") or {}
    versions = (
        registry.list_versions(reg["name"], limit=1000).get("versions", [])
        if reg.get("name")
        else []
    )
    if any(v["version"] == reg.get("version") for v in versions):
        return {
            "error": f"run {run_id[:8]} backs {reg['name']} v{reg['version']} — delete that model first",
            "status": 409,
        }
    shutil.rmtree(folder)
    return {"deleted": run_id}
