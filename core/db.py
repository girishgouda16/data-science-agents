"""
core/db.py

Persistence for completed training runs:
  - metadata + hold-out predictions → SQL, via SQLAlchemy Core (no ORM —
    this is write-mostly, no query layer built on top yet).
  - fitted model pipeline (.pkl)    → local disk, via joblib.

DATABASE_URL controls the metadata backend (see core/config.py):
  - unset / default → local SQLite file (data/training_runs.db). Zero setup,
    good enough for a demo or single-machine dev box.
  - postgresql://...  → same tables, same code path — just point
    DATABASE_URL at a real Postgres instance.

ARTIFACTS_DIR controls where .pkl files land (default data/artifacts/) —
one per run, at <ARTIFACTS_DIR>/<session_id>/<task_id>/model.pkl, path
recorded in training_runs.artifact_path.

Every write is best-effort — a DB or disk hiccup must never fail a training
run, so callers wrap save_training_run() and let it log-and-continue on error.

a2a_tasks.artifacts is a JSON blob and doubles as the durable home for the
agent's decision transcript (which tools it called, what the validator
rejected, what it decided at each gate) — no separate schema needed for that;
see agents/classification/a2a.py's use of `_data_part({"agent_transcript": ...})`.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
from sqlalchemy import (
    JSON,
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    MetaData,
    String,
    Table,
    create_engine,
)
from sqlalchemy.engine import Engine

from core.config import settings

logger = logging.getLogger(__name__)

metadata = MetaData()

training_runs = Table(
    "training_runs",
    metadata,
    Column("id", Integer, primary_key=True),
    Column("session_id", String, nullable=False, index=True),
    Column("task_id", String, nullable=False, index=True),
    Column("target_column", String, nullable=False),
    Column("best_model", String, nullable=False),
    Column("metric", String, nullable=False),
    Column("best_score", Float, nullable=False),
    Column("test_auc", Float),
    Column("cv_to_test_gap", Float),
    Column("overfitting_warning", Boolean),
    Column("all_model_scores", JSON),
    Column("best_params", JSON),
    Column("feature_importance", JSON),
    Column("label_classes", JSON),
    Column("artifact_path", String),
    Column("created_at", DateTime(timezone=True), nullable=False),
)

predictions = Table(
    "predictions",
    metadata,
    Column("id", Integer, primary_key=True),
    Column(
        "run_id", Integer, ForeignKey("training_runs.id"), nullable=False, index=True
    ),
    Column("row_index", Integer, nullable=False),
    Column("y_true", String, nullable=False),
    Column("y_pred", String, nullable=False),
)

# A2A task lifecycle — survives a restart, unlike a plain in-process dict.
# One row per A2A task_id (see agents/classification/a2a.py); state is the
# A2A task state ("working" | "input-required" | "completed" | "failed").
a2a_tasks = Table(
    "a2a_tasks",
    metadata,
    Column("task_id", String, primary_key=True),
    Column("session_id", String, index=True),
    Column("state", String, nullable=False),
    Column("artifacts", JSON),
    Column("created_at", DateTime(timezone=True), nullable=False),
    Column("updated_at", DateTime(timezone=True), nullable=False),
)


def save_model_artifact(session_id: str, task_id: str, pipeline: Any) -> str:
    """
    joblib-dump a fitted sklearn Pipeline to data/artifacts/<session_id>/<task_id>/model.pkl
    (path configurable via ARTIFACTS_DIR). Returns the path for storage in
    training_runs.artifact_path.
    """
    run_dir = Path(settings.artifacts_dir) / session_id / task_id
    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / "model.pkl"
    joblib.dump(pipeline, path)
    return str(path)


_engine: Engine | None = None


def get_engine() -> Engine:
    global _engine
    if _engine is None:
        _engine = create_engine(settings.database_url, pool_pre_ping=True)
    return _engine


def init_db() -> None:
    """
    Verify the schema is migrated, called once at app startup.

    Schema changes are owned by Alembic (`alembic upgrade head`), not the
    app process — running DDL from every app instance on boot races under
    multiple workers/replicas. This only checks readiness and logs loudly
    if someone forgot to run migrations.
    """
    engine = get_engine()
    try:
        with engine.connect() as conn:
            conn.execute(training_runs.select().limit(1))
            conn.execute(a2a_tasks.select().limit(1))
    except Exception:
        logger.exception(
            "Schema not migrated — run `alembic upgrade head` "
            "before starting the app (%s)",
            engine.url,
        )
        raise
    logger.info("Training run persistence ready (%s)", engine.url)


def save_training_run(
    session_id: str,
    task_id: str,
    target_column: str,
    result: Any,  # agents.classification.pipeline.TrainingResult
    artifact_path: str | None = None,
) -> None:
    """Persist run metadata + hold-out predictions."""
    engine = get_engine()

    with engine.begin() as conn:
        run_id = conn.execute(
            training_runs.insert().values(
                session_id=session_id,
                task_id=task_id,
                target_column=target_column,
                best_model=result.best_model_name,
                metric=result.metric,
                best_score=result.best_score,
                test_auc=result.test_auc,
                cv_to_test_gap=result.cv_to_test_gap,
                overfitting_warning=result.overfitting_warning,
                all_model_scores=result.all_model_scores,
                best_params=result.best_params,
                feature_importance=result.feature_importance,
                label_classes=result.label_classes,
                artifact_path=artifact_path,
                created_at=datetime.now(timezone.utc),
            )
        ).inserted_primary_key[0]

        if result.y_true and result.y_pred:
            conn.execute(
                predictions.insert(),
                [
                    {
                        "run_id": run_id,
                        "row_index": i,
                        "y_true": str(yt),
                        "y_pred": str(yp),
                    }
                    for i, (yt, yp) in enumerate(zip(result.y_true, result.y_pred))
                ],
            )


def save_a2a_task(
    task_id: str,
    session_id: str | None,
    state: str,
    artifacts: list[dict[str, Any]] | None = None,
) -> None:
    """Upsert an A2A task's lifecycle state — replaces the old in-memory dict."""
    engine = get_engine()
    now = datetime.now(timezone.utc)
    with engine.begin() as conn:
        existing = conn.execute(
            a2a_tasks.select().where(a2a_tasks.c.task_id == task_id)
        ).first()
        values = dict(
            session_id=session_id,
            state=state,
            artifacts=artifacts or [],
            updated_at=now,
        )
        if existing:
            conn.execute(
                a2a_tasks.update()
                .where(a2a_tasks.c.task_id == task_id)
                .values(**values)
            )
        else:
            conn.execute(
                a2a_tasks.insert().values(task_id=task_id, created_at=now, **values)
            )


def get_a2a_task(task_id: str) -> dict[str, Any] | None:
    engine = get_engine()
    with engine.connect() as conn:
        row = conn.execute(
            a2a_tasks.select().where(a2a_tasks.c.task_id == task_id)
        ).first()
    return dict(row._mapping) if row else None


def list_a2a_tasks_for_user(user_sub: str) -> list[dict[str, Any]]:
    """Every session belonging to one user, most-recently-updated first —
    what a chat UI's sidebar needs. gateway/app.py stores the JWT subject in
    the session_id column (its own gateway-level session id is task_id —
    see save_a2a_task's docstring), so filtering by session_id == user_sub
    is filtering by user."""
    engine = get_engine()
    with engine.connect() as conn:
        rows = conn.execute(
            a2a_tasks.select()
            .where(a2a_tasks.c.session_id == user_sub)
            .order_by(a2a_tasks.c.updated_at.desc())
        ).fetchall()
    return [dict(r._mapping) for r in rows]


def delete_a2a_task(task_id: str) -> None:
    engine = get_engine()
    with engine.begin() as conn:
        conn.execute(a2a_tasks.delete().where(a2a_tasks.c.task_id == task_id))


def list_a2a_tasks_in_state(state: str) -> list[dict[str, Any]]:
    """Every session in one state across users — the gateway resumes following
    the "working" ones after a restart."""
    with get_engine().connect() as conn:
        rows = conn.execute(
            a2a_tasks.select().where(a2a_tasks.c.state == state)
        ).fetchall()
    return [dict(r._mapping) for r in rows]
